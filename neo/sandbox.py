"""A fast, isolated Python sandbox for rollouts.

Why not a fresh `python script.py` per call? That costs 100-300 ms once
numpy/scipy/sympy are imported, and TRL runs a batch's tool calls one after
another. At 256 rollouts x ~1.5 calls per GRPO step that is most of a minute of
idle H100 per step, i.e. real money.

So each worker here is a long-lived "zygote" process that imports the heavy
modules once and then fork()s a child per execution. A child starts in ~1 ms
with numpy already loaded, runs the code under resource limits, and exits.

Isolation, given that this runs model-written code next to the checkpoints:
  * every child calls setsid() and is killed as a process group on timeout;
  * when started as root, children drop to the `nobody` user before running
    anything, so they cannot write to the Modal Volume or the repo;
  * RLIMIT_AS / RLIMIT_CPU / RLIMIT_FSIZE / RLIMIT_NPROC bound memory, CPU,
    file size and fork bombs; stdin is /dev/null.
BLAS is pinned to one thread before numpy is imported: forking a process whose
OpenBLAS thread pool is already running can deadlock the child.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional

DEFAULT_TIMEOUT = 15
MAX_OUTPUT_CHARS = 4000
PRELOAD = (
    "math", "cmath", "fractions", "decimal", "statistics", "itertools", "functools", "collections",
    "json", "re", "random", "datetime", "numpy", "scipy", "scipy.optimize", "scipy.integrate",
    "scipy.constants", "scipy.special", "scipy.stats", "sympy",
)


def strip_code_fences(text: str) -> str:
    """Strip a leading ```lang and trailing ``` fence if the model added one."""
    lines = (text or "").strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def format_result(stdout: str, stderr: str, duration: float, timed_out: bool, timeout: float,
                  max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """What the model sees: plain stdout, then stderr, like a terminal or a
    notebook cell. Deliberately harness-neutral."""
    if timed_out:
        partial = _clip(stdout.strip(), max_chars // 2) if stdout and stdout.strip() else ""
        msg = f"TimeoutError: execution exceeded {timeout:g} seconds"
        return f"{partial}\n{msg}" if partial else msg
    parts = []
    if stdout and stdout.strip():
        parts.append(_clip(stdout.rstrip(), max_chars))
    if stderr and stderr.strip():
        parts.append(_clip(stderr.rstrip(), max_chars))
    return "\n".join(parts) if parts else "(no output)"


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2:]
    return f"{head}\n... [output truncated: {len(text) - max_chars} characters omitted] ...\n{tail}"


# ---------------------------------------------------------------------------
# Zygote (runs in its own process)
# ---------------------------------------------------------------------------

def _child_run(code: str, workdir: str, timeout: float, drop_to: Optional[int]) -> None:
    """Runs inside the forked child. Never returns."""
    import resource
    import traceback

    try:
        os.setsid()
        os.chdir(workdir)
        out_fd = os.open(os.path.join(workdir, ".stdout"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
        err_fd = os.open(os.path.join(workdir, ".stderr"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.dup2(out_fd, 1)
        os.dup2(err_fd, 2)
        sys.stdin = open(0, "r", closefd=False)
        sys.stdout = open(1, "w", buffering=1, closefd=False)
        sys.stderr = open(2, "w", buffering=1, closefd=False)

        def limit(kind, value):
            try:
                resource.setrlimit(kind, (value, value))
            except (ValueError, OSError):
                pass

        limit(resource.RLIMIT_CPU, int(timeout) + 2)
        limit(resource.RLIMIT_FSIZE, 64 * 1024 * 1024)
        limit(resource.RLIMIT_AS, 8 * 1024 ** 3)
        if drop_to is not None:
            os.setgroups([])
            os.setgid(drop_to)
            os.setuid(drop_to)
            limit(resource.RLIMIT_NPROC, 256)
        os.environ.update({"HOME": workdir, "TMPDIR": workdir, "TEMP": workdir, "TMP": workdir,
                           "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    except Exception as exc:  # noqa: BLE001 - report setup failure as a script error
        os.write(2, f"[sandbox setup failed: {exc}]".encode())
        os._exit(1)

    status = 0
    glb = {"__name__": "__main__", "__builtins__": __builtins__, "__file__": os.path.join(workdir, "exec_script.py")}
    try:
        import linecache
        # Tracebacks then show source lines, as a real `python exec_script.py` run does.
        linecache.cache["exec_script.py"] = (len(code), None, code.splitlines(True), "exec_script.py")
        compiled = compile(code, "exec_script.py", "exec")
        exec(compiled, glb)  # noqa: S102 - this is the sandbox
    except SystemExit as exc:
        if exc.code not in (None, 0):
            if not isinstance(exc.code, int):
                print(exc.code, file=sys.stderr)
            status = exc.code if isinstance(exc.code, int) else 1
    except BaseException as exc:  # noqa: BLE001
        tb = exc.__traceback__
        # Hide the sandbox's own frame so the traceback reads like a plain
        # `python exec_script.py` run.
        while tb is not None and tb.tb_frame.f_code.co_filename != "exec_script.py":
            tb = tb.tb_next
        sys.stderr.write("Traceback (most recent call last):\n" if tb is not None else "")
        sys.stderr.write("".join(traceback.format_tb(tb)) if tb is not None else "")
        sys.stderr.write("".join(traceback.format_exception_only(type(exc), exc)))
        status = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(status)


def _nobody_uid() -> Optional[int]:
    if os.geteuid() != 0 or os.environ.get("NEO_SANDBOX_NO_DROP") == "1":
        return None
    try:
        import pwd
        return pwd.getpwnam("nobody").pw_uid
    except (KeyError, ImportError):
        return 65534


def _can_run_as(uid: int) -> bool:
    """Can an unprivileged child still read the interpreter's packages?

    Dropping to `nobody` is pointless if the site-packages tree is not world
    readable (lazy imports then fail inside model code, which would look like
    model errors and poison the reward). Check once, in a throwaway child.
    """
    import sysconfig

    paths = {sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib"), sys.prefix}
    pid = os.fork()
    if pid == 0:
        try:
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)
            ok = all(os.access(p, os.R_OK | os.X_OK) for p in paths if p)
            os._exit(0 if ok else 1)
        except BaseException:  # noqa: BLE001
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def serve(preload: Iterable[str] = PRELOAD) -> None:
    """Zygote loop: one JSON request per stdin line, one JSON reply per line."""
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    import importlib
    import shutil
    import signal
    import tempfile

    for name in preload:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - a missing optional module is fine
            pass
    drop_to = _nobody_uid()
    if drop_to is not None and not _can_run_as(drop_to):
        sys.stderr.write("neo.sandbox: site-packages not readable by nobody; running without privilege drop\n")
        drop_to = None
    base = tempfile.mkdtemp(prefix="neo_sandbox_")
    os.chmod(base, 0o711)
    out = sys.stdout
    out.write(json.dumps({"ready": True}) + "\n")
    out.flush()

    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        code = strip_code_fences(req.get("code", ""))
        timeout = float(req.get("timeout", DEFAULT_TIMEOUT))
        max_chars = int(req.get("max_chars", MAX_OUTPUT_CHARS))
        workdir = tempfile.mkdtemp(dir=base)
        os.chmod(workdir, 0o777)
        started = time.perf_counter()
        pid = os.fork()
        if pid == 0:
            _child_run(code, workdir, timeout, drop_to)
        timed_out = False
        while True:
            done, _status = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.perf_counter() - started > timeout:
                timed_out = True
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.waitpid(pid, 0)
                break
            time.sleep(0.002)
        duration = time.perf_counter() - started

        def read(name):
            try:
                with open(os.path.join(workdir, name), "r", encoding="utf-8", errors="replace") as fh:
                    return fh.read(1_000_000)
            except OSError:
                return ""

        stdout, stderr = read(".stdout"), read(".stderr")
        shutil.rmtree(workdir, ignore_errors=True)
        result = format_result(stdout, stderr, duration, timed_out, timeout, max_chars)
        out.write(json.dumps({"result": result, "stdout": stdout[:max_chars], "stderr": stderr[:max_chars],
                              "timed_out": timed_out, "duration": duration}) + "\n")
        out.flush()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class _Worker:
    def __init__(self, preload: Iterable[str]):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                          env.get("PYTHONPATH", "")]))
        env["NEO_SANDBOX_PRELOAD"] = ",".join(preload)
        # A clean interpreter: never inherit CUDA state or the trainer's threads.
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "neo.sandbox", "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env,
        )
        ready = self.proc.stdout.readline()
        if not ready or not json.loads(ready).get("ready"):
            raise RuntimeError("sandbox worker failed to start")

    def run(self, code: str, timeout: float, max_chars: int) -> dict:
        self.proc.stdin.write(json.dumps({"code": code, "timeout": timeout, "max_chars": max_chars}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("sandbox worker died")
        return json.loads(line)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


class SandboxPool:
    """Thread-safe pool of zygote workers.

    `run(code)` returns the formatted result string; `run_detailed(code)`
    returns the raw dict (stdout, stderr, timed_out, duration) as well.
    """

    def __init__(self, workers: int = 4, timeout: float = DEFAULT_TIMEOUT,
                 max_chars: int = MAX_OUTPUT_CHARS, preload: Iterable[str] = PRELOAD):
        self.timeout = timeout
        self.max_chars = max_chars
        self.preload = tuple(preload)
        self._idle: "queue.Queue[_Worker]" = queue.Queue()
        self._all: list[_Worker] = []
        self._lock = threading.Lock()
        for _ in range(max(1, workers)):
            self._spawn()

    def _spawn(self) -> None:
        worker = _Worker(self.preload)
        with self._lock:
            self._all.append(worker)
        self._idle.put(worker)

    def run_detailed(self, code: str, timeout: Optional[float] = None) -> dict:
        worker = self._idle.get()
        try:
            try:
                return worker.run(code, timeout or self.timeout, self.max_chars)
            except (RuntimeError, BrokenPipeError, OSError, json.JSONDecodeError):
                # The worker itself died (should not happen: code runs in a
                # child). Replace it and report the failure to the model.
                with self._lock:
                    if worker in self._all:
                        self._all.remove(worker)
                worker.close()
                worker = _Worker(self.preload)
                with self._lock:
                    self._all.append(worker)
                return {"result": "SystemError: sandbox worker restarted", "stdout": "",
                        "stderr": "", "timed_out": False, "duration": 0.0}
        finally:
            self._idle.put(worker)

    def run(self, code: str, timeout: Optional[float] = None) -> str:
        return self.run_detailed(code, timeout)["result"]

    def run_many(self, codes: list[str], timeout: Optional[float] = None) -> list[dict]:
        if not codes:
            return []
        with ThreadPoolExecutor(max_workers=max(1, len(self._all))) as ex:
            return list(ex.map(lambda c: self.run_detailed(c, timeout), codes))

    def close(self) -> None:
        with self._lock:
            workers, self._all = self._all, []
        for w in workers:
            w.close()


_DEFAULT_POOL: Optional[SandboxPool] = None
_DEFAULT_LOCK = threading.Lock()


def default_pool(workers: Optional[int] = None) -> SandboxPool:
    """Process-wide pool, created on first use."""
    global _DEFAULT_POOL
    with _DEFAULT_LOCK:
        if _DEFAULT_POOL is None:
            n = workers or int(os.environ.get("NEO_SANDBOX_WORKERS", "4"))
            _DEFAULT_POOL = SandboxPool(workers=n)
        return _DEFAULT_POOL


if __name__ == "__main__":
    if "--serve" in sys.argv:
        names = os.environ.get("NEO_SANDBOX_PRELOAD")
        serve(names.split(",") if names else PRELOAD)
