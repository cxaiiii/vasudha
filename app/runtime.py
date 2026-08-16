"""Running Python from inside a frozen build.

The sandbox executes generated code with `subprocess.run([sys.executable, ...])`.
That is correct from source, where sys.executable is python.exe. Inside a
PyInstaller bundle sys.executable is Vasudha.exe, so the same call would launch
a second copy of the whole application instead of running the script — and
python_tool is the difference between 4/6 and 0/6 on scripts/bench_numeric.py.
Shipping that would mean shipping the broken version.

Rather than bundling a second Python (an embeddable distribution is ~15 MB and
another thing to keep in step), the frozen executable doubles as its own
interpreter: invoked with the sentinel flag below it runs the given script and
exits, reusing the Python runtime already inside the bundle.

Trade-off worth knowing: code run this way sees only modules bundled into the
executable. The standard library is there (math, statistics, decimal,
fractions, json), so engineering arithmetic is fine, but numpy and scipy are
not present unless they are explicitly bundled in the .spec.
"""
from __future__ import annotations

import os
import runpy
import sys

#: Argument that turns the frozen executable into a plain script runner.
RUN_SCRIPT_FLAG = "--vasudha-exec-script"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def interpreter_argv() -> list[str]:
    """Command prefix that runs a Python script file passed as the next arg."""
    if is_frozen():
        return [sys.executable, RUN_SCRIPT_FLAG]
    return [sys.executable]


def _bind_std_streams() -> None:
    """Point sys.stdout/stderr at the real file descriptors.

    A windowed PyInstaller build (console=False) starts with sys.stdout and
    sys.stderr as None or a null writer, because a GUI-subsystem process has no
    console. But this process was launched by the sandbox with pipes, so fds 1
    and 2 are perfectly valid — only Python's wrappers are missing.

    Without this, every print() in generated code goes nowhere, capture_output
    returns empty, and python_tool silently produces no output: the shipped app
    would look like it ran your code and computed nothing. Rebinding costs
    nothing and makes console=False safe.
    """
    for fd, name, mode in ((1, "stdout", "w"), (2, "stderr", "w")):
        stream = getattr(sys, name, None)
        if stream is not None and hasattr(stream, "fileno"):
            try:
                stream.fileno()
                continue  # already wired to a real descriptor
            except (OSError, ValueError):
                pass
        try:
            setattr(sys, name, os.fdopen(fd, mode, buffering=1,
                                         encoding="utf-8", errors="replace",
                                         closefd=False))
        except OSError:
            # No usable descriptor (not launched with pipes). Leave as-is
            # rather than crashing the script before it starts.
            pass


def clear_mark_of_the_web() -> None:
    """Strip the "downloaded from the internet" flag from the bundle's own files.

    Windows Explorer stamps every file it extracts from a downloaded .zip with a
    Zone.Identifier alternate data stream (ZoneId=3). The .NET Framework then
    refuses to resolve entry points in a managed assembly carrying that stream,
    so pywebview's WinForms/EdgeChromium backend dies during import with:

        RuntimeError: Failed to resolve Python.Runtime.Loader.Initialize
        from ...\\pythonnet\\runtime\\Python.Runtime.dll

    which is a crash on launch, before any window appears, for every user who
    unzips the release the ordinary way. Measured on a real download of the CI
    artefact: 306 of the bundle's files were flagged. 7-Zip does not propagate
    the zone, which is exactly why this survives testing on a developer machine
    that has 7-Zip installed and a local build that was never zipped at all.

    Removing an ADS is deleting the "file:stream" path. Cheap — it touches no
    file contents and rewrites nothing — so it runs unconditionally on a frozen
    Windows build rather than trying to detect the failure first.

    The real fix is an Authenticode signature, which exempts the binary from
    this entirely. Until there is a certificate to sign with, this is the
    difference between an app that starts and one that does not.
    """
    if not is_frozen() or sys.platform != "win32":
        return
    root = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    for directory, _subdirs, files in os.walk(root):
        for name in files:
            try:
                os.remove(os.path.join(directory, name) + ":Zone.Identifier")
            except OSError:
                # Not flagged, already gone, or locked — all normal. This is a
                # best-effort repair and must never keep the app from starting.
                pass
    # The executable itself sits beside _internal, outside _MEIPASS.
    try:
        os.remove(sys.executable + ":Zone.Identifier")
    except OSError:
        pass


def maybe_run_as_interpreter() -> None:
    """Call first thing in the frozen entry point.

    If the sentinel flag is present, run the requested script and exit without
    ever creating a window. Must run before any GUI import so a sandboxed
    execution never flashes a window or initialises WebView2.
    """
    if len(sys.argv) >= 3 and sys.argv[1] == RUN_SCRIPT_FLAG:
        script = sys.argv[2]
        _bind_std_streams()
        # Present the script's own argv, as a real interpreter would.
        sys.argv = [script] + sys.argv[3:]
        code = 0
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        except BaseException:  # noqa: BLE001 - mimic an interpreter: report and exit 1
            import traceback
            traceback.print_exc()
            code = 1
        finally:
            for name in ("stdout", "stderr"):
                stream = getattr(sys, name, None)
                if stream is not None:
                    try:
                        stream.flush()
                    except (OSError, ValueError):
                        pass
        # os._exit avoids interpreter teardown, which in a frozen build can
        # re-enter PyInstaller's atexit handling and hang the child process.
        os._exit(code)
