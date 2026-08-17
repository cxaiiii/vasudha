import hashlib
import os
import re
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


def _interpreter_argv() -> list:
    """Command prefix for running a generated script.

    Plain [sys.executable] from source. Inside a PyInstaller build that would
    be Vasudha.exe, which would relaunch the app instead of running the script,
    so the frozen executable exposes a script-runner mode instead — see
    app/runtime.py. Imported defensively so this module keeps working for the
    Flask app, which has no dependency on the desktop package.
    """
    try:
        from app.runtime import interpreter_argv
        return interpreter_argv()
    except ImportError:
        return [os.sys.executable]


def _strip_code_fences(text: str) -> str:
    """Strip a leading/trailing ```lang / ``` markdown fence if present."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


#: Above this, read_file refuses and says to process the file in code instead.
#: 60 KB is roughly 15k tokens — already most of an 8k window, and a file that
#: size is never something the model needs *verbatim*; it needs an answer about
#: it. Refusing here is what turns "attach a 40 MB CSV" from a context overflow
#: into a two-line pandas script.
MAX_READ_BYTES = 60_000

#: Extensions the canvas can display. Anything else is described, not shown.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}

_TABULAR = {".csv", ".tsv", ".txt", ".log", ".md", ".json", ".jsonl", ".py",
            ".yaml", ".yml", ".xml", ".html"}


def describe_file(path: Path, preview_lines: int = 12) -> str:
    """What the model is told about an attached file — never its contents.

    The whole design rests on this. A 40 MB CSV cannot enter the context and
    does not need to: what the model needs in order to write the right code is
    the shape of the thing — how big, what columns, what the first few rows
    look like. Handing it a sample and a row count costs a couple of hundred
    tokens and answers the same questions that inlining the file would, except
    it also works when the file is a gigabyte.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"{path.name} — could not be read ({exc})"

    human = (f"{size/1e9:.2f} GB" if size >= 1e9 else
             f"{size/1e6:.1f} MB" if size >= 1e6 else
             f"{size/1e3:.1f} KB" if size >= 1e3 else f"{size} bytes")
    suffix = path.suffix.lower()

    if suffix in IMAGE_SUFFIXES:
        return f"`{path.name}` — image, {human}"

    if suffix not in _TABULAR:
        return (f"`{path.name}` — {human}, binary or unrecognised type. "
                "Open it in code to find out what it holds.")

    # Counted by streaming rather than by reading the file in: the point of
    # this function is that the file never has to fit in memory, and a
    # read_text() here would defeat it on exactly the files that matter.
    try:
        lines = 0
        sample: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle):
                lines += 1
                if i < preview_lines:
                    sample.append(line.rstrip("\n")[:200])
    except OSError as exc:
        return f"`{path.name}` — {human}, unreadable ({exc})"

    header = ""
    if suffix in (".csv", ".tsv") and sample:
        delimiter = "\t" if suffix == ".tsv" else ","
        columns = sample[0].split(delimiter)
        header = f", {len(columns)} columns"

    body = "\n".join(sample)
    more = f"\n… and {lines - preview_lines:,} more lines" if lines > preview_lines else ""
    return (f"`{path.name}` — {human}, {lines:,} lines{header}\n"
            f"First {min(lines, preview_lines)} lines:\n```\n{body}\n```{more}")


def packages_dir() -> str:
    """Where pip_tool installs, and where the sandbox looks for imports.

    One directory for the whole app rather than one per chat: a package is
    expensive to fetch and identical whoever asked for it, and re-downloading
    numpy for every new conversation would be its own bug.
    """
    try:
        from app.paths import app_data_dir
        root = app_data_dir() / "packages"
    except ImportError:
        root = Path(tempfile.gettempdir()) / "vasudha_packages"
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _sandbox_env(work_dir: str) -> dict:
    """Environment for anything the model runs.

    USERPROFILE/HOME/HOMEDRIVE/HOMEPATH are deliberately pointed at the sandbox
    dir itself, not left unset: without them, os.path.expanduser("~") on Windows
    silently returns "~" unexpanded (documented behavior, not an error) instead
    of raising — code that assumes a home directory then writes to a literal "~"
    path inside the sandbox with no indication anything went wrong. Giving it a
    real but contained home means "~" resolves somewhere that exists, without
    granting access to the real user profile.

    PYTHONPATH carries the pip_tool install directory, which is what makes an
    installed package importable on the *next* python_tool call, and the
    working directory itself, which is what makes `import script` find the
    model's own file. Without the second entry the workspace is the process's
    cwd but not on the import path — Python uses the *script's* directory for
    that, and the script lives in a scratch dir — so a model that wrote
    script.py and then ran `from script import *` got an import error for code
    that was correct.
    """
    return {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "COMSPEC": os.environ.get("COMSPEC", ""),
        "PATHEXT": os.environ.get("PATHEXT", ""),
        "TEMP": work_dir,
        "TMP": work_dir,
        "USERPROFILE": work_dir,
        "HOME": work_dir,
        "HOMEDRIVE": os.path.splitdrive(work_dir)[0],
        "HOMEPATH": os.path.splitdrive(work_dir)[1],
        "PYTHONPATH": os.pathsep.join([work_dir, packages_dir()]),
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    }


#: A package name pip will accept, and nothing else. Not a general argument
#: parser: allowing arbitrary text here would let a request smuggle in
#: --index-url and fetch from somewhere other than PyPI.
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
                         r"([=<>!~]=?[A-Za-z0-9.*+!-]+)?$")


class PackageInstaller:
    """pip install, into a directory the sandbox can import from.

    Exists because the sandbox ships the standard library and little else, and
    a model that reaches for textblob or pandas otherwise spends its turns
    discovering their absence and inventing worse substitutes — which is
    exactly what one user watched it do.
    """

    def __init__(self, timeout: int = 180) -> None:
        self.timeout = timeout

    def install(self, packages: str) -> str:
        names = [p.strip() for p in re.split(r"[\s,]+", packages or "") if p.strip()]
        if not names:
            return "[error] pip_tool needs at least one package name"
        bad = [n for n in names if not _PACKAGE_RE.match(n)]
        if bad:
            return (f"[error] not valid package names: {', '.join(bad)}. "
                    "Give plain names or name==version, nothing else.")

        target = packages_dir()

        # `python -m pip`, whether or not this is a frozen build. The shipped
        # executable has no `-m`, so app/runtime.py grew a flag that runs a
        # module through runpy — and pip itself is bundled (11 MB). Installing
        # with --target means pip only downloads and unpacks into a directory,
        # which is the mode least dependent on there being a real environment
        # around it.
        try:
            from app.runtime import module_argv
            argv = module_argv() + ["pip"]
        except ImportError:
            argv = [os.sys.executable, "-m", "pip"]

        try:
            started = time.time()
            process = subprocess.run(
                argv + ["install", "--target", target,
                        "--no-input", "--disable-pip-version-check", "--quiet",
                        *names],
                capture_output=True, text=True, timeout=self.timeout,
                env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
            elapsed = time.time() - started
        except subprocess.TimeoutExpired:
            return f"[error] pip install timed out after {self.timeout}s"
        except Exception as exc:  # noqa: BLE001
            return f"[error] could not run pip: {exc}"

        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "").strip()
            if "No module named pip" in detail:
                return ("[error] this build does not include pip, so packages cannot "
                        "be installed. Use the standard library instead: math, "
                        "statistics, decimal, fractions, json, re and csv.")
            return f"[error] pip install failed:\n{detail[-1500:]}"
        return (f"[installed {', '.join(names)} in {elapsed:.1f}s] "
                "They are importable from the next python_tool call onwards.")


class ShellRunner:
    """Run a shell command in the workspace.

    Same containment as python_tool — workspace cwd, stripped environment,
    hard timeout — because it is the same risk: this app already executes
    arbitrary generated Python, so a shell is not a new category of power. It
    is bounded rather than free: no interactive programs, and a timeout that
    kills anything waiting on input it will never get.
    """

    def __init__(self, timeout: int = 60) -> None:
        self.timeout = timeout

    def run(self, command: str, cwd: Optional[str] = None) -> str:
        command = (command or "").strip()
        if not command:
            return "[error] shell_tool needs a command"

        work_dir = cwd or tempfile.mkdtemp(prefix="vasudha_shell_")
        try:
            started = time.time()
            process = subprocess.run(
                command, shell=True, cwd=work_dir, env=_sandbox_env(work_dir),
                capture_output=True, text=True, timeout=self.timeout,
                stdin=subprocess.DEVNULL)
            elapsed = time.time() - started
        except subprocess.TimeoutExpired:
            return (f"[error] command exceeded {self.timeout}s and was killed. "
                    "Interactive commands never finish here — stdin is closed.")
        except Exception as exc:  # noqa: BLE001
            return f"[error] could not run the command: {exc}"

        chunks = []
        if process.stdout.strip():
            chunks.append(process.stdout.strip())
        if process.stderr.strip():
            chunks.append(f"[stderr]\n{process.stderr.strip()}")
        body = "\n\n".join(chunks) or "(no output)"
        return f"{body}\n\n[exit {process.returncode}, {elapsed:.2f}s]"


class SandboxedCodeExecutor:
    """
    Safely executes Python code inside a sandboxed workspace. Strips away
    sensitive environment variables and enforces rigorous timeouts.
    """
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def execute_python(self, code_str: str, cwd: Optional[str] = None) -> str:
        """cwd=None (default): a fresh ephemeral tempdir, cleaned up when
        this call returns — the original one-shot-computation behavior,
        unchanged for existing callers.

        cwd=<path>: runs inside that directory instead, and does NOT clean
        it up — this is the project-creation path, where write_file wrote
        real files there and this code needs to see them (and anything it
        writes itself needs to still be there for the *next* python_tool
        call).

        The script itself is written OUTSIDE the workspace and passed by
        absolute path. It used to live at <workspace>/_exec_script.py, which
        collided with the model in a way that cost a real user four turns: a
        traceback named the file, the model tried to read_file it (already
        deleted, since each run removes it), then write_file'd its own version
        of it — and the next python_tool call silently overwrote that with the
        new snippet. Keeping the scratch file out of the workspace makes the
        collision impossible rather than merely unlikely.
        """
        clean_code = _strip_code_fences(code_str)

        scratch_dir = tempfile.mkdtemp(prefix="vasudha_exec_")
        # NOT "script.py". Python puts the running script's own directory at
        # sys.path[0], so a scratch file called script.py shadows a workspace
        # file of the same name — and `from script import *`, which is how a
        # model naturally runs the file it has been editing, imported the
        # scratch file into itself. Observed as a recursive self-import
        # reporting "NameError: name 'a_star' is not defined" from code that
        # was perfectly correct. The leading underscore and the prefix make an
        # accidental import essentially impossible.
        script_path = os.path.join(scratch_dir, "_vasudha_run.py")

        if cwd is not None:
            temp_dir = cwd          # where the code runs and writes its files
            persistent = True       # the workspace outlives this call
        else:
            temp_dir = tempfile.mkdtemp(prefix="vasudha_sandbox_")
            persistent = False

        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(clean_code)

            # Restrict environment variables to essential runtime items.
            # USERPROFILE/HOME/HOMEDRIVE/HOMEPATH are deliberately pointed at
            # the sandbox dir itself, not left unset: without them,
            # os.path.expanduser("~") on Windows silently returns "~"
            # unexpanded (documented behavior, not an error) instead of
            # raising — code that assumes a home directory then writes to a
            # literal "~" path inside the sandbox with no indication
            # anything went wrong. Giving it a real (but still contained)
            # home directory means "~" resolves to somewhere that actually
            # exists, without granting access to the real user profile.
            safe_env = _sandbox_env(temp_dir)

            start_time = time.time()
            process = subprocess.run(
                _interpreter_argv() + [script_path],
                cwd=temp_dir,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            duration = time.time() - start_time

            output_chunks = []
            if process.stdout:
                output_chunks.append(f"[Stdout]:\n{process.stdout.strip()}")
            if process.stderr:
                output_chunks.append(f"[Stderr / Errors]:\n{process.stderr.strip()}")
            
            if not output_chunks:
                return f"[Execution completed successfully in {duration:.2f}s with no terminal output]"

            return "\n\n".join(output_chunks) + f"\n\n[Finished in {duration:.2f}s]"

        except subprocess.TimeoutExpired:
            return f"[Error: Execution halted - Script exceeded maximum allowed runtime ({self.timeout}s)]"
        except Exception as e:
            return f"[System Error during execution]: {str(e)}"
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
            if not persistent:
                shutil.rmtree(temp_dir, ignore_errors=True)


def prepare_render_html(html_str: str) -> str:
    """Strip markdown fences from a <render_tool> block. No execution here —
    this is client-side only, rendered in a sandboxed iframe by the frontend,
    never run server-side."""
    return _strip_code_fences(html_str)


class WebSearcher:
    """Real web search via DuckDuckGo (ddgs) — no API key required."""
    def __init__(self, max_results: int = 5, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "[Error: empty search query]"
        try:
            results = DDGS().text(query, max_results=self.max_results)
        except Exception as e:
            return f"[Search failed]: {str(e)}"

        if not results:
            return f"[No results found for: {query}]"

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()
            href = r.get("href", "").strip()
            body = r.get("body", "").strip()
            formatted.append(f"{i}. {title}\n   {href}\n   {body}")
        return "\n\n".join(formatted)


class PageFetcher:
    """Fetches one real URL and extracts its readable text. search_tool
    alone only returns short result-page snippets, which often don't
    contain the specific fact needed (an exact version number, a precise
    date) even when the linked page does — this reads the actual page."""
    def __init__(self, timeout: int = 10, max_chars: int = 4000):
        self.timeout = timeout
        self.max_chars = max_chars

    def fetch(self, url: str) -> str:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return f"[Error: not a valid URL: {url!r}]"
        try:
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; VasudhaBot/1.0)"},
            )
            resp.raise_for_status()
        except Exception as e:
            return f"[Fetch failed]: {str(e)}"

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        if not text:
            return f"[No readable text extracted from {url}]"
        if len(text) > self.max_chars:
            return text[:self.max_chars] + " [...truncated]"
        return text


class WorkspaceManager:
    """One persistent directory per chat session — the missing piece that
    made real project creation impossible before: SandboxedCodeExecutor's
    default behavior wipes its tempdir after every single call, so a file
    written by one python_tool call was already gone by the next. A
    session_id (generated once per chat by the frontend, regenerated on
    "Reset / New Chat") maps to one workspace directory that survives
    across the whole conversation.

    Cleanup is a lazy TTL sweep on access, not a background thread or cron
    — this is a single-process dev server, not a service that needs to
    notice an idle session the moment it goes stale.
    """
    def __init__(self, root: Optional[Path] = None, ttl_seconds: int = 2 * 60 * 60):
        self.root = root or (Path(tempfile.gettempdir()) / "vasudha_workspaces")
        self.ttl_seconds = ttl_seconds
        self.root.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        # session_id comes straight from the client — never trust it as a
        # raw path component (a malicious "../../etc" session_id would
        # otherwise resolve outside self.root entirely). Hashing it always
        # produces a safe, flat directory name regardless of input.
        safe_name = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:32]
        return self.root / safe_name

    def get(self, session_id: str) -> Path:
        """Returns this session's workspace, creating it if new. Touches
        the directory's mtime so the TTL sweep treats it as active."""
        self._sweep_stale()
        path = self._session_dir(session_id)
        path.mkdir(parents=True, exist_ok=True)
        os.utime(path, None)
        return path

    def _sweep_stale(self) -> None:
        now = time.time()
        try:
            for child in self.root.iterdir():
                if child.is_dir():
                    try:
                        if (now - child.stat().st_mtime) > self.ttl_seconds:
                            shutil.rmtree(child, ignore_errors=True)
                    except FileNotFoundError:
                        continue
        except FileNotFoundError:
            pass


class WorkspaceFileTools:
    """write_file / read_file / list_directory scoped to one workspace
    directory (real files, real filesystem — nothing simulated). Every
    path is resolved relative to the workspace root and verified to still
    land inside it after resolution: a rejected write is returned to the
    model as a visible error it can react to, not silently clamped or
    silently written somewhere unexpected.
    """
    MAX_FILE_BYTES = 512 * 1024   # plenty for source files; not a dumping ground
    MAX_FILES = 200                # guards against a runaway loop spraying the disk

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()

    def _resolve(self, rel_path: str) -> Path:
        rel_path = (rel_path or "").strip().lstrip("/\\")
        if not rel_path:
            raise ValueError("empty path")
        candidate = (self.workspace / rel_path).resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"path {rel_path!r} escapes the workspace — not allowed")
        return candidate

    def write_file(self, rel_path: str, content: str) -> str:
        try:
            target = self._resolve(rel_path)
        except ValueError as e:
            return f"[Error: {e}]"
        if len(content.encode("utf-8")) > self.MAX_FILE_BYTES:
            return f"[Error: content exceeds the {self.MAX_FILE_BYTES}-byte file limit]"
        if not target.exists():
            existing_files = sum(1 for p in self.workspace.rglob("*") if p.is_file())
            if existing_files >= self.MAX_FILES:
                return f"[Error: workspace file limit ({self.MAX_FILES}) reached]"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            return f"[Error writing {rel_path}]: {e}"
        return f"[Wrote {len(content)} chars to {rel_path}]"

    def read_file(self, rel_path: str) -> str:
        try:
            target = self._resolve(rel_path)
        except ValueError as e:
            return f"[Error: {e}]"
        if not target.exists():
            return f"[Error: {rel_path} does not exist]"
        if not target.is_file():
            return f"[Error: {rel_path} is not a file]"

        # A large file is refused rather than truncated. Truncation looks like
        # success: the model reads the first 60 KB of a 40 MB CSV, sees plausible
        # rows, and answers about 0.15% of the data without knowing it. Describing
        # the file and pointing at code is the only honest response, and it is
        # also the one that actually works.
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            return (f"[{rel_path} is {size/1e6:.1f} MB — too large to read into the "
                    f"conversation, and reading part of it would answer about a "
                    f"fraction of the data without saying so.\n\n"
                    f"{describe_file(target)}\n\n"
                    "Process it with python_tool instead — open it, iterate, and "
                    "print only the result you need.]")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return (f"[Error: {rel_path} is not text. {describe_file(target)}]")

    def list_directory(self, rel_path: str = ".") -> str:
        if rel_path in ("", ".", "/", "\\", None):
            target = self.workspace
        else:
            try:
                target = self._resolve(rel_path)
            except ValueError as e:
                return f"[Error: {e}]"
        if not target.exists():
            return f"[Error: {rel_path} does not exist]"
        entries = []
        for p in sorted(target.rglob("*")):
            # No longer skips "_exec_script.py". The executor used to write its
            # scratch script there and this hid it; the script now lives outside
            # the workspace entirely, so a file with that name is the model's
            # own and hiding it would make write_file look like it silently
            # failed.
            kind = "dir " if p.is_dir() else "file"
            entries.append(f"{kind}  {p.relative_to(self.workspace)}")
        if not entries:
            return "[workspace is empty]"
        return "\n".join(entries)
