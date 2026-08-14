import hashlib
import os
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
        call). Cleanup for that case is WorkspaceManager's TTL sweep, not
        per-call, since the whole point is persistence across calls."""
        clean_code = _strip_code_fences(code_str)

        if cwd is not None:
            temp_dir = cwd
            # Leading underscore keeps this out of the way of whatever
            # filenames the model actually chose for the project, and
            # signals "generated scratch file" if the model lists the dir.
            script_path = os.path.join(temp_dir, "_exec_script.py")
            persistent = True
        else:
            temp_dir = tempfile.mkdtemp(prefix="vasudha_sandbox_")
            script_path = os.path.join(temp_dir, "exec_script.py")
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
            safe_env = {
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "TEMP": temp_dir,
                "TMP": temp_dir,
                "USERPROFILE": temp_dir,
                "HOME": temp_dir,
                "HOMEDRIVE": os.path.splitdrive(temp_dir)[0],
                "HOMEPATH": os.path.splitdrive(temp_dir)[1],
                "PYTHONUNBUFFERED": "1"
            }

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
            if persistent:
                # Only remove the scratch script — everything else in the
                # workspace is the project itself and must survive for the
                # next tool call.
                try:
                    os.remove(script_path)
                except OSError:
                    pass
            else:
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
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"[Error: {rel_path} is not a readable text file]"

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
            if p.name == "_exec_script.py":
                continue  # the executor's own scratch file, not part of the project
            kind = "dir " if p.is_dir() else "file"
            entries.append(f"{kind}  {p.relative_to(self.workspace)}")
        if not entries:
            return "[workspace is empty]"
        return "\n".join(entries)
