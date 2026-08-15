"""Vasudha desktop — pywebview shell + JS bridge.

Why pywebview and not Electron/Tauri:
  * Electron would ship a second Chromium (~120-150 MB) to render 950 lines of
    HTML. WebView2 is already present on Win10/11.
  * Tauri needs a Rust toolchain, which this build machine does not have.
  * pywebview keeps the whole app in one Python process, which matters because
    python_tool executes generated code with sys.executable — the sandbox that
    took the model from 0/6 to 4/6 on scripts/bench_numeric.py needs a real
    interpreter present. A "native" app without Python would ship the 0/6
    version in a nicer window.

There is no HTTP server here. The UI reaches Python through pywebview's js_api
bridge, so nothing binds a port and nothing is reachable from the network — the
privacy claim is structural, not a promise.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Must happen before webview is imported: when the frozen executable is invoked
# as a script runner by the sandbox, it has to behave like a plain interpreter
# and never initialise a GUI. See app/runtime.py.
from app.runtime import maybe_run_as_interpreter  # noqa: E402

maybe_run_as_interpreter()

import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from typing import Optional  # noqa: E402

import webview  # noqa: E402

from app.backends import Backend, LlamaCppBackend, OllamaBackend, select_backend  # noqa: E402
from app.bootstrap import ModelStore, DownloadProgress, app_data_dir  # noqa: E402
from app.history import Chat, ChatStore  # noqa: E402
from app import personas  # noqa: E402
from app.session import ChatSession, CORE_RULES  # noqa: E402
from app.settings import Settings, SettingsStore  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vasudha.app")

def _find_ui_dir() -> Path:
    """Locate the UI assets from source or from inside a frozen bundle.

    Candidates are tried in order and the first that actually contains
    index.html wins, rather than trusting one layout: getting this wrong is
    invisible until runtime, where it surfaces as a 404 page instead of the
    app. See the note in packaging/Vasudha.spec about pywebview serving its
    HTTP root from sys._MEIPASS.
    """
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates += [Path(meipass) / "ui", Path(meipass) / "app" / "ui"]
    candidates.append(Path(__file__).resolve().parent / "ui")

    for candidate in candidates:
        if (candidate / "index.html").exists():
            return candidate

    # Nothing found: return the source-tree path so the failure names a real
    # location instead of an empty one.
    return Path(__file__).resolve().parent / "ui"


UI_DIR = _find_ui_dir()


class Api:
    """Everything the front end may call. Method names are the JS API surface."""

    def __init__(self) -> None:
        self._window: Optional[webview.Window] = None
        self._store = ModelStore()
        self._chats = ChatStore()
        self._chat = Chat()
        self._settings_store = SettingsStore()
        self._settings = self._settings_store.load()
        self._backend: Optional[Backend] = None
        self._session: Optional[ChatSession] = None
        self._maximised = False
        self._needs_setup = False
        self._lock = threading.Lock()

    # -- plumbing ----------------------------------------------------------

    def _emit(self, payload: dict) -> None:
        """Push one event into the page. Serialised via json.dumps so a stray
        quote in tool output cannot break out of the JS string."""
        if not self._window:
            return
        try:
            self._window.evaluate_js(f"window.vasudha.onEvent({json.dumps(payload)})")
        except Exception:  # noqa: BLE001 - the window may be closing
            logger.debug("emit failed (window gone?)", exc_info=True)

    def _call_js(self, fn: str, payload: dict) -> None:
        if not self._window:
            return
        try:
            self._window.evaluate_js(f"window.vasudha.{fn}({json.dumps(payload)})")
        except Exception:  # noqa: BLE001
            logger.debug("call_js %s failed", fn, exc_info=True)

    # -- lifecycle ---------------------------------------------------------

    def prepare(self) -> None:
        """Select and load a backend. Runs behind the splash, before the main
        window is shown, so the chat is answerable the moment it appears.

        Touches no JS: the main window's page has not loaded yet.
        """
        configured = self._settings.model_path or None
        model_path = Path(configured) if configured and Path(configured).exists() else None
        if model_path is None:
            model_path = self._store.installed_model()

        prefer = None if self._settings.backend == "auto" else self._settings.backend
        if not model_path and not OllamaBackend.probe(self._settings.ollama_url):
            self._needs_setup = True
            return

        try:
            self._backend = select_backend(
                str(model_path) if model_path else None,
                ollama_model_hint=self._settings.ollama_model or "vasudha",
                prefer=prefer)
        except RuntimeError:
            self._needs_setup = True
            return

        self._session = ChatSession(
            self._backend,
            system_prompt=personas.build_system_prompt(self._settings.persona, CORE_RULES))
        self._apply_settings()
        self._needs_setup = False

    def ready(self) -> None:
        """Called from the page once it has loaded. Publishes state that
        prepare() already worked out."""
        self._publish()

    def _boot(self) -> None:
        """Re-run selection after something changed (a download finished, a
        model was chosen), then publish. Safe to call from a worker thread."""
        self.prepare()
        self._publish()

    def _publish(self) -> None:
        self._call_js("onSettings", asdict(self._settings))
        self._call_js("setDataPath", str(app_data_dir()))

        if self._needs_setup or not self._backend:
            self._call_js("showFirstRun", True)
            return

        # Persona choice comes after the engine is up: it is a preference, and
        # asking for it while the app might still be unusable would be rude.
        if not self._settings.onboarded:
            self._call_js("showOnboarding", {"personas": personas.as_cards()})

        fast = isinstance(self._backend, OllamaBackend) or (
            isinstance(self._backend, LlamaCppBackend) and "GPU" in self._backend.display_name)
        self._call_js("onBackend", {
            "label": self._backend.display_name,
            "fast": fast,
            "detail": ("Running on your GPU." if fast else
                       "Running on CPU — answers take longer. Installing Ollama "
                       "would use your graphics card instead."),
        })
        self._call_js("showFirstRun", False)
        self._push_chat_list()

    # -- window controls ---------------------------------------------------

    def minimise(self) -> None:
        if self._window:
            self._window.minimize()

    def close(self) -> None:
        if self._window:
            self._window.destroy()

    def get_bounds(self) -> dict:
        """Current position and size, so a resize drag can be computed against
        a known origin rather than accumulating rounding drift."""
        if not self._window:
            return {"x": 0, "y": 0, "width": 1180, "height": 780}
        return {"x": self._window.x, "y": self._window.y,
                "width": self._window.width, "height": self._window.height}

    def set_bounds(self, x: int, y: int, width: int, height: int) -> None:
        if not self._window:
            return
        # move before resize: growing from the top-left edge otherwise shows a
        # frame at the old origin for one paint, which reads as a flicker.
        try:
            self._window.move(int(x), int(y))
            self._window.resize(int(width), int(height))
        except Exception:  # noqa: BLE001 - a drag past a screen edge can throw
            logger.debug("set_bounds failed", exc_info=True)

    def toggle_maximise(self) -> None:
        if not self._window:
            return
        self._maximised = not getattr(self, "_maximised", False)
        if self._maximised:
            self._window.maximize()
        else:
            self._window.restore()

    # -- settings ----------------------------------------------------------

    def get_settings(self) -> dict:
        return asdict(self._settings)

    def choose_persona(self, key: str) -> dict:
        """Record the first-run choice and apply it to the live session."""
        self._settings.persona = personas.get(key).key
        self._settings.onboarded = True
        self._settings_store.save(self._settings)
        if self._session:
            self._session.system_prompt = personas.build_system_prompt(
                self._settings.persona, CORE_RULES)
        return asdict(self._settings)

    def list_personas(self) -> list:
        return personas.as_cards()

    def save_settings(self, values: dict) -> dict:
        merged = asdict(self._settings)
        merged.update({k: v for k, v in (values or {}).items() if k in merged})
        self._settings = Settings(**merged).clamp()
        self._settings_store.save(self._settings)
        self._apply_settings()
        return asdict(self._settings)

    def reset_settings(self) -> dict:
        self._settings = Settings()
        self._settings_store.save(self._settings)
        self._apply_settings()
        return asdict(self._settings)

    def _apply_settings(self) -> None:
        """Push live-changeable values into the running session. Backend and
        model path are not applied here — switching engines means reloading a
        2.3 GB model, so it takes effect on the next launch rather than
        stalling the window mid-conversation."""
        if self._session:
            self._session.options = self._settings.generation_options()
            self._session.max_iterations = self._settings.max_iterations
            self._session.set_tool_timeout(self._settings.tool_timeout)
            self._session.system_prompt = personas.build_system_prompt(
                self._settings.persona, CORE_RULES)

    def save_document(self, doc: dict) -> str:
        """Save-as for a canvas document. Returns the chosen path, or ""."""
        if not self._window or not doc:
            return ""
        suggested = doc.get("filename") or "document.md"
        chosen = self._window.create_file_dialog(
            webview.SAVE_DIALOG, save_filename=suggested)
        if not chosen:
            return ""
        target = Path(chosen if isinstance(chosen, str) else chosen[0])
        try:
            target.write_text(doc.get("content", ""), encoding="utf-8")
        except OSError as exc:
            logger.exception("could not save document")
            self._emit({"kind": "error", "text": f"Could not save: {exc}"})
            return ""
        return str(target)

    def open_data_folder(self) -> None:
        self._open_folder(app_data_dir())

    def open_personas_folder(self) -> None:
        """Seed the folder first, so it is never opened empty on a fresh install."""
        self._open_folder(personas.ensure_installed())

    @staticmethod
    def _open_folder(path: Path) -> None:
        from app.paths import open_folder
        open_folder(path)

    def clear_history(self) -> None:
        for row in self._chats.list_summaries(limit=10_000):
            self._chats.delete(row["id"])
        self._chat = Chat()
        self._call_js("loadChat", {"events": []})
        self._push_chat_list()

    # -- first run ---------------------------------------------------------

    def start_download(self) -> None:
        threading.Thread(target=self._download, daemon=True).start()

    def _download(self) -> None:
        def on_progress(p: DownloadProgress) -> None:
            self._call_js("onDownloadProgress", asdict(p))

        try:
            self._store.download(on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001 - show the user anything that goes wrong
            logger.exception("model download failed")
            # Onto the first-run screen, not the chat: the chat is behind this
            # overlay, so an error written there is invisible and the button
            # simply looks dead.
            self._call_js("onDownloadError", {"message": str(exc)})
            return
        self._boot()

    def locate_model(self) -> str:
        """Let the user point at a GGUF they already have rather than
        re-downloading 2.3 GB they may already be storing for ollama.

        Returns the chosen path so the settings field can show it immediately.
        """
        if not self._window:
            return ""
        chosen = self._window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False,
            file_types=("GGUF model (*.gguf)", "All files (*.*)"))
        if not chosen:
            return ""
        path = Path(chosen[0])
        self._store.adopt(path)
        self._settings.model_path = str(path)
        self._settings_store.save(self._settings)
        threading.Thread(target=self._boot, daemon=True).start()
        return str(path)

    # -- chat --------------------------------------------------------------

    def ask(self, text: str) -> None:
        """Run one turn. Blocking work happens on a worker thread so the bridge
        call returns immediately and the UI keeps painting."""
        threading.Thread(target=self._ask, args=(text,), daemon=True).start()

    def _ask(self, text: str) -> None:
        if not self._session:
            self._emit({"kind": "error", "text": "No model loaded yet."})
            self._emit({"kind": "done"})
            return

        with self._lock:
            self._chats.title_for(self._chat, text)
            self._chat.events.append({"kind": "user", "text": text})
            try:
                for event in self._session.ask(text):
                    self._chat.events.append(event)
                    self._emit(event)
            except Exception as exc:  # noqa: BLE001 - never kill the UI thread
                logger.exception("turn failed")
                self._emit({"kind": "error", "text": str(exc)})
            finally:
                # Save even on failure: a turn that errored halfway is still
                # part of the conversation, and losing the user's question
                # because the model fell over would be its own bug.
                self._chat.messages = list(self._session.history)
                try:
                    self._chats.save(self._chat)
                except OSError:
                    logger.exception("could not persist chat %s", self._chat.id)
                self._emit({"kind": "done"})
                self._push_chat_list()

    def _push_chat_list(self) -> None:
        rows = self._chats.list_summaries()
        for row in rows:
            row["active"] = row["id"] == self._chat.id
        self._call_js("setChats", rows)

    def new_chat(self) -> None:
        self._chat = Chat()
        if self._session:
            self._session.reset()
        self._call_js("loadChat", {"events": []})
        self._push_chat_list()

    def open_chat(self, chat_id: str) -> None:
        chat = self._chats.load(chat_id)
        if chat is None:
            return
        self._chat = chat
        if self._session:
            self._session.history = list(chat.messages)
        self._call_js("loadChat", {"events": chat.events})
        self._push_chat_list()


#: Minimum time the splash stays up. Boot is often faster than this, but a
#: window that flashes for 300ms reads as a glitch rather than a launch. The
#: ceiling is comfort, not necessity — if the engine is slower than this the
#: splash simply stays up longer, it never cuts away from unfinished work.
SPLASH_MIN_SECONDS = 3.5


def main() -> int:
    api = Api()

    splash = webview.create_window(
        "Vasudha",
        str(UI_DIR / "splash.html"),
        width=520, height=400,
        frameless=True, easy_drag=False,
        on_top=True, resizable=False,
        background_color="#FDF0E4",
    )

    window = webview.create_window(
        "Vasudha",
        str(UI_DIR / "index.html"),
        js_api=api,
        width=1180,
        height=780,
        min_size=(900, 600),
        frameless=True,
        easy_drag=False,          # dragging is scoped to the titlebar strip
        background_color="#FCE0CC",
        hidden=True,              # revealed once the engine is up
    )
    api._window = window

    def boot() -> None:
        """Warm the engine behind the splash, then swap windows.

        The work is done here rather than after the main window appears so the
        user never sees an empty chat that cannot yet answer.
        """
        started = time.time()

        def say(text: str) -> None:
            try:
                splash.evaluate_js(f"window.splash.setStatus({json.dumps(text)})")
            except Exception:  # noqa: BLE001 - splash may already be gone
                pass

        try:
            say("Looking for a model…")
            api.prepare()
            say("Ready")
        except Exception:  # noqa: BLE001 - never strand the user on the splash
            logger.exception("boot failed")
            say("Starting anyway…")

        remaining = SPLASH_MIN_SECONDS - (time.time() - started)
        if remaining > 0:
            time.sleep(remaining)

        window.show()
        try:
            splash.destroy()
        except Exception:  # noqa: BLE001
            logger.debug("splash already closed", exc_info=True)

    webview.start(boot, debug=bool(os.environ.get("VASUDHA_DEBUG")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
