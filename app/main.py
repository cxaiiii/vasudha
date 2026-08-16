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
from app.runtime import clear_mark_of_the_web, maybe_run_as_interpreter  # noqa: E402

maybe_run_as_interpreter()

# Before `import webview` below, which is where a downloaded-and-unzipped build
# crashes: .NET will not load pythonnet's assembly while Windows has it flagged
# as internet-sourced. See clear_mark_of_the_web for the full failure.
clear_mark_of_the_web()

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
from app.memory import InteractionLog, MemoryBook  # noqa: E402
from app import personas  # noqa: E402
from app.session import ChatSession, CORE_RULES  # noqa: E402
from app.settings import Settings, SettingsStore  # noqa: E402

def _configure_logging() -> None:
    """Log to a file as well as the console.

    A windowed build has console=False, so sys.stderr goes nowhere and every
    traceback the app produces is lost. That is how a model failing to load
    turned into "the app shows the download screen" with no way to find out
    why. The file is small, capped, and the first thing to ask a user for.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        from logging.handlers import RotatingFileHandler
        from app.paths import app_data_dir
        handlers.append(RotatingFileHandler(
            app_data_dir() / "vasudha.log", maxBytes=512_000, backupCount=1,
            encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unwritable data dir must not stop launch
        pass
    # force=True because basicConfig is a no-op when the root logger already
    # has a handler, and something in the import chain of a frozen build
    # installs one. Without it this produced a log file of exactly zero bytes,
    # which is worse than no log at all: it looks like the app got far enough
    # to log and had nothing to say.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True)
    logging.getLogger("vasudha.app").info("--- Vasudha starting ---")


_configure_logging()
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
        # Shared across chats on purpose: a lesson learned in one
        # conversation is worthless if the next one cannot see it.
        self._memory = MemoryBook()
        self._interactions = InteractionLog()
        self._backend: Optional[Backend] = None
        self._session: Optional[ChatSession] = None
        self._maximised = False
        self._needs_setup = False
        #: True until prepare() has finished once. Guards ready() from
        #: publishing a half-built state as though it were the final one.
        self._starting = True
        #: Why the engine would not start, when a model was present but unusable.
        #: Empty means "no model yet", which is the only case the download
        #: screen actually answers.
        self._load_error = ""
        self._failed_model = ""
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
        # Release whatever is already loaded, first. prepare() is not only a
        # startup path — locate_model and the post-download boot both re-enter
        # it — and a llama.cpp model holds its VRAM until it is closed. Loading
        # a second 2.7 GB model beside the first one fails on any 6 GB card, and
        # fails with "Failed to load model from file", which reads as a corrupt
        # download rather than as "you already have this open".
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:  # noqa: BLE001 - closing is best effort
                logger.debug("closing the previous backend failed", exc_info=True)
            self._backend = None
            self._session = None
            import gc
            gc.collect()

        configured = self._settings.model_path or None
        model_path = Path(configured) if configured and Path(configured).exists() else None
        if model_path is None:
            model_path = self._store.installed_model()

        # Before select_backend, which is the first thing to import llama_cpp:
        # the ggml backends register at library load and a device filtered
        # afterwards is already initialised.
        from app.backends import set_gpu_device
        set_gpu_device(self._settings.gpu_device)

        prefer = None if self._settings.backend == "auto" else self._settings.backend
        if not model_path and not OllamaBackend.probe(self._settings.ollama_url):
            self._needs_setup = True
            self._starting = False
            return

        try:
            # num_ctx has to reach the engine here, at construction: the
            # built-in backend's window is fixed when the model is loaded, and
            # passing it only in per-request options (which is all that used to
            # happen) left it at the 8192 default while the user's setting said
            # 16384. Nothing reported the mismatch — llama.cpp simply shifted
            # the oldest tokens out mid-conversation.
            self._backend = select_backend(
                str(model_path) if model_path else None,
                ollama_model_hint=self._settings.ollama_model or "vasudha",
                prefer=prefer,
                n_ctx=self._settings.num_ctx,
                n_batch=self._settings.n_batch,
                n_threads=self._settings.n_threads)
        except Exception as exc:  # noqa: BLE001 - see below
            # Every failure to build an engine used to land the user on the
            # download screen, whatever the cause: `except RuntimeError` caught
            # a model that would not load, and anything else (an ImportError
            # from a missing native library, a VRAM allocation that failed)
            # escaped prepare() entirely and left self._backend as None, which
            # _publish reads as "no model" too.
            #
            # So a machine with a perfectly good 2.3 GB GGUF already on disk was
            # told to download 2.3 GB, and downloading again could not possibly
            # fix it. Record what actually went wrong instead; _publish decides
            # what to show, and it now has something true to show.
            logger.exception("could not start an engine")
            self._needs_setup = True
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._failed_model = str(model_path) if model_path else ""
            self._starting = False
            return
        self._load_error = ""
        self._failed_model = ""

        self._session = ChatSession(
            self._backend,
            system_prompt=personas.build_system_prompt(self._settings.persona, CORE_RULES),
            workspace=str(self._workspace_for(self._chat.id)),
            memory=self._memory,
            interaction_log=self._interactions)
        self._apply_settings()
        self._needs_setup = False
        self._starting = False
        threading.Thread(target=self._warm, daemon=True).start()

    @staticmethod
    def _workspace_for(chat_id: str) -> Path:
        """One durable directory per chat, under the app's data folder.

        The session used to be built with no workspace at all, which sent every
        file the model wrote — and every document it produced — to a temporary
        directory the OS is free to delete. That was survivable when documents
        were the only output; with read_file/write_file/edit_file it means a
        project the model builds across several turns can vanish underneath it,
        and the user has nowhere to look for the files afterwards.

        Keyed by chat id so two conversations cannot overwrite each other's
        files, and so reopening a chat finds the work it produced.
        """
        path = app_data_dir() / "workspaces" / (chat_id or "default")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _rebind_workspace(self) -> None:
        """Point the live session at the current chat's workspace."""
        if not self._session:
            return
        self._session.set_workspace(str(self._workspace_for(self._chat.id)))

    def _warm(self) -> None:
        """Prefill the static prompt prefix while the user is still reading the
        window.

        Deliberately not on the splash: warming costs about as long as one
        prefill (~37s on CPU for this prompt), and a 40-second splash to save
        40 seconds later is not a saving. On a background thread the cost is
        hidden entirely if the user takes that long to type, and if they do not,
        it is the same work their question would have paid for anyway.

        Takes the same lock as a turn, so a question asked mid-warm waits
        rather than driving the same llama context from two threads.
        """
        if not self._session or not self._backend:
            return
        try:
            with self._lock:
                self._backend.warm(
                    [{"role": "system", "content": self._session.system_prompt}],
                    self._session.schemas, self._session.options)
        except Exception:  # noqa: BLE001 - an optimisation, never a precondition
            logger.debug("warm failed", exc_info=True)

    def ready(self) -> None:
        """Called from the page once it has loaded. Publishes state that
        prepare() already worked out — unless it has not finished yet.

        The page loads when the window is *created*, not when it is shown, so
        this fires while prepare() is still loading a 2.7 GB model behind the
        splash. Publishing then found self._backend still None and showed the
        first-run screen on a machine whose model was loading perfectly well —
        and clicking "I already have the file" on that screen re-entered
        prepare() and tried to load a second copy alongside the first.
        """
        if self._starting:
            return          # _publish runs at the end of prepare() instead
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
            # Two genuinely different situations, which looked identical before:
            # there is no model, or there is one and it will not load. Only the
            # first is fixed by downloading, so only the first gets a download
            # screen that promises it will help.
            self._call_js("showFirstRun", {
                "error": self._load_error,
                "model": self._failed_model,
            })
            return

        # Persona choice comes after the engine is up: it is a preference, and
        # asking for it while the app might still be unusable would be rude.
        if not self._settings.onboarded:
            self._call_js("showOnboarding", {"personas": personas.as_cards()})

        fast = isinstance(self._backend, OllamaBackend) or (
            isinstance(self._backend, LlamaCppBackend) and "GPU" in self._backend.display_name)
        detail = ("Running on your GPU." if fast else
                  "Running on CPU — answers take longer. Installing Ollama "
                  "would use your graphics card instead.")
        # A context smaller than the one in Settings has to be admitted. This
        # app has already shipped one bug where the engine quietly ran at half
        # the configured window and nothing said so.
        downgraded = getattr(self._backend, "downgraded_from", None)
        actual = getattr(self._backend, "context_limit", None)
        if downgraded and actual:
            detail += (f" The {downgraded:,}-token context did not fit in memory, "
                       f"so this chat holds {actual:,} tokens.")
        self._call_js("onBackend", {
            "label": self._backend.display_name,
            "fast": fast,
            "detail": detail,
        })
        self._call_js("showFirstRun", False)
        self._push_chat_list()

    # -- window controls ---------------------------------------------------

    def minimise(self) -> None:
        if self._window:
            self._window.minimize()

    def close(self) -> None:
        # Before destroying the window: a headless Chromium is a separate
        # process and survives the one that forgot about it, leaving the user
        # with a browser they cannot see and did not know they started.
        self.shutdown()
        if self._window:
            self._window.destroy()

    def shutdown(self) -> None:
        """Release anything that outlives this process if left alone."""
        if self._session:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001 - never block the window closing
                logger.debug("session close failed", exc_info=True)

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
        # create_file_dialog returns a tuple of paths on most pywebview
        # backends and a bare string on some. Indexing [0] unconditionally
        # takes the first *character* of the string form — "C" — which becomes
        # a path that does not exist, gets written to settings and to
        # models.json, and sends the app straight back to this screen having
        # apparently ignored the file the user just picked. save_document
        # already guards this; this one did not.
        path = Path(chosen if isinstance(chosen, str) else chosen[0])
        if not path.is_file():
            self._call_js("onDownloadError",
                          {"message": f"That is not a readable file: {path}"})
            return ""

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
            # Each _emit is an evaluate_js round trip across the webview
            # bridge. One per token is affordable at 6 tok/s on CPU and is not
            # at 60 tok/s on a GPU, where the bridge, not the model, becomes
            # the bottleneck and the text arrives in visible steps. Deltas are
            # therefore coalesced into ~50 ms batches, which is below the
            # threshold where streaming stops reading as continuous anyway.
            buffered: dict[str, list[str]] = {"token": [], "thinking_token": []}
            last_flush = time.monotonic()

            def flush() -> None:
                nonlocal last_flush
                for kind, parts in buffered.items():
                    if parts:
                        self._emit({"kind": kind, "text": "".join(parts)})
                        parts.clear()
                last_flush = time.monotonic()

            try:
                for event in self._session.ask(text):
                    kind = event.get("kind")

                    # Token deltas are for the live view only. The finished
                    # 'text' event carries the same content, so persisting both
                    # would store every reply twice — once as prose and once as
                    # a few hundred one-word fragments — and replay it doubled
                    # when the chat is reopened.
                    if kind in ("token", "thinking_token"):
                        buffered[kind].append(event.get("text", ""))
                        if time.monotonic() - last_flush >= 0.05:
                            flush()
                        continue

                    flush()   # ordering: never let a delta land after the
                              # event that concludes it
                    self._chat.events.append(event)
                    self._emit(event)
                flush()
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
            # Drop the browser with the conversation. Carrying its cookies and
            # logged-in sessions into an unrelated chat is a privacy leak, not
            # a convenience.
            self._session.close()
        self._rebind_workspace()
        if self._session:
            self._session.chat_id = self._chat.id
        self._call_js("loadChat", {"events": []})
        self._push_chat_list()

    def open_chat(self, chat_id: str) -> None:
        chat = self._chats.load(chat_id)
        if chat is None:
            return
        self._chat = chat
        if self._session:
            self._session.history = list(chat.messages)
        # Follow the chat: reopening a conversation should find the files it
        # created, not the previous chat's.
        self._rebind_workspace()
        self._call_js("loadChat", {"events": chat.events})
        self._push_chat_list()

    def attach_file(self) -> dict:
        """Let the user pick a file for the model to work on.

        The file is copied into the workspace and *described* to the model —
        size, line count, columns, a dozen sample rows. Its contents never
        enter the conversation, which is what makes attaching a 40 MB CSV a
        sensible thing to do rather than an instant context overflow.
        """
        if not self._window or not self._session:
            return {"ok": False, "error": "no chat is open yet"}
        chosen = self._window.create_file_dialog(webview.OPEN_DIALOG,
                                                 allow_multiple=True)
        if not chosen:
            return {"ok": False, "error": ""}
        # Same string-or-tuple guard as locate_model: some pywebview backends
        # return a bare string, and indexing it yields one character.
        paths = [chosen] if isinstance(chosen, str) else list(chosen)

        attached, failed = [], []
        for path in paths:
            result = self._session.attach(path)
            (attached if result.get("ok") else failed).append(result)
        return {"ok": bool(attached), "attached": attached,
                "error": "; ".join(f.get("error", "") for f in failed)}

    def image_data_url(self, path: str) -> str:
        """A picture as a data: URI.

        There is no HTTP server to serve files from — that is the whole point of
        the js_api bridge — so an image reaches the canvas as bytes or not at
        all. Bounded because a data URI is base64 and lands in the DOM.
        """
        import base64
        import mimetypes
        try:
            target = Path(path)
            if target.stat().st_size > 12_000_000:
                return ""
            mime = mimetypes.guess_type(target.name)[0] or "image/png"
            return (f"data:{mime};base64,"
                    + base64.b64encode(target.read_bytes()).decode("ascii"))
        except (OSError, ValueError):
            logger.warning("could not read image %s", path, exc_info=True)
            return ""

    def open_workspace_folder(self) -> None:
        """Show the user where the model's files actually are."""
        self._open_folder(self._workspace_for(self._chat.id))


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
            api._starting = False
            say("Starting anyway…")

        # ready() returns early while _starting is set, so the final state has
        # to be published from here. Both orderings are covered: if the page
        # loaded first its ready() was a no-op and this publishes; if it loads
        # after, this call fails silently against a page that is not there yet
        # and ready() publishes instead.
        api._starting = False
        api._publish()

        remaining = SPLASH_MIN_SECONDS - (time.time() - started)
        if remaining > 0:
            time.sleep(remaining)

        window.show()
        try:
            splash.destroy()
        except Exception:  # noqa: BLE001
            logger.debug("splash already closed", exc_info=True)

    # Also on the window's own close event, since the titlebar X and the OS
    # both bypass Api.close().
    window.events.closing += api.shutdown

    webview.start(boot, debug=bool(os.environ.get("VASUDHA_DEBUG")))
    api.shutdown()          # belt and braces: normal exit path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
