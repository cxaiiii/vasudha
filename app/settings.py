"""User settings, persisted to %LOCALAPPDATA%\\Vasudha\\settings.json.

Every value here is one a user might genuinely need to change on their own
machine — not a dumping ground for knobs. Notably absent: a switch to disable
python_tool. The measured difference between tool use and no tool use on this
model is 0/6 versus 4/6 (scripts/bench_numeric.py); an option to turn it off
would only ever be an option to get wrong answers.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _settings_path() -> Path:
    from app.paths import app_data_dir
    return app_data_dir() / "settings.json"


#: One knob for how hard to try. Each mode sets the four things that actually
#: bound a turn: how much the model may write, how much it may remember, how
#: many tool calls it gets, and how long any one of them may run.
#:
#: num_ctx is the odd one out and deliberately so — the built-in engine fixes
#: its window when the model loads, so a change here takes effect on the next
#: launch rather than mid-conversation. Everything else applies immediately.
EFFORT_MODES: dict[str, dict] = {
    "low": dict(
        label="Low", num_predict=512, num_ctx=4096,
        max_iterations=3, tool_timeout=15,
        blurb="Quick answers. One or two tools at most."),
    "medium": dict(
        label="Medium", num_predict=2048, num_ctx=8192,
        max_iterations=8, tool_timeout=30,
        blurb="The default. Enough room to search, compute and answer."),
    "high": dict(
        label="High", num_predict=4096, num_ctx=16384,
        max_iterations=16, tool_timeout=60,
        blurb="Longer research. Multi-step work with several sources."),
    "max": dict(
        label="Max", num_predict=8192, num_ctx=32768,
        max_iterations=24, tool_timeout=120,
        blurb="Everything it has. Slow, and worth it for real reports."),
    "yolo": dict(
        label="YOLO", num_predict=16384, num_ctx=65536,
        max_iterations=40, tool_timeout=300,
        blurb="No limits worth the name. It will run until it is finished "
              "or you stop it — and on a 4B that can be a very long time."),
}

EFFORT_ORDER = ["low", "medium", "high", "max", "yolo"]


@dataclass
class Settings:
    # -- engine ----------------------------------------------------------
    backend: str = "auto"          # auto | ollama | builtin
    model_path: str = ""           # explicit GGUF for the built-in backend
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""         # blank = first match containing "vasudha"

    # -- generation ------------------------------------------------------
    temperature: float = 0.6
    top_p: float = 0.9
    num_predict: int = 2048        # see session.py: 1024 truncated tool calls
    num_ctx: int = 16384

    # -- engine tuning ----------------------------------------------------
    # Defaults are the measured best on the shipped 4B Q4_K_M (see the table in
    # app/backends.py). n_threads=0 means "detect physical cores" — the right
    # answer varies per machine and hyperthreads measurably hurt, so this is
    # derived rather than guessed at a fixed number.
    n_batch: int = 1024
    n_threads: int = 0
    # Pin work to one GPU when the machine has more than one. -1 = leave it to
    # llama.cpp, which splits layers across every visible device and lets the
    # slowest one throttle the pass. Setting the discrete card's index measured
    # +27% prefill on a 740M/RTX-4050 laptop. The startup banner lists devices
    # found, not the device chosen — set VASUDHA_DEBUG to see it.
    gpu_device: int = -1

    #: Which EFFORT_MODES entry is selected. The individual values above stay
    #: authoritative — a mode writes into them — so hand-editing one still
    #: works and is not silently reverted on next launch.
    effort: str = "medium"

    # -- identity --------------------------------------------------------
    persona: str = "engineer"
    #: False until the first-run picker has been completed once
    onboarded: bool = False

    # -- behaviour -------------------------------------------------------
    show_reasoning: bool = False   # expand <think> blocks by default
    open_tool_cards: bool = True   # expand the code+stdout evidence by default
    # Seconds a python_tool run may take. 15 was sized for arithmetic and is
    # too tight now that pip_tool exists: importing a freshly installed package
    # for the first time can take longer than that on its own (textblob pulls in
    # nltk and its corpora), and the model reads the timeout as "my code is
    # wrong" and rewrites working code.
    tool_timeout: int = 30
    max_iterations: int = 12       # tool calls per turn before giving up

    # -- interface -------------------------------------------------------
    # The only request this app makes that the user did not ask for. A
    # plain GET of the public releases API, carrying no identifier and no
    # version, once a day. Off means no request at all.
    #: Largest context measured to actually load, keyed by
    #: "<model path>|<gpu>". Measured rather than estimated: the
    #: arithmetic was 4x too conservative on a real machine, refusing
    #: 32k on a card that loads it. Probed once in the background and
    #: reused, since the answer only changes with the model or the card.
    probed_contexts: dict = field(default_factory=dict)

    check_updates: bool = True
    last_update_check: float = 0.0

    reduce_motion: bool = False
    send_on_enter: bool = True

    def clamp(self) -> "Settings":
        """Keep hand-edited or stale values inside ranges the app can honour.

        settings.json is a plain file a user may edit; a negative num_ctx or a
        temperature of 40 should not be able to wedge the app on next launch.
        """
        self.temperature = min(max(float(self.temperature), 0.0), 2.0)
        self.top_p = min(max(float(self.top_p), 0.01), 1.0)
        self.num_predict = min(max(int(self.num_predict), 256), 16384)
        self.num_ctx = min(max(int(self.num_ctx), 2048), 262144)
        # Ceilings sized so the loudest EFFORT_MODES entry survives them. They
        # used to be 120 and 32, which silently cut YOLO's declared 300s and 40
        # iterations down — a setting promising something the app would not
        # honour, which is the failure this file exists to prevent.
        self.tool_timeout = min(max(int(self.tool_timeout), 1), 600)
        self.max_iterations = min(max(int(self.max_iterations), 1), 64)
        self.n_batch = min(max(int(self.n_batch), 32), 4096)
        # 0 stays 0: it is the sentinel for "detect", not a value to clamp up.
        self.n_threads = max(int(self.n_threads), 0)
        # -1 is the "leave it alone" sentinel; anything below that is a typo.
        self.gpu_device = max(int(self.gpu_device), -1)
        if self.backend not in ("auto", "ollama", "builtin"):
            self.backend = "auto"
        if self.effort not in EFFORT_MODES:
            self.effort = "medium"
        return self

    def apply_effort(self, name: str) -> "Settings":
        """Set the four bounds a mode controls, and record which mode it was."""
        mode = EFFORT_MODES.get(name)
        if not mode:
            return self
        self.effort = name
        self.num_predict = mode["num_predict"]
        self.num_ctx = mode["num_ctx"]
        self.max_iterations = mode["max_iterations"]
        self.tool_timeout = mode["tool_timeout"]
        return self.clamp()

    def generation_options(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }


class SettingsStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or _settings_path()
        self._known = {f.name for f in fields(Settings)}

    def load(self) -> Settings:
        if not self.path.exists():
            return Settings()
        try:
            data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("settings.json unreadable; using defaults", exc_info=True)
            return Settings()
        # Ignore unknown keys rather than raising: a settings file written by a
        # newer build must not stop an older one from starting.
        return Settings(**{k: v for k, v in data.items() if k in self._known}).clamp()

    def save(self, settings: Settings) -> None:
        settings.clamp()
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
        os.replace(temp, self.path)
