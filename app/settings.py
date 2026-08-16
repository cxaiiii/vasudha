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
    # Which GPU to use, when the machine has more than one. -1 = let llama.cpp
    # choose, which takes device 0 — the integrated GPU on a switchable-graphics
    # laptop. Set to the discrete card's index (usually 1) to use it instead;
    # the device list is printed at startup when VASUDHA_DEBUG is set.
    gpu_device: int = -1

    # -- identity --------------------------------------------------------
    persona: str = "engineer"
    #: False until the first-run picker has been completed once
    onboarded: bool = False

    # -- behaviour -------------------------------------------------------
    show_reasoning: bool = False   # expand <think> blocks by default
    open_tool_cards: bool = True   # expand the code+stdout evidence by default
    tool_timeout: int = 15         # seconds a python_tool run may take
    max_iterations: int = 12       # tool calls per turn before giving up

    # -- interface -------------------------------------------------------
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
        self.tool_timeout = min(max(int(self.tool_timeout), 1), 120)
        self.max_iterations = min(max(int(self.max_iterations), 1), 32)
        self.n_batch = min(max(int(self.n_batch), 32), 4096)
        # 0 stays 0: it is the sentinel for "detect", not a value to clamp up.
        self.n_threads = max(int(self.n_threads), 0)
        # -1 is the "leave it alone" sentinel; anything below that is a typo.
        self.gpu_device = max(int(self.gpu_device), -1)
        if self.backend not in ("auto", "ollama", "builtin"):
            self.backend = "auto"
        return self

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
