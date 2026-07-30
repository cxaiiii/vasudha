"""
Vasudha logging utilities.

Wraps Python's standard logging with Rich for beautiful, structured output.
Provides a consistent logging interface across the entire Vasudha codebase.

Design decisions:
  - Single global logger hierarchy under "vasudha" namespace
  - Rich handler for human-readable terminal output
  - File handler optional (for long training runs)
  - Log level configurable via VASUDHA_LOG_LEVEL env var or setup_logging()
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ── Vasudha's custom console theme ────────────────────────────────────────────
_VASUDHA_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "bold green",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "repr.number": "bold cyan",
        "repr.str": "green",
    }
)

_console = Console(theme=_VASUDHA_THEME, stderr=True)

# ── Module-level flag to avoid re-initialization ──────────────────────────────
_logging_initialized: bool = False


def setup_logging(
    level: str | int = "INFO",
    log_file: Optional[str | Path] = None,
    rich_tracebacks: bool = True,
) -> None:
    """
    Initialize Vasudha's global logging configuration.

    Should be called once at the start of a script or training run.
    Safe to call multiple times (idempotent after first call).

    Args:
        level: Logging level. Can be string ("DEBUG", "INFO", etc.) or int.
               Overridden by VASUDHA_LOG_LEVEL environment variable if set.
        log_file: Optional path to a log file. Useful for training runs on Colab.
        rich_tracebacks: Whether to use Rich's enhanced traceback formatting.

    Example:
        >>> setup_logging(level="DEBUG", log_file="train.log")
    """
    global _logging_initialized  # noqa: PLW0603

    # Environment variable override
    env_level = os.environ.get("VASUDHA_LOG_LEVEL", "").upper()
    if env_level:
        level = env_level

    # Normalize level
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level

    # Configure root Vasudha logger
    root_logger = logging.getLogger("vasudha")
    root_logger.setLevel(numeric_level)

    # Avoid adding handlers multiple times
    if not root_logger.handlers or not _logging_initialized:
        # Remove existing handlers to avoid duplicates
        root_logger.handlers.clear()

        # ── Rich console handler ───────────────────────────────────────────
        rich_handler = RichHandler(
            console=_console,
            rich_tracebacks=rich_tracebacks,
            show_time=True,
            show_level=True,
            show_path=False,  # Keep output clean
            markup=True,
            log_time_format="[%H:%M:%S]",
        )
        rich_handler.setLevel(numeric_level)
        root_logger.addHandler(rich_handler)

        # ── Optional file handler ──────────────────────────────────────────
        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root_logger.addHandler(file_handler)

        # Don't propagate to root logger (avoids duplicate output)
        root_logger.propagate = False
        _logging_initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a Vasudha-namespaced logger for a module.

    Args:
        name: Module name (typically __name__). Will be prefixed with "vasudha."
              if not already.

    Returns:
        Logger instance configured to use Vasudha's Rich handler.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Model loaded successfully")
    """
    # Ensure logging is initialized with defaults if not already done
    if not _logging_initialized:
        setup_logging()

    # Namespace all loggers under vasudha.
    if not name.startswith("vasudha"):
        full_name = f"vasudha.{name}"
    else:
        full_name = name

    return logging.getLogger(full_name)


def log_banner(title: str, subtitle: str = "") -> None:
    """
    Print a banner to the console (not logger). Used for training start/end.

    Args:
        title: Main banner title.
        subtitle: Optional subtitle line.
    """
    _console.rule(f"[bold blue]{title}[/bold blue]")
    if subtitle:
        _console.print(f"[dim]{subtitle}[/dim]", justify="center")
