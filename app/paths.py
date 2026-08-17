"""Where things live, per operating system.

This logic was copy-pasted into bootstrap, history, settings and personas —
four chances to fix a platform bug in three places and miss the fourth. It
lives here now.

Each OS has a convention, and using the wrong one is not merely untidy: on
macOS a folder under ~/.local/share is invisible in Finder, so "Open folder"
would appear to do nothing.

    Windows   %LOCALAPPDATA%\\Vasudha
    macOS     ~/Library/Application Support/Vasudha
    Linux     $XDG_DATA_HOME/Vasudha, else ~/.local/share/Vasudha
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "Vasudha"


def app_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_folder(path: Path) -> None:
    """Reveal a folder in the system file manager.

    os.startfile exists only on Windows — calling it elsewhere is an
    AttributeError, not a graceful no-op.
    """
    try:
        if sys.platform == "win32":
            os.startfile(str(path))       # noqa: S606 - the user's own folder
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except (OSError, AttributeError):
        logger.exception("could not open %s", path)


def executable_name() -> str:
    """What the shipped binary is called on this platform."""
    return "Vasudha.exe" if sys.platform == "win32" else "Vasudha"
