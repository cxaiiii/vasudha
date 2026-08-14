"""First-run model acquisition.

The binary ships without the 2.3 GB GGUF — bundling it would make the installer
enormous and force a re-download of the whole app for every model revision.
Instead the model lands in %LOCALAPPDATA%\\Vasudha\\models on first launch, and
stays there across app updates.

Three ways to get one, cheapest first:
  1. adopt() an existing file the user already has (very common: ollama users
     already store this exact GGUF, so re-downloading it is pure waste).
  2. download() from the configured release URL, resumable.
  3. Neither — the app falls back to ollama if it is running.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

# Where the weights are hosted.
#
# NOT GitHub: release assets are capped at 2.00 GiB and every quant of this
# model is larger (Q3_K_M is 2.11 GiB, Q4_K_M is 2.52 GiB). HuggingFace has no
# such cap, serves over a CDN, and honours HTTP range requests, which is what
# makes the resume below work.
#
# `?download=true` asks HuggingFace for the file rather than the HTML page.
DEFAULT_MODEL_URL = os.environ.get("VASUDHA_MODEL_URL", "")
DEFAULT_MODEL_NAME = "vasudha-leaprunning-4b-v3-Q3_K_M.gguf"

#: Published SHA-256 of the release build. A 2.2 GB download that is silently
#: truncated or corrupted does not fail cleanly — llama.cpp reports a confusing
#: tensor error, or worse, loads and produces nonsense. Checking the digest
#: turns that into one clear message.
DEFAULT_MODEL_SHA256 = os.environ.get("VASUDHA_MODEL_SHA256", "")
DEFAULT_MODEL_BYTES = int(os.environ.get("VASUDHA_MODEL_BYTES", "0") or 0)

_CHUNK = 1 << 20  # 1 MiB


@dataclass
class DownloadProgress:
    done_mb: float
    total_mb: float
    percent: float
    mbps: float
    #: "downloading" | "verifying" | "done"; the UI shows a different line for
    #: verification because hashing 2.2 GB takes a few seconds and a bar that
    #: sits at 100% doing nothing looks like a hang.
    stage: str = "downloading"
    eta_seconds: float = 0.0


def sha256_of(path: Path, on_progress=None) -> str:
    """Digest of a file, read in chunks so a 2.2 GB model is not held in RAM."""
    digest = hashlib.sha256()
    total = path.stat().st_size or 1
    done = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)  # 4 MiB
            if not block:
                break
            digest.update(block)
            done += len(block)
            if on_progress:
                on_progress(DownloadProgress(
                    done_mb=done / 1e6, total_mb=total / 1e6,
                    percent=100.0 * done / total, mbps=0.0, stage="verifying"))
    return digest.hexdigest()


def _load_source_config() -> dict:
    """Where to fetch the model from, resolved at runtime.

    Looked up in order: %LOCALAPPDATA%\\Vasudha\\model_source.json, then a file
    beside the executable, then the copy inside the bundle. This exists so the
    download URL can be corrected after a build has shipped — the binary is
    212 MB and rebuilding it to change one string is not a reasonable way to
    fix a moved link. Environment variables still win over all of it.
    """
    candidates = []
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    candidates.append(Path(base) / "Vasudha" / "model_source.json")
    candidates.append(Path(sys.executable).parent / "model_source.json")
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "model_source.json")
    candidates.append(Path(__file__).resolve().parent.parent / "model_source.json")

    for path in candidates:
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    logger.info("model source from %s", path)
                    return data
        except (json.JSONDecodeError, OSError):
            logger.warning("ignoring unreadable %s", path, exc_info=True)
    return {}


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    path = Path(base) / "Vasudha"
    path.mkdir(parents=True, exist_ok=True)
    return path


class ModelStore:
    """Where the GGUF lives, and how it gets there."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or (app_data_dir() / "models")
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = app_data_dir() / "models.json"

    # -- lookup ------------------------------------------------------------

    def installed_model(self) -> Optional[Path]:
        """The GGUF to use, or None. An adopted path is honoured first so the
        user's existing copy wins over anything we downloaded."""
        if self.index_path.exists():
            try:
                record = json.loads(self.index_path.read_text(encoding="utf-8"))
                adopted = record.get("path")
                if adopted and Path(adopted).exists():
                    return Path(adopted)
            except (json.JSONDecodeError, OSError):
                logger.warning("models.json unreadable; ignoring", exc_info=True)

        candidates = sorted(self.root.glob("*.gguf"), key=lambda p: p.stat().st_size,
                            reverse=True)
        return candidates[0] if candidates else None

    def adopt(self, path: Path) -> None:
        """Record a GGUF the user already has, in place — no copy. Copying
        2.3 GB to say 'installed' wastes the disk space the user was trying to
        avoid spending twice."""
        self.index_path.write_text(json.dumps({"path": str(path)}, indent=2),
                                   encoding="utf-8")

    # -- download ----------------------------------------------------------

    def download(self, url: str = "", name: str = DEFAULT_MODEL_NAME,
                 on_progress: Optional[Callable[[DownloadProgress], None]] = None,
                 sha256: str = "") -> Path:
        """Resumable, verified download.

        Writes to a .part file and only renames on success, so an interrupted
        download can never be mistaken for a usable model. If a digest is
        published, the file is checked before it is accepted — a truncated GGUF
        does not fail cleanly at load time, it fails confusingly.
        """
        config = _load_source_config()
        url = url or DEFAULT_MODEL_URL or str(config.get("url", ""))
        sha256 = (sha256 or DEFAULT_MODEL_SHA256
                  or str(config.get("sha256", ""))).strip().lower()
        if name == DEFAULT_MODEL_NAME and config.get("filename"):
            name = str(config["filename"])

        if not url:
            raise RuntimeError(
                "No download link is set up yet. Use 'I already have the file' to "
                "point Vasudha at a .gguf on this machine, or add a url to "
                "model_source.json beside the application.")

        target = self.root / name
        partial = target.with_suffix(target.suffix + ".part")
        existing = partial.stat().st_size if partial.exists() else 0

        headers = {"Range": f"bytes={existing}-"} if existing else {}
        with requests.get(url, stream=True, headers=headers, timeout=60) as response:
            if existing and response.status_code == 200:
                # Server ignored the Range header — start over rather than
                # appending fresh bytes onto a partial file and corrupting it.
                existing = 0
                partial.unlink(missing_ok=True)
            elif existing and response.status_code != 206:
                response.raise_for_status()
            else:
                response.raise_for_status()

            remaining = int(response.headers.get("Content-Length", 0))
            total = existing + remaining
            done = existing
            started = time.time()

            mode = "ab" if existing else "wb"
            with open(partial, mode) as handle:
                for chunk in response.iter_content(chunk_size=_CHUNK):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    done += len(chunk)
                    if on_progress and total:
                        elapsed = max(time.time() - started, 1e-6)
                        rate = ((done - existing) / 1e6) / elapsed
                        on_progress(DownloadProgress(
                            done_mb=done / 1e6,
                            total_mb=total / 1e6,
                            percent=100.0 * done / total,
                            mbps=rate,
                            stage="downloading",
                            eta_seconds=((total - done) / 1e6) / rate if rate > 0.01 else 0.0,
                        ))

        if sha256:
            actual = sha256_of(partial, on_progress=on_progress)
            if actual != sha256:
                # Keep the bad file out of the way rather than leaving something
                # that looks installed but will fail strangely at load.
                partial.unlink(missing_ok=True)
                raise RuntimeError(
                    "The downloaded file did not match its published checksum, so it "
                    "was discarded. This usually means the download was interrupted or "
                    "the connection altered it. Try again.\n"
                    f"expected {sha256}\ngot      {actual}")

        if target.exists():
            target.unlink()
        shutil.move(str(partial), str(target))
        self.adopt(target)
        if on_progress:
            size_mb = target.stat().st_size / 1e6
            on_progress(DownloadProgress(done_mb=size_mb, total_mb=size_mb,
                                         percent=100.0, mbps=0.0, stage="done"))
        return target
