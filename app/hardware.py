"""What this machine can actually hold, and what context size fits in it.

A context window is not a preference. Weights, KV cache and compute buffer all
have to sit in the same memory at once, and asking for more than fits does not
degrade — it fails at load with `Failed to load model from file`, which reads
like a corrupt download and sends the user off to re-fetch a file that was
never the problem. That happened, repeatedly, and then happened again the
moment an effort mode was allowed to request 65,536 tokens on a 6 GB card.

So the requested size is capped against measured hardware before the engine
ever sees it. The arithmetic is not subtle:

    KV bytes = 2 (K and V) x layers x kv_heads x head_dim x n_ctx x bytes/elem

Everything here degrades to a safe default rather than raising. A machine we
cannot measure gets the conservative answer, which is worse than optimal and
better than a crash on launch.
"""
from __future__ import annotations

import logging
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Used when nothing can be measured. Small enough to load on any machine that
#: can hold the model at all.
FALLBACK_CONTEXT = 4096

#: Room left for llama.cpp's compute buffers, which scale with batch size and
#: are not part of the KV arithmetic. Measured at ~1.0 GB for a 4B at
#: n_batch=1024; 1.2 GB leaves margin for a larger model without being so
#: cautious that a 6 GB card loses a useful chunk of its context.
COMPUTE_BUFFER_BYTES = 1_200_000_000

#: Not all of a GPU is available — the desktop compositor, the browser, and
#: whatever else is running got there first. Measured on a 6.14 GB laptop card
#: with an ordinary desktop: about 0.5 GB gone before anything loads.
GPU_HEADROOM_BYTES = 600_000_000


# ── GGUF header ───────────────────────────────────────────────────────────────

_GGUF_TYPES = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
    5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8),
    12: ("d", 8),
}


def read_gguf_metadata(path: str, limit: int = 512) -> dict:
    """Key/value pairs from a GGUF header, without touching the tensors.

    gguf-py's GGUFReader memory-maps the whole file and took 7.4s on a 2.7 GB
    model here — far too slow for something on the launch path, and it is a
    dependency a frozen build would rather not carry. The header is a simple
    length-prefixed format sitting in the first few kilobytes, so it is parsed
    directly: milliseconds, and no import.
    """
    values: dict = {}
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"GGUF":
                return {}
            struct.unpack("<I", handle.read(4))[0]          # version
            struct.unpack("<Q", handle.read(8))[0]          # tensor count
            kv_count = struct.unpack("<Q", handle.read(8))[0]

            def read_string() -> str:
                length = struct.unpack("<Q", handle.read(8))[0]
                return handle.read(length).decode("utf-8", "replace")

            def read_value(kind: int):
                if kind == 8:
                    return read_string()
                if kind == 9:                                # array
                    item_kind = struct.unpack("<I", handle.read(4))[0]
                    count = struct.unpack("<Q", handle.read(8))[0]
                    # Skipped rather than collected: the arrays in a GGUF header
                    # are the tokenizer vocabulary, which is hundreds of
                    # thousands of strings and none of this function's business.
                    for _ in range(count):
                        read_value(item_kind)
                    return None
                fmt, size = _GGUF_TYPES.get(kind, (None, 0))
                if fmt is None:
                    raise ValueError(f"unknown gguf value type {kind}")
                return struct.unpack("<" + fmt, handle.read(size))[0]

            for _ in range(min(kv_count, limit)):
                key = read_string()
                kind = struct.unpack("<I", handle.read(4))[0]
                value = read_value(kind)
                if value is not None:
                    values[key] = value
    except (OSError, struct.error, ValueError, UnicodeDecodeError):
        logger.debug("could not read gguf header from %s", path, exc_info=True)
        return {}
    return values


def kv_bytes_per_token(metadata: dict, bits: int = 16) -> Optional[int]:
    """Bytes of KV cache one token costs, or None if the header did not say.

    Read from the model rather than assumed, because the whole point is to work
    for whatever GGUF is dropped in. Falls back to embedding_length/head_count
    when key_length is absent, which is how most architectures express it.
    """
    def find(*suffixes) -> Optional[int]:
        for key, value in metadata.items():
            if isinstance(value, int) and key.endswith(suffixes):
                return value
        return None

    layers = find(".block_count")
    kv_heads = find(".attention.head_count_kv")
    head_dim = find(".attention.key_length")
    if head_dim is None:
        hidden, heads = find(".embedding_length"), find(".attention.head_count")
        head_dim = (hidden // heads) if (hidden and heads) else None
    if not (layers and kv_heads and head_dim):
        return None
    return 2 * layers * kv_heads * head_dim * (bits // 8)


# ── available memory ──────────────────────────────────────────────────────────

def gpu_memory_bytes() -> Optional[int]:
    """Total VRAM of the card llama.cpp will use, or None if not discoverable.

    nvidia-smi only, deliberately. There is no portable way to ask a Vulkan
    device its size from Python, and a wrong number here is worse than no
    number: it would cap the context using memory that does not exist. AMD and
    Intel therefore fall through to the system-RAM path, which is conservative
    and safe.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        # The device the app pins to, when it pins to one.
        pinned = os.environ.get("GGML_VK_VISIBLE_DEVICES")
        sizes = [int(line.strip()) for line in out.stdout.splitlines() if line.strip()]
        if not sizes:
            return None
        if pinned and pinned.isdigit() and int(pinned) < len(sizes):
            return sizes[int(pinned)] * 1024 * 1024
        return max(sizes) * 1024 * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def system_memory_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 - psutil is optional
        return None


# ── the answer ────────────────────────────────────────────────────────────────

def max_context(model_path: str, on_gpu: bool, requested: int,
                floor: int = 2048) -> tuple[int, str]:
    """The largest context that will actually load, and why it was capped.

    Returns (n_ctx, reason). reason is empty when the request was granted, so a
    caller can stay quiet in the ordinary case and explain itself in the one
    that matters.
    """
    if requested <= floor:
        return requested, ""

    metadata = read_gguf_metadata(model_path)
    per_token = kv_bytes_per_token(metadata)
    if not per_token:
        # Unknown geometry: honour the request rather than guessing. A wrong
        # cap is a silent downgrade, which this app has already shipped once.
        return requested, ""

    try:
        weights = os.path.getsize(model_path)
    except OSError:
        return requested, ""

    if on_gpu:
        total = gpu_memory_bytes()
        if total is None:
            # A GPU we cannot measure: leave the request alone rather than
            # capping against the wrong pool of memory entirely.
            return requested, ""
        available = total - GPU_HEADROOM_BYTES - weights - COMPUTE_BUFFER_BYTES
        where = "your GPU"
    else:
        total = system_memory_bytes()
        if total is None:
            return requested, ""
        # Half of RAM, because everything else on the machine needs the rest.
        available = (total // 2) - weights - COMPUTE_BUFFER_BYTES
        where = "this machine's memory"

    if available <= 0:
        return floor, (f"{Path(model_path).name} barely fits in {where}, so the "
                       f"context is set to {floor:,} tokens.")

    fits = int(available // per_token)
    # Rounded down to a power of two: llama.cpp is happier with round batch
    # geometry, and a precise-looking 13,417 invites the question of whether it
    # was measured or guessed.
    rounded = 1 << (fits.bit_length() - 1) if fits >= floor else floor
    allowed = max(min(requested, rounded), floor)

    if allowed >= requested:
        return requested, ""
    return allowed, (
        f"{requested:,} tokens of context need about "
        f"{(requested * per_token) / 1e9:.1f} GB of memory for the cache alone, "
        f"which does not fit in {where} alongside the model. Running at "
        f"{allowed:,} instead.")
