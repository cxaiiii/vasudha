"""
Vasudha VRAM and memory monitoring utilities.

Provides real-time tracking of GPU memory usage, which is critical for
working within Google Colab Free Tier's 15GB VRAM limit.

Key features:
  - Snapshot-based peak VRAM tracking
  - Context manager for measuring memory cost of operations
  - Human-readable byte formatting
  - Warning thresholds for proactive OOM prevention
"""

from __future__ import annotations

import contextlib
import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch

from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

# Colab T4 VRAM ceiling — used for percentage calculations
_T4_VRAM_BYTES: int = 15 * 1024**3  # 15 GiB


@dataclass
class MemoryStats:
    """
    Snapshot of GPU memory statistics at a point in time.

    All values are in bytes.
    """

    allocated: int = 0
    """Currently allocated memory."""

    reserved: int = 0
    """Memory reserved by PyTorch's caching allocator (includes allocated)."""

    peak_allocated: int = 0
    """Peak allocated since last reset_peak_memory_stats()."""

    peak_reserved: int = 0
    """Peak reserved since last reset_peak_memory_stats()."""

    total: int = 0
    """Total GPU memory."""

    free: int = 0
    """Free GPU memory."""

    device: str = "cuda:0"
    """Device this snapshot was taken from."""

    timestamp: float = field(default_factory=time.time)
    """Unix timestamp when this snapshot was taken."""

    @property
    def utilization_pct(self) -> float:
        """Percentage of total GPU memory currently allocated."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.allocated / self.total

    @property
    def t4_pct(self) -> float:
        """Percentage of T4's 15GB used (useful for Colab budget tracking)."""
        return 100.0 * self.allocated / _T4_VRAM_BYTES

    def __repr__(self) -> str:
        return (
            f"MemoryStats("
            f"allocated={format_bytes(self.allocated)}, "
            f"reserved={format_bytes(self.reserved)}, "
            f"peak={format_bytes(self.peak_allocated)}, "
            f"total={format_bytes(self.total)}, "
            f"util={self.utilization_pct:.1f}%)"
        )


def format_bytes(num_bytes: int) -> str:
    """
    Format a byte count as a human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted string like "1.23 GiB" or "512.0 MiB".

    Example:
        >>> format_bytes(1073741824)
        '1.00 GiB'
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0  # type: ignore[assignment]
    return f"{num_bytes:.2f} PiB"


def get_memory_stats(device: Optional[str | torch.device] = None) -> MemoryStats:
    """
    Get a memory statistics snapshot for the specified device.

    Returns a MemoryStats with all zeros if CUDA is not available (CPU mode).

    Args:
        device: CUDA device identifier. Defaults to "cuda:0" if CUDA available.

    Returns:
        MemoryStats snapshot.
    """
    if not torch.cuda.is_available():
        return MemoryStats(device="cpu")

    if device is None:
        device_idx = torch.cuda.current_device()
        device_str = f"cuda:{device_idx}"
    else:
        device_str = str(device)
        device_idx = int(device_str.split(":")[-1]) if ":" in device_str else 0

    total, free_mem = torch.cuda.mem_get_info(device_idx)
    stats = torch.cuda.memory_stats(device_str)

    return MemoryStats(
        allocated=torch.cuda.memory_allocated(device_str),
        reserved=torch.cuda.memory_reserved(device_str),
        peak_allocated=stats.get("allocated_bytes.all.peak", 0),
        peak_reserved=stats.get("reserved_bytes.all.peak", 0),
        total=total + (total - free_mem),  # total = free + used at query time
        free=free_mem,
        device=device_str,
    )


def reset_memory_stats(device: Optional[str | torch.device] = None) -> None:
    """Reset peak memory statistics for accurate per-operation benchmarking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


class VRAMMonitor:
    """
    Context manager and standalone utility for tracking VRAM usage.

    Designed for Colab T4 constraints — logs warnings when approaching
    the 15GB ceiling to help avoid OOM crashes mid-training.

    Usage:
        # As context manager:
        with VRAMMonitor("Forward pass") as mon:
            output = model(input_ids)
        print(mon.delta_allocated)  # bytes consumed by forward pass

        # Standalone periodic logging:
        monitor = VRAMMonitor(warning_threshold_pct=80.0)
        monitor.log_status()  # logs current VRAM usage
    """

    def __init__(
        self,
        label: str = "",
        device: Optional[str | torch.device] = None,
        warning_threshold_pct: float = 85.0,
        gc_before: bool = False,
    ) -> None:
        """
        Args:
            label: Human-readable label for this measurement (e.g., "forward pass").
            device: CUDA device to monitor. Defaults to current device.
            warning_threshold_pct: Log a warning if allocated VRAM exceeds this
                                   percentage of total GPU memory.
            gc_before: If True, run Python GC and empty CUDA cache before measurement.
                       Useful for clean baseline measurements.
        """
        self.label = label
        self.device = device
        self.warning_threshold_pct = warning_threshold_pct
        self.gc_before = gc_before

        self._start_stats: Optional[MemoryStats] = None
        self._end_stats: Optional[MemoryStats] = None

    @property
    def start_stats(self) -> Optional[MemoryStats]:
        """Memory stats at the start of the monitored block."""
        return self._start_stats

    @property
    def end_stats(self) -> Optional[MemoryStats]:
        """Memory stats at the end of the monitored block."""
        return self._end_stats

    @property
    def delta_allocated(self) -> int:
        """Change in allocated memory during the monitored block (bytes)."""
        if self._start_stats is None or self._end_stats is None:
            return 0
        return self._end_stats.allocated - self._start_stats.allocated

    def get_stats(self) -> dict[str, float]:
        """
        Current VRAM snapshot as a plain dict.

        Used by the training loop, which logs scalars rather than MemoryStats
        objects and must stay silent on CPU-only runs.
        """
        if not torch.cuda.is_available():
            return {"used_mb": 0.0, "peak_mb": 0.0, "total_mb": 0.0, "percent_used": 0.0}

        stats = get_memory_stats(self.device)
        return {
            "used_mb": stats.allocated / 1e6,
            "peak_mb": stats.peak_allocated / 1e6,
            "total_mb": stats.total / 1e6,
            "percent_used": stats.utilization_pct,
        }

    def __enter__(self) -> "VRAMMonitor":
        if self.gc_before:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        reset_memory_stats(self.device)
        self._start_stats = get_memory_stats(self.device)
        return self

    def __exit__(self, *args: object) -> None:
        self._end_stats = get_memory_stats(self.device)

        if self._start_stats and self._end_stats:
            delta = self.delta_allocated
            peak = self._end_stats.peak_allocated

            prefix = f"[{self.label}] " if self.label else ""
            logger.debug(
                f"{prefix}VRAM delta: {format_bytes(delta)}, "
                f"peak: {format_bytes(peak)}, "
                f"now: {format_bytes(self._end_stats.allocated)} "
                f"({self._end_stats.utilization_pct:.1f}%)"
            )

            # Warn if approaching OOM
            if self._end_stats.utilization_pct >= self.warning_threshold_pct:
                logger.warning(
                    f"{prefix}VRAM usage is {self._end_stats.utilization_pct:.1f}% "
                    f"({format_bytes(self._end_stats.allocated)} / "
                    f"{format_bytes(self._end_stats.total)}). "
                    f"Approaching OOM threshold!"
                )

    def log_status(self, prefix: str = "") -> MemoryStats:
        """
        Log current VRAM status and return stats snapshot.

        Args:
            prefix: Optional prefix for the log message.

        Returns:
            Current MemoryStats snapshot.
        """
        stats = get_memory_stats(self.device)
        label = prefix or self.label
        msg_prefix = f"[{label}] " if label else ""

        logger.info(
            f"{msg_prefix}VRAM: {format_bytes(stats.allocated)} allocated "
            f"/ {format_bytes(stats.total)} total "
            f"({stats.utilization_pct:.1f}%)"
        )

        if stats.utilization_pct >= self.warning_threshold_pct:
            logger.warning(
                f"{msg_prefix}VRAM utilization ({stats.utilization_pct:.1f}%) "
                f"exceeds warning threshold ({self.warning_threshold_pct:.0f}%)!"
            )

        return stats

    def __repr__(self) -> str:
        return (
            f"VRAMMonitor(label='{self.label}', "
            f"warning_threshold={self.warning_threshold_pct}%)"
        )


@contextmanager
def track_vram(
    label: str = "",
    device: Optional[str | torch.device] = None,
    gc_before: bool = False,
) -> Generator[VRAMMonitor, None, None]:
    """
    Convenience context manager for VRAM tracking.

    Args:
        label: Label for the tracked operation.
        device: CUDA device to monitor.
        gc_before: Run GC before measurement.

    Yields:
        VRAMMonitor instance with stats populated after exit.

    Example:
        >>> with track_vram("model forward") as mon:
        ...     output = model(input_ids)
        >>> print(f"Used {format_bytes(mon.delta_allocated)}")
    """
    monitor = VRAMMonitor(label=label, device=device, gc_before=gc_before)
    with monitor:
        yield monitor
