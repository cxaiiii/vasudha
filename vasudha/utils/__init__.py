"""
Vasudha utilities package.

Provides:
  - Rich-based structured logging
  - VRAM / memory monitoring
  - dtype utilities
"""

from vasudha.utils.logging import get_logger, setup_logging
from vasudha.utils.memory import VRAMMonitor, get_memory_stats, format_bytes
from vasudha.utils.dtype import (
    str_to_dtype,
    dtype_to_str,
    get_compute_dtype,
    is_bf16_supported,
)

__all__ = [
    "get_logger",
    "setup_logging",
    "VRAMMonitor",
    "get_memory_stats",
    "format_bytes",
    "str_to_dtype",
    "dtype_to_str",
    "get_compute_dtype",
    "is_bf16_supported",
]
