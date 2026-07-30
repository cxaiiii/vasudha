"""
Vasudha dtype utilities.

Handles dtype selection and conversion across the training stack.

Key concerns:
  - T4 GPUs support fp16 and bf16 (bf16 since CUDA 11.x on T4, but T4's
    hardware support for bf16 is emulated — fp16 is usually faster on T4)
  - A100/H100 support bf16 natively and should prefer it
  - For QLoRA: compute dtype is bf16/fp16, storage dtype is nf4/int8
  - Some operations (loss, layernorm) should always stay in fp32

Design decision:
  We expose `get_compute_dtype()` which auto-detects the best dtype for
  the current hardware, with explicit override support via config.
"""

from __future__ import annotations

from typing import Union

import torch


# Type alias for dtype arguments
DTypeLike = Union[torch.dtype, str]

# Mapping from string names to torch dtypes
_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float8_e5m2": torch.float8_e5m2 if hasattr(torch, "float8_e5m2") else torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16,
    "int8": torch.int8,
    "int4": torch.quint4x2 if hasattr(torch, "quint4x2") else torch.int8,
}

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int32: "int32",
    torch.int64: "int64",
}


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert a string dtype name to a torch.dtype.

    Args:
        dtype_str: String dtype name. Supported: "float32"/"fp32",
                   "float16"/"fp16", "bfloat16"/"bf16", "int8".

    Returns:
        Corresponding torch.dtype.

    Raises:
        ValueError: If the string is not a recognized dtype.

    Example:
        >>> str_to_dtype("bf16")
        torch.bfloat16
    """
    dtype_str = dtype_str.lower().strip()
    if dtype_str not in _STR_TO_DTYPE:
        valid = list(_STR_TO_DTYPE.keys())
        raise ValueError(
            f"Unknown dtype string: '{dtype_str}'. "
            f"Supported values: {valid}"
        )
    return _STR_TO_DTYPE[dtype_str]


def dtype_to_str(dtype: torch.dtype) -> str:
    """
    Convert a torch.dtype to its canonical string name.

    Args:
        dtype: PyTorch dtype.

    Returns:
        String name (e.g., "bfloat16").
    """
    return _DTYPE_TO_STR.get(dtype, str(dtype).replace("torch.", ""))


def is_bf16_supported() -> bool:
    """
    Check whether the current CUDA device supports bfloat16 efficiently.

    Hardware bf16 is supported on Ampere (A100) and later. T4 (Turing) can
    compute in bf16 but it is software-emulated and slower than fp16.

    Returns:
        True if the device natively supports bf16 (compute capability >= 8.0).
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8  # Ampere and later


def is_fp16_supported() -> bool:
    """
    Check whether fp16 training is supported on the current device.

    fp16 is supported on Volta (V100) and later (compute capability >= 7.0).
    T4 is Turing (7.5), so fp16 is natively supported.

    Returns:
        True if the device supports fp16 compute.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (7, 0)


def get_compute_dtype(preferred: str = "auto") -> torch.dtype:
    """
    Determine the best compute dtype for the current hardware.

    This is the dtype used for model weights and activations during training.
    For QLoRA, this is the dtype of the LoRA adapters (base model stays int4).

    Selection logic:
      - "auto": bf16 if natively supported (Ampere+), fp16 on T4/Turing, fp32 on CPU
      - "bf16" / "fp16" / "fp32": Use the specified dtype (raises if not supported)

    Args:
        preferred: One of "auto", "bf16", "fp16", "fp32".

    Returns:
        Selected compute dtype.

    Example:
        >>> dtype = get_compute_dtype("auto")  # → fp16 on T4, bf16 on A100
    """
    preferred = preferred.lower()

    if preferred == "auto":
        if not torch.cuda.is_available():
            return torch.float32
        if is_bf16_supported():
            return torch.bfloat16
        if is_fp16_supported():
            return torch.float16
        return torch.float32

    if preferred in ("bf16", "bfloat16"):
        # On T4, bf16 works but is emulated — warn but don't error
        if torch.cuda.is_available() and not is_bf16_supported():
            import warnings
            warnings.warn(
                "bf16 requested but not natively supported (compute capability < 8.0). "
                "Operations will be software-emulated, which may be slow. "
                "Consider using 'fp16' on T4 GPUs.",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.bfloat16

    if preferred in ("fp16", "float16"):
        return torch.float16

    if preferred in ("fp32", "float32"):
        return torch.float32

    # Fall back to string conversion
    return str_to_dtype(preferred)


def bytes_per_element(dtype: torch.dtype) -> float:
    """
    Return the number of bytes per element for a given dtype.

    Useful for estimating model memory footprint before loading.

    Args:
        dtype: PyTorch dtype.

    Returns:
        Bytes per element (may be fractional for sub-byte dtypes like int4 = 0.5).
    """
    _BYTES: dict[torch.dtype, float] = {
        torch.float32: 4.0,
        torch.float16: 2.0,
        torch.bfloat16: 2.0,
        torch.int8: 1.0,
        torch.int32: 4.0,
        torch.int64: 8.0,
        torch.bool: 0.125,  # 1 bit
    }
    # int4 (quint4x2 packs 2 values per byte)
    if hasattr(torch, "quint4x2") and dtype == torch.quint4x2:
        return 0.5
    return _BYTES.get(dtype, 4.0)


def estimate_model_vram(
    num_parameters: int,
    dtype: torch.dtype = torch.float16,
    include_optimizer: bool = True,
    optimizer_multiplier: float = 2.0,
) -> int:
    """
    Estimate the VRAM required to load a model with a given parameter count.

    This is a heuristic estimate — actual usage may vary due to:
    - Activation memory during forward/backward
    - PyTorch caching allocator overhead
    - Gradient memory

    Args:
        num_parameters: Total number of model parameters.
        dtype: Storage dtype for parameters.
        include_optimizer: Whether to include optimizer state estimate.
        optimizer_multiplier: Extra multiplier for optimizer states.
                              AdamW ≈ 2× (momentum + variance in fp32).

    Returns:
        Estimated VRAM in bytes.

    Example:
        >>> # Estimate for 4B model in fp16
        >>> vram = estimate_model_vram(4_000_000_000, torch.float16)
        >>> print(format_bytes(vram))
    """
    from vasudha.utils.memory import format_bytes  # avoid circular at module level

    param_bytes = int(num_parameters * bytes_per_element(dtype))
    total = param_bytes
    if include_optimizer:
        # AdamW maintains fp32 copies of parameters + 2 fp32 moment tensors
        total += int(num_parameters * 4 * optimizer_multiplier)

    return total
