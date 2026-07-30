import time
from typing import Tuple
import torch
import torch.nn as nn
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning("Triton not found. Using PyTorch fallback for RMSNorm.")

def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    PyTorch reference implementation for RMSNorm.
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x

if HAS_TRITON:
    @triton.jit
    def _rms_norm_fwd_kernel(
        x_ptr, y_ptr, w_ptr,
        stride_x_row, stride_y_row,
        N, eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Map program ID to row
        row_idx = tl.program_id(0)
        x_row_ptr = x_ptr + row_idx * stride_x_row
        y_row_ptr = y_ptr + row_idx * stride_y_row

        # Create offsets and mask
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # Load x and w
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # Compute variance
        x_sq = x * x
        var = tl.sum(x_sq, axis=0) / N
        rsqrt = tl.math.rsqrt(var + eps)

        # Normalize and apply weight
        y = x * rsqrt * w

        # Store result
        tl.store(y_row_ptr + cols, y.to(x_ptr.dtype.element_ty), mask=mask)

def fused_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Applies RMSNorm using Triton if available, else falls back to PyTorch.
    """
    if not HAS_TRITON or not x.is_cuda:
        logger.debug("RMSNorm: using pytorch_fallback")
        return rms_norm_ref(x, weight, eps)
    
    logger.debug("RMSNorm: using triton")
    
    # Reshape input to 2D
    x_shape = x.shape
    x_2d = x.view(-1, x_shape[-1])
    M, N = x_2d.shape

    # Allocate output
    y = torch.empty_like(x_2d)

    # Triton constants
    # Max block size for T4 is generally around 1024 or 2048 depending on registers
    # We round up N to next power of 2, up to max sizes
    MAX_FUSED_SIZE = 65536
    if N > MAX_FUSED_SIZE:
        return rms_norm_ref(x, weight, eps) # Fallback for extremely large hidden sizes

    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Adjust warps for T4 compute capability 7.5
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    
    _rms_norm_fwd_kernel[(M,)](
        x_2d, y, weight,
        x_2d.stride(0), y.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return y.view(*x_shape)

class VasudhaRMSNorm(nn.Module):
    """
    RMSNorm module that uses Triton-fused kernel under the hood.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rms_norm(x, self.weight, self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hidden_size={self.weight.shape[0]}, eps={self.eps})"

def benchmark_rms_norm(hidden_size: int = 4096, batch_size: int = 32, seq_len: int = 128):
    """
    Benchmark Triton vs PyTorch RMSNorm implementation.
    """
    if not torch.cuda.is_available():
        print("CUDA not available for benchmark.")
        return

    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    weight = torch.randn(hidden_size, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(10):
        _ = rms_norm_ref(x, weight)
        if HAS_TRITON:
            _ = fused_rms_norm(x, weight)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = rms_norm_ref(x, weight)
    torch.cuda.synchronize()
    pt_time = (time.time() - start) / 100

    triton_time = None
    if HAS_TRITON:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = fused_rms_norm(x, weight)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100

    print(f"RMSNorm PT Time: {pt_time * 1000:.3f} ms")
    if triton_time:
        print(f"RMSNorm Triton Time: {triton_time * 1000:.3f} ms")
        print(f"Speedup: {pt_time / triton_time:.2f}x")
