import os
import time
import torch
import torch.nn.functional as F
from typing import Tuple
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

# Fused kernels here are forward-only. Opt in explicitly; never in training.
_ALLOW_TRITON = os.environ.get("VASUDHA_USE_TRITON") == "1"

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning("Triton not found. Using PyTorch fallback for SwiGLU.")

def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation for SwiGLU.
    """
    return F.silu(gate) * up

def swiglu_inplace_ref(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference for in-place SwiGLU splitting.
    """
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up

if HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr, up_ptr, out_ptr,
        stride_gate_row, stride_up_row, stride_out_row,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        gate_row_ptr = gate_ptr + row_idx * stride_gate_row
        up_row_ptr = up_ptr + row_idx * stride_up_row
        out_row_ptr = out_ptr + row_idx * stride_out_row

        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        gate = tl.load(gate_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # SiLU: x * sigmoid(x)
        sigmoid_gate = 1.0 / (1.0 + tl.exp(-gate))
        silu_gate = gate * sigmoid_gate
        out = silu_gate * up

        tl.store(out_row_ptr + cols, out.to(gate_ptr.dtype.element_ty), mask=mask)

def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Applies SwiGLU using Triton if available, else PyTorch.
    """
    # The Triton kernel is forward-only: it writes into a fresh `torch.empty`,
    # so its output carries no grad_fn and every FFN gradient is silently
    # dropped. Training didn't crash because the residual branch still had one.
    #
    # Checking `is_grad_enabled()` is not sufficient: gradient checkpointing runs
    # its first pass under no_grad and recomputes under grad, which would put the
    # two passes on different code paths. Default to eager and make Triton an
    # explicit opt-in (VASUDHA_USE_TRITON=1) for inference-only work.
    needs_grad = torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad)
    if not _ALLOW_TRITON or not HAS_TRITON or not gate.is_cuda or needs_grad:
        logger.debug("SwiGLU: using pytorch_fallback")
        return swiglu_ref(gate, up)

    logger.debug("SwiGLU: using triton")
    
    gate_2d = gate.view(-1, gate.shape[-1])
    up_2d = up.view(-1, up.shape[-1])
    M, N = gate_2d.shape

    out = torch.empty_like(gate_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    
    _swiglu_fwd_kernel[(M,)](
        gate_2d, up_2d, out,
        gate_2d.stride(0), up_2d.stride(0), out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out.view(*gate.shape)

def fused_swiglu_inplace(x: torch.Tensor) -> torch.Tensor:
    """
    Applies SwiGLU on a single concatenated tensor of shape (*, 2*I).
    """
    if not HAS_TRITON or not x.is_cuda:
        logger.debug("SwiGLU inplace: using pytorch_fallback")
        return swiglu_inplace_ref(x)
    
    gate, up = x.chunk(2, dim=-1)
    return fused_swiglu(gate, up)

def benchmark_swiglu(hidden_size: int = 4096, batch_size: int = 32, seq_len: int = 128):
    """
    Benchmark Triton vs PyTorch SwiGLU implementation.
    """
    if not torch.cuda.is_available():
        return

    gate = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    up = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(10):
        _ = swiglu_ref(gate, up)
        if HAS_TRITON:
            _ = fused_swiglu(gate, up)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = swiglu_ref(gate, up)
    torch.cuda.synchronize()
    pt_time = (time.time() - start) / 100

    triton_time = None
    if HAS_TRITON:
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = fused_swiglu(gate, up)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 100

    print(f"SwiGLU PT Time: {pt_time * 1000:.3f} ms")
    if triton_time:
        print(f"SwiGLU Triton Time: {triton_time * 1000:.3f} ms")
        print(f"Speedup: {pt_time / triton_time:.2f}x")
