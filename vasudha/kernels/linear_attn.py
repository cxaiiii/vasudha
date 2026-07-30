import time
import torch
import torch.nn as nn
from typing import Tuple, Optional
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning("Triton not found. Using PyTorch fallback for GLA Linear Attn.")

def gla_chunkwise_ref(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, 
    chunk_size: int = 64, initial_state: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch reference for GLA Chunkwise Parallel Scan.
    Shapes:
    - q: (B, H, L, d_head)
    - k, v, g: (B, Hkv, L, d_head)
    """
    B, H, L, D = q.shape
    Hkv = k.shape[1]
    
    # Very slow naive elementwise scanning for reference correctness
    out = torch.zeros_like(q)
    
    # State tracking per KV head: (B, Hkv, D, D)
    state = torch.zeros(B, Hkv, D, D, device=q.device, dtype=q.dtype)
    if initial_state is not None:
        state = initial_state.clone()

    for i in range(L):
        # Update state with current k, v, and decay g
        q_i = q[:, :, i, :] # (B, H, D)
        k_i = k[:, :, i, :] # (B, Hkv, D)
        v_i = v[:, :, i, :] # (B, Hkv, D)
        g_i = g[:, :, i, :] # (B, Hkv, D)

        # Broadcast KV to H for query
        repeats = H // Hkv
        
        # State decay and add
        k_i_exp = k_i.unsqueeze(-1) # (B, Hkv, D, 1)
        v_i_exp = v_i.unsqueeze(-2) # (B, Hkv, 1, D)
        kv_i = k_i_exp @ v_i_exp    # (B, Hkv, D, D)
        
        g_i_exp = torch.exp(g_i).unsqueeze(-1)
        state = state * g_i_exp + kv_i
        
        # Query out
        state_repeated = state.repeat_interleave(repeats, dim=1) # (B, H, D, D)
        out_i = (q_i.unsqueeze(-2) @ state_repeated).squeeze(-2) # (B, H, D)
        out[:, :, i, :] = out_i
        
    return out, state

# Simplified forward triton kernel for GLA
if HAS_TRITON:
    @triton.jit
    def _gla_fwd_kernel(
        q_ptr, k_ptr, v_ptr, g_ptr, out_ptr, state_ptr,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_kb, stride_kh, stride_kl, stride_kd,
        stride_ob, stride_oh, stride_ol, stride_od,
        B, H, L, D: tl.constexpr, Hkv: tl.constexpr,
        BLOCK_L: tl.constexpr
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        kv_head_idx = head_idx // (H // Hkv)

        # Offsets
        q_offset = batch_idx * stride_qb + head_idx * stride_qh
        k_offset = batch_idx * stride_kb + kv_head_idx * stride_kh
        o_offset = batch_idx * stride_ob + head_idx * stride_oh

        # Simplified state accumulation (single block L)
        state = tl.zeros((D, D), dtype=tl.float32)
        
        d_cols = tl.arange(0, D)
        for i in range(L):
            q_i = tl.load(q_ptr + q_offset + i * stride_ql + d_cols)
            k_i = tl.load(k_ptr + k_offset + i * stride_kl + d_cols)
            v_i = tl.load(v_ptr + k_offset + i * stride_kl + d_cols) # v uses k offsets for KV
            g_i = tl.load(g_ptr + k_offset + i * stride_kl + d_cols)
            
            # Decay state (elementwise simplified)
            decay = tl.exp(g_i)
            # Outer product of k and v
            kv_add = k_i[:, None] * v_i[None, :]
            
            # State update
            state = state * decay[:, None] + kv_add
            
            # Query
            out_i = tl.sum(q_i[:, None] * state, axis=0)
            tl.store(out_ptr + o_offset + i * stride_ol + d_cols, out_i.to(out_ptr.dtype.element_ty))

def gla_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, 
                chunk_size: int = 64, initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TRITON or not q.is_cuda:
        logger.debug("GLA Linear Attn: using pytorch_fallback")
        return gla_chunkwise_ref(q, k, v, g, chunk_size, initial_state)
        
    logger.debug("GLA Linear Attn: using triton")
    
    B, H, L, D = q.shape
    Hkv = k.shape[1]
    out = torch.empty_like(q)
    final_state = torch.zeros(B, Hkv, D, D, device=q.device, dtype=q.dtype) # Dummy state return
    
    # We round D to power of 2 for simplicity
    D_PAD = triton.next_power_of_2(D)
    
    _gla_fwd_kernel[(B, H)](
        q, k, v, g, out, final_state,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, L, D_PAD, Hkv,
        BLOCK_L=chunk_size
    )
    
    return out, final_state

def benchmark_linear_attn(B=2, H=8, L=256, D=64):
    if not torch.cuda.is_available(): return
    Hkv = 2
    q = torch.randn(B, H, L, D, device='cuda')
    k = torch.randn(B, Hkv, L, D, device='cuda')
    v = torch.randn(B, Hkv, L, D, device='cuda')
    g = -torch.rand(B, Hkv, L, D, device='cuda')

    start = time.time()
    gla_chunkwise_ref(q, k, v, g)
    pt_time = time.time() - start

    if HAS_TRITON:
        start = time.time()
        gla_forward(q, k, v, g)
        tr_time = time.time() - start
        print(f"GLA PT: {pt_time*1000:.2f}ms, Triton: {tr_time*1000:.2f}ms")
