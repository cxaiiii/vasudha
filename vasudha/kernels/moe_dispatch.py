import time
import torch
from typing import Tuple
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning("Triton not found. Using PyTorch fallback for MoE Dispatch.")

def dispatch_ref(tokens: torch.Tensor, routing_weights: torch.Tensor, expert_indices: torch.Tensor, num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch reference for MoE Dispatch.
    """
    N, D = tokens.shape
    k = expert_indices.shape[1]
    
    # Expand tokens for each chosen expert
    dispatched = tokens.unsqueeze(1).expand(-1, k, -1).reshape(N * k, D)
    
    # Flatten expert assignments
    flat_indices = expert_indices.flatten()
    
    # Sort to group by expert
    sorted_expert_indices, sort_idx = flat_indices.sort()
    
    dispatched_sorted = dispatched[sort_idx]
    
    return dispatched_sorted, sorted_expert_indices, sort_idx

if HAS_TRITON:
    @triton.jit
    def _dispatch_kernel(
        tokens_ptr, sorted_idx_ptr, out_ptr,
        stride_tn, stride_td, stride_on, stride_od,
        N, K, D: tl.constexpr,
        BLOCK_D: tl.constexpr
    ):
        row_idx = tl.program_id(0) # ranges from 0 to N*K
        
        # Get original index from sort
        orig_row = tl.load(sorted_idx_ptr + row_idx)
        orig_token_idx = orig_row // K
        
        # Load token and store
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        
        token = tl.load(tokens_ptr + orig_token_idx * stride_tn + cols, mask=mask)
        tl.store(out_ptr + row_idx * stride_on + cols, token, mask=mask)

def moe_dispatch(tokens: torch.Tensor, routing_weights: torch.Tensor, expert_indices: torch.Tensor, num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not HAS_TRITON or not tokens.is_cuda:
        logger.debug("MoE Dispatch: using pytorch_fallback")
        return dispatch_ref(tokens, routing_weights, expert_indices, num_experts)
        
    logger.debug("MoE Dispatch: using triton")
    
    N, D = tokens.shape
    K = expert_indices.shape[1]
    
    flat_indices = expert_indices.flatten()
    sorted_expert_indices, sort_idx = flat_indices.sort()
    
    out = torch.empty((N * K, D), device=tokens.device, dtype=tokens.dtype)
    
    BLOCK_D = triton.next_power_of_2(D)
    
    _dispatch_kernel[(N * K,)](
        tokens, sort_idx, out,
        tokens.stride(0), tokens.stride(1), out.stride(0), out.stride(1),
        N, K, D,
        BLOCK_D=BLOCK_D
    )
    
    return out, sorted_expert_indices, sort_idx

def benchmark_moe_dispatch(N=4096, D=1024, num_experts=8, k=2):
    if not torch.cuda.is_available(): return
    tokens = torch.randn(N, D, device='cuda')
    weights = torch.rand(N, k, device='cuda')
    indices = torch.randint(0, num_experts, (N, k), device='cuda')
    
    start = time.time()
    for _ in range(10): dispatch_ref(tokens, weights, indices, num_experts)
    pt_time = (time.time() - start) / 10
    
    if HAS_TRITON:
        start = time.time()
        for _ in range(10): moe_dispatch(tokens, weights, indices, num_experts)
        tr_time = (time.time() - start) / 10
        print(f"Dispatch PT: {pt_time*1000:.2f}ms, Triton: {tr_time*1000:.2f}ms")
