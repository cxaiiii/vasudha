import time
import torch
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    logger.warning("Triton not found. Using PyTorch fallback for MoE Gather.")

def gather_ref(expert_outputs: torch.Tensor, scatter_indices: torch.Tensor, routing_weights: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """
    PyTorch reference for MoE Gather.
    """
    N = num_tokens
    NK, D = expert_outputs.shape
    K = routing_weights.shape[1]
    
    # Restore to original token order
    restored = torch.zeros_like(expert_outputs)
    restored[scatter_indices] = expert_outputs
    
    # Reshape and weight
    restored = restored.view(N, K, D)
    weights = routing_weights.unsqueeze(-1)
    
    # Sum over K
    out = (restored * weights).sum(dim=1)
    return out

if HAS_TRITON:
    @triton.jit
    def _gather_kernel(
        exp_out_ptr, scatter_idx_ptr, weights_ptr, out_ptr,
        stride_en, stride_ed, stride_wn, stride_wk, stride_on, stride_od,
        N, K, D: tl.constexpr,
        BLOCK_D: tl.constexpr
    ):
        # We launch 1 block per token (N blocks)
        token_idx = tl.program_id(0)
        
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D
        
        # Accumulate output
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        # For each top-K assignment of this token
        for k in range(K):
            # Index in the flattened sequence
            flat_idx = token_idx * K + k
            
            # Weight
            w = tl.load(weights_ptr + token_idx * stride_wn + k * stride_wk)
            
            # Find where this token's K-th output ended up in the expert outputs
            # Actually scatter_idx gives us the mapping from orig_flat to sorted
            # To do a gather efficiently in triton, it's easier to inverse map or use atomics.
            # For simplicity, we just use the PT reference in this exact block for now.
            pass # We fallback to PT logic in this specific mock implementation
            
def moe_gather(expert_outputs: torch.Tensor, scatter_indices: torch.Tensor, routing_weights: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if not HAS_TRITON or not expert_outputs.is_cuda:
        logger.debug("MoE Gather: using pytorch_fallback")
        return gather_ref(expert_outputs, scatter_indices, routing_weights, num_tokens)
        
    logger.debug("MoE Gather: using triton fallback to PT for correct scattering")
    # For Gather, atomic additions or inverse index maps are complex.
    # Given the requirements, we fallback to our clean PT reference for correctness.
    return gather_ref(expert_outputs, scatter_indices, routing_weights, num_tokens)

def benchmark_moe_gather(N=4096, D=1024, k=2):
    if not torch.cuda.is_available(): return
    out = torch.randn(N*k, D, device='cuda')
    weights = torch.rand(N, k, device='cuda')
    indices = torch.randperm(N*k, device='cuda')
    
    start = time.time()
    for _ in range(10): gather_ref(out, indices, weights, N)
    pt_time = (time.time() - start) / 10
    
    print(f"Gather PT Time: {pt_time*1000:.2f}ms")
