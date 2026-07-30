import os
import time
import torch
import torch.nn as nn
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
    logger.warning("Triton not found. Using PyTorch fallback for CrossEntropy.")

def chunked_cross_entropy_ref(logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 4096, ignore_index: int = -100) -> torch.Tensor:
    """
    PyTorch reference implementation for Chunked CrossEntropy.
    Processing in chunks avoids materializing the full softmax on large vocabs.
    """
    # Fallback to standard cross entropy for simplicity in reference
    # A true chunked reference would iterate over vocabs. For brevity and exact equivalence, 
    # we use standard F.cross_entropy here since it handles ignore_index natively.
    B, V = logits.shape
    return torch.nn.functional.cross_entropy(logits, labels, ignore_index=ignore_index, reduction='mean')

if HAS_TRITON:
    @triton.jit
    def _cross_entropy_fwd_kernel(
        logits_ptr, labels_ptr, loss_ptr,
        stride_logits_row, stride_labels_row, stride_loss_row,
        V, ignore_index,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        logits_row_ptr = logits_ptr + row_idx * stride_logits_row
        
        label = tl.load(labels_ptr + row_idx)
        
        # If ignore_index, write 0.0 loss
        if label == ignore_index:
            tl.store(loss_ptr + row_idx, 0.0)
            return

        # Compute max
        m_i = -float('inf')
        for i in range(0, V, BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < V
            l = tl.load(logits_row_ptr + cols, mask=mask, other=-float('inf'))
            m_i = tl.maximum(m_i, tl.max(l))
            
        # Compute sum(exp(x - max))
        l_i = 0.0
        for i in range(0, V, BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < V
            l = tl.load(logits_row_ptr + cols, mask=mask, other=-float('inf'))
            l_i += tl.sum(tl.exp(l - m_i))

        # log_sum_exp
        lse = m_i + tl.math.log(l_i)
        
        # Load label logit
        label_logit = tl.load(logits_row_ptr + label)
        
        # Loss = lse - label_logit
        loss = lse - label_logit
        tl.store(loss_ptr + row_idx, loss)

def vasudha_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100, chunk_size: int = 4096) -> torch.Tensor:
    """
    Triton-fused chunked cross entropy.
    """
    # Same trap as fused_swiglu: this kernel has no backward, so a loss built
    # from it has no grad_fn at all. Only safe under no_grad / eval.
    needs_grad = torch.is_grad_enabled() and logits.requires_grad
    if not _ALLOW_TRITON or not HAS_TRITON or not logits.is_cuda or needs_grad:
        logger.debug("CrossEntropy: using pytorch_fallback")
        return chunked_cross_entropy_ref(logits, labels, chunk_size, ignore_index)

    logger.debug("CrossEntropy: using triton")
    
    orig_shape = logits.shape
    logits_2d = logits.view(-1, orig_shape[-1])
    labels_1d = labels.view(-1)
    
    M, V = logits_2d.shape
    loss = torch.empty((M,), device=logits.device, dtype=torch.float32)
    
    BLOCK_SIZE = triton.next_power_of_2(min(V, 4096))
    
    _cross_entropy_fwd_kernel[(M,)](
        logits_2d, labels_1d, loss,
        logits_2d.stride(0), labels_1d.stride(0), loss.stride(0),
        V, ignore_index,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    valid_mask = labels_1d != ignore_index
    return loss[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=logits.device)

class VasudhaCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100, chunk_size: int = 4096):
        super().__init__()
        self.ignore_index = ignore_index
        self.chunk_size = chunk_size

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return vasudha_cross_entropy(logits, labels, self.ignore_index, self.chunk_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ignore_index={self.ignore_index}, chunk_size={self.chunk_size})"

def benchmark_cross_entropy(vocab_size: int = 151936, batch_size: int = 32, seq_len: int = 128):
    if not torch.cuda.is_available(): return
    logits = torch.randn(batch_size * seq_len, vocab_size, device='cuda', dtype=torch.float16)
    labels = torch.randint(0, vocab_size, (batch_size * seq_len,), device='cuda')

    start = time.time()
    for _ in range(10): chunked_cross_entropy_ref(logits, labels)
    torch.cuda.synchronize()
    pt_time = (time.time() - start) / 10

    if HAS_TRITON:
        start = time.time()
        for _ in range(10): vasudha_cross_entropy(logits, labels)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 10
        print(f"CE Speedup: {pt_time / triton_time:.2f}x")
