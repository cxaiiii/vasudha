import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any

try:
    from vasudha.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from vasudha.kernels.swiglu import fused_swiglu
except ImportError:
    fused_swiglu = None
    logger.info("fused_swiglu not available. Using pure PyTorch SwiGLU.")

class VasudhaExpert(nn.Module):
    """
    A single SwiGLU Feed Forward Network (FFN) expert.
    
    This acts as one of the many experts in the MoE layer, or as the dense FFN
    in standard non-MoE layers.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the expert.
        Output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
        """
        if fused_swiglu is not None:
            # Assumes fused_swiglu takes gate and up projections
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            hidden = fused_swiglu(gate, up)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            hidden = F.silu(gate) * up
            
        return self.down_proj(hidden)
        
    def __repr__(self) -> str:
        return f"VasudhaExpert(hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size})"

class VasudhaExpertGroup(nn.Module):
    """
    A group of N experts sharing batched weights to allow expert-parallel GEMM.
    
    Instead of running N sequential Linear layers, we batch the weights and use 
    torch.bmm or einsum for efficient execution when processing multiple experts.
    """
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Shape: (num_experts, hidden_size, intermediate_size)
        self.gate_weight = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.up_weight = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        
        # Shape: (num_experts, intermediate_size, hidden_size)
        self.down_weight = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # standard initialization
        nn.init.kaiming_uniform_(self.gate_weight, a=torch.math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_weight, a=torch.math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_weight, a=torch.math.sqrt(5))

    def forward_single_expert(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        Processes inputs through a single expert specified by expert_idx.
        """
        gate_w = self.gate_weight[expert_idx]
        up_w = self.up_weight[expert_idx]
        down_w = self.down_weight[expert_idx]
        
        gate = torch.matmul(x, gate_w)
        up = torch.matmul(x, up_w)
        
        if fused_swiglu is not None:
            hidden = fused_swiglu(gate, up)
        else:
            hidden = F.silu(gate) * up
            
        return torch.matmul(hidden, down_w)
        
    def forward_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Processes tokens assigned to experts in mask.
        mask is a boolean tensor of shape (num_experts, num_tokens).
        """
        # This implementation falls back to iterating over active experts for simplicity,
        # but in practice, a grouped GEMM kernel is preferred.
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            expert_mask = mask[i]
            if not expert_mask.any():
                continue
                
            tokens = x[expert_mask]
            out = self.forward_single_expert(tokens, i)
            output[expert_mask] = out
            
        return output
        
    def __repr__(self) -> str:
        return (f"VasudhaExpertGroup(num_experts={self.num_experts}, "
                f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size})")
