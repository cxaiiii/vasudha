import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    from vasudha.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from vasudha.kernels.moe_dispatch import moe_dispatch
    from vasudha.kernels.moe_gather import moe_gather
except ImportError:
    moe_dispatch = None
    moe_gather = None
    logger.info("moe_dispatch or moe_gather kernels not available, using PyTorch fallback.")

from .expert import VasudhaExpert, VasudhaExpertGroup
from .router import VasudhaRouter
from .capacity import CapacityManager

class VasudhaMoELayer(nn.Module):
    """
    The full Mixture of Experts (MoE) FFN layer.
    
    Includes a router, capacity management, and expert group routing.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # Each expert is a *slice* of the dense FFN, not a copy of it. Using
        # intermediate_size here would give every expert the full dense width,
        # inflating the 4B config from ~3.6B to ~10.3B parameters.
        self.moe_intermediate_size = getattr(
            config, "moe_intermediate_size", config.intermediate_size
        )
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.capacity_factor = getattr(config, 'capacity_factor', 1.25)
        
        self.router = VasudhaRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.num_experts_per_tok
        )
        
        self.capacity_manager = CapacityManager()
        
        # Grouped experts for efficient batched execution
        self.experts = VasudhaExpertGroup(
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MoE layer.
        
        Returns:
            output (Tensor): Processed hidden states
            aux_loss (Tensor): Scalar loss for load balancing and z-loss
        """
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            flat_hidden = hidden_states.view(-1, self.hidden_size)
        else:
            flat_hidden = hidden_states

        num_tokens = flat_hidden.size(0)
        
        # 1. Router
        router_output = self.router(flat_hidden)
        routing_weights = router_output.routing_weights
        expert_indices = router_output.expert_indices
        aux_loss = router_output.aux_loss
        
        # 2. Capacity Constraints
        capacity = self.capacity_manager.compute_capacity(
            num_tokens=num_tokens,
            capacity_factor=self.capacity_factor,
            num_experts=self.num_experts
        )
        
        expert_indices, pruned_weights, overflow_mask = self.capacity_manager.apply_capacity(
            expert_indices=expert_indices,
            routing_weights=routing_weights,
            capacity=capacity
        )
        
        # 3, 4, 5, 6. Dispatch, Execute, Gather, and Weight
        if moe_dispatch is not None and moe_gather is not None:
            # Use optimized kernels
            # dispatched_tokens, token_to_expert_map, sort_idx = moe_dispatch(flat_hidden, pruned_weights, expert_indices, self.num_experts)
            # Placeholder for kernel-based expert group call
            pass

        # Fallback PyTorch implementation
        final_output = torch.zeros_like(flat_hidden)
        
        for i in range(self.num_experts_per_tok):
            # For each top-k choice
            indices = expert_indices[:, i]
            weights = pruned_weights[:, i].unsqueeze(-1) # (N, 1)
            
            # Mask of valid tokens (not overflowed)
            valid_mask = ~overflow_mask[:, i]
            
            # Iterate through experts and process their tokens
            for expert_idx in range(self.num_experts):
                expert_mask = (indices == expert_idx) & valid_mask
                if not expert_mask.any():
                    continue
                    
                tokens = flat_hidden[expert_mask]
                expert_out = self.experts.forward_single_expert(tokens, expert_idx)
                
                # Apply routing weight and add to output
                final_output[expert_mask] += expert_out * weights[expert_mask]
                
        if len(orig_shape) == 3:
            final_output = final_output.view(*orig_shape)
            
        return final_output, aux_loss

    def __repr__(self) -> str:
        return (f"VasudhaMoELayer(hidden_size={self.hidden_size}, num_experts={self.num_experts}, "
                f"top_k={self.num_experts_per_tok})")

class VasudhaDecoderLayerFFN(nn.Module):
    """
    Selects between MoE FFN and Dense FFN based on the layer index.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Typical config assumes `moe_layers` is a list of odd indices
        if hasattr(config, 'moe_layers') and layer_idx in config.moe_layers:
            self.ffn = VasudhaMoELayer(config)
            self.is_moe = True
        else:
            self.ffn = VasudhaExpert(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size
            )
            self.is_moe = False
            
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_moe:
            return self.ffn(hidden_states)
        else:
            # Dense layer doesn't produce aux loss
            zero_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype, requires_grad=True)
            return self.ffn(hidden_states), zero_loss

    def __repr__(self) -> str:
        return f"VasudhaDecoderLayerFFN(layer_idx={self.layer_idx}, is_moe={self.is_moe})"
