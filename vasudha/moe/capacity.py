import torch
from dataclasses import dataclass
from typing import Tuple, Dict

try:
    from vasudha.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

@dataclass
class CapacityStats:
    tokens_dropped: int
    overflow_fraction: float
    per_expert_counts: torch.Tensor

class CapacityManager:
    """
    Handles token overflow when experts exceed their capacity constraint.
    """
    def __init__(self):
        pass

    def compute_capacity(self, num_tokens: int, capacity_factor: float, num_experts: int) -> int:
        """
        Computes the maximum number of tokens an expert can process.
        """
        return int((capacity_factor * num_tokens) / num_experts)

    def apply_capacity(self, expert_indices: torch.Tensor, routing_weights: torch.Tensor, capacity: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Drops tokens assigned to overloaded experts by setting their weight to 0.
        
        Args:
            expert_indices: (N, k) indices of selected experts
            routing_weights: (N, k) weights for the selected experts
            capacity: max tokens per expert
            
        Returns:
            pruned_indices: (N, k)
            pruned_weights: (N, k)
            overflow_mask: (N, k) boolean mask of dropped tokens
        """
        N, k = expert_indices.shape
        num_experts = int(expert_indices.max().item()) + 1
        
        # Initialize counts
        expert_counts = torch.zeros(num_experts, dtype=torch.long, device=expert_indices.device)
        overflow_mask = torch.zeros_like(expert_indices, dtype=torch.bool)
        
        pruned_weights = routing_weights.clone()
        
        # Iterating over top-k
        for i in range(k):
            # For each token's i-th choice
            indices = expert_indices[:, i]
            
            # Simple fallback loop for overflow computation (can be parallelized with scatter_add)
            for token_idx in range(N):
                exp_idx = indices[token_idx].item()
                if expert_counts[exp_idx] < capacity:
                    expert_counts[exp_idx] += 1
                else:
                    overflow_mask[token_idx, i] = True
                    pruned_weights[token_idx, i] = 0.0
                    
        # Re-normalize weights for non-dropped tokens if a choice was dropped
        # (Optional, but helps keep weight sum to 1 if one of top-k is dropped)
        weight_sums = pruned_weights.sum(dim=-1, keepdim=True)
        # Avoid division by zero
        weight_sums[weight_sums == 0] = 1.0
        pruned_weights = pruned_weights / weight_sums
        
        return expert_indices, pruned_weights, overflow_mask

    def log_utilization(self, expert_indices: torch.Tensor, num_experts: int) -> Dict[int, int]:
        """
        Returns a dictionary of per-expert token counts.
        """
        counts = torch.bincount(expert_indices.flatten(), minlength=num_experts)
        return {i: counts[i].item() for i in range(num_experts)}
        
    def __repr__(self) -> str:
        return "CapacityManager()"
