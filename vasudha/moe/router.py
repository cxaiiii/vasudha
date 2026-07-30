import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

try:
    from vasudha.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

@dataclass
class RouterOutput:
    routing_weights: torch.Tensor  # (N, k)
    expert_indices: torch.Tensor   # (N, k)
    aux_loss: torch.Tensor         # scalar
    expert_utilization: torch.Tensor # (num_experts,)

class VasudhaRouter(nn.Module):
    """
    Top-k soft routing for Mixture of Experts.
    
    Includes load balancing loss and z-loss to encourage balanced expert usage
    and stable routing logits.
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, router_jitter_noise: float = 0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise
        
        self.router_weights = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> RouterOutput:
        """
        Forward pass for the router.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) or (N, hidden_size)
            
        Returns:
            RouterOutput containing weights, indices, aux_loss, and utilization.
        """
        # Flatten hidden states to (N, hidden_size)
        original_shape = hidden_states.shape
        if len(original_shape) > 2:
            hidden_states = hidden_states.view(-1, self.hidden_size)
            
        N = hidden_states.size(0)
        
        if self.training and self.router_jitter_noise > 0.0:
            # Apply uniform jitter noise to inputs for exploration
            noise = torch.empty_like(hidden_states).uniform_(
                1.0 - self.router_jitter_noise, 
                1.0 + self.router_jitter_noise
            )
            hidden_states_noisy = hidden_states * noise
        else:
            hidden_states_noisy = hidden_states
            
        logits = self.router_weights(hidden_states_noisy) # (N, num_experts)
        
        # Softmax over all experts for loss computation
        router_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        
        # Top-k selection
        routing_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Re-normalize routing weights over the selected top-k experts
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # --- Auxiliary Losses ---
        # 1. Load Balancing Loss
        # Create a one-hot tensor of shape (N, num_experts) for the selected experts
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float() # (N, k, num_experts)
        expert_mask = expert_mask.sum(dim=1) # (N, num_experts)
        
        # Fraction of tokens routed to each expert (f_i)
        tokens_per_expert = expert_mask.mean(dim=0) # (num_experts,)
        
        # Mean routing probability for each expert (P_i)
        mean_router_probs = router_probs.mean(dim=0) # (num_experts,)
        
        # Balance loss = num_experts * sum(f_i * P_i)
        balance_loss = self.num_experts * torch.sum(tokens_per_expert * mean_router_probs)
        
        # 2. Router Z-Loss
        # z_loss = mean(log(sum(exp(logits)))^2)
        log_z = torch.logsumexp(logits, dim=-1) # (N,)
        z_loss = torch.mean(log_z ** 2)
        
        # Total aux loss (weights can be adjusted in config, assuming 1.0 here or scaled externally)
        aux_loss = balance_loss + 0.001 * z_loss
        
        return RouterOutput(
            routing_weights=routing_weights,
            expert_indices=expert_indices,
            aux_loss=aux_loss,
            expert_utilization=tokens_per_expert
        )

    def __repr__(self) -> str:
        return f"VasudhaRouter(hidden_size={self.hidden_size}, num_experts={self.num_experts}, top_k={self.top_k})"
