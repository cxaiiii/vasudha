from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

try:
    from vasudha.utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from .router import RouterOutput

class DifficultyLevel(Enum):
    EASY = 0
    MEDIUM = 1
    HARD = 2

@dataclass
class AdaptiveRouterOutput(RouterOutput):
    difficulty_logits: torch.Tensor  # (B, 3)
    difficulty_level: DifficultyLevel
    active_k: int

class AdaptiveRouter(nn.Module):
    """
    Difficulty-based dynamic Top-k routing.
    
    Predicts whether current input is Easy/Medium/Hard and selects k experts accordingly.
    """
    def __init__(self, hidden_size: int, num_experts: int, min_k: int = 1, mid_k: int = 2, max_k: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k_map = {0: min_k, 1: mid_k, 2: max_k}
        
        # Difficulty Predictor: 2-layer MLP
        self.difficulty_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3)
        )
        
        # Standard router weights
        self.router_weights = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, training: bool = True) -> AdaptiveRouterOutput:
        """
        Forward pass for adaptive router.
        """
        B = hidden_states.size(0)
        
        # Average pooling over sequence length if 3D tensor
        if hidden_states.dim() == 3:
            pooled_state = hidden_states.mean(dim=1) # (B, H)
        else:
            pooled_state = hidden_states # (B, H)
            
        difficulty_logits = self.difficulty_predictor(pooled_state) # (B, 3)
        
        if training:
            # Gumbel-Softmax for differentiable hard selection
            difficulty_probs = F.gumbel_softmax(difficulty_logits, tau=1.0, hard=True)
            # Find chosen level
            chosen_level = difficulty_probs.argmax(dim=-1) # (B,)
        else:
            # Hard argmax
            chosen_level = difficulty_logits.argmax(dim=-1) # (B,)
            
        # For simplicity, we assume a single difficulty level per batch based on mode
        # In a more advanced setup, this could be per-token dynamic k
        mode_level = chosen_level.mode().values.item()
        active_k = self.k_map[mode_level]
        difficulty_level_enum = DifficultyLevel(mode_level)
        
        # Flatten for standard routing
        if hidden_states.dim() == 3:
            flat_hidden = hidden_states.view(-1, self.hidden_size)
        else:
            flat_hidden = hidden_states
            
        logits = self.router_weights(flat_hidden) # (N, num_experts)
        router_probs = F.softmax(logits, dim=-1)
        
        routing_weights, expert_indices = torch.topk(router_probs, active_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Load balancing loss and Z-loss calculation (simplified here)
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float().sum(dim=1)
        tokens_per_expert = expert_mask.mean(dim=0)
        mean_router_probs = router_probs.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(tokens_per_expert * mean_router_probs)
        
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = torch.mean(log_z ** 2)
        aux_loss = balance_loss + 0.001 * z_loss
        
        return AdaptiveRouterOutput(
            routing_weights=routing_weights,
            expert_indices=expert_indices,
            aux_loss=aux_loss,
            expert_utilization=tokens_per_expert,
            difficulty_logits=difficulty_logits,
            difficulty_level=difficulty_level_enum,
            active_k=active_k
        )
        
    def __repr__(self) -> str:
        return f"AdaptiveRouter(hidden_size={self.hidden_size}, num_experts={self.num_experts})"
