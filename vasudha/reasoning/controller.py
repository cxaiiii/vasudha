import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

class DifficultyLevel(IntEnum):
    EASY = 0
    MEDIUM = 1
    HARD = 2

@dataclass
class ControllerOutput:
    difficulty: DifficultyLevel
    logits: torch.Tensor          # (3,) raw difficulty logits
    expert_k: int                 # Recommended top-k for this input
    confidence: float             # Max softmax probability
    
    def __repr__(self) -> str:
        return f"ControllerOutput(difficulty={self.difficulty.name}, expert_k={self.expert_k}, confidence={self.confidence:.4f})"

class ReasoningController(nn.Module):
    """2-layer MLP difficulty predictor.
    
    Takes the final hidden state (pooled over sequence) and predicts
    difficulty: Easy / Medium / Hard.
    
    The predicted difficulty drives:
    - Number of active experts (k=1/2/4)
    - Attention strategy selection
    - Number of reasoning passes (future)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 3)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        pool: str = "mean",
    ) -> ControllerOutput:
        """Predict difficulty from hidden states."""
        if pool == "mean":
            pooled = hidden_states.mean(dim=1)
        elif pool == "last":
            pooled = hidden_states[:, -1, :]
        elif pool == "first":
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {pool}")
            
        logits = self.mlp(pooled)
        seq_logits = logits[0] if logits.dim() > 1 else logits
        probs = torch.softmax(seq_logits, dim=-1)
        confidence, max_idx = torch.max(probs, dim=-1)
        
        difficulty = DifficultyLevel(max_idx.item())
        expert_k = self.get_recommended_k(difficulty, self.config)
        
        return ControllerOutput(
            difficulty=difficulty,
            logits=seq_logits,
            expert_k=expert_k,
            confidence=confidence.item()
        )
        
    def get_recommended_k(self, difficulty: DifficultyLevel, config) -> int:
        """Map difficulty to expert count."""
        min_k = getattr(config, 'min_k', 1)
        mid_k = getattr(config, 'mid_k', 2)
        max_k = getattr(config, 'max_k', 4)
        
        if difficulty == DifficultyLevel.EASY:
            return min_k
        elif difficulty == DifficultyLevel.MEDIUM:
            return mid_k
        else:
            return max_k
            
    def compute_auxiliary_loss(
        self,
        difficulty_logits: torch.Tensor,
        pseudo_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Auxiliary loss: cross-entropy if pseudo_labels, else entropy regularization."""
        if pseudo_labels is not None:
            criterion = nn.CrossEntropyLoss()
            return criterion(difficulty_logits, pseudo_labels)
        else:
            probs = torch.softmax(difficulty_logits, dim=-1)
            log_probs = torch.log_softmax(difficulty_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            return -entropy
            
    def __repr__(self) -> str:
        return f"ReasoningController(hidden_size={self.config.hidden_size})"
