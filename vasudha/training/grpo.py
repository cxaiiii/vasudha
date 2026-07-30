from __future__ import annotations

import re
import subprocess
from typing import Any, Callable, Dict, List, Optional
from vasudha.utils.logging import get_logger

try:
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    GRPOTrainer, GRPOConfig = object, Any

logger = get_logger(__name__)

class VasudhaGRPOHooks:
    """Hooks for GRPO training with Vasudha models."""
    
    @staticmethod
    def math_reward_fn(completions: list[str], references: list[str]) -> list[float]:
        """Rule-based reward: 1.0 if answer matches, 0.0 otherwise.
        Extracts \\boxed{} answers and checks exact match.
        """
        rewards = []
        for comp, ref in zip(completions, references):
            # Try to extract content inside \boxed{}
            match = re.search(r"\\boxed{([^}]*)}", comp)
            if match:
                extracted_ans = match.group(1).strip()
                if extracted_ans == ref.strip():
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                rewards.append(0.0)
        return rewards
    
    @staticmethod
    def format_reward_fn(completions: list[str]) -> list[float]:
        """Reward for following thinking format: <think>...</think> Answer."""
        rewards = []
        for comp in completions:
            if "<think>" in comp and "</think>" in comp:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    
    @staticmethod
    def code_reward_fn(completions: list[str], test_cases: list[Dict[str, str]]) -> list[float]:
        """Reward for code: runs test cases in subprocess, returns pass rate."""
        # Simple mock implementation for safety in actual execution.
        rewards = []
        for comp, t_cases in zip(completions, test_cases):
            # In a real environment, you'd write comp to a file and run subprocess
            # Here we provide a basic stub to avoid arbitrary execution by default.
            passed = 0
            # Mock evaluation logic here
            rewards.append(float(passed) / max(len(t_cases), 1))
        return rewards

def build_grpo_trainer(
    model: Any,
    train_dataset: Any,
    reward_fn: Callable | list[Callable],
    grpo_config: Any,
    tokenizer: Any
) -> Any:
    """Builds a GRPO Trainer instance."""
    if GRPOTrainer is object:
        raise ImportError("trl is required for GRPO training.")
        
    return GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
