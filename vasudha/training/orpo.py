from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    from trl import ORPOTrainer
except ImportError:
    ORPOTrainer = object

try:
    from transformers import TrainingArguments, PreTrainedTokenizer
except ImportError:
    TrainingArguments, PreTrainedTokenizer = Any, Any

from .lora_utils import QLoRAConfig, prepare_model_for_qlora
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ORPOConfig:
    lambda_orpo: float = 0.1   # ORPO loss weight
    max_length: int = 1024
    max_prompt_length: int = 512
    output_dir: str = "./orpo_checkpoints"
    learning_rate: float = 8e-6
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True

    def __repr__(self) -> str:
        return f"ORPOConfig(lambda_orpo={self.lambda_orpo}, lr={self.learning_rate})"


class VasudhaORPOTrainer(ORPOTrainer):
    """Wraps TRL's ORPOTrainer with QLoRA support."""
    pass


def build_orpo_trainer(
    model: Any,
    train_dataset: Any,
    orpo_config: ORPOConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    qlora_config: Optional[QLoRAConfig] = None
) -> VasudhaORPOTrainer:
    """Builds an ORPO Trainer instance."""
    if ORPOTrainer is object:
        raise ImportError("trl must be installed for ORPO training.")
        
    if qlora_config is not None:
        model = prepare_model_for_qlora(model, qlora_config)
        
    args = TrainingArguments(
        output_dir=orpo_config.output_dir,
        learning_rate=orpo_config.learning_rate,
        num_train_epochs=orpo_config.num_train_epochs,
        per_device_train_batch_size=orpo_config.per_device_train_batch_size,
        gradient_accumulation_steps=orpo_config.gradient_accumulation_steps,
        gradient_checkpointing=orpo_config.gradient_checkpointing,
        remove_unused_columns=False,
    )
    
    return VasudhaORPOTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=orpo_config.max_length,
        max_prompt_length=orpo_config.max_prompt_length,
    )
