from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    from trl import DPOTrainer
except ImportError:
    DPOTrainer = object

try:
    from transformers import TrainingArguments, PreTrainedTokenizer
except ImportError:
    TrainingArguments, PreTrainedTokenizer = Any, Any

from .lora_utils import QLoRAConfig, prepare_model_for_qlora
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class DPOConfig:
    beta: float = 0.1           # KL regularization weight
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    label_smoothing: float = 0.0
    reference_free: bool = False
    max_length: int = 1024
    max_prompt_length: int = 512
    output_dir: str = "./dpo_checkpoints"
    learning_rate: float = 5e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True

    def __repr__(self) -> str:
        return f"DPOConfig(beta={self.beta}, loss_type='{self.loss_type}', lr={self.learning_rate})"


class VasudhaDPOTrainer(DPOTrainer):
    """Wraps TRL's DPOTrainer with QLoRA support."""
    pass

def build_dpo_trainer(
    model: Any,
    ref_model: Any,
    train_dataset: Any,
    dpo_config: DPOConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    qlora_config: Optional[QLoRAConfig] = None
) -> VasudhaDPOTrainer:
    """Builds a DPO Trainer instance."""
    if DPOTrainer is object:
        raise ImportError("trl must be installed for DPO training.")
        
    if qlora_config is not None:
        model = prepare_model_for_qlora(model, qlora_config)
        # Typically ref_model is not quantized for LoRA but can be handled differently
        
    args = TrainingArguments(
        output_dir=dpo_config.output_dir,
        learning_rate=dpo_config.learning_rate,
        num_train_epochs=dpo_config.num_train_epochs,
        per_device_train_batch_size=dpo_config.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_config.gradient_accumulation_steps,
        gradient_checkpointing=dpo_config.gradient_checkpointing,
        remove_unused_columns=False,
    )
    
    return VasudhaDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        beta=dpo_config.beta,
        loss_type=dpo_config.loss_type,
        label_smoothing=dpo_config.label_smoothing,
        max_length=dpo_config.max_length,
        max_prompt_length=dpo_config.max_prompt_length,
    )
