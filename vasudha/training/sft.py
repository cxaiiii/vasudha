from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

try:
    from transformers import TrainingArguments
except ImportError:
    TrainingArguments = None

@dataclass
class SFTConfig:
    # Core
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 1
    max_steps: int = -1
    
    # Batch size (T4 optimized)
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    
    # Memory
    gradient_checkpointing: bool = True
    max_seq_length: int = 2048
    packing: bool = True
    
    # Optimizer  
    learning_rate: float = 2e-4
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    report_to: str = "tensorboard"
    
    # Precision
    compute_dtype: str = "auto"
    
    # Resume
    resume_from_checkpoint: bool = True

    def __repr__(self) -> str:
        return f"SFTConfig(output_dir='{self.output_dir}', epochs={self.num_train_epochs}, lr={self.learning_rate})"

def build_training_args(sft_config: SFTConfig) -> Any:
    """
    Build the args object for VasudhaTrainer.

    Returns a trl.SFTConfig when TRL is available, since modern TRL moved
    max_length/packing out of the SFTTrainer constructor and into the args
    object. Falls back to plain TrainingArguments otherwise.
    """
    if TrainingArguments is None:
        raise ImportError("transformers is required to build TrainingArguments.")

    fp16 = False
    bf16 = False

    if sft_config.compute_dtype == "auto":
        # torch.cuda.is_bf16_supported() returns True on T4 — it reports CUDA
        # support, not native hardware support, and T4 emulates bf16 in software.
        # is_bf16_supported() gates on compute capability >= 8.0 instead.
        from vasudha.utils.dtype import is_bf16_supported

        if is_bf16_supported():
            bf16 = True
        elif torch.cuda.is_available():
            fp16 = True
    elif sft_config.compute_dtype in ("bfloat16", "bf16"):
        bf16 = True
    elif sft_config.compute_dtype in ("float16", "fp16"):
        fp16 = True

    gradient_checkpointing_kwargs = {"use_reentrant": False} if sft_config.gradient_checkpointing else None

    args_cls: Any = TrainingArguments
    extra: dict[str, Any] = {}
    try:
        from trl import SFTConfig as TRLSFTConfig

        args_cls = TRLSFTConfig
        # TRL renamed max_seq_length -> max_length; support both vintages.
        import dataclasses

        fields = {f.name for f in dataclasses.fields(TRLSFTConfig)}
        extra["max_length" if "max_length" in fields else "max_seq_length"] = (
            sft_config.max_seq_length
        )
        extra["packing"] = sft_config.packing
    except ImportError:
        pass

    return args_cls(
        **extra,
        output_dir=sft_config.output_dir,
        num_train_epochs=sft_config.num_train_epochs,
        max_steps=sft_config.max_steps,
        per_device_train_batch_size=sft_config.per_device_train_batch_size,
        per_device_eval_batch_size=sft_config.per_device_eval_batch_size,
        gradient_accumulation_steps=sft_config.gradient_accumulation_steps,
        gradient_checkpointing=sft_config.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        learning_rate=sft_config.learning_rate,
        optim=sft_config.optim,
        lr_scheduler_type=sft_config.lr_scheduler_type,
        warmup_ratio=sft_config.warmup_ratio,
        weight_decay=sft_config.weight_decay,
        max_grad_norm=sft_config.max_grad_norm,
        logging_steps=sft_config.logging_steps,
        eval_steps=sft_config.eval_steps,
        save_steps=sft_config.save_steps,
        save_total_limit=sft_config.save_total_limit,
        report_to=sft_config.report_to,
        fp16=fp16,
        bf16=bf16,
    )
