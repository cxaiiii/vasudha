from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig, get_peft_model, prepare_model_for_kbit_training = None, None, None

try:
    from bitsandbytes import nn as bnb_nn
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig, bnb_nn = None, None

@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    def __repr__(self) -> str:
        return f"QLoRAConfig(r={self.r}, lora_alpha={self.lora_alpha}, target_modules={self.target_modules})"

def get_bnb_config(qlora_config: QLoRAConfig) -> "BitsAndBytesConfig":
    """Builds BitsAndBytes config for 4-bit loading."""
    if BitsAndBytesConfig is None:
        raise ImportError("bitsandbytes and transformers are required for QLoRA.")
    import torch
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    compute_dtype = dtype_map.get(qlora_config.bnb_4bit_compute_dtype, torch.float16)
    
    return BitsAndBytesConfig(
        load_in_4bit=qlora_config.load_in_4bit,
        bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
    )

def get_lora_config(qlora_config: QLoRAConfig) -> "LoraConfig":
    """Builds LoraConfig from QLoRAConfig."""
    if LoraConfig is None:
        raise ImportError("peft is required for QLoRA. Install it via pip install peft.")
    return LoraConfig(
        r=qlora_config.r,
        lora_alpha=qlora_config.lora_alpha,
        lora_dropout=qlora_config.lora_dropout,
        bias=qlora_config.bias,
        task_type=qlora_config.task_type,
        target_modules=qlora_config.target_modules,
    )

def prepare_model_for_qlora(model: Any, qlora_config: QLoRAConfig) -> Any:
    """Prepares model for kbit training and applies LoRA."""
    if get_peft_model is None or prepare_model_for_kbit_training is None:
        raise ImportError("peft is required for QLoRA. Install it via pip install peft.")
    
    # prepare_model_for_kbit_training upcasts every parameter to fp32. That is
    # correct for a quantized base (the weights are 4-bit blobs; the fp32 copies
    # are only the norms and head) but catastrophic for an unquantized one —
    # a bf16 4B model would balloon from 8GB to 16GB before training starts.
    # Unquantized runs only need the checkpointing-compatible input grads.
    if getattr(qlora_config, "load_in_4bit", False) or getattr(qlora_config, "load_in_8bit", False):
        logger.info("Preparing model for kbit training...")
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Unquantized base — skipping kbit upcast, enabling input grads")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = get_lora_config(qlora_config)
    logger.info("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # prepare_model_for_kbit_training upcasts to fp32, but it ran before the
    # adapters existed, so they inherit the base layer's compute dtype. A bf16
    # trainable param breaks the fp16 GradScaler on T4
    # ("_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
    # BFloat16"), and fp32 adapters are the standard QLoRA recipe anyway —
    # 22M params in fp32 costs ~90MB.
    import torch

    recast = 0
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            recast += 1
    if recast:
        logger.info(f"Cast {recast} trainable tensors to fp32 for stable grad scaling")

    return model

def print_trainable_parameters(model: Any) -> None:
    """Prints trainable vs frozen parameters using Rich."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percent_trainable = 100 * trainable_params / all_param
    
    if Console:
        console = Console()
        table = Table(title="Model Parameters")
        table.add_column("Type", justify="left", style="cyan")
        table.add_column("Parameters", justify="right", style="magenta")
        table.add_column("Percentage", justify="right", style="green")
        
        table.add_row("Trainable", f"{trainable_params:,}", f"{percent_trainable:.4f}%")
        table.add_row("Frozen", f"{all_param - trainable_params:,}", f"{100 - percent_trainable:.4f}%")
        table.add_row("Total", f"{all_param:,}", "100.0000%")
        
        console.print(table)
    else:
        logger.info(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {percent_trainable:.4f}")
