from __future__ import annotations

import time
import torch
from typing import Any, Dict, Optional, Union

try:
    from transformers import TrainerCallback, PreTrainedTokenizer
except ImportError:
    TrainerCallback, PreTrainedTokenizer = object, Any

try:
    from trl import SFTTrainer
except ImportError:
    SFTTrainer = object

from vasudha.utils.logging import get_logger
from vasudha.utils.memory import VRAMMonitor
from .lora_utils import QLoRAConfig, prepare_model_for_qlora, print_trainable_parameters
from .sft import SFTConfig, build_training_args

logger = get_logger(__name__)

class VasudhaCallback(TrainerCallback):
    """Callback for tracking VRAM and timing during training."""
    def __init__(self):
        self.vram_monitor = VRAMMonitor()
        self.step_start_time = 0.0

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        logger.info("=== Starting Vasudha Training ===")
        print_trainable_parameters(kwargs.get("model"))

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.step_start_time = time.time()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        step_time = time.time() - self.step_start_time
        vram_stats = self.vram_monitor.get_stats()
        
        # Check if VRAM is > 85% of total
        if vram_stats.get("percent_used", 0) > 85.0:
            logger.warning(f"High VRAM usage: {vram_stats['percent_used']}% used.")
            
        kwargs["logs"] = kwargs.get("logs", {})
        kwargs["logs"]["step_time"] = step_time
        kwargs["logs"]["vram_used_mb"] = vram_stats.get("used_mb", 0)

class VasudhaTrainer(SFTTrainer):
    def __init__(
        self,
        model: Any | str,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        qlora_config: Optional[QLoRAConfig] = None,
        sft_config: Optional[SFTConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        data_collator: Optional[Any] = None,
        **kwargs: Any,
    ):
        if SFTTrainer is object:
            raise ImportError("trl must be installed to use VasudhaTrainer.")
            
        if sft_config is None:
            sft_config = SFTConfig()
            
        if qlora_config is not None and not isinstance(model, str):
            model = prepare_model_for_qlora(model, qlora_config)
            
        args = build_training_args(sft_config)

        # TRL >= 0.12 renamed `tokenizer` to `processing_class` and moved
        # max_seq_length/packing into the args object (handled in
        # build_training_args). Detect rather than pin, so both vintages work.
        import inspect

        trainer_params = inspect.signature(SFTTrainer.__init__).parameters
        tokenizer_kwarg = "processing_class" if "processing_class" in trainer_params else "tokenizer"

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **{tokenizer_kwarg: tokenizer},
            **kwargs
        )
        
        self.add_callback(VasudhaCallback())
        self.vram_monitor = VRAMMonitor()

    def compute_loss(self, model: Any, inputs: Dict[str, Union[torch.Tensor, Any]], return_outputs: bool = False, num_items_in_batch: Optional[int] = None) -> Union[torch.Tensor, tuple[torch.Tensor, Any]]:
        """Computes loss and handles MoE aux_loss."""
        outputs = model(**inputs)
        loss = outputs.loss
        
        if hasattr(outputs, "aux_loss") and outputs.aux_loss is not None:
            # We don't add aux_loss here if it's already added inside model.forward(),
            # but we log it for tracking.
            self.log({"aux_loss": outputs.aux_loss.item()})
            
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: Any, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """Perform a training step and log VRAM usage occasionally."""
        loss = super().training_step(model, inputs, num_items_in_batch)

        if self.state.global_step % getattr(self.args, "logging_steps", 10) == 0:
            vram_stats = self.vram_monitor.get_stats()
            self.log({"vram_used_mb_step": vram_stats.get("used_mb", 0)})
            
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Adds VRAM stats to tensorboard logs."""
        vram_stats = self.vram_monitor.get_stats()
        logs["vram_used_mb"] = vram_stats.get("used_mb", 0)
        if start_time is None:
            super().log(logs)
        else:
            super().log(logs, start_time)
