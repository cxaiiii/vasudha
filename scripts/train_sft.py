"""
Vasudha SFT Training Script.

Run supervised fine-tuning (SFT) with optional QLoRA on Qwen3-4B or Qwen3-8B.

Usage:
    # Basic QLoRA SFT on Qwen3-4B
    python scripts/train_sft.py

    # With Hydra overrides
    python scripts/train_sft.py model=qwen3_8b training.args.max_seq_length=4096

    # Full precision (requires more VRAM, not recommended for T4)
    python scripts/train_sft.py training.quantization.load_in_4bit=false

Google Colab usage:
    !python vasudha/scripts/train_sft.py model=qwen3_4b
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path when running as script
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from vasudha.utils.logging import get_logger, setup_logging, log_banner
from vasudha.utils.memory import format_bytes, get_memory_stats

logger = get_logger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main SFT training entry point.

    All configuration is handled by Hydra. The config is composed from:
      configs/config.yaml (root)
      configs/model/{model}.yaml
      configs/training/sft_qlora.yaml
      configs/data/math_reasoning.yaml
      configs/eval/standard.yaml

    Args:
        cfg: Hydra-composed configuration DictConfig.
    """
    # ── Setup ──────────────────────────────────────────────────────────────────
    setup_logging(
        level="INFO",
        log_file=os.path.join(cfg.log_dir, "train.log") if hasattr(cfg, "log_dir") else None,
    )
    log_banner("Vasudha SFT Training", f"Model: {cfg.model.hf_model_name}")

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ── Hardware detection ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_stats = get_memory_stats()
        logger.info(
            f"GPU: {device_name} | "
            f"VRAM: {format_bytes(memory_stats.total)} | "
            f"CUDA: {torch.version.cuda}"
        )
    else:
        logger.warning("No CUDA GPU detected! Training will be very slow on CPU.")

    # ── Set seed ───────────────────────────────────────────────────────────────
    torch.manual_seed(cfg.get("seed", 42))

    # ── Load tokenizer ─────────────────────────────────────────────────────────
    hf_model_name: str = cfg.model.hf_model_name
    logger.info(f"Loading tokenizer from '{hf_model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    # ── Load model ─────────────────────────────────────────────────────────────
    from vasudha.models.vasudha_model import VasudhaForCausalLM

    training_cfg = cfg.training
    quant_cfg = training_cfg.quantization
    load_in_4bit: bool = quant_cfg.get("load_in_4bit", True)
    load_in_8bit: bool = quant_cfg.get("load_in_8bit", False)

    vasudha_path: str | None = cfg.model.get("vasudha_path", None)
    if not vasudha_path:
        raise ValueError(
            "cfg.model.vasudha_path is not set. Convert the Qwen3 weights into a "
            "Vasudha checkpoint first — quantized training loads from disk:\n\n"
            f"    python scripts/convert_qwen3_to_vasudha.py --model {hf_model_name} "
            "--out ./vasudha-4b-init\n\n"
            "then re-run with model.vasudha_path=./vasudha-4b-init"
        )

    logger.info(
        f"Loading Vasudha checkpoint '{vasudha_path}' "
        f"({'4-bit' if load_in_4bit else '8-bit' if load_in_8bit else 'full precision'})..."
    )

    from vasudha.utils.dtype import get_compute_dtype

    # T4 (Turing) emulates bf16 in software — get_compute_dtype picks fp16 there
    # and bf16 on Ampere+, so the quantized compute dtype tracks the actual GPU.
    compute_dtype = get_compute_dtype(cfg.hardware.get("compute_dtype", "auto"))

    load_kwargs: dict = {"torch_dtype": compute_dtype, "device_map": {"": 0}}

    # The current GLA implementation is a correctness reference with a Python
    # loop over sequence positions.  An H100 cannot accelerate that loop, so
    # an SDPA override is the practical training path until the chunked Triton
    # kernel exists.  Qwen-derived q/k/v/o weights load unchanged; only the
    # experimental GLA gate projections are omitted.
    attention_override = cfg.model.get("attention_override", None)
    if attention_override and attention_override != "checkpoint":
        from vasudha.models.config import VasudhaConfig

        model_config = VasudhaConfig.from_pretrained(vasudha_path)
        model_config.attention_type = str(attention_override)
        model_config.use_cache = False
        load_kwargs["config"] = model_config
        logger.warning(
            "Overriding checkpoint attention '%s' -> '%s' for training.",
            "hybrid", attention_override,
        )
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )

    model = VasudhaForCausalLM.from_pretrained(vasudha_path, **load_kwargs)
    # Caches are useful only for autoregressive inference.  Keeping them during
    # SFT wastes HBM bandwidth and activation memory on every layer.
    model.config.use_cache = False

    logger.info(f"Model loaded: {model!r}")

    # ── Apply LoRA ─────────────────────────────────────────────────────────────
    # Adapters are what make this trainable on one GPU, quantized or not: full
    # fine-tuning of 4B params needs ~64GB for Adam state alone. On a 24GB card
    # bf16 weights + LoRA fit comfortably and run faster than 4-bit, which pays
    # a dequantization cost on every matmul. So attach LoRA whenever the mode
    # asks for it, not only when bitsandbytes is in play.
    if load_in_4bit or load_in_8bit or training_cfg.get("mode", "sft_qlora") == "sft_qlora":
        from vasudha.training.lora_utils import QLoRAConfig, prepare_model_for_qlora

        lora_cfg = training_cfg.lora
        qlora_config = QLoRAConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("lora_alpha", 32),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
            modules_to_save=list(lora_cfg["modules_to_save"]) if lora_cfg.get("modules_to_save") else None,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

        model = prepare_model_for_qlora(model, qlora_config)
        from vasudha.training.lora_utils import print_trainable_parameters
        print_trainable_parameters(model)

    # ── Load dataset ───────────────────────────────────────────────────────────
    from vasudha.datasets import build_mixture

    data_cfg = cfg.data
    sources: dict[str, float] = {}
    for dataset_spec in data_cfg.mixture:
        sources[dataset_spec.name] = dataset_spec.weight

    # A prepared dataset is a finite, on-disk Arrow table — no shuffle buffer of
    # raw rows in RAM, and no re-download on every restart. See
    # scripts/prepare_dataset.py.
    prepared_path = data_cfg.get("prepared_path", None)
    if prepared_path:
        from datasets import load_from_disk

        train_dataset = load_from_disk(str(prepared_path))
        logger.info(f"Loaded prepared dataset '{prepared_path}' ({len(train_dataset)} rows)")
    else:
        logger.info(f"Building dataset mixture: {sources}")
        train_dataset = build_mixture(
            sources=sources,
            streaming=True,
            seed=cfg.get("seed", 42),
            buffer_size=data_cfg.loader.get("buffer_size", 10000),
        )

    # ── Build training args ────────────────────────────────────────────────────
    from vasudha.training.sft import SFTConfig, build_training_args

    args_cfg = training_cfg.args
    sft_config = SFTConfig(
        output_dir=str(cfg.get("checkpoint_dir", "./checkpoints")),
        num_train_epochs=args_cfg.get("num_train_epochs", 1),
        # Streaming datasets have no __len__, so transformers cannot derive a
        # schedule from num_train_epochs alone — max_steps has to reach the
        # trainer or it refuses to start.
        max_steps=args_cfg.get("max_steps", -1),
        per_device_train_batch_size=args_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=args_cfg.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=args_cfg.get("gradient_checkpointing", True),
        max_seq_length=args_cfg.get("max_seq_length", 2048),
        packing=args_cfg.get("packing", True),
        learning_rate=args_cfg.get("learning_rate", 2e-4),
        optim=args_cfg.get("optim", "paged_adamw_32bit"),
        logging_steps=args_cfg.get("logging_steps", 10),
        eval_steps=args_cfg.get("eval_steps", 100),
        save_steps=args_cfg.get("save_steps", 200),
        save_total_limit=args_cfg.get("save_total_limit", 3),
        report_to="tensorboard",
        compute_dtype=cfg.hardware.get("compute_dtype", "auto"),
        resume_from_checkpoint=args_cfg.get("resume_from_checkpoint", True),
    )

    training_args = build_training_args(sft_config)

    # ── Check for existing checkpoint ──────────────────────────────────────────
    from vasudha.training.checkpoint import CheckpointManager
    ckpt_manager = CheckpointManager(sft_config.output_dir)
    resume_checkpoint = ckpt_manager.get_latest_checkpoint()
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")

    # ── Initialize trainer ─────────────────────────────────────────────────────
    from vasudha.training.trainer import VasudhaTrainer

    trainer = VasudhaTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        sft_config=sft_config,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    log_banner("Training Start")
    logger.info(
        f"Starting training: "
        f"{sft_config.num_train_epochs} epoch(s), "
        f"batch_size={sft_config.per_device_train_batch_size} × "
        f"accum={sft_config.gradient_accumulation_steps} = "
        f"effective_batch={sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}"
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ── Save final model ────────────────────────────────────────────────────────
    final_save_dir = os.path.join(sft_config.output_dir, "final_model")
    trainer.save_model(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    logger.info(f"Model saved to {final_save_dir}")
    log_banner("Training Complete")


if __name__ == "__main__":
    main()
