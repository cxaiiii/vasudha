"""
Vasudha evaluation script.

Run benchmark evaluations on a trained Vasudha model.

Usage:
    python scripts/evaluate.py \\
        --model_path ./checkpoints/final_model \\
        --benchmarks gsm8k math500 humaneval \\
        --output_dir ./eval_results

    # Quick eval (100 samples per benchmark)
    python scripts/evaluate.py \\
        --model_path Qwen/Qwen3-4B \\
        --benchmarks gsm8k \\
        --max_samples 100 \\
        --load_in_4bit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import torch
from rich.console import Console
from transformers import AutoTokenizer

from vasudha.utils.logging import get_logger, setup_logging, log_banner
from vasudha.utils.memory import get_memory_stats, format_bytes

console = Console()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vasudha evaluation runner")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (local dir or HuggingFace model ID)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k"],
        choices=["gsm8k", "math500", "humaneval", "mbpp", "longbench", "needle"],
        help="Benchmarks to evaluate on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per benchmark (None = full benchmark)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Base model for a LoRA adapter. Defaults to the path recorded in "
             "adapter_config.json, which is usually correct.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    """
    Load a model for evaluation.

    Three shapes turn up here and they load differently:
      - a LoRA training output: only adapter_model.safetensors, base elsewhere
      - a converted Vasudha checkpoint: full weights + config.json
      - a stock Qwen3 checkpoint

    The architecture is read from the checkpoint's own config rather than
    assumed, so evaluating an sdpa model doesn't silently build a hybrid one.
    """
    import json
    import os

    from vasudha.models.vasudha_model import VasudhaForCausalLM

    path = args.model_path
    adapter_cfg = os.path.join(path, "adapter_config.json")

    load_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if os.path.exists(adapter_cfg):
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base = args.base_path or json.load(f).get("base_model_name_or_path")
        if not base or not os.path.exists(base):
            raise SystemExit(
                f"'{path}' is a LoRA adapter but its base model could not be located "
                f"(got {base!r}). Pass --base_path pointing at the converted checkpoint."
            )
        logger.info(f"LoRA adapter detected — loading base '{base}'")
        model = VasudhaForCausalLM.from_pretrained(base, **load_kwargs)

        from peft import PeftModel

        logger.info(f"Applying adapter from '{path}'")
        model = PeftModel.from_pretrained(model, path)
        # Folding the adapter into the base removes the per-layer LoRA branch
        # from every forward — meaningful when generating 512 tokens x 200 problems.
        model = model.merge_and_unload()
    else:
        logger.info(f"Loading full checkpoint '{path}'")
        model = VasudhaForCausalLM.from_pretrained(path, **load_kwargs)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if torch.cuda.is_available():
        stats = get_memory_stats()
        logger.info(f"Model loaded. VRAM: {format_bytes(stats.allocated)}")

    return model, tokenizer


def main() -> None:
    setup_logging()
    args = parse_args()

    log_banner("Vasudha Evaluation", f"Model: {args.model_path}")
    logger.info(f"Benchmarks: {args.benchmarks}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args)

    # Run evaluation
    from evaluation.runner import EvaluationRunner
    from evaluation.report import EvalReport

    runner = EvaluationRunner(
        model=model,
        tokenizer=tokenizer,
        benchmarks=args.benchmarks,
        batch_size=args.batch_size,
    )

    results = runner.run(
        benchmarks=args.benchmarks,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    # Print and save report
    report = EvalReport(results)
    report.print_table()
    report.save_json(os.path.join(args.output_dir, "results.json"))
    report.save_markdown(os.path.join(args.output_dir, "results.md"))

    log_banner("Evaluation Complete")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
