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
    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load model for evaluation."""
    from vasudha.models.vasudha_model import VasudhaForCausalLM

    logger.info(f"Loading model from '{args.model_path}'...")

    model = VasudhaForCausalLM.from_qwen3_pretrained(
        args.model_path,
        attention_type="hybrid",
        use_moe=True,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
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
