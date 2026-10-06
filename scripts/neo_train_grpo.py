"""Stage 2 — GRPO with verifiable rewards and a real Python sandbox.

Tasks are routed per example to a tool environment (a Python interpreter
under one of several common names) or to the tool-less "text" environment.
Rewards come from neo.verify: unit-aware numeric grading, math-verify,
sandboxed unit tests, exact function-call matching, IFEval checkers.

    python scripts/neo_train_grpo.py --model /vol/neo/models/neo-stage1-text \
        --data /vol/neo/data/rl_selected.jsonl --out /vol/neo/runs/grpo \
        --merge-to /vol/neo/models/neo-final-text --max-minutes 105
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-to", default=None)
    ap.add_argument("--max-minutes", type=float, default=105)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--num-generations", type=int, default=8)
    # 256 completions (32 prompts x 8) per optimizer step, generated in one
    # vLLM call. The micro-batch stays small because TRL's GRPO loss builds
    # full logits: 248k vocab x 1.5k tokens is ~0.76 GB per completion in bf16,
    # and the backward doubles it.
    ap.add_argument("--batch", type=int, default=4, help="per-device micro-batch (completions)")
    ap.add_argument("--accum", type=int, default=64, help="micro-batches per optimizer step")
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tool-iterations", type=int, default=4)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--cpu", action="store_true", help="CPU, fp32, transformers generation (tests only)")
    ap.add_argument("--save-steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="tensorboard")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig

    from neo.budget import make_time_budget_callback
    from neo.data import prompt_tokens, read_jsonl
    from neo.envs import ENVIRONMENTS, PYTHON_ENVS, python_tool_schema
    from neo.trl_ext import NeoGRPOTrainer, lora_config, neo_reward

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = read_jsonl(args.data)[: args.limit] if args.limit else read_jsonl(args.data)
    keep = []
    for r in rows:
        tools = [python_tool_schema(r["environment"])] if r["environment"] in PYTHON_ENVS else None
        if prompt_tokens(tokenizer, r["prompt"], tools, args.thinking) <= args.max_prompt_tokens:
            keep.append({"prompt": r["prompt"], "environment": r["environment"], "task": r["task"],
                         "source": r["source"]})
    print(f"RL tasks after length filter: {len(keep)}", flush=True)
    dataset = Dataset.from_list(keep)

    cfg = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_generations=args.num_generations,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup,
        max_steps=args.max_steps,
        max_grad_norm=1.0,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=not args.cpu,
        gradient_checkpointing=True,
        report_to=args.report_to,
        seed=args.seed,
        model_init_kwargs={"dtype": "float32" if args.cpu else "bfloat16"},
        max_completion_length=args.max_completion,
        max_tool_calling_iterations=args.max_tool_iterations,
        temperature=1.0,
        top_p=1.0,
        beta=args.beta,
        epsilon=0.2,
        epsilon_high=0.28,
        loss_type="dapo",
        scale_rewards="batch",
        mask_truncated_completions=True,
        chat_template_kwargs={"enable_thinking": args.thinking},
        use_vllm=not args.cpu,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_max_model_length=args.max_prompt_tokens + args.max_completion,
        log_completions=True,
        num_completions_to_print=2,
        shuffle_dataset=True,
        use_cpu=args.cpu,
    )
    trainer = NeoGRPOTrainer(model=args.model, reward_funcs=[neo_reward], args=cfg, train_dataset=dataset,
                             processing_class=tokenizer, peft_config=lora_config(args.lora_r, args.lora_alpha),
                             environment_factory=ENVIRONMENTS)
    elapsed = time.time() - started
    trainer.add_callback(make_time_budget_callback(max(60.0, args.max_minutes * 60 - elapsed)))
    # Resume if a previous attempt of this stage left checkpoints behind.
    has_ckpt = os.path.isdir(args.out) and any(d.startswith("checkpoint-") for d in os.listdir(args.out))
    trainer.train(resume_from_checkpoint=True if has_ckpt else None)

    adapter_dir = os.path.join(args.out, "adapter")
    trainer.save_model(adapter_dir)
    print(f"adapter saved to {adapter_dir}", flush=True)
    if args.merge_to:
        model = trainer.accelerator.unwrap_model(trainer.model)
        merged = model.merge_and_unload()
        merged.save_pretrained(args.merge_to, safe_serialization=True)
        tokenizer.save_pretrained(args.merge_to)
        print(f"merged model saved to {args.merge_to}", flush=True)
    with open(os.path.join(args.out, "neo_stage.json"), "w", encoding="utf-8") as fh:
        json.dump({"stage": "grpo", "steps": trainer.state.global_step, "minutes": (time.time() - started) / 60,
                   "args": vars(args)}, fh, indent=2)


if __name__ == "__main__":
    main()
