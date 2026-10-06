"""Stage 1 — on-policy distillation from a larger Qwen3.5 into Vasudha Neo.

The student samples its own completions (vLLM, colocated); the teacher scores
every token the student wrote; the loss is the per-token reverse KL between
the two next-token distributions (TRL DistillationTrainer, beta=1). No text
written by anyone but the student is ever trained on.

    python scripts/neo_train_opd.py --student /vol/neo/models/qwen3.5-4b-text \
        --teacher /vol/neo/models/qwen3.5-9b-text --data /vol/neo/data/opd.jsonl \
        --out /vol/neo/runs/opd --merge-to /vol/neo/models/neo-stage1-text --max-minutes 80
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
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-to", default=None)
    ap.add_argument("--max-minutes", type=float, default=80)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8, help="per-device micro-batch (prompts)")
    ap.add_argument("--accum", type=int, default=16, help="micro-batches per optimizer step")
    ap.add_argument("--max-completion", type=int, default=1536)
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    ap.add_argument("--beta", type=float, default=1.0, help="1.0 = reverse KL, 0.5 = JSD, 0.0 = forward KL")
    ap.add_argument("--thinking", action="store_true", help="distil thinking mode instead of the fast mode")
    ap.add_argument("--vllm-mem", type=float, default=0.30)
    ap.add_argument("--cpu", action="store_true", help="CPU, fp32, transformers generation (tests only)")
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="tensorboard")
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import DistillationConfig, DistillationTrainer

    from neo.budget import make_time_budget_callback
    from neo.data import prompt_tokens, read_jsonl
    from neo.trl_ext import lora_config

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    rows = read_jsonl(args.data)[: args.limit] if args.limit else read_jsonl(args.data)
    rows = [{"prompt": r["prompt"], "source": r["source"]} for r in rows
            if prompt_tokens(tokenizer, r["prompt"], thinking=args.thinking) <= args.max_prompt_tokens]
    print(f"OPD prompts after length filter: {len(rows)}", flush=True)
    dataset = Dataset.from_list(rows)

    cfg = DistillationConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
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
        teacher_model_init_kwargs={"dtype": "float32" if args.cpu else "bfloat16"},
        max_completion_length=args.max_completion,
        temperature=1.0,
        beta=args.beta,
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
    trainer = DistillationTrainer(model=args.student, teacher_model=args.teacher, args=cfg, train_dataset=dataset,
                                  processing_class=tokenizer, peft_config=lora_config(args.lora_r, args.lora_alpha))
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
        print(f"merged stage-1 model saved to {args.merge_to}", flush=True)
    with open(os.path.join(args.out, "neo_stage.json"), "w", encoding="utf-8") as fh:
        json.dump({"stage": "opd", "steps": trainer.state.global_step, "minutes": (time.time() - started) / 60,
                   "args": vars(args)}, fh, indent=2)


if __name__ == "__main__":
    main()
