"""Fast QLoRA SFT for the practical Vasudha Engineering branch.

This intentionally trains plain Qwen3-8B with Unsloth.  It is a separate,
compatible model from the experimental Vasudha hybrid checkpoint.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=1250)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--merge", action="store_true", help="Save a standalone merged 16-bit model")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    import torch
    from datasets import load_from_disk
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=128,
        # Keep dropout at zero: this allows Unsloth to fuse the LoRA path and
        # avoids a measurable slowdown. Regularisation is provided by the
        # dataset mix and short training schedule.
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing=False if args.no_gradient_checkpointing else "unsloth",
        random_state=42,
    )
    dataset = load_from_disk(args.data)

    from trl import SFTConfig, SFTTrainer

    fields = {f.name for f in __import__("dataclasses").fields(SFTConfig)}
    train_args = dict(
        output_dir=args.out,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=250,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="tensorboard",
        seed=42,
    )
    train_args["max_length" if "max_length" in fields else "max_seq_length"] = args.max_seq_length
    if "dataloader_num_workers" in fields:
        train_args["dataloader_num_workers"] = 2
    if "packing" in fields:
        train_args["packing"] = True

    trainer_kwargs = {
        "model": model,
        "args": SFTConfig(**train_args),
        "train_dataset": dataset,
        "dataset_text_field": "text",
    }
    params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    # Without this, loss is computed over the ENTIRE flattened text — system
    # prompt, synthetic tool-output wrapper text, everything — not just what
    # the assistant actually generates. With packing=True concatenating
    # multiple examples per sequence, an identical system prompt (and, for
    # the tool-use dataset, identical follow-up wrapper phrasing) gets
    # predicted-and-lossed on repeatedly across the whole dataset: wasted
    # LoRA capacity memorizing fixed boilerplate instead of the actual
    # skill. Masking to response-only tokens is the standard fix (this is
    # exactly what this helper is for), not a reason to change how the data
    # itself is generated. Markers match ChatFormatter._format_qwen3's
    # literal "<|im_start|>{role}\n" turn boundaries exactly.
    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.merge:
        model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
        print(f"Saved standalone Vasudha model to {args.out}")
    else:
        model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        print(f"Saved Vasudha Engineering Qwen adapter to {args.out}")


if __name__ == "__main__":
    main()
