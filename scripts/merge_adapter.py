"""
Merge a trained LoRA adapter into its base model, producing one standalone
checkpoint directory (no separate base + adapter to keep track of).

Usage:
    python scripts/merge_adapter.py \
        --adapter_path /vol/ckpt/runs/sdpa-bf16/checkpoints/final_model \
        --base_path /vol/ckpt/vasudha-4b-sdpa \
        --out /vol/ckpt/vasudha-4b-merged
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from vasudha.models.vasudha_model import VasudhaForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, default=None,
                         help="Defaults to the path recorded in adapter_config.json.")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--attention-override", choices=["checkpoint", "sdpa"], default="checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adapter_cfg_path = os.path.join(args.adapter_path, "adapter_config.json")
    with open(adapter_cfg_path, "r", encoding="utf-8") as f:
        base_path = args.base_path or json.load(f).get("base_model_name_or_path")
    if not base_path or not os.path.exists(base_path):
        raise SystemExit(f"Base model not found at {base_path!r} — pass --base_path.")

    print(f"Loading base from '{base_path}'")
    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.attention_override != "checkpoint":
        from vasudha.models.config import VasudhaConfig
        config = VasudhaConfig.from_pretrained(base_path)
        config.attention_type = args.attention_override
        config.use_cache = False
        load_kwargs["config"] = config
    model = VasudhaForCausalLM.from_pretrained(base_path, **load_kwargs)

    print(f"Applying adapter from '{args.adapter_path}'")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    # Training deliberately disables KV cache; the standalone artifact is for
    # inference, where cache reuse is essential for token-by-token generation.
    model.config.use_cache = True

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.out)

    print(f"Merged model saved to '{args.out}'")


if __name__ == "__main__":
    main()
