"""Generate the same engineering prompts from both Vasudha branches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPTS = [
    "Design a cantilever beam for a 200 N tip load. State assumptions, equations, Python simulation, safety factor, and limitations.",
    "Estimate pressure drop in a water pipe and show a Python validation workflow.",
    "Create a 5 V RC filter and explain how you would simulate and test it.",
    "Generate a parametric gear housing in CadQuery and explain how to validate the STEP output.",
    "Design a small PCB power circuit and describe the netlist, ERC, and bench-validation steps.",
]


def load_branch(base: str, adapter: str | None, vasudha: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if vasudha:
        from vasudha.models.config import VasudhaConfig
        from vasudha.models.vasudha_model import VasudhaForCausalLM
        config = VasudhaConfig.from_pretrained(base)
        config.attention_type = "sdpa"
        config.use_cache = True
        model = VasudhaForCausalLM.from_pretrained(base, config=config, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)


def generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True)
    return tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-base", default="Qwen/Qwen3-8B")
    parser.add_argument("--qwen-adapter", required=True)
    parser.add_argument("--vasudha-base", required=True)
    parser.add_argument("--vasudha-adapter", required=True)
    parser.add_argument("--out", default="engineering_comparison.json")
    args = parser.parse_args()
    qwen, qwen_tok = load_branch(args.qwen_base, args.qwen_adapter, False)
    vasudha, vasudha_tok = load_branch(args.vasudha_base, args.vasudha_adapter, True)
    results = []
    for prompt in PROMPTS:
        results.append({"prompt": prompt, "qwen_unsloth": generate(qwen, qwen_tok, prompt), "vasudha_hybrid": generate(vasudha, vasudha_tok, prompt)})
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} prompt comparisons to {args.out}")


if __name__ == "__main__":
    main()
