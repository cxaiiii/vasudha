"""Tiny interactive chat bench for the practical Vasudha Engineering adapter.

Local example:
    python scripts/chat_bench.py --adapter ./unsloth-qwen3-8b/checkpoint-250

The base can be a local Qwen3-8B directory or ``Qwen/Qwen3-8B``.
"""
from __future__ import annotations

import argparse


SYSTEM = """You are Vasudha, an engineering-focused AI assistant. Think like a careful engineer: identify governing equations, state assumptions, use Python or approved tools when useful, interpret results, validate them, and discuss limitations. Never claim to have executed code unless a tool result is provided."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory, e.g. checkpoint-250")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--prompt", default=None, help="Answer one prompt and exit (useful from Windows via Modal).")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Vasudha Engineering…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kwargs)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()
    print("Ready. Commands: /reset, /quit\n")

    if args.prompt:
        print(f"Vasudha: {generate_one(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)}")
        return

    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit"}:
            return
        if prompt.lower() == "/reset":
            history = [{"role": "system", "content": SYSTEM}]
            print("[conversation reset]")
            continue
        history.append({"role": "user", "content": prompt})
        answer = generate_one(model, tokenizer, prompt, args.max_new_tokens, args.temperature, history)
        print(f"Vasudha: {answer}\n")
        history.append({"role": "assistant", "content": answer})


def generate_one(model, tokenizer, prompt, max_new_tokens, temperature, history=None):
    import torch
    messages = history or [{"role": "system", "content": SYSTEM}]
    messages = messages + [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    # Recent Transformers returns BatchEncoding; older versions may return a
    # tensor.  Normalize both forms before calling generate().
    inputs = encoded["input_ids"] if hasattr(encoded, "__getitem__") and not hasattr(encoded, "shape") else encoded
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=0.9, use_cache=True)
    return tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    main()
