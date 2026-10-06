"""Checkpoint plumbing.

Training runs on a text-only copy of Qwen3.5 (Qwen3_5ForCausalLM): no vision
tower in GPU memory, no multimodal code paths in TRL or vLLM. The deliverable,
though, should be the full model the base was — vision encoder and MTP head
included — so the trained language weights are written back into the
original checkpoint's tensors at the end. Nothing outside the language model
changes; LoRA never touched it.

    make_text_checkpoint  original hub model -> text-only bf16 copy
    merge_adapter         text base + LoRA adapter -> merged text model
    export_full           merged text model -> original layout, full model
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Optional

TEXT_PREFIX = "model."
FULL_PREFIX = "model.language_model."


def make_text_checkpoint(src: str, out_dir: str, dtype: str = "bfloat16") -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(src, dtype=getattr(torch, dtype), low_cpu_mem_usage=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(out_dir)
    try:
        from transformers import GenerationConfig
        GenerationConfig.from_pretrained(src).save_pretrained(out_dir)
    except Exception:  # noqa: BLE001 - optional
        pass
    with open(os.path.join(out_dir, "neo_source.json"), "w", encoding="utf-8") as fh:
        json.dump({"source": src, "class": type(model).__name__}, fh)
    return out_dir


def merge_adapter(base_dir: str, adapter_dir: str, out_dir: str, device: Optional[str] = None) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 device_map=device)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(out_dir)
    for extra in ("generation_config.json", "neo_source.json"):
        if os.path.exists(os.path.join(base_dir, extra)):
            shutil.copy(os.path.join(base_dir, extra), os.path.join(out_dir, extra))
    return out_dir


def _read_tensors(path_dir: str) -> dict:
    """All tensors of a text checkpoint, keyed in text-model form ("model.X").

    transformers 5 saves a Qwen3.5 text model converted from the multimodal
    checkpoint back in the original key layout ("model.language_model.X"),
    while a model built from a text config saves "model.X"; accept both."""
    from safetensors.torch import load_file

    tensors = {}
    for name in sorted(os.listdir(path_dir)):
        if name.endswith(".safetensors"):
            for key, value in load_file(os.path.join(path_dir, name)).items():
                if key.startswith(FULL_PREFIX):
                    key = TEXT_PREFIX + key[len(FULL_PREFIX):]
                tensors[key] = value
    return tensors


def export_full(text_dir: str, original_dir: str, out_dir: str) -> dict:
    """Write the trained language-model tensors into a copy of the original
    full checkpoint. Returns a small report; raises if anything does not line
    up (a missing or misshapen tensor must never be shipped silently)."""
    from safetensors.torch import load_file, save_file

    trained = _read_tensors(text_dir)
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(original_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
    else:
        shards = [n for n in os.listdir(original_dir) if n.endswith(".safetensors")]

    used, replaced, kept = set(), 0, 0
    for shard in shards:
        tensors = load_file(os.path.join(original_dir, shard))
        out = {}
        for key, value in tensors.items():
            if key.startswith(FULL_PREFIX):
                text_key = TEXT_PREFIX + key[len(FULL_PREFIX):]
                if text_key not in trained:
                    raise KeyError(f"trained checkpoint has no tensor for {key} ({text_key})")
                new = trained[text_key]
                if tuple(new.shape) != tuple(value.shape):
                    raise ValueError(f"shape mismatch for {key}: {tuple(new.shape)} vs {tuple(value.shape)}")
                out[key] = new.to(value.dtype).contiguous()
                used.add(text_key)
                replaced += 1
            else:
                out[key] = value
                kept += 1
        save_file(out, os.path.join(out_dir, shard), metadata={"format": "pt"})

    leftover = sorted(k for k in trained if k not in used and not k.startswith("lm_head."))
    if leftover:
        raise ValueError(f"{len(leftover)} trained tensors have no home in the original layout, e.g. {leftover[:3]}")
    for name in os.listdir(original_dir):
        src = os.path.join(original_dir, name)
        if name.endswith(".safetensors") or os.path.isdir(src):
            continue
        shutil.copy(src, os.path.join(out_dir, name))
    report = {"replaced": replaced, "kept_unchanged": kept, "shards": len(shards)}
    with open(os.path.join(out_dir, "neo_export.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report
