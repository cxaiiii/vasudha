"""
Isolate where a converted Vasudha checkpoint diverges from its Qwen3 source.

Runs inference only — no optimizer, no LoRA, no TRL — so the number it prints is
attributable to the model code and the checkpoint alone. Loads the two models
sequentially and frees each, so peak VRAM is one model, not two.

Usage:
    python scripts/diagnose_conversion.py ./vasudha-4b-dense --qwen Qwen/Qwen3-4B
    python scripts/diagnose_conversion.py ./vasudha-4b-dense --load-in-4bit
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

# Running `python scripts/diagnose_conversion.py` puts scripts/ on sys.path, not
# the repo root, so `vasudha` is unimportable without this — same as train_sft.
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

DEFAULT_TEXT = (
    "The capital of France is Paris. The capital of Japan is Tokyo. "
    "If a train travels 60 kilometers in one hour, it travels 120 kilometers "
    "in two hours."
)


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _manual_ce(logits: torch.Tensor, ids: torch.Tensor) -> float:
    """Cross-entropy computed outside the model, bypassing the fused kernel."""
    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = ids[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)).item()


def _bisect_layer0(model, ref_hidden: list[torch.Tensor]) -> None:
    """
    Split layer 0 into attention and FFN and score each against a local reference
    built from the layer's own weights, so a mismatch means the *code* is wrong,
    not the checkpoint.
    """
    layer = model.model.layers[0]
    dev = next(model.parameters()).device
    x = ref_hidden[0].to(dev, torch.float16)
    expected = ref_hidden[1].to(dev, torch.float16)
    pos = torch.arange(x.shape[1], device=dev).unsqueeze(0)

    print("\n── layer 0 bisect ──")
    with torch.no_grad():
        out = layer(hidden_states=x, attention_mask=None, position_ids=pos, use_cache=False)
        got = out[0] if isinstance(out, (tuple, list)) else out.hidden_states
    print(f"full layer                 : {_rel(got, expected):.4f}")

    attn = layer.self_attn
    with torch.no_grad():
        xn = layer.input_layernorm(x)
        a = attn(hidden_states=xn, attention_mask=None, position_ids=pos, use_cache=False)
        a = a[0] if isinstance(a, (tuple, list)) else a

        # Canonical GQA written out longhand, using HF's own rotary helper.
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb as hf_rope,
        )

        b, s, _ = xn.shape
        hd, nh, nkv = attn.head_dim, attn.num_heads, attn.num_key_value_heads
        q = attn.q_proj(xn).view(b, s, nh, hd)
        k = attn.k_proj(xn).view(b, s, nkv, hd)
        v = attn.v_proj(xn).view(b, s, nkv, hd)
        if attn.q_norm is not None:
            q, k = attn.q_norm(q), attn.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        inv_freq = 1.0 / (
            attn.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=dev) / hd)
        )
        freqs = torch.outer(pos[0].float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(xn.dtype)[None], emb.sin().to(xn.dtype)[None]
        q_r, k_r = hf_rope(q, k, cos, sin, unsqueeze_dim=1)

        n_rep = nh // nkv
        k_r = k_r.repeat_interleave(n_rep, dim=1)
        v_r = v.repeat_interleave(n_rep, dim=1)
        ref_a = F.scaled_dot_product_attention(q_r, k_r, v_r, is_causal=True)
        ref_a = attn.o_proj(ref_a.transpose(1, 2).reshape(b, s, nh * hd))
    print(f"attention vs longhand      : {_rel(a, ref_a):.4f}")

    with torch.no_grad():
        xm = layer.post_attention_layernorm(x)
        m = layer.mlp(xm)
        g, u = layer.mlp.gate_proj(xm), layer.mlp.up_proj(xm)
        ref_m = layer.mlp.down_proj(F.silu(g) * u)
    print(f"mlp fused vs eager swiglu  : {_rel(m, ref_m):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vasudha_path")
    parser.add_argument("--qwen", default="Qwen/Qwen3-4B")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    device = {"": 0} if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.vasudha_path)
    ids = tok(args.text, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        ids = ids.cuda()
    print(f"tokens: {ids.shape[-1]}")

    # ── Stock Qwen3 first, so its per-layer activations survive for the diff ──
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(
        args.qwen, torch_dtype=torch.float16, device_map=device
    )
    ref.eval()
    with torch.no_grad():
        ref_out = ref(input_ids=ids, labels=ids, output_hidden_states=True)
    ref_loss = ref_out.loss.item()
    ref_manual = _manual_ce(ref_out.logits, ids)
    # Keep on CPU in fp32: 37 tokens x 2560 x 37 layers is negligible.
    ref_hidden = [h.detach().float().cpu() for h in ref_out.hidden_states]
    print(f"qwen3 loss            : {ref_loss:.4f}")
    print(f"qwen3 loss (manual)   : {ref_manual:.4f}")

    del ref, ref_out
    _free()

    # ── Vasudha ───────────────────────────────────────────────────────────────
    from vasudha.models.vasudha_model import VasudhaForCausalLM

    load_kwargs = {"torch_dtype": torch.float16, "device_map": device}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model, info = VasudhaForCausalLM.from_pretrained(
        args.vasudha_path, output_loading_info=True, **load_kwargs
    )
    model.eval()

    # Anything here is a weight that silently stayed at its random init.
    for label in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        keys = info.get(label) or []
        print(f"{label}: {len(keys)}  {keys[:8]}")

    # The uint8 trap: under 4-bit a Linear4bit reports .weight.dtype == uint8,
    # and vasudha_model casts hidden_states to it before the matmul.
    lm_head = model.lm_head
    print(f"lm_head: {type(lm_head).__name__}  weight.dtype={lm_head.weight.dtype}")
    print(f"config: attention_type={model.config.attention_type} use_moe={model.config.use_moe}")

    with torch.no_grad():
        out = model(input_ids=ids, labels=ids, output_hidden_states=True)
    logits = out.logits
    print(f"logits: dtype={logits.dtype} finite={torch.isfinite(logits).all().item()} "
          f"std={logits.float().std().item():.4f}")
    print(f"vasudha loss (model)  : {out.loss.item():.4f}")
    print(f"vasudha loss (manual) : {_manual_ce(logits, ids):.4f}")

    # ── Per-layer divergence ──────────────────────────────────────────────────
    # hidden_states[0] is the embedding output; [i] is the output of layer i-1.
    # The first index where relative error jumps is the module that broke.
    got = [h.detach().float().cpu() for h in out.hidden_states]
    print(f"\nhidden_states: qwen={len(ref_hidden)} vasudha={len(got)}")
    print("layer  rel_err   note")
    first_bad = None
    for i, (a, b) in enumerate(zip(got, ref_hidden)):
        if a.shape != b.shape:
            print(f"{i:>5}  SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
            first_bad = first_bad if first_bad is not None else i
            break
        rel = ((a - b).norm() / b.norm().clamp_min(1e-6)).item()
        flag = ""
        if rel > 0.05 and first_bad is None:
            first_bad = i
            flag = "  <-- first divergence"
        # Print the head, the tail, and anything anomalous.
        if i < 4 or i >= len(got) - 2 or flag:
            print(f"{i:>5}  {rel:>7.4f}{flag}")

    if first_bad is None:
        print("\nAll hidden states match — divergence is in lm_head or the loss.")
    elif first_bad == 0:
        print("\nEmbeddings differ: the checkpoint's embed_tokens is not Qwen3's.")
    else:
        print(f"\nFirst bad layer: {first_bad - 1} (hidden_states[{first_bad}]). "
              "Layers before it are exact, so the bug is inside that layer.")

    if first_bad == 1:
        _bisect_layer0(model, ref_hidden)


if __name__ == "__main__":
    main()
