"""
Convert Qwen3 pretrained weights into a Vasudha checkpoint on disk.

This is a one-off, CPU-only, low-RAM conversion step. It exists because the
in-process path (`VasudhaForCausalLM.from_qwen3_pretrained`) cannot work on a
free-tier Colab runtime:

  * It materializes a ~3.6B parameter Vasudha model in fp32 on CPU (~14.5 GB),
    which exceeds the 12.7 GB of RAM a free Colab VM has. The process gets
    OOM-killed before any weight is copied.
  * When `load_in_4bit=True`, the *source* Qwen3 model is quantized, so its
    state dict holds packed uint8 blobs whose shapes never match Vasudha's
    fp16/fp32 parameters. Every projection silently fails the shape check and
    stays randomly initialized.

This script avoids both problems: it never instantiates either model with real
storage. Vasudha is built on the `meta` device (names + shapes, zero bytes) and
the Qwen3 weights are streamed tensor-by-tensor straight from the safetensors
shards. Output is written as fp16 shards, so peak RAM stays around the size of
one output shard.

Usage:
    python scripts/convert_qwen3_to_vasudha.py \
        --model Qwen/Qwen3-4B \
        --out ./vasudha-4b-init \
        --num-experts 8 \
        --moe-intermediate-size 864

Then train against the converted checkpoint (this is the step that quantizes):
    python scripts/train_sft.py model.vasudha_path=./vasudha-4b-init
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vasudha.models.qwen3_compat import _qwen3_config_to_vasudha
from vasudha.utils.logging import get_logger, log_banner, setup_logging

logger = get_logger(__name__)

# Keep shards comfortably under the RAM budget of a free Colab VM.
DEFAULT_SHARD_SIZE_BYTES = 1_800_000_000


# ══════════════════════════════════════════════════════════════════════════════
# Source-side: lazy access to the Qwen3 safetensors shards
# ══════════════════════════════════════════════════════════════════════════════


class Qwen3WeightSource:
    """
    Random access to a Qwen3 checkpoint's tensors without loading it into RAM.

    Opens each safetensors shard lazily and caches at most one open handle, so
    the resident set stays at roughly one tensor rather than one model.
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._key_to_file: dict[str, Path] = {}
        self._open_path: Optional[Path] = None
        self._open_handle: Any = None

        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            for key, filename in index["weight_map"].items():
                self._key_to_file[key] = model_dir / filename
        else:
            single = model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(
                    f"No safetensors checkpoint found in {model_dir}. "
                    "Only safetensors sources are supported."
                )
            with safe_open(single, framework="pt") as f:
                for key in f.keys():
                    self._key_to_file[key] = single

        logger.info(
            f"Source checkpoint: {len(self._key_to_file)} tensors across "
            f"{len(set(self._key_to_file.values()))} shard(s)"
        )

    def __contains__(self, key: str) -> bool:
        return key in self._key_to_file

    def keys(self) -> Iterator[str]:
        return iter(self._key_to_file)

    def get(self, key: str) -> torch.Tensor:
        """Read a single tensor from its shard."""
        path = self._key_to_file[key]
        if path != self._open_path:
            if self._open_handle is not None:
                self._open_handle.__exit__(None, None, None)
            self._open_handle = safe_open(path, framework="pt")
            self._open_handle.__enter__()
            self._open_path = path
        return self._open_handle.get_tensor(key)

    def close(self) -> None:
        if self._open_handle is not None:
            self._open_handle.__exit__(None, None, None)
            self._open_handle = None
            self._open_path = None


# ══════════════════════════════════════════════════════════════════════════════
# Destination-side: per-parameter resolution
# ══════════════════════════════════════════════════════════════════════════════


def _init_missing(
    key: str,
    shape: torch.Size,
    dtype: torch.dtype,
    initializer_range: float,
    num_layers: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Initialize a Vasudha parameter that has no Qwen3 counterpart.

    Mirrors `VasudhaPreTrainedModel._init_weights`: normal(0, initializer_range)
    for projections, with the 1/sqrt(2L) depth scaling on residual projections
    (o_proj / down_proj), ones for RMSNorm gammas, zeros for biases.
    """
    if key.endswith(".bias"):
        return torch.zeros(shape, dtype=dtype)

    # RMSNorm gammas are the only 1-D weights in the model.
    if len(shape) == 1:
        return torch.ones(shape, dtype=dtype)

    std = initializer_range
    if ".o_proj." in key or ".down_proj." in key or key.endswith("down_weight"):
        std = initializer_range / math.sqrt(2 * num_layers)

    # Sample in fp32 then cast — normal_ on fp16 loses resolution in the tails.
    out = torch.empty(shape, dtype=torch.float32)
    out.normal_(mean=0.0, std=std, generator=generator)
    return out.to(dtype)


def _resolve(
    dst_key: str,
    dst_shape: torch.Size,
    dtype: torch.dtype,
    src: Qwen3WeightSource,
    initializer_range: float,
    num_layers: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, str]:
    """
    Produce the value for one Vasudha parameter.

    Returns (tensor, provenance) where provenance is one of
    "copied", "upcycled", "router-zero", "random".
    """
    # ── MoE expert upcycling: replicate the dense FFN across experts ──────────
    # Each expert receives a contiguous slice of the dense intermediate dim, so
    # concatenating all experts reproduces the original FFN exactly.
    # MoE layers nest as `mlp.ffn.experts.*`; the dense layers they replace are
    # `mlp.{gate,up,down}_proj.weight`, so the `.ffn.` level drops out.
    if dst_key.endswith(".experts.gate_weight") or dst_key.endswith(".experts.up_weight"):
        src_key = dst_key.replace(".ffn.experts.gate_weight", ".gate_proj.weight").replace(
            ".ffn.experts.up_weight", ".up_proj.weight"
        )
        if src_key in src:
            src_tensor = src.get(src_key)  # (E*I, H)
            E, H, I = dst_shape
            if src_tensor.shape == (E * I, H):
                upcycled = src_tensor.t().reshape(H, E, I).transpose(0, 1).contiguous()
                return upcycled.to(dtype), "upcycled"
            logger.warning(
                f"Cannot upcycle '{src_key}' {tuple(src_tensor.shape)} into "
                f"'{dst_key}' {tuple(dst_shape)} — expected {(E * I, H)}. "
                "Check num_experts * moe_intermediate_size == intermediate_size."
            )

    elif dst_key.endswith(".experts.down_weight"):
        src_key = dst_key.replace(".ffn.experts.down_weight", ".down_proj.weight")
        if src_key in src:
            src_tensor = src.get(src_key)  # (H, E*I)
            E, I, H = dst_shape
            if src_tensor.shape == (H, E * I):
                upcycled = src_tensor.t().reshape(E, I, H).contiguous()
                return upcycled.to(dtype), "upcycled"
            logger.warning(
                f"Cannot upcycle '{src_key}' {tuple(src_tensor.shape)} into "
                f"'{dst_key}' {tuple(dst_shape)} — expected {(H, E * I)}."
            )

    # ── Router: zero init so every expert is weighted equally at step 0 ───────
    # Combined with upcycled experts, this makes the MoE layer's output
    # identical to the dense FFN it replaces, so conversion is loss-preserving.
    elif dst_key.endswith(".router.router_weights.weight"):
        return torch.zeros(dst_shape, dtype=dtype), "router-zero"

    # ── Tied embeddings: a tied source omits lm_head entirely ────────────────
    # Qwen3-4B ties lm_head to embed_tokens, so `lm_head.weight` is absent from
    # its checkpoint. Write an explicit copy rather than letting it fall through
    # to random init, which would destroy the output head.
    if dst_key == "lm_head.weight" and dst_key not in src:
        embed_key = "model.embed_tokens.weight"
        if embed_key in src:
            src_tensor = src.get(embed_key)
            if src_tensor.shape == dst_shape:
                return src_tensor.to(dtype).clone(), "copied"

    # ── Direct name match: Vasudha follows Qwen3 naming for shared modules ────
    if dst_key in src:
        src_tensor = src.get(dst_key)
        if src_tensor.shape == dst_shape:
            return src_tensor.to(dtype), "copied"
        logger.warning(
            f"Shape mismatch for '{dst_key}': src={tuple(src_tensor.shape)}, "
            f"dst={tuple(dst_shape)}. Initializing randomly instead."
        )

    return (
        _init_missing(dst_key, dst_shape, dtype, initializer_range, num_layers, generator),
        "random",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Conversion driver
# ══════════════════════════════════════════════════════════════════════════════


def convert(
    model_name_or_path: str,
    out_dir: Path,
    attention_type: str,
    use_moe: bool,
    num_experts: int,
    moe_intermediate_size: Optional[int],
    dtype: torch.dtype,
    shard_size: int,
    seed: int,
) -> None:
    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    from vasudha.models.vasudha_model import VasudhaForCausalLM

    # ── Fetch the source checkpoint (weights only, no model instantiation) ────
    if os.path.isdir(model_name_or_path):
        model_dir = Path(model_name_or_path)
    else:
        logger.info(f"Downloading '{model_name_or_path}' (weights + config only)...")
        model_dir = Path(
            snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            )
        )

    src = Qwen3WeightSource(model_dir)
    qwen3_config = AutoConfig.from_pretrained(model_dir)

    # ── Derive the expert width from the source, don't trust a hardcoded value ─
    # Upcycling only works when the experts exactly partition the dense FFN, and
    # intermediate_size varies across Qwen3 sizes (4B is 9728, not the 6912 the
    # repo config assumed). Deriving it makes the flag impossible to get wrong.
    if use_moe:
        dense_intermediate = getattr(qwen3_config, "intermediate_size")
        if moe_intermediate_size is None:
            if dense_intermediate % num_experts != 0:
                raise ValueError(
                    f"intermediate_size ({dense_intermediate}) is not divisible by "
                    f"num_experts ({num_experts}); experts cannot partition the FFN. "
                    "Pick a num_experts that divides it evenly."
                )
            moe_intermediate_size = dense_intermediate // num_experts
            logger.info(
                f"Derived moe_intermediate_size={moe_intermediate_size} "
                f"({dense_intermediate} / {num_experts})"
            )
        elif num_experts * moe_intermediate_size != dense_intermediate:
            raise ValueError(
                f"num_experts ({num_experts}) * moe_intermediate_size "
                f"({moe_intermediate_size}) = {num_experts * moe_intermediate_size}, "
                f"but the source FFN is {dense_intermediate} wide. Upcycling would "
                "silently fall back to random init and discard the pretrained FFN. "
                f"Use --moe-intermediate-size {dense_intermediate // num_experts} "
                "or omit the flag to derive it."
            )

    vasudha_config = _qwen3_config_to_vasudha(
        qwen3_config,
        attention_type=attention_type,
        use_moe=use_moe,
        num_experts=num_experts,
        moe_intermediate_size=moe_intermediate_size,
    )
    vasudha_config.torch_dtype = str(dtype).replace("torch.", "")
    logger.info(f"Target config: {vasudha_config!r}")

    # ── Build the target model on `meta`: names and shapes, zero bytes ────────
    with init_empty_weights():
        meta_model = VasudhaForCausalLM(vasudha_config)
    meta_sd = meta_model.state_dict()
    logger.info(f"Target model: {len(meta_sd)} tensors (built on meta device)")

    # ── Plan shards up front so filenames are stable ──────────────────────────
    plan: list[list[str]] = [[]]
    running = 0
    for key, tensor in meta_sd.items():
        nbytes = tensor.numel() * dtype.itemsize
        if running > 0 and running + nbytes > shard_size:
            plan.append([])
            running = 0
        plan[-1].append(key)
        running += nbytes

    total_shards = len(plan)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)

    weight_map: dict[str, str] = {}
    total_bytes = 0
    counts = {"copied": 0, "upcycled": 0, "router-zero": 0, "random": 0}
    random_keys: list[str] = []

    # ── Fill and flush one shard at a time ────────────────────────────────────
    for shard_idx, keys in enumerate(plan, start=1):
        filename = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        shard: dict[str, torch.Tensor] = {}

        for key in keys:
            tensor, provenance = _resolve(
                key,
                meta_sd[key].shape,
                dtype,
                src,
                vasudha_config.initializer_range,
                vasudha_config.num_hidden_layers,
                generator,
            )
            shard[key] = tensor.contiguous()
            weight_map[key] = filename
            counts[provenance] += 1
            if provenance == "random":
                random_keys.append(key)
            total_bytes += tensor.numel() * tensor.element_size()

        save_file(shard, out_dir / filename, metadata={"format": "pt"})
        logger.info(f"Wrote {filename} ({len(shard)} tensors)")
        del shard
        gc.collect()

    src.close()

    # ── Index + config + tokenizer ────────────────────────────────────────────
    with open(out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            f,
            indent=2,
        )

    vasudha_config.save_pretrained(out_dir)
    try:
        AutoTokenizer.from_pretrained(model_dir).save_pretrained(out_dir)
    except Exception as exc:  # tokenizer is a convenience, not a requirement
        logger.warning(f"Could not copy tokenizer: {exc}")

    log_banner("Conversion Complete", str(out_dir))
    logger.info(
        f"{counts['copied']} copied, {counts['upcycled']} upcycled, "
        f"{counts['router-zero']} routers zeroed, {counts['random']} randomly initialized"
    )
    if random_keys:
        # These are the genuinely new modules (GLA gates, etc.) — expected, but
        # worth eyeballing, since anything unexpected here is lost pretraining.
        preview = ", ".join(random_keys[:5])
        logger.info(f"Randomly initialized (first 5): {preview}")
    logger.info(f"Total size: {total_bytes / 1e9:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HF model id or local dir")
    parser.add_argument("--out", default="./vasudha-4b-init", help="Output directory")
    parser.add_argument("--attention-type", default="hybrid")
    parser.add_argument("--no-moe", action="store_true")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=None,
        help="Expert FFN width. Defaults to intermediate_size // num_experts, "
             "which is the only value that upcycles losslessly.",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE_BYTES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(level="INFO")
    log_banner("Qwen3 → Vasudha Conversion", f"{args.model} → {args.out}")

    convert(
        model_name_or_path=args.model,
        out_dir=Path(args.out),
        attention_type=args.attention_type,
        use_moe=not args.no_moe,
        num_experts=args.num_experts,
        moe_intermediate_size=args.moe_intermediate_size,
        dtype=getattr(torch, args.dtype),
        shard_size=args.shard_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
