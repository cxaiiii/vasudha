"""
Qwen3 weight loading compatibility.

Maps Qwen3-4B and Qwen3-8B HuggingFace weights into Vasudha's naming convention.

Key naming differences:
  Qwen3                      Vasudha
  ──────────────────────────────────────────
  model.embed_tokens         model.embed_tokens     (same)
  model.layers.{i}.         model.layers.{i}.
    self_attn.q_proj           self_attn.q_proj     (same)
    self_attn.k_proj           self_attn.k_proj     (same)
    self_attn.v_proj           self_attn.v_proj     (same)
    self_attn.o_proj           self_attn.o_proj     (same)
    self_attn.q_norm           self_attn.q_norm     (same)
    self_attn.k_norm           self_attn.k_norm     (same)
    mlp.gate_proj              mlp.gate_proj        (same for dense layers)
    mlp.up_proj                mlp.up_proj          (same)
    mlp.down_proj              mlp.down_proj        (same)
    input_layernorm            input_layernorm      (same)
    post_attention_layernorm   post_attention_layernorm (same)
  model.norm                 model.norm             (same)
  lm_head                    lm_head                (same)

Qwen3-4B/8B are dense models, so all FFN weights map directly to Vasudha's
dense SwiGLU layers. MoE layers in Vasudha will be randomly initialized
(since Qwen3-4B/8B have no MoE weights to load).

Strategy:
  1. Load Qwen3 model with from_pretrained (into Qwen3ForCausalLM)
  2. Build VasudhaForCausalLM with matching config
  3. Copy overlapping weights by name
  4. Log which weights were loaded vs. randomly initialized
"""

from __future__ import annotations

import re
from typing import Any, Optional, TYPE_CHECKING

import torch

from vasudha.models.config import VasudhaConfig
from vasudha.utils.logging import get_logger

if TYPE_CHECKING:
    from vasudha.models.vasudha_model import VasudhaForCausalLM

logger = get_logger(__name__)


def load_from_qwen3(
    cls: type,
    model_name_or_path: str,
    attention_type: str = "hybrid",
    use_moe: bool = True,
    num_experts: int = 64,
    moe_intermediate_size: Optional[int] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    **kwargs: Any,
) -> "VasudhaForCausalLM":
    """
    Load Qwen3 pretrained weights into a VasudhaForCausalLM.

    This function:
    1. Downloads/loads the Qwen3 model from HuggingFace
    2. Extracts its config to build a compatible VasudhaConfig
    3. Instantiates VasudhaForCausalLM with random weights
    4. Copies matching weights from Qwen3 → Vasudha
    5. Returns the partially-initialized model (MoE layers are random)

    Args:
        cls: VasudhaForCausalLM class.
        model_name_or_path: HuggingFace model ID or local path.
        attention_type: Vasudha attention type to use.
        use_moe: Whether to add MoE layers.
        load_in_4bit: QLoRA mode.
        load_in_8bit: 8-bit mode.
        torch_dtype: Compute dtype.
        device_map: Device placement strategy.
        **kwargs: Extra args forwarded to transformers.from_pretrained.

    Returns:
        VasudhaForCausalLM with Qwen3 weights loaded.

    Warning:
        This path instantiates the full Vasudha model in fp32 on CPU (~14.5GB for
        the 4B geometry) and only quantizes the *source* Qwen3 model, so the
        returned Vasudha model is unquantized. It is unusable on a free Colab VM.
        Use scripts/convert_qwen3_to_vasudha.py instead, which streams weights
        with a bounded memory footprint and writes a checkpoint that
        bitsandbytes can quantize on load.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    logger.info(f"Loading Qwen3 weights from '{model_name_or_path}'...")

    # ── Reject quantized sources ───────────────────────────────────────────────
    # A quantized source model exposes packed uint8 blobs in its state dict, so
    # every projection would fail the shape check in _copy_weights and be left at
    # its random init — a silently untrained model. Refuse rather than mislead.
    if load_in_4bit or load_in_8bit:
        raise ValueError(
            "load_from_qwen3 cannot copy weights out of a quantized source model. "
            "Convert on CPU first, then quantize on load:\n"
            "    python scripts/convert_qwen3_to_vasudha.py "
            f"--model {model_name_or_path} --out ./vasudha-init\n"
            "    python scripts/train_sft.py model.vasudha_path=./vasudha-init"
        )

    # ── Load Qwen3 source model ────────────────────────────────────────────────
    load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": "auto" if torch_dtype == "auto" else torch_dtype,
    }
    load_kwargs.update(kwargs)

    qwen3_config = AutoConfig.from_pretrained(model_name_or_path)
    qwen3_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, **load_kwargs
    )
    logger.info(f"Qwen3 model loaded. Config: {qwen3_config.model_type}")

    # ── Build Vasudha config from Qwen3 config ─────────────────────────────────
    vasudha_config = _qwen3_config_to_vasudha(
        qwen3_config,
        attention_type=attention_type,
        use_moe=use_moe,
        num_experts=num_experts,
        moe_intermediate_size=moe_intermediate_size,
    )
    logger.info(f"Vasudha config: {vasudha_config!r}")

    # ── Instantiate Vasudha model ──────────────────────────────────────────────
    # Load on CPU first for weight copying
    vasudha_model: "VasudhaForCausalLM" = cls(vasudha_config)
    vasudha_model.eval()

    # ── Copy weights ───────────────────────────────────────────────────────────
    num_loaded, num_skipped, num_missing = _copy_weights(
        src_model=qwen3_model,
        dst_model=vasudha_model,
    )

    logger.info(
        f"Weight loading complete: "
        f"{num_loaded} loaded, "
        f"{num_skipped} skipped (shape mismatch), "
        f"{num_missing} randomly initialized (MoE / new layers)"
    )

    # Free Qwen3 model memory
    del qwen3_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vasudha_model


def _qwen3_config_to_vasudha(
    qwen3_config: Any,
    attention_type: str = "hybrid",
    use_moe: bool = True,
    num_experts: int = 64,
    moe_intermediate_size: Optional[int] = None,
) -> VasudhaConfig:
    """
    Build a VasudhaConfig that matches the given Qwen3 config's geometry.

    Inspects the Qwen3 config attributes and maps them to VasudhaConfig params.
    Handles Qwen3 config naming (which may differ slightly between versions).

    Args:
        qwen3_config: A Qwen3 PretrainedConfig (AutoConfig result).
        attention_type: Attention type for Vasudha.
        use_moe: Whether to enable MoE in Vasudha.

    Returns:
        VasudhaConfig with matching architectural parameters.
    """
    # Safe attribute access with fallbacks
    def _get(attr: str, default: Any) -> Any:
        return getattr(qwen3_config, attr, default)

    hidden_size: int = _get("hidden_size", 2560)
    num_layers: int = _get("num_hidden_layers", 36)
    num_heads: int = _get("num_attention_heads", 32)
    num_kv_heads: int = _get("num_key_value_heads", 8)
    intermediate_size: int = _get("intermediate_size", 6912)
    head_dim: int = _get("head_dim", hidden_size // num_heads)
    vocab_size: int = _get("vocab_size", 151936)
    max_pos: int = _get("max_position_embeddings", 32768)
    rope_theta: float = _get("rope_theta", 1_000_000.0)
    rms_norm_eps: float = _get("rms_norm_eps", 1e-6)
    attention_bias: bool = _get("attention_bias", False)
    tie_embeddings: bool = _get("tie_word_embeddings", False)
    bos_token_id: int = _get("bos_token_id", 151643)
    eos_token_id: Any = _get("eos_token_id", 151645)

    # Qwen3 always has QK-Norm (introduced in Qwen3)
    qk_norm: bool = True

    return VasudhaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_pos,
        rope_theta=rope_theta,
        rms_norm_eps=rms_norm_eps,
        attention_bias=attention_bias,
        qk_norm=qk_norm,
        attention_type=attention_type,
        hybrid_pattern=["gla", "gla", "gla", "sdpa"],
        use_moe=use_moe,
        num_experts=num_experts,
        num_experts_per_tok=2,
        moe_intermediate_size=moe_intermediate_size if moe_intermediate_size is not None else intermediate_size,
        tie_word_embeddings=tie_embeddings,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )


def _copy_weights(
    src_model: Any,
    dst_model: "VasudhaForCausalLM",
) -> tuple[int, int, int]:
    """
    Copy weights from Qwen3 model to VasudhaForCausalLM by matching parameter names.

    Qwen3 and Vasudha share the same naming convention for most parameters
    (since Vasudha was designed to be Qwen3-compatible). This makes copying
    straightforward: we simply iterate over Vasudha's state dict and look for
    matching keys in Qwen3's state dict.

    Parameters that exist in Vasudha but not Qwen3 (e.g., MoE router weights,
    GLA gate projections) are left at their randomly initialized values.

    Args:
        src_model: Source Qwen3 model.
        dst_model: Destination VasudhaForCausalLM.

    Returns:
        Tuple of (num_loaded, num_skipped, num_missing).
    """
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()

    num_loaded = 0
    num_skipped = 0
    num_missing = 0

    new_sd = dst_sd.copy()

    for dst_key in dst_sd:
        # ── Upcycling logic for MoE ──
        if dst_key.endswith(".experts.gate_weight") or dst_key.endswith(".experts.up_weight"):
            # MoE layers nest as `mlp.ffn.experts.*`; the dense FFN they replace
            # is `mlp.{gate,up}_proj.weight`, so the `.ffn.` level drops out.
            src_key = dst_key.replace(".ffn.experts.gate_weight", ".gate_proj.weight").replace(".ffn.experts.up_weight", ".up_proj.weight")
            if src_key in src_sd:
                src_tensor = src_sd[src_key]  # (intermediate_size, hidden_size)
                dst_tensor = dst_sd[dst_key]  # (num_experts, hidden_size, expert_intermediate_size)
                
                E, H, I = dst_tensor.shape
                # PyTorch linear weight is (out_features, in_features).
                # Transpose to (hidden, intermediate), reshape to (hidden, E, I), and transpose to (E, hidden, I)
                if src_tensor.shape[0] == E * I and src_tensor.shape[1] == H:
                    upcycled = src_tensor.t().reshape(H, E, I).transpose(0, 1)
                    new_sd[dst_key] = upcycled.to(dst_tensor.dtype)
                    num_loaded += 1
                    logger.debug(f"Upcycled {src_key} -> {dst_key}")
                    continue
                    
        elif dst_key.endswith(".experts.down_weight"):
            src_key = dst_key.replace(".ffn.experts.down_weight", ".down_proj.weight")
            if src_key in src_sd:
                src_tensor = src_sd[src_key]  # (hidden_size, intermediate_size)
                dst_tensor = dst_sd[dst_key]  # (num_experts, expert_intermediate_size, hidden_size)
                
                E, I, H = dst_tensor.shape
                # Transpose to (intermediate, hidden) then reshape to (E, I, H)
                if src_tensor.shape[1] == E * I and src_tensor.shape[0] == H:
                    upcycled = src_tensor.t().reshape(E, I, H)
                    new_sd[dst_key] = upcycled.to(dst_tensor.dtype)
                    num_loaded += 1
                    logger.debug(f"Upcycled {src_key} -> {dst_key}")
                    continue
                    
        elif dst_key.endswith(".router.router_weights.weight"):
            # Initialize router to all zeros to route equally to all experts initially
            dst_tensor = dst_sd[dst_key]
            new_sd[dst_key] = torch.zeros_like(dst_tensor)
            num_loaded += 1
            logger.debug(f"Initialized zero routing for {dst_key}")
            continue

        # Direct name match (most weights)
        if dst_key in src_sd:
            src_tensor = src_sd[dst_key]
            dst_tensor = dst_sd[dst_key]

            if src_tensor.shape == dst_tensor.shape:
                new_sd[dst_key] = src_tensor.to(dst_tensor.dtype)
                num_loaded += 1
            else:
                logger.warning(
                    f"Shape mismatch for '{dst_key}': "
                    f"src={src_tensor.shape}, dst={dst_tensor.shape}. Skipping."
                )
                num_skipped += 1
        else:
            # This weight doesn't exist in Qwen3 — it's new in Vasudha
            # (e.g., GLA g_proj, MoE router, reasoning controller)
            logger.debug(f"Missing in Qwen3: '{dst_key}' — will be randomly initialized")
            num_missing += 1

    # Load the new state dict
    dst_model.load_state_dict(new_sd, strict=False)

    return num_loaded, num_skipped, num_missing


def get_qwen3_weight_map(
    qwen3_model_name: str,
) -> dict[str, str]:
    """
    Return the full key mapping from Qwen3 to Vasudha parameter names.

    Useful for debugging or custom weight copying logic.

    Args:
        qwen3_model_name: Model name (e.g., "Qwen/Qwen3-4B").

    Returns:
        Dict mapping Qwen3 key → Vasudha key.
    """
    from transformers import AutoConfig
    qwen3_config = AutoConfig.from_pretrained(qwen3_model_name)
    n_layers = qwen3_config.num_hidden_layers

    mapping: dict[str, str] = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for i in range(n_layers):
        layer_keys = [
            ("self_attn.q_proj.weight", "self_attn.q_proj.weight"),
            ("self_attn.k_proj.weight", "self_attn.k_proj.weight"),
            ("self_attn.v_proj.weight", "self_attn.v_proj.weight"),
            ("self_attn.o_proj.weight", "self_attn.o_proj.weight"),
            ("self_attn.q_norm.weight", "self_attn.q_norm.weight"),
            ("self_attn.k_norm.weight", "self_attn.k_norm.weight"),
            ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
            ("mlp.up_proj.weight", "mlp.up_proj.weight"),
            ("mlp.down_proj.weight", "mlp.down_proj.weight"),
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("post_attention_layernorm.weight", "post_attention_layernorm.weight"),
        ]
        prefix = f"model.layers.{i}."
        for qwen_key, vasudha_key in layer_keys:
            mapping[prefix + qwen_key] = prefix + vasudha_key

    return mapping
