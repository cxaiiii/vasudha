"""
Vasudha model configuration.

VasudhaConfig is the single source of truth for all architectural hyperparameters.
It extends HuggingFace's PretrainedConfig for full ecosystem compatibility.

Design Principles:
  1. Every architectural choice is a config parameter — nothing is hardcoded.
  2. Sensible defaults match Qwen3-4B geometry for Colab T4 compatibility.
  3. Attention type, MoE usage, hybrid pattern all controlled from config.
  4. Configs are YAML-serializable and Hydra-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from transformers import PretrainedConfig


class VasudhaConfig(PretrainedConfig):
    """
    Configuration class for Vasudha language models.

    Extends HuggingFace's PretrainedConfig to remain compatible with the
    transformers ecosystem (from_pretrained, push_to_hub, etc.).

    Architectural Overview:
      - Hybrid attention: alternates between Gated Linear Attention (GLA) and
        full Grouped Query Attention (GQA) at a configurable ratio.
      - Sparse MoE: every other transformer layer uses MoE FFN; the rest use
        standard SwiGLU dense FFN.
      - RMSNorm with optional QK-Norm (Qwen3-style training stability).
      - RoPE positional embeddings with configurable theta.

    Args:
        vocab_size: Vocabulary size of the tokenizer.
        hidden_size: Dimension of the hidden states (embedding dimension).
        intermediate_size: Hidden dimension of the SwiGLU FFN (before gating).
            If None, defaults to 8/3 * hidden_size rounded to nearest 64.
        num_hidden_layers: Total number of transformer decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of KV heads for Grouped Query Attention.
            Must divide num_attention_heads. Smaller values save KV cache memory.
        head_dim: Dimension of each attention head. Defaults to hidden_size //
            num_attention_heads if not specified.
        max_position_embeddings: Maximum sequence length for RoPE precomputation.
        rope_theta: RoPE base frequency. Qwen3 uses 1,000,000 for long context.
        rope_scaling: Optional dict for RoPE scaling (e.g., YaRN for >32K context).
        rms_norm_eps: Epsilon for RMSNorm numerical stability.
        attention_bias: Whether to add bias to attention QKV projections.
            Qwen3 removes QKV bias (unlike Qwen2). Set False for Qwen3 compat.
        qk_norm: Whether to apply QK-Norm (per-head RMSNorm on Q and K before
            attention). Qwen3 innovation for training stability at large scale.

        attention_type: Default attention type for all layers. One of:
            - "sdpa": PyTorch scaled dot-product attention (always available)
            - "flash": FlashAttention-2 (CUDA, optional dep)
            - "gla": Pure GLA (our reference implementation)
            - "gla_triton": GLA with Triton kernels (CUDA, requires triton)
            - "sliding_window": Sliding window local attention
            - "hybrid": Use hybrid_pattern to define per-layer attention type
        hybrid_pattern: If attention_type == "hybrid", specifies the repeating
            pattern of attention types. Example: ["gla", "gla", "gla", "sdpa"]
            means every 4th layer uses full GQA, others use GLA.
            The pattern repeats for all num_hidden_layers layers.
        sliding_window_size: Window size for sliding window attention.

        use_moe: Whether to use Sparse MoE FFN in alternating layers.
        num_experts: Total number of expert FFNs (when use_moe=True).
        num_experts_per_tok: Number of experts activated per token (Top-k).
        moe_intermediate_size: Hidden dim of each expert's SwiGLU FFN.
            If None, defaults to intermediate_size // num_experts_per_tok.
        moe_layers: Which layer indices use MoE FFN. If None and use_moe=True,
            defaults to every other layer (1, 3, 5, ...).
        router_aux_loss_coef: Weight for the load balancing auxiliary loss.
        router_z_loss_coef: Weight for the router Z-loss (prevents logit collapse).
        router_jitter_noise: Noise std for router logits during training
            (Noisy Top-k for better exploration).
        expert_capacity_factor: Maximum fraction of tokens each expert can process.
            1.0 = perfectly balanced; >1.0 = overflow buffer.
        use_adaptive_routing: Enable difficulty-adaptive Top-k routing.
            When True, the number of experts per token varies based on a
            difficulty predictor (Easy→k=1, Medium→k=2, Hard→k=4).
        adaptive_routing_min_k: Minimum k for adaptive routing (Easy mode).
        adaptive_routing_max_k: Maximum k for adaptive routing (Hard mode).

        use_reasoning_controller: Enable the lightweight difficulty predictor
            that drives adaptive routing and optional thinking mode.

        tie_word_embeddings: Whether to tie input and output embeddings.
        initializer_range: Std for weight initialization.
        use_cache: Whether to use KV caching during inference.

    Example:
        >>> # Qwen3-4B-compatible config
        >>> config = VasudhaConfig(
        ...     vocab_size=151936,
        ...     hidden_size=2560,
        ...     num_hidden_layers=36,
        ...     num_attention_heads=32,
        ...     num_key_value_heads=8,
        ...     attention_type="hybrid",
        ...     use_moe=True,
        ... )
    """

    model_type = "vasudha"

    def __init__(
        self,
        # ── Vocabulary ─────────────────────────────────────────────────────────
        vocab_size: int = 151936,
        # ── Core dimensions ────────────────────────────────────────────────────
        hidden_size: int = 2560,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 36,
        # ── Attention heads ────────────────────────────────────────────────────
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: Optional[int] = None,
        # ── Position embeddings ────────────────────────────────────────────────
        max_position_embeddings: int = 32768,
        rope_theta: float = 1_000_000.0,
        rope_scaling: Optional[dict[str, Any]] = None,
        # ── Normalization ──────────────────────────────────────────────────────
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        qk_norm: bool = True,
        # ── Attention type & hybrid pattern ────────────────────────────────────
        attention_type: str = "hybrid",
        hybrid_pattern: Optional[list[str]] = None,
        sliding_window_size: Optional[int] = 4096,
        # ── Sparse MoE ────────────────────────────────────────────────────────
        use_moe: bool = True,
        num_experts: int = 64,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: Optional[int] = None,
        moe_layers: Optional[list[int]] = None,
        router_aux_loss_coef: float = 1e-3,
        router_z_loss_coef: float = 1e-4,
        router_jitter_noise: float = 0.0,
        expert_capacity_factor: float = 1.25,
        # ── Adaptive routing ──────────────────────────────────────────────────
        use_adaptive_routing: bool = False,
        adaptive_routing_min_k: int = 1,
        adaptive_routing_max_k: int = 4,
        # ── Reasoning controller ──────────────────────────────────────────────
        use_reasoning_controller: bool = False,
        # ── Output / misc ─────────────────────────────────────────────────────
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 151643,
        eos_token_id: Union[int, list[int]] = 151645,
        **kwargs: Any,
    ) -> None:
        # ── Store all parameters ───────────────────────────────────────────────
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # head_dim defaults to hidden_size // num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)

        # intermediate_size: 8/3 * hidden_size is the standard SwiGLU ratio,
        # rounded to nearest multiple of 64 for hardware alignment
        if intermediate_size is None:
            _raw = int(8 / 3 * hidden_size)
            self.intermediate_size = (_raw + 63) // 64 * 64
        else:
            self.intermediate_size = intermediate_size

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.qk_norm = qk_norm

        # ── Attention type ─────────────────────────────────────────────────────
        _valid_attn_types = {
            "sdpa", "flash", "gla", "gla_triton", "sliding_window", "hybrid"
        }
        if attention_type not in _valid_attn_types:
            raise ValueError(
                f"attention_type='{attention_type}' is not valid. "
                f"Choose from: {sorted(_valid_attn_types)}"
            )
        self.attention_type = attention_type

        # Default hybrid pattern: [GLA, GLA, GLA, SDPA] — 3:1 ratio
        if hybrid_pattern is None:
            self.hybrid_pattern = ["gla", "gla", "gla", "sdpa"]
        else:
            self.hybrid_pattern = hybrid_pattern

        self.sliding_window_size = sliding_window_size

        # ── Sparse MoE ─────────────────────────────────────────────────────────
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Expert FFN intermediate size: typically same as dense FFN
        if moe_intermediate_size is None:
            self.moe_intermediate_size = self.intermediate_size
        else:
            self.moe_intermediate_size = moe_intermediate_size

        # Default: MoE at odd-indexed layers (1, 3, 5, ..., num_hidden_layers-1)
        if moe_layers is None and use_moe:
            self.moe_layers = list(range(1, num_hidden_layers, 2))
        else:
            self.moe_layers = moe_layers or []

        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef
        self.router_jitter_noise = router_jitter_noise
        self.expert_capacity_factor = expert_capacity_factor

        # ── Adaptive routing ───────────────────────────────────────────────────
        self.use_adaptive_routing = use_adaptive_routing
        self.adaptive_routing_min_k = adaptive_routing_min_k
        self.adaptive_routing_max_k = adaptive_routing_max_k

        # ── Reasoning controller ───────────────────────────────────────────────
        self.use_reasoning_controller = use_reasoning_controller

        # ── Pass to parent (handles tie_word_embeddings, pad/bos/eos ids) ─────
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        # Note: initializer_range set after super().__init__ to avoid kwarg collision
        self.initializer_range = initializer_range
        self.use_cache = use_cache

    @property
    def num_q_heads(self) -> int:
        """Alias for num_attention_heads."""
        return self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        """Alias for num_key_value_heads."""
        return self.num_key_value_heads

    @property
    def gqa_ratio(self) -> int:
        """Number of Q heads per KV head (for GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    def get_attention_type_for_layer(self, layer_idx: int) -> str:
        """
        Return the attention type for a given layer index.

        For hybrid attention, cycles through hybrid_pattern.
        For non-hybrid types, returns attention_type directly.

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            Attention type string for this layer.

        Example:
            >>> config = VasudhaConfig(attention_type="hybrid",
            ...                        hybrid_pattern=["gla", "gla", "gla", "sdpa"])
            >>> config.get_attention_type_for_layer(3)
            'sdpa'
            >>> config.get_attention_type_for_layer(4)
            'gla'
        """
        if self.attention_type != "hybrid":
            return self.attention_type
        pattern = self.hybrid_pattern
        return pattern[layer_idx % len(pattern)]

    def is_moe_layer(self, layer_idx: int) -> bool:
        """
        Return True if the given layer index uses MoE FFN.

        Args:
            layer_idx: Zero-based layer index.

        Returns:
            True if this layer should use Sparse MoE FFN.
        """
        return self.use_moe and (layer_idx in self.moe_layers)

    def estimate_parameters(self) -> dict[str, int]:
        """
        Estimate the total parameter count broken down by component.

        This is a rough estimate useful for planning memory usage before
        instantiating the full model.

        Returns:
            Dict with parameter count by component and 'total'.
        """
        H = self.hidden_size
        I = self.intermediate_size
        V = self.vocab_size
        L = self.num_hidden_layers
        Q = self.num_attention_heads
        KV = self.num_key_value_heads
        Hd = self.head_dim
        E = self.num_experts
        EI = self.moe_intermediate_size

        # Embedding
        embed = V * H

        # Attention per layer: Q, K, V, O projections
        # Q: (H, Q*Hd), K: (H, KV*Hd), V: (H, KV*Hd), O: (Q*Hd, H)
        attn_per_layer = H * Q * Hd + H * KV * Hd + H * KV * Hd + Q * Hd * H

        # Dense FFN per layer: gate + up + down (SwiGLU)
        dense_ffn_per_layer = H * I + H * I + I * H

        # MoE FFN per layer: E experts × (gate + up + down) + router
        moe_ffn_per_layer = E * (H * EI + H * EI + EI * H) + H * E

        # Layer norms (per layer: 2 norms)
        norms_per_layer = 2 * H

        # Final norm + LM head
        final = H + (0 if self.tie_word_embeddings else V * H)

        # Sum over layers
        num_moe = len(self.moe_layers)
        num_dense = L - num_moe

        total_attn = L * attn_per_layer
        total_dense_ffn = num_dense * dense_ffn_per_layer
        total_moe_ffn = num_moe * moe_ffn_per_layer
        total_norms = L * norms_per_layer

        total = embed + total_attn + total_dense_ffn + total_moe_ffn + total_norms + final

        return {
            "embeddings": embed,
            "attention": total_attn,
            "dense_ffn": total_dense_ffn,
            "moe_ffn": total_moe_ffn,
            "norms": total_norms,
            "lm_head": final,
            "total": total,
        }

    @classmethod
    def for_qwen3_4b(cls, **kwargs: Any) -> "VasudhaConfig":
        """
        Return a config matching Qwen3-4B geometry.

        Qwen3-4B specs (from model card):
          - 4.02B parameters (dense)
          - 36 layers, hidden=2560, heads=32, KV heads=8
          - intermediate_size=6912 (SwiGLU)
          - rope_theta=1,000,000
          - max_position: 32768 (extended to 128K via YaRN)

        Args:
            **kwargs: Override any config parameter.

        Returns:
            VasudhaConfig pre-set for Qwen3-4B geometry.
        """
        defaults = dict(
            vocab_size=151936,
            hidden_size=2560,
            intermediate_size=6912,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=32768,
            rope_theta=1_000_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            qk_norm=True,
            attention_type="hybrid",
            hybrid_pattern=["gla", "gla", "gla", "sdpa"],
            use_moe=True,
            num_experts=64,
            num_experts_per_tok=2,
            moe_intermediate_size=6912,
            router_aux_loss_coef=1e-3,
            router_z_loss_coef=1e-4,
            tie_word_embeddings=False,
            bos_token_id=151643,
            eos_token_id=151645,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_qwen3_8b(cls, **kwargs: Any) -> "VasudhaConfig":
        """
        Return a config matching Qwen3-8B geometry.

        Qwen3-8B specs:
          - 8.19B parameters (dense)
          - 36 layers, hidden=4096, heads=32, KV heads=8
          - intermediate_size=14336
        """
        defaults = dict(
            vocab_size=151936,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=32768,
            rope_theta=1_000_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            qk_norm=True,
            attention_type="hybrid",
            hybrid_pattern=["gla", "gla", "gla", "sdpa"],
            use_moe=True,
            num_experts=64,
            num_experts_per_tok=2,
            moe_intermediate_size=14336,
            router_aux_loss_coef=1e-3,
            router_z_loss_coef=1e-4,
            tie_word_embeddings=False,
            bos_token_id=151643,
            eos_token_id=151645,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_debug(cls, **kwargs: Any) -> "VasudhaConfig":
        """
        Return a tiny config for fast unit testing on CPU.

        Uses minimal dimensions that fit in CPU memory for CI/testing.
        """
        defaults = dict(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=512,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            qk_norm=True,
            attention_type="hybrid",
            hybrid_pattern=["gla", "gla", "gla", "sdpa"],
            use_moe=True,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=512,
            router_aux_loss_coef=1e-3,
            router_z_loss_coef=1e-4,
            tie_word_embeddings=False,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    def __repr__(self) -> str:
        params = self.estimate_parameters()
        total_m = params["total"] / 1e6
        return (
            f"VasudhaConfig("
            f"hidden={self.hidden_size}, "
            f"layers={self.num_hidden_layers}, "
            f"heads={self.num_attention_heads}/{self.num_key_value_heads}, "
            f"attn={self.attention_type}, "
            f"moe={'on' if self.use_moe else 'off'} "
            f"[{len(self.moe_layers)}/{self.num_hidden_layers} layers, "
            f"E={self.num_experts}/k={self.num_experts_per_tok}], "
            f"~{total_m:.0f}M params"
            f")"
        )


# ── Register config with HuggingFace auto-mapping ─────────────────────────────
# This is done lazily in models/__init__.py to avoid import cycles
