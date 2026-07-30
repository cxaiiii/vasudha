"""
Vasudha model architecture.

The core transformer model, combining:
  - Hybrid attention (GLA + GQA, 3:1 ratio by default)
  - Sparse MoE FFN (every other layer)
  - RMSNorm with optional QK-Norm
  - RoPE positional embeddings

Exports:
  - VasudhaModel: Base transformer without LM head
  - VasudhaForCausalLM: Full causal LM (for training and inference)

Architecture follows the Qwen3 layout:
  [Embedding] → [N × DecoderLayer] → [RMSNorm] → [LM Head]

Each DecoderLayer:
  [RMSNorm] → [Attention] → [Residual]
  [RMSNorm] → [FFN: Dense or MoE] → [Residual]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin

from vasudha.models.base import VasudhaPreTrainedModel
from vasudha.models.config import VasudhaConfig
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)


# ── Import attention modules (with fallback chain) ─────────────────────────────
try:
    from vasudha.attention.registry import build_attention
    from vasudha.attention.sdpa import VasudhaRMSNorm, VasudhaRotaryEmbedding
    _ATTENTION_AVAILABLE = True
except ImportError:
    logger.warning("Attention modules not yet built — using inline fallback implementations.")
    _ATTENTION_AVAILABLE = False


# ── Import MoE modules (with fallback to dense) ────────────────────────────────
try:
    from vasudha.moe.moe_layer import VasudhaDecoderLayerFFN
    _MOE_AVAILABLE = True
except ImportError:
    logger.warning("MoE modules not yet built — all layers will use dense FFN.")
    _MOE_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# Inline fallback implementations (used when attention/moe modules not built)
# These ensure the model can be instantiated even before all modules exist.
# ══════════════════════════════════════════════════════════════════════════════

class _FallbackRMSNorm(nn.Module):
    """Fallback RMSNorm used when kernels/attention module is not available."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

    def __repr__(self) -> str:
        return f"RMSNorm(hidden_size={self.hidden_size}, eps={self.eps})"


class _FallbackRotaryEmbedding(nn.Module):
    """Fallback RoPE used when attention module is not available."""

    def __init__(self, dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        # Not a registered buffer — see VasudhaRotaryEmbedding for why a
        # persistent=False buffer comes back as uninitialized memory after
        # from_pretrained. Derived on access instead.
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    @property
    def inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(x.device)
        # position_ids: (batch, seq_len)
        # Compute freqs for each position
        freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)  # (batch, seq_len, dim)
        sin = emb.sin().to(x.dtype)  # (batch, seq_len, dim)
        return cos, sin



def _get_norm_cls(config: VasudhaConfig) -> type:
    """Return the appropriate RMSNorm class."""
    if _ATTENTION_AVAILABLE:
        return VasudhaRMSNorm
    return _FallbackRMSNorm


def _get_rope_cls(config: VasudhaConfig) -> type:
    """Return the appropriate RoPE class."""
    if _ATTENTION_AVAILABLE:
        return VasudhaRotaryEmbedding
    return _FallbackRotaryEmbedding


# ══════════════════════════════════════════════════════════════════════════════
# Dense FFN (SwiGLU) — used in non-MoE layers
# ══════════════════════════════════════════════════════════════════════════════

class VasudhaDenseSwiGLU(nn.Module):
    """
    Dense SwiGLU feed-forward network.

    Used in even-indexed layers where MoE is not active.
    SwiGLU = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Dimensions:
      hidden_size → intermediate_size (gate + up), intermediate_size → hidden_size (down)
    """

    def __init__(self, config: VasudhaConfig) -> None:
        super().__init__()
        self.config = config
        H = config.hidden_size
        I = config.intermediate_size

        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)

        # Mark down_proj as residual projection for scaled init
        self.down_proj._is_residual_proj = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Try fused SwiGLU kernel first
        try:
            from vasudha.kernels.swiglu import fused_swiglu
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            hidden = fused_swiglu(gate, up)
        except ImportError:
            # PyTorch fallback
            gate = F.silu(self.gate_proj(x))
            hidden = gate * self.up_proj(x)

        return self.down_proj(hidden)

    def __repr__(self) -> str:
        return (
            f"VasudhaDenseSwiGLU("
            f"hidden={self.config.hidden_size}, "
            f"intermediate={self.config.intermediate_size})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Decoder Layer
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DecoderLayerOutput:
    """Output container for a single decoder layer."""
    hidden_states: torch.Tensor
    """Transformed hidden states. Shape: (batch, seq_len, hidden_size)."""

    present_key_value: Optional[Tuple[torch.Tensor, ...]] = None
    """KV cache state for this layer (or recurrent state for GLA layers)."""

    aux_loss: Optional[torch.Tensor] = None
    """MoE auxiliary loss (None for dense layers)."""

    self_attn_weights: Optional[torch.Tensor] = None
    """Optional attention weights (only for full GQA layers when requested)."""


class VasudhaDecoderLayer(nn.Module):
    """
    A single Vasudha transformer decoder layer.

    Structure:
      input
        → input_layernorm (RMSNorm)
        → Self-attention (GLA or Full GQA depending on layer_idx)
        → Residual add
        → post_attention_layernorm (RMSNorm)
        → FFN (Dense SwiGLU or Sparse MoE depending on layer_idx)
        → Residual add
      → output

    Args:
        config: VasudhaConfig
        layer_idx: Zero-based index of this layer. Determines:
                   - Attention type (hybrid pattern lookup)
                   - FFN type (MoE at odd indices, Dense at even)
    """

    def __init__(self, config: VasudhaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # ── Normalization layers ───────────────────────────────────────────────
        NormCls = _get_norm_cls(config)
        self.input_layernorm = NormCls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NormCls(config.hidden_size, eps=config.rms_norm_eps)

        # ── Attention module (selected by hybrid pattern) ──────────────────────
        self._attn_type = config.get_attention_type_for_layer(layer_idx)
        if _ATTENTION_AVAILABLE:
            self.self_attn = build_attention(config, layer_idx)
        else:
            # Fallback: import-free stub that will be replaced when attention is built
            logger.warning(
                f"Layer {layer_idx}: Attention modules not available. "
                f"Using placeholder — build vasudha.attention first."
            )
            self.self_attn = _AttentionPlaceholder(config, layer_idx)

        # ── FFN (MoE at odd layers, Dense at even layers) ──────────────────────
        self._is_moe = config.is_moe_layer(layer_idx)
        if self._is_moe and _MOE_AVAILABLE:
            self.mlp = VasudhaDecoderLayerFFN(config, layer_idx)
        else:
            # Dense SwiGLU
            self.mlp = VasudhaDenseSwiGLU(config)
            if self._is_moe and not _MOE_AVAILABLE:
                logger.warning(
                    f"Layer {layer_idx}: MoE requested but module not available. "
                    f"Using dense FFN."
                )

        # ── Gradient checkpointing ────────────────────────────────────────────
        self.gradient_checkpointing: bool = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> DecoderLayerOutput:
        """
        Forward pass for a single decoder layer.

        Args:
            hidden_states: Input tensor. Shape: (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask (additive, 0/-inf values).
            position_ids: Token positions for RoPE. Shape: (batch, seq_len).
            past_key_value: KV cache from previous step (or GLA recurrent state).
            output_attentions: Whether to return attention weights.
            use_cache: Whether to compute and return updated KV cache.
            cache_position: Position indices for cache (used in static cache).

        Returns:
            DecoderLayerOutput with updated hidden_states and cache.
        """
        residual = hidden_states

        # ── Self-attention block ────────────────────────────────────────────────
        hidden_states = self.input_layernorm(hidden_states)

        # Use gradient checkpointing for attention if enabled
        if self.gradient_checkpointing and self.training:
            attn_output = torch.utils.checkpoint.checkpoint(
                self.self_attn,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                use_reentrant=False,
            )
        else:
            attn_output = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        # Unpack attention output
        if isinstance(attn_output, (tuple, list)):
            hidden_states, present_key_value = attn_output[0], attn_output[1] if len(attn_output) > 1 else None
            attn_weights = attn_output[2] if output_attentions and len(attn_output) > 2 else None
        else:
            hidden_states = attn_output
            present_key_value = None
            attn_weights = None

        hidden_states = residual + hidden_states

        # ── FFN block ──────────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MoE FFN returns (output, aux_loss); Dense FFN returns just output
        aux_loss: Optional[torch.Tensor] = None
        if self.gradient_checkpointing and self.training:
            ffn_output = torch.utils.checkpoint.checkpoint(
                self.mlp,
                hidden_states,
                use_reentrant=False,
            )
        else:
            ffn_output = self.mlp(hidden_states)

        if isinstance(ffn_output, (tuple, list)):
            hidden_states, aux_loss = ffn_output[0], ffn_output[1]
        else:
            hidden_states = ffn_output

        hidden_states = residual + hidden_states

        return DecoderLayerOutput(
            hidden_states=hidden_states,
            present_key_value=present_key_value,
            aux_loss=aux_loss,
            self_attn_weights=attn_weights,
        )

    def __repr__(self) -> str:
        return (
            f"VasudhaDecoderLayer("
            f"idx={self.layer_idx}, "
            f"attn={self._attn_type}, "
            f"ffn={'moe' if self._is_moe else 'dense'})"
        )


class _AttentionPlaceholder(nn.Module):
    """Placeholder attention module for when attention package is not yet built."""

    def __init__(self, config: VasudhaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        H = config.hidden_size
        # Minimal Q, K, V, O projections as placeholder
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.v_proj = nn.Linear(H, H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, None]:
        B, L, H = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # Simple scaled dot-product attention (no masking — placeholder only)
        scale = math.sqrt(H)
        attn = torch.softmax(torch.bmm(q, k.transpose(-1, -2)) / scale, dim=-1)
        out = self.o_proj(torch.bmm(attn, v))
        return out, None


# ══════════════════════════════════════════════════════════════════════════════
# VasudhaModel — Base transformer (no LM head)
# ══════════════════════════════════════════════════════════════════════════════

class VasudhaModel(VasudhaPreTrainedModel):
    """
    Vasudha transformer model without language modeling head.

    This is the base model that computes hidden state representations.
    Use VasudhaForCausalLM for language modeling.

    Architecture:
      embed_tokens → [N × VasudhaDecoderLayer] → norm → output

    The model returns the final hidden states and optionally:
      - All hidden states (for representation analysis)
      - All attention weights (for interpretability)
      - KV cache (for autoregressive generation)
      - MoE auxiliary losses (summed over all MoE layers, for training)
    """

    def __init__(self, config: VasudhaConfig) -> None:
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # ── Token embeddings ───────────────────────────────────────────────────
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        # ── Rotary position embeddings (shared across all attention layers) ────
        RopeCls = _get_rope_cls(config)
        if _ATTENTION_AVAILABLE:
            self.rotary_emb = RopeCls(
                dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rotary_emb = RopeCls(
                dim=config.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )

        # ── Decoder layers ─────────────────────────────────────────────────────
        self.layers = nn.ModuleList(
            [VasudhaDecoderLayer(config, layer_idx=i)
             for i in range(config.num_hidden_layers)]
        )

        # ── Final layer norm ───────────────────────────────────────────────────
        NormCls = _get_norm_cls(config)
        self.norm = NormCls(config.hidden_size, eps=config.rms_norm_eps)

        # ── Optional reasoning controller ─────────────────────────────────────
        self.reasoning_controller = None
        if config.use_reasoning_controller:
            try:
                from vasudha.reasoning.controller import ReasoningController
                self.reasoning_controller = ReasoningController(config)
            except ImportError:
                logger.warning("Reasoning controller module not available.")

        # Initialize weights
        self.post_init()
        VasudhaPreTrainedModel._mark_residual_projections(self)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[Optional[Tuple[torch.Tensor, ...]]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPast, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the Vasudha base model.

        Args:
            input_ids: Token IDs. Shape: (batch, seq_len). Mutually exclusive
                       with inputs_embeds.
            attention_mask: Attention mask. 1 = attend, 0 = mask.
                            Shape: (batch, seq_len).
            position_ids: Position indices for RoPE. If None, computed as
                          arange(seq_len) offset by cache length.
            past_key_values: Per-layer KV cache from previous generation steps.
                             Length = num_hidden_layers.
            inputs_embeds: Pre-computed token embeddings (alternative to input_ids).
            use_cache: Whether to return updated KV cache.
            output_attentions: Whether to return attention weights per layer.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a ModelOutput dict or a plain tuple.

        Returns:
            BaseModelOutputWithPast (or tuple) containing:
              - last_hidden_state
              - past_key_values (if use_cache=True)
              - hidden_states (if output_hidden_states=True)
              - attentions (if output_attentions=True)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions or False
        output_hidden_states = output_hidden_states or False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ── Input validation ───────────────────────────────────────────────────
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        # ── Embedding lookup ───────────────────────────────────────────────────
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape

        # ── Convert DynamicCache to list-of-tuples ─────────────────────────────
        # Newer transformers passes a DynamicCache object during generate().
        # Convert it once here so all downstream code can use list indexing.
        cache_len = 0
        if past_key_values is not None:
            try:
                from transformers.cache_utils import DynamicCache
                if isinstance(past_key_values, DynamicCache):
                    cache_len = past_key_values.get_seq_length()
                    num_cached_layers = len(past_key_values.layers)
                    past_key_values_list: list = []
                    for layer_idx in range(len(self.layers)):
                        if layer_idx < num_cached_layers and past_key_values.layers[layer_idx].is_initialized:
                            past_key_values_list.append(
                                (past_key_values.layers[layer_idx].keys,
                                 past_key_values.layers[layer_idx].values)
                            )
                        else:
                            past_key_values_list.append(None)
                    past_key_values = past_key_values_list
                else:
                    # Legacy list-of-tuples format.
                    cache_len = self._real_kv_cache_len(past_key_values)
            except ImportError:
                pass

        # ── Position IDs ───────────────────────────────────────────────────────
        if position_ids is None:
            position_ids = torch.arange(
                cache_len,
                cache_len + seq_len,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)

        # ── Rotary embeddings ──────────────────────────────────────────────────
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)

        # ── Causal attention mask ──────────────────────────────────────────────
        causal_mask = self._prepare_attention_mask(
            attention_mask, inputs_embeds, past_key_values, seq_len, batch_size
        )

        # ── Initialize cache ───────────────────────────────────────────────────
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # ── Forward through decoder layers ─────────────────────────────────────
        hidden_states = inputs_embeds
        all_hidden_states: Optional[tuple] = () if output_hidden_states else None
        all_self_attentions: Optional[tuple] = () if output_attentions else None
        next_decoder_cache: list = []
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        has_aux_loss = False

        for layer_idx, (decoder_layer, past_kv) in enumerate(
            zip(self.layers, past_key_values)
        ):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

            layer_output: DecoderLayerOutput = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_output.hidden_states

            if use_cache:
                next_decoder_cache.append(layer_output.present_key_value)

            if layer_output.aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_output.aux_loss
                has_aux_loss = True

            if output_attentions and layer_output.self_attn_weights is not None:
                all_self_attentions = all_self_attentions + (layer_output.self_attn_weights,)  # type: ignore[operator]

        # ── Final norm ─────────────────────────────────────────────────────────
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

        # ── Apply reasoning controller (optional) ──────────────────────────────
        if self.reasoning_controller is not None:
            hidden_states = self.reasoning_controller(hidden_states)

        # Store aux loss as an attribute so the LM head can access it
        self._last_aux_loss = total_aux_loss if has_aux_loss else None

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _real_kv_cache_len(self, past_key_values: list) -> int:
        """
        Cached sequence length of the first layer that actually keeps one.

        Cannot be inferred from tensor shape alone: GLA's recurrent state is
        `(batch, kv_heads, head_dim, head_dim)` — also rank 4, so a naive
        `dim() == 4` check mistakes it for a real (key, value) cache and reads
        `head_dim` off dim 2 as if it were a sequence length. head_dim is a
        small constant (commonly < the actual cached length), so this doesn't
        merely give a wrong number — it silently truncates the attention mask
        below the SDPA layer's real key length, which surfaces during
        multi-step generation as a shape mismatch in scaled_dot_product_attention.
        The per-layer attention type from config, not tensor shape, is the
        only reliable discriminator.
        """
        for i, past in enumerate(past_key_values):
            if past is None:
                continue
            if self.config.get_attention_type_for_layer(i) == "gla":
                continue
            if isinstance(past, (tuple, list)) and past[0] is not None and hasattr(past[0], "shape"):
                return int(past[0].shape[2])
        return 0

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[list],
        seq_len: int,
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """
        Build a 4D causal attention mask compatible with scaled_dot_product_attention.

        For most cases, we let SDPA handle causality with is_causal=True,
        which is more memory-efficient than materializing the full mask.

        When attention_mask is provided (e.g., for padding), we convert it to
        a 4D additive mask (0 = attend, -inf = ignore).

        Returns:
            4D mask of shape (batch, 1, seq_len, total_len) or None.
        """
        if attention_mask is None:
            return None

        # Convert 2D padding mask to 4D additive mask
        # Input: (batch, seq_len) with 1=attend, 0=pad
        # Output: (batch, 1, seq_len, seq_len) with 0/-inf values

        total_len = seq_len
        if past_key_values is not None:
            total_len += self._real_kv_cache_len(past_key_values)

        # Build causal mask
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        min_val = torch.finfo(dtype).min

        # (batch, 1, seq_len, total_len)
        mask_4d = torch.zeros(batch_size, 1, seq_len, total_len, device=device, dtype=dtype)

        # Causal component. Without this the mask is padding-only and every token
        # attends to its own future, including the token it is predicting — which
        # makes pretrained (causally-trained) weights produce garbage.
        # Query i sits at absolute position (total_len - seq_len + i), so it may
        # attend to keys 0 .. (total_len - seq_len + i): tril with that offset.
        causal = torch.ones(seq_len, total_len, device=device, dtype=torch.bool).tril(
            diagonal=total_len - seq_len
        )

        # Pad the key-side mask to cover cached tokens (always attendable).
        if attention_mask.shape[-1] < total_len:
            pad_len = total_len - attention_mask.shape[-1]
            attention_mask = F.pad(attention_mask, (pad_len, 0), value=1)

        # Combine causal and padding as booleans, then fill once. Adding two
        # min_val terms would overflow to -inf and can NaN out softmax rows.
        keep = causal.unsqueeze(0).unsqueeze(0) & attention_mask[:, None, None, :].bool()
        mask_4d = mask_4d.masked_fill(~keep, min_val)

        return mask_4d


# ══════════════════════════════════════════════════════════════════════════════
# VasudhaForCausalLM — Full causal language model
# ══════════════════════════════════════════════════════════════════════════════

class VasudhaForCausalLM(VasudhaPreTrainedModel, GenerationMixin):
    """
    Vasudha causal language model for text generation and training.

    Adds an LM head (unembedding layer) on top of VasudhaModel.
    Handles MoE auxiliary loss aggregation and cross-entropy loss computation.

    This is the primary class for:
      - Supervised fine-tuning (SFT)
      - QLoRA training
      - Autoregressive inference / generation
      - Evaluation on text benchmarks

    Weight tying:
      If config.tie_word_embeddings=True, the LM head weight is tied to the
      embedding table weight (saves memory, standard in many LLMs).
    """

    # transformers >= 5 expects a {target: source} mapping here, not a list. The
    # list form only survives when tie_word_embeddings=False, because the tying
    # code returns early — Qwen3-4B ties its embeddings, so it would crash.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: VasudhaConfig) -> None:
        super().__init__(config)
        self.model = VasudhaModel(config)
        self.vocab_size = config.vocab_size

        # LM head: projects hidden_size → vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def get_decoder(self) -> VasudhaModel:
        return self.model

    def set_decoder(self, decoder: VasudhaModel) -> None:
        self.model = decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[Optional[Tuple[torch.Tensor, ...]]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[CausalLMOutputWithPast, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the causal LM.

        When labels are provided, computes cross-entropy loss with optional
        MoE auxiliary loss. The total loss is:
            total_loss = ce_loss + router_aux_loss + router_z_loss

        Args:
            input_ids: Input token IDs. Shape: (batch, seq_len).
            attention_mask: Padding mask. Shape: (batch, seq_len).
            position_ids: Token positions for RoPE.
            past_key_values: KV cache from previous steps.
            inputs_embeds: Alternative to input_ids.
            labels: Target token IDs for computing loss. -100 = ignore.
                    Shape: (batch, seq_len). Standard causal LM: shifted input_ids.
            use_cache: Whether to return KV cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return ModelOutput or plain tuple.

        Returns:
            CausalLMOutputWithPast containing loss (if labels provided),
            logits, and optional cache/hidden states/attentions.
        """
        output_attentions = output_attentions or False
        output_hidden_states = output_hidden_states or False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ── Base model forward ─────────────────────────────────────────────────
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # ── Language modeling head ─────────────────────────────────────────────
        # Compute logits in fp32 for numerical stability
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        logits = logits.float()  # Always compute loss in fp32

        # ── Loss computation ───────────────────────────────────────────────────
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Chunked cross-entropy to save VRAM (avoids materializing full softmax)
            try:
                from vasudha.kernels.cross_entropy import vasudha_cross_entropy
                loss = vasudha_cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            except ImportError:
                # Standard cross-entropy fallback
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            # ── Add MoE auxiliary losses ───────────────────────────────────────
            aux_loss = getattr(self.model, "_last_aux_loss", None)
            if aux_loss is not None:
                loss = loss + aux_loss

        if not return_dict:
            output = (logits,) + tuple(
                v for v in [
                    outputs.past_key_values,
                    outputs.hidden_states,
                    outputs.attentions,
                ] if v is not None
            )
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """
        Self-contained greedy/sampling loop, bypassing GenerationMixin.generate().

        Vasudha's per-layer cache is heterogeneous by construction: SDPA layers
        cache (key, value) tensors shaped (batch, heads, seq, head_dim), GLA
        layers cache a single recurrent-state tensor with no such seq axis.
        transformers' Cache classes (DynamicCache and friends) assume every
        layer's cache has the same shape and can be stacked/indexed uniformly,
        so GenerationMixin.generate() — which in current transformers requires
        past_key_values to be a Cache instance — cannot represent this model's
        cache at all and fails inside its own bookkeeping (surfacing as an
        opaque 'NoneType' object has no attribute 'dim'). This loop drives the
        model with the plain list-of-tuples cache format forward() already
        produces and consumes, which scripts/diagnose_conversion.py verified
        byte-for-byte against stock Qwen3.

        Supports exactly what the evaluation benchmarks use: a single prompt
        per call, greedy or temperature/top-p sampling, EOS-stopping. Not a
        general replacement for GenerationMixin (no beam search, no
        num_return_sequences).
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generated = input_ids
        step_input_ids = input_ids
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            outputs = self(
                input_ids=step_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if do_sample:
                probs = torch.softmax(next_token_logits / max(temperature, 1e-5), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = cum - sorted_probs > top_p
                    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
                    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token = next_token_logits.argmax(dim=-1)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished, torch.full_like(next_token, pad_token_id), next_token
                )

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
            )
            step_input_ids = next_token.unsqueeze(-1)

            if eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if bool(finished.all()):
                    break

        return generated

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Prepare inputs for autoregressive generation.

        When KV cache is active, only process the newly generated token
        (not the entire sequence) for efficiency.
        """
        # If cache exists, only process the last new token
        has_cache = False
        if past_key_values is not None:
            try:
                from transformers.cache_utils import DynamicCache
                if isinstance(past_key_values, DynamicCache):
                    has_cache = past_key_values.get_seq_length() > 0
                else:
                    has_cache = past_key_values[0] is not None
            except (ImportError, TypeError, IndexError):
                has_cache = False
        if has_cache:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Generate position_ids from attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @classmethod
    def from_qwen3_pretrained(
        cls,
        model_name_or_path: str,
        attention_type: str = "hybrid",
        use_moe: bool = True,
        num_experts: int = 64,
        moe_intermediate_size: Optional[int] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "auto",
        **kwargs: Any,
    ) -> "VasudhaForCausalLM":
        """
        Load Qwen3 weights into a VasudhaForCausalLM model.

        This convenience method:
        1. Determines the Vasudha config from the Qwen3 model name
        2. Loads Qwen3 weights and remaps them to Vasudha naming
        3. Returns an initialized VasudhaForCausalLM

        Args:
            model_name_or_path: HuggingFace model ID (e.g., "Qwen/Qwen3-4B")
                                 or local path.
            attention_type: Attention type to use in Vasudha ("hybrid", "sdpa", etc.)
            use_moe: Whether to add sparse MoE (won't load MoE weights from Qwen3
                     since Qwen3-4B/8B are dense — initializes MoE randomly).
            load_in_4bit: Whether to load in 4-bit (QLoRA mode).
            load_in_8bit: Whether to load in 8-bit.
            torch_dtype: "auto", "float16", "bfloat16", "float32".
            **kwargs: Additional kwargs passed to from_pretrained.

        Returns:
            Initialized VasudhaForCausalLM with Qwen3 weights.
        """
        from vasudha.models.qwen3_compat import load_from_qwen3
        return load_from_qwen3(
            cls,
            model_name_or_path=model_name_or_path,
            attention_type=attention_type,
            use_moe=use_moe,
            num_experts=num_experts,
            moe_intermediate_size=moe_intermediate_size,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    def __repr__(self) -> str:
        params = self.config.estimate_parameters()
        return (
            f"VasudhaForCausalLM(\n"
            f"  config={self.config!r},\n"
            f"  params~{params['total']/1e6:.0f}M\n"
            f")"
        )
