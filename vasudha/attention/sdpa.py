import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .registry import AttentionRegistry

logger = logging.getLogger(__name__)

class VasudhaRMSNorm(nn.Module):
    """
    Standard RMSNorm with no bias and no mean subtraction (matches Qwen3 style).
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.variance_epsilon})"


class VasudhaRotaryEmbedding(nn.Module):
    """
    Precomputes cos and sin tables for Rotary Positional Embeddings.
    """
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 1000000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # These tables are deliberately NOT registered buffers. `from_pretrained`
        # builds the model on the meta device and materializes only tensors that
        # appear in the checkpoint; a `persistent=False` buffer is in neither
        # place, so it came back as uninitialized memory — cos/sin ≈ 0, which
        # zeroes every query and key and silently destroys attention. Plain
        # attributes are invisible to that machinery and are rebuilt on demand.
        self.max_seq_len_cached = 0
        self.cos_cached: Optional[torch.Tensor] = None
        self.sin_cached: Optional[torch.Tensor] = None

    @property
    def inv_freq(self) -> torch.Tensor:
        """Derived from `base`/`dim` on every access, so it cannot go stale."""
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        inv_freq = self.inv_freq.to(device)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin for given position_ids.

        Args:
            x: Any tensor on the target device (used for device/dtype).
            position_ids: (batch, seq_len) position indices.

        Returns:
            cos, sin: Each shaped (1, seq_len, dim) for broadcasting.
        """
        # Index by the largest *position*, not the sequence length — with a KV
        # cache the positions run past the current chunk's length.
        needed = int(position_ids.max().item()) + 1
        if (
            self.cos_cached is None
            or needed > self.max_seq_len_cached
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != x.dtype
        ):
            self._set_cos_sin_cache(
                seq_len=max(needed, self.max_seq_len_cached), device=x.device, dtype=x.dtype
            )
        # position_ids shape: (batch, seq_len)
        # cos_cached shape: (max_seq_len, dim)
        # We index by position_ids and return (batch, seq_len, dim)
        cos = self.cos_cached[position_ids].to(dtype=x.dtype)  # (batch, seq_len, dim)
        sin = self.sin_cached[position_ids].to(dtype=x.dtype)  # (batch, seq_len, dim)
        return cos, sin

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim}, base={self.base})"



def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embeddings to query and key tensors.

    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_kv_heads, seq_len, head_dim)
        cos: (batch, seq_len, head_dim) -- already indexed by position
        sin: (batch, seq_len, head_dim) -- already indexed by position
        position_ids: unused (kept for API compatibility)
    """
    # cos/sin are already (batch, seq_len, dim). Unsqueeze for head broadcast.
    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, dim)
    sin = sin.unsqueeze(1)  # (batch, 1, seq_len, dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expands KV heads to match Q heads for Grouped Query Attention.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@AttentionRegistry.register("sdpa")
class VasudhaGQAAttention(nn.Module):
    """
    Pure PyTorch SDPA implementation for full GQA attention.
    Supports QK-Norm and GQA head expansion.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 1000000.0)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if getattr(config, "qk_norm", True):
            self.q_norm = VasudhaRMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
            self.k_norm = VasudhaRMSNorm(self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-6))
        else:
            self.q_norm = None
            self.k_norm = None

        self.rotary_emb = VasudhaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_values is not None:
            kv_seq_len += past_key_values[0].shape[-2]

        if position_ids is None:
            position_ids = torch.arange(kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True if attention_mask is None and q_len > 1 else False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value, None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer_idx={self.layer_idx}, heads={self.num_heads}, kv_heads={self.num_key_value_heads})"
