import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

from .registry import AttentionRegistry
from .sdpa import VasudhaGQAAttention, apply_rotary_pos_emb

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


@AttentionRegistry.register("flash_attn")
class VasudhaFlashAttention(VasudhaGQAAttention):
    """
    Flash Attention 2 implementation with graceful fallback to SDPA if unavailable.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if not HAS_FLASH_ATTN:
            logger.warning("flash_attn is not installed. VasudhaFlashAttention will fall back to PyTorch SDPA.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        
        # Fallback to pure SDPA if flash_attn is missing, or during complex masking/inference
        if not HAS_FLASH_ATTN or past_key_values is not None or attention_mask is not None:
            return super().forward(hidden_states, attention_mask, position_ids, past_key_values, output_attentions, use_cache)

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        kv_seq_len = key_states.shape[1]
        
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        # RoPE applies on transposed tensors
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        past_key_value = (key_states.transpose(1, 2), value_states.transpose(1, 2)) if use_cache else None

        # flash_attn_func supports native GQA by providing q with num_heads and k/v with num_key_value_heads
        attn_output = flash_attn_func(query_states, key_states, value_states, dropout_p=0.0, causal=True)
        
        attn_output = attn_output.contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value, None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer_idx={self.layer_idx}, heads={self.num_heads}, kv_heads={self.num_key_value_heads}, flash_attn={HAS_FLASH_ATTN})"


def build_flash_attention(config, layer_idx: int) -> nn.Module:
    """Factory builder specifically for flash attention."""
    if HAS_FLASH_ATTN:
        return VasudhaFlashAttention(config, layer_idx)
    else:
        logger.warning("flash_attn not installed. Falling back to VasudhaGQAAttention.")
        return VasudhaGQAAttention(config, layer_idx)
