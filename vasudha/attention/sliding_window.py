import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

from .registry import AttentionRegistry
from .sdpa import VasudhaGQAAttention

logger = logging.getLogger(__name__)

@AttentionRegistry.register("sliding_window")
class VasudhaSlidingWindowAttention(VasudhaGQAAttention):
    """
    Sliding window attention on top of pure PyTorch GQA Attention.
    Tokens can only attend to previous tokens within the configured window.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.window_size = getattr(config, "sliding_window_size", 4096)

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

        # If sequence is smaller than window and we are not doing inference with cache
        if q_len <= self.window_size and past_key_values is None:
            return super().forward(hidden_states, attention_mask, position_ids, past_key_values, output_attentions, use_cache)

        kv_seq_len = q_len
        if past_key_values is not None:
            kv_seq_len += past_key_values[0].shape[-2]
            
        mask = torch.ones(q_len, kv_seq_len, dtype=torch.bool, device=hidden_states.device)
        mask = torch.tril(mask, diagonal=kv_seq_len - q_len)
        mask = torch.triu(mask, diagonal=kv_seq_len - q_len - self.window_size + 1)
        
        if attention_mask is not None:
            attention_mask = attention_mask.bool() & mask.unsqueeze(0).unsqueeze(0)
        else:
            attention_mask = mask.unsqueeze(0).unsqueeze(0)
            
        return super().forward(hidden_states, attention_mask, position_ids, past_key_values, output_attentions, use_cache)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer_idx={self.layer_idx}, heads={self.num_heads}, kv_heads={self.num_key_value_heads}, window_size={self.window_size})"
