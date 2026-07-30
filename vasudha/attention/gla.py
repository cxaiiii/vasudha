import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

from .registry import AttentionRegistry
from .sdpa import VasudhaRMSNorm, VasudhaRotaryEmbedding, apply_rotary_pos_emb

logger = logging.getLogger(__name__)


@AttentionRegistry.register("gla")
class VasudhaGLAAttention(nn.Module):
    """
    Gated Linear Attention (GLA) - Pure PyTorch Reference Implementation.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = getattr(config, "rope_theta", 1000000.0)
        self.chunk_size = getattr(config, "gla_chunk_size", 64)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
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
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        g = self.g_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        g = torch.sigmoid(g.transpose(1, 2))

        kv_seq_len = q_len
        if past_key_values is not None:
            kv_seq_len += 1 # in GLA, past_key_values maintains the recurrent state, conceptually 1 timestep step

        if position_ids is None:
            position_ids = torch.arange(kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if q_len == 1:
            # Inference step: Purely recurrent
            if past_key_values is not None:
                S = past_key_values[0]
            else:
                S = torch.zeros(bsz, self.num_key_value_heads, self.head_dim, self.head_dim, device=q.device, dtype=q.dtype)

            g_vec = g[:, :, 0, :]
            k_vec = k[:, :, 0, :]
            v_vec = v[:, :, 0, :]
            q_vec = q[:, :, 0, :]

            G = torch.einsum('bhd,bhe->bhde', g_vec, g_vec)
            kv_outer = torch.einsum('bhd,bhe->bhde', k_vec, v_vec)
            
            S_new = G * S + kv_outer
            
            num_rep = self.num_heads // self.num_key_value_heads
            S_rep = S_new.unsqueeze(2).expand(bsz, self.num_key_value_heads, num_rep, self.head_dim, self.head_dim)
            S_rep = S_rep.reshape(bsz, self.num_heads, self.head_dim, self.head_dim)
            
            out = torch.einsum('bhd,bhde->bhe', q_vec, S_rep)
            out = out.unsqueeze(2)
            
            past_key_value = (S_new,) if use_cache else None

        else:
            # Full sequence forward step (simulated chunkwise recurrent form)
            S = torch.zeros(bsz, self.num_key_value_heads, self.head_dim, self.head_dim, device=q.device, dtype=q.dtype)
            outputs = []
            
            for t in range(q_len):
                g_vec = g[:, :, t, :]
                k_vec = k[:, :, t, :]
                v_vec = v[:, :, t, :]
                q_vec = q[:, :, t, :]
                
                G = torch.einsum('bhd,bhe->bhde', g_vec, g_vec)
                kv_outer = torch.einsum('bhd,bhe->bhde', k_vec, v_vec)
                
                S = G * S + kv_outer

                # Group the query heads instead of broadcasting the state across
                # them. The old form materialized S at (bsz, num_heads, d, d) on
                # every timestep — num_rep times larger than S itself, saved for
                # backward at each of q_len steps. Mathematically identical:
                # head h = kv*num_rep + r reads state slice kv either way.
                num_rep = self.num_heads // self.num_key_value_heads
                q_grouped = q_vec.view(bsz, self.num_key_value_heads, num_rep, self.head_dim)
                out_t = torch.einsum('bkrd,bkde->bkre', q_grouped, S)
                out_t = out_t.reshape(bsz, self.num_heads, self.head_dim)
                outputs.append(out_t.unsqueeze(2))
            
            out = torch.cat(outputs, dim=2)
            past_key_value = (S,) if use_cache else None

        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        return out, past_key_value, None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(layer_idx={self.layer_idx}, heads={self.num_heads}, kv_heads={self.num_key_value_heads}, chunk_size={self.chunk_size})"
