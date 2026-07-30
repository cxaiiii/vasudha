import torch
import torch.nn as nn
from typing import Tuple

class LatentKVCache(nn.Module):
    """Learned KV compression that projects K,V into a lower-dimensional space.
    
    Inspired by DeepSeek-V2's Multi-head Latent Attention (MLA).
    Projects (B, L, H, D) KV tensors down to (B, L, H, D_latent)
    and decompresses at attention time.
    """
    
    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        
        self.down_proj_k = nn.Linear(head_dim, latent_dim, bias=False)
        self.up_proj_k = nn.Linear(latent_dim, head_dim, bias=False)
        
        self.down_proj_v = nn.Linear(head_dim, latent_dim, bias=False)
        self.up_proj_v = nn.Linear(latent_dim, head_dim, bias=False)
        
    def compress(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress K,V to latent space for caching."""
        k_latent = self.down_proj_k(k)
        v_latent = self.down_proj_v(v)
        return k_latent, v_latent
        
    def decompress(self, k_latent: torch.Tensor, v_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompress K,V from latent space for attention."""
        k_reconstructed = self.up_proj_k(k_latent)
        v_reconstructed = self.up_proj_v(v_latent)
        return k_reconstructed, v_reconstructed
        
    def __repr__(self) -> str:
        return f"LatentKVCache(head_dim={self.head_dim}, latent_dim={self.latent_dim})"
