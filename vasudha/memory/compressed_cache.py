import torch
from typing import Tuple

class CompressedKVCache:
    """4-bit quantized KV cache using absmax scaling per-head."""
    
    def __init__(
        self,
        max_seq_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        bits: int = 4,
        device: str = "cuda",
    ):
        self.bits = bits
        self.device = device
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        
        dtype = torch.uint8
        # shape: (num_layers, max_seq_len, num_kv_heads, head_dim // (8//bits))
        packed_dim = head_dim // (8 // bits)
        self.k_cache = torch.zeros((num_layers, max_seq_len, num_kv_heads, packed_dim), dtype=dtype, device=device)
        self.v_cache = torch.zeros((num_layers, max_seq_len, num_kv_heads, packed_dim), dtype=dtype, device=device)
        self.k_scales = torch.zeros((num_layers, max_seq_len, num_kv_heads, 1), dtype=torch.float16, device=device)
        self.v_scales = torch.zeros((num_layers, max_seq_len, num_kv_heads, 1), dtype=torch.float16, device=device)
        
    def quantize_kv(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, token_idx: int) -> None:
        """Quantize and store K,V for a token."""
        k_q, k_s = self.quantize_tensor(k, self.bits)
        v_q, v_s = self.quantize_tensor(v, self.bits)
        
        self.k_cache[layer_idx, token_idx] = k_q
        self.v_cache[layer_idx, token_idx] = v_q
        self.k_scales[layer_idx, token_idx] = k_s
        self.v_scales[layer_idx, token_idx] = v_s
        
    def dequantize_kv(self, layer_idx: int, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize K,V for all stored tokens."""
        k_q = self.k_cache[layer_idx, :num_tokens]
        k_s = self.k_scales[layer_idx, :num_tokens]
        v_q = self.v_cache[layer_idx, :num_tokens]
        v_s = self.v_scales[layer_idx, :num_tokens]
        
        k = self.dequantize_tensor(k_q, k_s, self.bits)
        v = self.dequantize_tensor(v_q, v_s, self.bits)
        return k, v
        
    @staticmethod
    def quantize_tensor(x: torch.Tensor, bits: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-row absmax quantization."""
        amax = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        scale = amax / ((2 ** (bits - 1)) - 1)
        x_q = torch.round(x / scale).to(torch.int8)
        # Assuming simple mapping for uint8 storage (simplification for prototype)
        x_q = (x_q + (2 ** (bits - 1))).to(torch.uint8)
        return x_q, scale
        
    @staticmethod  
    def dequantize_tensor(x_q: torch.Tensor, scale: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """Dequantize a per-row quantized tensor."""
        x_q_int = x_q.to(torch.int32) - (2 ** (bits - 1))
        return x_q_int * scale
        
    def __repr__(self) -> str:
        return f"CompressedKVCache(bits={self.bits}, device={self.device})"
