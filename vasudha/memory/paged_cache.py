import torch
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class PagedCacheConfig:
    block_size: int = 16
    num_blocks: int = 512
    num_heads: int = 8
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

class PagedKVCache:
    """Paged KV cache with free list management."""
    
    def __init__(self, config: PagedCacheConfig):
        self.config = config
        self.k_cache = torch.zeros(
            (config.num_blocks, config.block_size, config.num_heads, config.head_dim),
            dtype=config.dtype,
            device=config.device
        )
        self.v_cache = torch.zeros(
            (config.num_blocks, config.block_size, config.num_heads, config.head_dim),
            dtype=config.dtype,
            device=config.device
        )
        self._free_blocks = list(range(config.num_blocks))
        self.block_tables = {}  # seq_id -> list of block indices
        
    def allocate_sequence(self, seq_id: int, num_tokens: int) -> List[int]:
        """Allocate blocks for a new sequence. Returns block indices."""
        num_blocks_needed = (num_tokens + self.config.block_size - 1) // self.config.block_size
        if len(self._free_blocks) < num_blocks_needed:
            raise RuntimeError("Out of memory: Not enough free blocks available.")
            
        allocated = [self._free_blocks.pop(0) for _ in range(num_blocks_needed)]
        self.block_tables[seq_id] = allocated
        return allocated
        
    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks for a sequence."""
        if seq_id in self.block_tables:
            self._free_blocks.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]
            
    def write_kv(self, seq_id: int, token_pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write K,V for a single token position."""
        block_idx = token_pos // self.config.block_size
        offset = token_pos % self.config.block_size
        
        physical_block = self.block_tables[seq_id][block_idx]
        self.k_cache[physical_block, offset] = k
        self.v_cache[physical_block, offset] = v
        
    def read_kv(self, seq_id: int, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read K,V for all tokens in a sequence."""
        blocks = self.block_tables[seq_id]
        k_list, v_list = [], []
        
        for i, physical_block in enumerate(blocks):
            if i == len(blocks) - 1 and num_tokens % self.config.block_size != 0:
                valid_len = num_tokens % self.config.block_size
                k_list.append(self.k_cache[physical_block, :valid_len])
                v_list.append(self.v_cache[physical_block, :valid_len])
            else:
                k_list.append(self.k_cache[physical_block])
                v_list.append(self.v_cache[physical_block])
                
        k_tensor = torch.cat(k_list, dim=0)
        v_tensor = torch.cat(v_list, dim=0)
        return k_tensor, v_tensor
        
    @property
    def free_blocks(self) -> int:
        """Number of available blocks."""
        return len(self._free_blocks)
        
    @property
    def utilization(self) -> float:
        """Fraction of total blocks currently in use."""
        return 1.0 - (len(self._free_blocks) / self.config.num_blocks)
        
    def __repr__(self) -> str:
        return f"PagedKVCache(utilization={self.utilization*100:.1f}%, free_blocks={self.free_blocks}/{self.config.num_blocks})"
