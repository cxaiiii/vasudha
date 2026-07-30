import logging
from typing import Type, Any
from .registry import AttentionRegistry

logger = logging.getLogger(__name__)


class VasudhaHybridManager:
    """
    Manager class responsible for building and selecting the correct attention
    mechanism based on the hybrid configuration.
    """
    def __init__(self, config: Any):
        self.config = config
        # Pattern: every 4th layer is full GQA, others are GLA
        self.attention_types = ["gla", "gla", "gla", "sdpa"]
        
    def get_attention_type_for_layer(self, layer_idx: int) -> str:
        """Determines the attention type for a specific layer."""
        if hasattr(self.config, "get_attention_type_for_layer"):
            return self.config.get_attention_type_for_layer(layer_idx)
        return self.attention_types[layer_idx % len(self.attention_types)]

    def get_attention_module(self, layer_idx: int) -> Type:
        """Returns the requested Attention Module class with fallback support."""
        attn_type = self.get_attention_type_for_layer(layer_idx)
        
        # Determine fallback if missing plugins
        if attn_type == "flash":
            try:
                import flash_attn
                attn_type = "flash_attn"
            except ImportError:
                logger.warning("flash_attn unavailable. Falling back to sdpa for layer %s", layer_idx)
                attn_type = "sdpa"
                
        if attn_type == "gla_triton":
            try:
                import triton
                # Assume if triton is present gla_triton could be used, but since we only have pure pytorch right now
                attn_type = "gla"
            except ImportError:
                logger.warning("triton unavailable. Falling back to pure PyTorch gla for layer %s", layer_idx)
                attn_type = "gla"

        try:
            return AttentionRegistry.get(attn_type)
        except ValueError:
            logger.warning(f"Attention type '{attn_type}' not found in registry, falling back to 'sdpa'.")
            return AttentionRegistry.get("sdpa")

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        """Checks if the layer employs a linear attention (e.g. GLA)."""
        return self.get_attention_type_for_layer(layer_idx) in ("gla", "gla_triton")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pattern={self.attention_types})"
