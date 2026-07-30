import logging
from typing import Callable, Dict, Type, Any

logger = logging.getLogger(__name__)

class AttentionRegistry:
    """Registry for attention mechanisms."""
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register an attention class."""
        def wrapper(attention_cls: Type) -> Type:
            if name in cls._registry:
                logger.warning(f"Attention module {name} already registered. Overwriting.")
            cls._registry[name] = attention_cls
            return attention_cls
        return wrapper

    @classmethod
    def get(cls, name: str) -> Type:
        """Get an attention class from the registry by name."""
        if name not in cls._registry:
            raise ValueError(f"Attention type '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name]


def build_attention(config: Any, layer_idx: int):
    """
    Factory function that instantiates the correct attention class for a given layer.
    """
    from .hybrid import VasudhaHybridManager
    manager = VasudhaHybridManager(config)
    attention_cls = manager.get_attention_module(layer_idx)
    return attention_cls(config, layer_idx)
