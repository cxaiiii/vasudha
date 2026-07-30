"""
Vasudha — Research-Grade Efficient Reasoning Language Model Framework.

A modular, compute-efficient LLM platform targeting maximum reasoning per FLOP,
optimized for Google Colab Free Tier (Tesla T4, 15GB VRAM).

Architecture:
  - Hybrid Attention (GLA + Full GQA, 3:1 ratio)
  - Sparse Mixture of Experts (every other layer)
  - Triton-optimized kernels
  - Qwen3 compatible weight loading
  - HuggingFace native interface
"""

from vasudha.version import __version__

# Public API — populated as modules are implemented
from vasudha.models.config import VasudhaConfig
from vasudha.models.vasudha_model import VasudhaModel, VasudhaForCausalLM

__all__ = [
    "__version__",
    "VasudhaConfig",
    "VasudhaModel",
    "VasudhaForCausalLM",
]
