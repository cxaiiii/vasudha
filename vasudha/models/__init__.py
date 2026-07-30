"""
Vasudha models package.

Public exports:
  - VasudhaConfig: Architecture configuration
  - VasudhaModel: Base transformer (no LM head)
  - VasudhaForCausalLM: Full causal LM with weight-tied or separate LM head
"""

from vasudha.models.config import VasudhaConfig
from vasudha.models.vasudha_model import VasudhaModel, VasudhaForCausalLM

# Register with HuggingFace AutoConfig and AutoModel
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("vasudha", VasudhaConfig)
AutoModelForCausalLM.register(VasudhaConfig, VasudhaForCausalLM)

__all__ = [
    "VasudhaConfig",
    "VasudhaModel",
    "VasudhaForCausalLM",
]
