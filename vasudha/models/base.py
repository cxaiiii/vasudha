"""
Vasudha base model class.

Provides the VasudhaPreTrainedModel base class which extends HuggingFace's
PreTrainedModel with Vasudha-specific utilities:
  - Gradient checkpointing setup
  - MoE auxiliary loss handling
  - Weight initialization
  - VRAM-aware loading helpers
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import logging as hf_logging

from vasudha.models.config import VasudhaConfig
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)


class VasudhaPreTrainedModel(PreTrainedModel):
    """
    Base class for all Vasudha models.

    Provides:
    1. Correct weight initialization (scaled by 1/sqrt(2*num_layers) for residuals)
    2. Gradient checkpointing integration
    3. MoE auxiliary loss aggregation helpers
    4. Standard HuggingFace interface

    All Vasudha model variants (VasudhaModel, VasudhaForCausalLM) inherit this.
    """

    config_class = VasudhaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VasudhaDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for Vasudha modules.

        Follows the standard practice for Transformer initialization:
        - Linear layers: normal distribution with std = initializer_range
        - Embedding layers: normal distribution with std = initializer_range
        - Residual projections (o_proj, down_proj): scaled by 1/sqrt(2 * n_layers)
          This is the "depth scaling" used by GPT-2 and most modern LLMs to prevent
          residual stream variance from exploding with depth.
        - Bias terms: zero initialized
        - RMSNorm weight (gamma): ones initialized

        Args:
            module: The module to initialize.
        """
        std = self.config.initializer_range
        # Scale output projections by 1/sqrt(2*L) for residual stability
        residual_std = std / math.sqrt(2 * self.config.num_hidden_layers)

        if isinstance(module, nn.Linear):
            # Check if this is a residual output projection
            is_residual = getattr(module, "_is_residual_proj", False)
            init_std = residual_std if is_residual else std
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            # Zero out padding embedding if it exists
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(
        self, module: nn.Module, value: bool = False
    ) -> None:
        """
        Enable or disable gradient checkpointing for decoder layers.

        This trades compute for memory by recomputing activations during the
        backward pass instead of storing them. Essential for T4 (15GB VRAM).

        Args:
            module: Module to configure.
            value: True to enable gradient checkpointing.
        """
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @classmethod
    def _mark_residual_projections(cls, model: "VasudhaPreTrainedModel") -> None:
        """
        Mark output projections for scaled initialization.

        Output projections (o_proj, down_proj) should be initialized with
        smaller std to preserve residual stream scale at initialization.
        Modifies modules in-place by setting _is_residual_proj = True.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith(".o_proj") or name.endswith(".down_proj"):
                    module._is_residual_proj = True  # type: ignore[attr-defined]
