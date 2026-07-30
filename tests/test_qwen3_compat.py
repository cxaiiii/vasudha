"""
Tests for Qwen3 weight loading compatibility.

Verifies:
  1. Config conversion from Qwen3 → Vasudha
  2. Weight name mapping
  3. State dict compatibility check (without downloading the full model)
"""

from __future__ import annotations

import pytest
import torch

from vasudha.models.config import VasudhaConfig
from vasudha.models.qwen3_compat import _qwen3_config_to_vasudha, get_qwen3_weight_map


class TestConfigConversion:
    """Test Qwen3 → Vasudha config conversion."""

    def test_basic_conversion(self):
        """Test converting a mock Qwen3 config."""
        from types import SimpleNamespace

        # Simulate a Qwen3-4B config object
        mock_qwen3_config = SimpleNamespace(
            hidden_size=2560,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=6912,
            head_dim=128,
            vocab_size=151936,
            max_position_embeddings=32768,
            rope_theta=1_000_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            tie_word_embeddings=False,
            bos_token_id=151643,
            eos_token_id=151645,
        )

        config = _qwen3_config_to_vasudha(mock_qwen3_config, attention_type="hybrid")

        assert isinstance(config, VasudhaConfig)
        assert config.hidden_size == 2560
        assert config.num_hidden_layers == 36
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 8
        assert config.attention_type == "hybrid"
        assert config.vocab_size == 151936
        assert config.qk_norm is True  # Always True for Qwen3

    def test_moe_config_injection(self):
        """Test that MoE config is properly injected."""
        from types import SimpleNamespace

        mock_config = SimpleNamespace(
            hidden_size=2560,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=6912,
            head_dim=128,
            vocab_size=151936,
            max_position_embeddings=32768,
            rope_theta=1_000_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            tie_word_embeddings=False,
            bos_token_id=151643,
            eos_token_id=151645,
        )

        config_with_moe = _qwen3_config_to_vasudha(mock_config, use_moe=True)
        config_no_moe = _qwen3_config_to_vasudha(mock_config, use_moe=False)

        assert config_with_moe.use_moe is True
        assert len(config_with_moe.moe_layers) > 0
        assert config_no_moe.use_moe is False


class TestWeightMapping:
    """Test weight name mapping between Qwen3 and Vasudha."""

    def test_weight_map_structure(self):
        """Verify the weight map has expected structure without downloading."""
        from types import SimpleNamespace
        # We can't call get_qwen3_weight_map without downloading, so test manually
        from vasudha.models.qwen3_compat import _qwen3_config_to_vasudha

        mock_qwen3_config = SimpleNamespace(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=512,
            head_dim=64,
            vocab_size=1000,
            max_position_embeddings=512,
            rope_theta=10_000.0,
            rms_norm_eps=1e-6,
            attention_bias=False,
            tie_word_embeddings=False,
            bos_token_id=1,
            eos_token_id=2,
        )

        vasudha_config = _qwen3_config_to_vasudha(mock_qwen3_config)

        # Build the model and check key names
        from vasudha.models.vasudha_model import VasudhaForCausalLM
        model = VasudhaForCausalLM(vasudha_config)
        param_names = list(model.state_dict().keys())

        # Key names that should be present
        assert "model.embed_tokens.weight" in param_names
        assert "model.norm.weight" in param_names
        assert "lm_head.weight" in param_names

        # Layer 0 attention
        assert "model.layers.0.input_layernorm.weight" in param_names
        assert "model.layers.0.self_attn.q_proj.weight" in param_names

    def test_weight_copy_matching_shapes(self):
        """Test that overlapping weights can be copied without shape errors."""
        from vasudha.models.vasudha_model import VasudhaForCausalLM
        from vasudha.models.qwen3_compat import _copy_weights

        # Both models use same config for shape matching
        debug_config = VasudhaConfig.for_debug(use_moe=False, attention_type="sdpa")
        src_model = VasudhaForCausalLM(debug_config)
        dst_model = VasudhaForCausalLM(debug_config)

        # Initialize src with random values
        for param in src_model.parameters():
            torch.nn.init.normal_(param)

        # Copy
        num_loaded, num_skipped, num_missing = _copy_weights(src_model, dst_model)

        # All weights should be loaded (same architecture)
        total = num_loaded + num_skipped + num_missing
        assert num_loaded == total, f"Expected all weights loaded, got {num_loaded}/{total}"
        assert num_skipped == 0, f"No weights should be skipped for same architecture"
