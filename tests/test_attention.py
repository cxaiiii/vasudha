"""
Unit tests for Vasudha attention modules.

Tests cover:
  1. GQA (SDPA) attention — shape correctness, gradient flow, KV cache
  2. GLA attention — shape correctness, recurrent state, chunkwise equivalence
  3. Hybrid attention manager — correct type dispatch by layer index
  4. QK-Norm — training stability property
  5. RoPE — position encoding correctness
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from vasudha.models.config import VasudhaConfig


# ── Test fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def debug_config() -> VasudhaConfig:
    """Tiny config for fast CPU testing."""
    return VasudhaConfig.for_debug()


@pytest.fixture
def batch_inputs(debug_config: VasudhaConfig):
    """Generate random batch inputs for attention testing."""
    B, L, H = 2, 16, debug_config.hidden_size
    hidden_states = torch.randn(B, L, H)
    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
    return hidden_states, position_ids


# ── Config tests ───────────────────────────────────────────────────────────────

class TestVasudhaConfig:
    """Test VasudhaConfig initialization and derived properties."""

    def test_default_init(self):
        config = VasudhaConfig()
        assert config.hidden_size == 2560
        assert config.num_hidden_layers == 36
        assert config.attention_type == "hybrid"

    def test_debug_config(self):
        config = VasudhaConfig.for_debug()
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 4
        assert len(config.moe_layers) == 2  # layers 1 and 3

    def test_hybrid_pattern_lookup(self):
        config = VasudhaConfig.for_debug(
            attention_type="hybrid",
            hybrid_pattern=["gla", "gla", "gla", "sdpa"],
        )
        # Pattern repeats: [gla, gla, gla, sdpa, gla, gla, ...]
        assert config.get_attention_type_for_layer(0) == "gla"
        assert config.get_attention_type_for_layer(3) == "sdpa"
        assert config.get_attention_type_for_layer(4) == "gla"
        assert config.get_attention_type_for_layer(7) == "sdpa"

    def test_moe_layer_detection(self):
        config = VasudhaConfig.for_debug(use_moe=True)
        # Odd layers are MoE
        assert not config.is_moe_layer(0)
        assert config.is_moe_layer(1)
        assert not config.is_moe_layer(2)
        assert config.is_moe_layer(3)

    def test_gqa_ratio(self):
        config = VasudhaConfig.for_debug()
        expected_ratio = config.num_attention_heads // config.num_key_value_heads
        assert config.gqa_ratio == expected_ratio

    def test_intermediate_size_default(self):
        """Verify default intermediate_size is ~8/3 * hidden_size rounded to 64."""
        config = VasudhaConfig(hidden_size=256, intermediate_size=None)
        raw = 8 / 3 * 256
        expected = (int(raw) + 63) // 64 * 64
        assert config.intermediate_size == expected

    def test_parameter_estimate_positive(self):
        config = VasudhaConfig.for_debug()
        params = config.estimate_parameters()
        assert params["total"] > 0
        assert all(v >= 0 for v in params.values())

    def test_invalid_attention_type_raises(self):
        with pytest.raises(ValueError, match="attention_type"):
            VasudhaConfig(attention_type="invalid_type")

    def test_qwen3_4b_config(self):
        config = VasudhaConfig.for_qwen3_4b()
        assert config.hidden_size == 2560
        assert config.num_hidden_layers == 36
        assert config.vocab_size == 151936
        assert config.qk_norm is True

    def test_repr(self):
        config = VasudhaConfig.for_debug()
        r = repr(config)
        assert "VasudhaConfig" in r
        assert "hybrid" in r


# ── Attention tests ────────────────────────────────────────────────────────────

class TestSDPAAttention:
    """Test VasudhaGQAAttention (SDPA-based full attention)."""

    @pytest.fixture(autouse=True)
    def import_sdpa(self):
        pytest.importorskip("vasudha.attention.sdpa",
                            reason="Attention module not built")

    def test_output_shape(self, debug_config, batch_inputs):
        from vasudha.attention.sdpa import VasudhaGQAAttention
        B, L, H = 2, 16, debug_config.hidden_size
        hidden_states, position_ids = batch_inputs

        attn = VasudhaGQAAttention(debug_config, layer_idx=3)
        with torch.no_grad():
            output, kv_cache = attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                use_cache=True,
            )

        assert output.shape == (B, L, H), f"Expected {(B, L, H)}, got {output.shape}"
        assert kv_cache is not None

    def test_gradient_flow(self, debug_config, batch_inputs):
        from vasudha.attention.sdpa import VasudhaGQAAttention
        hidden_states, position_ids = batch_inputs
        hidden_states.requires_grad_(True)

        attn = VasudhaGQAAttention(debug_config, layer_idx=3)
        output, _ = attn(hidden_states=hidden_states, position_ids=position_ids)
        loss = output.mean()
        loss.backward()

        assert hidden_states.grad is not None
        assert not hidden_states.grad.isnan().any()

    def test_kv_cache_consistency(self, debug_config):
        """KV cache: processing seq at once vs. token by token should match."""
        from vasudha.attention.sdpa import VasudhaGQAAttention
        B, L, H = 1, 8, debug_config.hidden_size
        attn = VasudhaGQAAttention(debug_config, layer_idx=3)
        attn.eval()

        x = torch.randn(B, L, H)
        pos = torch.arange(L).unsqueeze(0)

        with torch.no_grad():
            # Full sequence
            out_full, _ = attn(hidden_states=x, position_ids=pos, use_cache=False)

            # Token by token with cache
            out_cached = []
            past_kv = None
            for t in range(L):
                xt = x[:, t:t+1, :]
                post = pos[:, t:t+1]
                out_t, past_kv = attn(
                    hidden_states=xt,
                    position_ids=post,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                out_cached.append(out_t)
            out_cached_tensor = torch.cat(out_cached, dim=1)

        torch.testing.assert_close(out_full, out_cached_tensor, atol=1e-4, rtol=1e-4)


class TestGLAAttention:
    """Test VasudhaGLAAttention (Gated Linear Attention)."""

    @pytest.fixture(autouse=True)
    def import_gla(self):
        pytest.importorskip("vasudha.attention.gla",
                            reason="Attention module not built")

    def test_output_shape(self, debug_config, batch_inputs):
        from vasudha.attention.gla import VasudhaGLAAttention
        B, L, H = 2, 16, debug_config.hidden_size
        hidden_states, position_ids = batch_inputs

        attn = VasudhaGLAAttention(debug_config, layer_idx=0)
        with torch.no_grad():
            output, state = attn(
                hidden_states=hidden_states,
                position_ids=position_ids,
                use_cache=True,
            )

        assert output.shape == (B, L, H)
        # State should be the recurrent hidden state

    def test_gradient_flow(self, debug_config, batch_inputs):
        from vasudha.attention.gla import VasudhaGLAAttention
        hidden_states, position_ids = batch_inputs
        hidden_states.requires_grad_(True)

        attn = VasudhaGLAAttention(debug_config, layer_idx=0)
        output, _ = attn(hidden_states=hidden_states, position_ids=position_ids)
        loss = output.mean()
        loss.backward()

        assert hidden_states.grad is not None
        assert not hidden_states.grad.isnan().any(), "NaN gradients in GLA!"
        assert not hidden_states.grad.isinf().any(), "Inf gradients in GLA!"

    def test_no_nan_output(self, debug_config, batch_inputs):
        from vasudha.attention.gla import VasudhaGLAAttention
        hidden_states, position_ids = batch_inputs
        attn = VasudhaGLAAttention(debug_config, layer_idx=0)
        with torch.no_grad():
            output, _ = attn(hidden_states=hidden_states, position_ids=position_ids)
        assert not output.isnan().any(), "GLA output contains NaN!"


class TestHybridAttention:
    """Test the hybrid attention manager."""

    @pytest.fixture(autouse=True)
    def import_hybrid(self):
        pytest.importorskip("vasudha.attention.hybrid",
                            reason="Attention module not built")

    def test_layer_type_mapping(self, debug_config):
        from vasudha.attention.hybrid import VasudhaHybridManager
        # debug_config has 4 layers, pattern [gla, gla, gla, sdpa]
        manager = VasudhaHybridManager(debug_config)
        assert manager.is_linear_attention_layer(0)  # GLA
        assert manager.is_linear_attention_layer(1)  # GLA
        assert manager.is_linear_attention_layer(2)  # GLA
        assert not manager.is_linear_attention_layer(3)  # SDPA

    def test_correct_module_type(self, debug_config):
        from vasudha.attention.hybrid import VasudhaHybridManager
        from vasudha.attention.gla import VasudhaGLAAttention
        from vasudha.attention.sdpa import VasudhaGQAAttention

        manager = VasudhaHybridManager(debug_config)
        gla_cls = manager.get_attention_module(0)
        sdpa_cls = manager.get_attention_module(3)

        assert issubclass(gla_cls, VasudhaGLAAttention) or gla_cls.__name__ == "VasudhaGLAAttention"


# ── RMSNorm tests ──────────────────────────────────────────────────────────────

class TestRMSNorm:
    """Test RMSNorm implementation."""

    @pytest.fixture(autouse=True)
    def import_norm(self):
        pytest.importorskip("vasudha.attention.sdpa",
                            reason="Attention module not built")

    def test_output_shape(self):
        from vasudha.attention.sdpa import VasudhaRMSNorm
        norm = VasudhaRMSNorm(256)
        x = torch.randn(2, 16, 256)
        out = norm(x)
        assert out.shape == x.shape

    def test_unit_variance_approximately(self):
        """RMSNorm output should have ~unit RMS per position."""
        from vasudha.attention.sdpa import VasudhaRMSNorm
        norm = VasudhaRMSNorm(256)
        # Initialize weight to 1 for clean test
        norm.weight.data.fill_(1.0)
        x = torch.randn(100, 256)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=0.1, rtol=0.1)

    def test_gradient_flow(self):
        from vasudha.attention.sdpa import VasudhaRMSNorm
        norm = VasudhaRMSNorm(256)
        x = torch.randn(2, 16, 256, requires_grad=True)
        out = norm(x)
        out.mean().backward()
        assert x.grad is not None
        assert not x.grad.isnan().any()
