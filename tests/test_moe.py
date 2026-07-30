"""
Unit tests for Vasudha MoE modules.

Tests cover:
  1. Router — Top-k selection, load balance loss, Z-loss
  2. Expert — Forward pass, gradient flow
  3. MoE Layer — Full forward with aux loss
  4. Capacity manager — Token overflow handling
  5. Adaptive router — Difficulty prediction, k selection
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vasudha.models.config import VasudhaConfig


@pytest.fixture
def debug_config() -> VasudhaConfig:
    return VasudhaConfig.for_debug()


@pytest.fixture
def router_input(debug_config):
    B, L, H = 2, 16, debug_config.hidden_size
    return torch.randn(B, L, H)


class TestVasudhaRouter:
    """Test the Top-k router."""

    @pytest.fixture(autouse=True)
    def import_router(self):
        pytest.importorskip("vasudha.moe.router", reason="MoE module not built")

    def test_output_shapes(self, debug_config, router_input):
        from vasudha.moe.router import VasudhaRouter
        router = VasudhaRouter(debug_config)
        B, L, H = router_input.shape
        N = B * L
        k = debug_config.num_experts_per_tok

        result = router(router_input)

        assert result.routing_weights.shape == (N, k), \
            f"Expected ({N}, {k}), got {result.routing_weights.shape}"
        assert result.expert_indices.shape == (N, k)
        assert result.aux_loss.ndim == 0, "aux_loss should be scalar"

    def test_routing_weights_sum_to_one(self, debug_config, router_input):
        from vasudha.moe.router import VasudhaRouter
        router = VasudhaRouter(debug_config)
        result = router(router_input)
        # Each token's routing weights should sum to ~1.0
        sums = result.routing_weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_expert_indices_in_range(self, debug_config, router_input):
        from vasudha.moe.router import VasudhaRouter
        router = VasudhaRouter(debug_config)
        result = router(router_input)
        E = debug_config.num_experts
        assert result.expert_indices.min() >= 0
        assert result.expert_indices.max() < E

    def test_aux_loss_positive(self, debug_config, router_input):
        from vasudha.moe.router import VasudhaRouter
        router = VasudhaRouter(debug_config)
        result = router(router_input)
        assert result.aux_loss.item() >= 0, "Aux loss should be non-negative"

    def test_gradient_through_routing_weights(self, debug_config, router_input):
        from vasudha.moe.router import VasudhaRouter
        router_input.requires_grad_(True)
        router = VasudhaRouter(debug_config)
        result = router(router_input)
        # routing_weights should be differentiable
        loss = result.routing_weights.mean() + result.aux_loss
        loss.backward()
        assert router_input.grad is not None

    def test_no_nan_in_aux_loss(self, debug_config, router_input):
        from vasudha.moe.router import VasudhaRouter
        router = VasudhaRouter(debug_config)
        result = router(router_input)
        assert not result.aux_loss.isnan(), "Router aux_loss is NaN!"


class TestVasudhaExpert:
    """Test individual SwiGLU experts."""

    @pytest.fixture(autouse=True)
    def import_expert(self):
        pytest.importorskip("vasudha.moe.expert", reason="MoE module not built")

    def test_output_shape(self, debug_config):
        from vasudha.moe.expert import VasudhaExpert
        expert = VasudhaExpert(debug_config)
        B, L = 2, 8
        x = torch.randn(B * L, debug_config.hidden_size)
        out = expert(x)
        assert out.shape == x.shape, f"Expert should preserve shape, got {out.shape}"

    def test_gradient_flow(self, debug_config):
        from vasudha.moe.expert import VasudhaExpert
        expert = VasudhaExpert(debug_config)
        x = torch.randn(8, debug_config.hidden_size, requires_grad=True)
        out = expert(x)
        out.mean().backward()
        assert x.grad is not None
        assert not x.grad.isnan().any()


class TestMoELayer:
    """Test full MoE layer."""

    @pytest.fixture(autouse=True)
    def import_moe(self):
        pytest.importorskip("vasudha.moe.moe_layer", reason="MoE module not built")

    def test_output_shape_and_aux_loss(self, debug_config, router_input):
        from vasudha.moe.moe_layer import VasudhaMoELayer
        moe = VasudhaMoELayer(debug_config)
        result = moe(router_input)

        if isinstance(result, (tuple, list)):
            output, aux_loss = result[0], result[1]
        else:
            output = result
            aux_loss = None

        B, L, H = router_input.shape
        assert output.shape == (B, L, H)

        if aux_loss is not None:
            assert aux_loss.ndim == 0, "Aux loss must be scalar"
            assert aux_loss.item() >= 0

    def test_gradient_through_moe(self, debug_config, router_input):
        from vasudha.moe.moe_layer import VasudhaMoELayer
        router_input.requires_grad_(True)
        moe = VasudhaMoELayer(debug_config)
        result = moe(router_input)

        if isinstance(result, (tuple, list)):
            output, aux_loss = result[0], result[1] if len(result) > 1 else None
        else:
            output, aux_loss = result, None

        loss = output.mean()
        if aux_loss is not None:
            loss = loss + aux_loss
        loss.backward()

        assert router_input.grad is not None
        assert not router_input.grad.isnan().any(), "NaN gradients in MoE layer!"


class TestCapacityManager:
    """Test token overflow handling."""

    @pytest.fixture(autouse=True)
    def import_capacity(self):
        pytest.importorskip("vasudha.moe.capacity", reason="MoE module not built")

    def test_capacity_computation(self, debug_config):
        from vasudha.moe.capacity import CapacityManager
        cm = CapacityManager(debug_config)
        capacity = cm.compute_capacity(
            num_tokens=32,
            capacity_factor=1.25,
            num_experts=debug_config.num_experts,
        )
        # With 32 tokens, 8 experts, factor 1.25: 32/8 * 1.25 = 5
        expected = int(32 / debug_config.num_experts * 1.25)
        assert capacity == expected, f"Expected capacity {expected}, got {capacity}"


class TestAdaptiveRouter:
    """Test difficulty-adaptive routing."""

    @pytest.fixture(autouse=True)
    def import_adaptive(self):
        pytest.importorskip("vasudha.moe.adaptive_router", reason="MoE module not built")

    def test_output_has_difficulty(self, debug_config, router_input):
        from vasudha.moe.adaptive_router import AdaptiveRouter
        config = VasudhaConfig.for_debug(use_adaptive_routing=True)
        router = AdaptiveRouter(config)
        result = router(router_input, training=False)

        assert hasattr(result, "difficulty_logits")
        assert result.difficulty_logits.shape[-1] == 3  # Easy/Medium/Hard
        assert result.active_k in (
            config.adaptive_routing_min_k,
            (config.adaptive_routing_min_k + config.adaptive_routing_max_k) // 2,
            config.adaptive_routing_max_k,
        )

    def test_k_bounds_respected(self, debug_config):
        from vasudha.moe.adaptive_router import AdaptiveRouter
        config = VasudhaConfig.for_debug(
            use_adaptive_routing=True,
            adaptive_routing_min_k=1,
            adaptive_routing_max_k=4,
        )
        router = AdaptiveRouter(config)
        x = torch.randn(2, 16, config.hidden_size)

        for _ in range(10):
            result = router(x, training=False)
            assert config.adaptive_routing_min_k <= result.active_k <= config.adaptive_routing_max_k
