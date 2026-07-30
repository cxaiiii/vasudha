"""
Tests for Triton kernels.

Verifies:
  1. Shape correctness for all kernels
  2. Numerical equivalence between Triton and PyTorch reference
  3. Gradient flow
  4. T4 compatibility (block size handling)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vasudha.kernels import get_kernel_status


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def kernel_status() -> dict[str, str]:
    return get_kernel_status()


class TestKernelStatus:
    """Test that kernel status is correctly reported."""

    def test_status_returns_dict(self, kernel_status):
        assert isinstance(kernel_status, dict)
        assert len(kernel_status) > 0

    def test_all_kernels_have_status(self, kernel_status):
        expected_kernels = {"rms_norm", "swiglu", "cross_entropy", "linear_attn",
                           "moe_dispatch", "moe_gather"}
        for k in expected_kernels:
            assert k in kernel_status, f"Kernel '{k}' not in status report"
            assert kernel_status[k] in ("triton", "pytorch_fallback")


class TestRMSNorm:
    """Test fused RMSNorm kernel."""

    @pytest.fixture(autouse=True)
    def import_kernel(self):
        pytest.importorskip("vasudha.kernels.rms_norm", reason="Kernels not built")

    def test_shape_preserved(self, device):
        from vasudha.kernels.rms_norm import fused_rms_norm
        x = torch.randn(8, 16, 256, device=device)
        w = torch.ones(256, device=device)
        out = fused_rms_norm(x, w)
        assert out.shape == x.shape

    def test_equivalence_with_pytorch(self, device):
        from vasudha.kernels.rms_norm import fused_rms_norm, rms_norm_ref
        x = torch.randn(4, 32, 512, device=device)
        w = torch.ones(512, device=device)

        ref = rms_norm_ref(x, w)
        ours = fused_rms_norm(x, w)
        torch.testing.assert_close(ours, ref, atol=1e-3, rtol=1e-3)

    def test_no_nan_output(self, device):
        from vasudha.kernels.rms_norm import fused_rms_norm
        x = torch.randn(8, 64, 2560, device=device)
        w = torch.ones(2560, device=device)
        out = fused_rms_norm(x, w)
        assert not out.isnan().any(), "RMSNorm produced NaN!"

    def test_gradient_flow(self, device):
        from vasudha.kernels.rms_norm import fused_rms_norm
        x = torch.randn(4, 16, 256, device=device, requires_grad=True)
        w = torch.ones(256, device=device, requires_grad=True)
        out = fused_rms_norm(x, w)
        out.mean().backward()
        assert x.grad is not None
        assert w.grad is not None
        assert not x.grad.isnan().any()


class TestSwiGLU:
    """Test fused SwiGLU kernel."""

    @pytest.fixture(autouse=True)
    def import_kernel(self):
        pytest.importorskip("vasudha.kernels.swiglu", reason="Kernels not built")

    def test_shape(self, device):
        from vasudha.kernels.swiglu import fused_swiglu
        gate = torch.randn(32, 1024, device=device)
        up = torch.randn(32, 1024, device=device)
        out = fused_swiglu(gate, up)
        assert out.shape == gate.shape

    def test_equivalence(self, device):
        from vasudha.kernels.swiglu import fused_swiglu
        gate = torch.randn(16, 512, device=device)
        up = torch.randn(16, 512, device=device)

        ref = F.silu(gate) * up
        ours = fused_swiglu(gate, up)
        torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)

    def test_no_nan(self, device):
        from vasudha.kernels.swiglu import fused_swiglu
        gate = torch.randn(64, 6912, device=device)
        up = torch.randn(64, 6912, device=device)
        out = fused_swiglu(gate, up)
        assert not out.isnan().any()


class TestChunkedCrossEntropy:
    """Test chunked cross-entropy."""

    @pytest.fixture(autouse=True)
    def import_kernel(self):
        pytest.importorskip("vasudha.kernels.cross_entropy", reason="Kernels not built")

    def test_equivalence_with_standard(self, device):
        from vasudha.kernels.cross_entropy import vasudha_cross_entropy
        N, V = 128, 32000
        logits = torch.randn(N, V, device=device)
        labels = torch.randint(0, V, (N,), device=device)

        std_loss = F.cross_entropy(logits, labels, ignore_index=-100)
        chunked_loss = vasudha_cross_entropy(logits, labels, ignore_index=-100)

        torch.testing.assert_close(chunked_loss, std_loss, atol=1e-4, rtol=1e-4)

    def test_ignore_index_respected(self, device):
        from vasudha.kernels.cross_entropy import vasudha_cross_entropy
        N, V = 64, 1000
        logits = torch.randn(N, V, device=device)
        labels = torch.randint(0, V, (N,), device=device)
        labels[:N//2] = -100  # Ignore half

        std_loss = F.cross_entropy(logits, labels, ignore_index=-100)
        chunked_loss = vasudha_cross_entropy(logits, labels, ignore_index=-100)

        torch.testing.assert_close(chunked_loss, std_loss, atol=1e-4, rtol=1e-4)


class TestGLAKernel:
    """Test GLA chunkwise scan kernel."""

    @pytest.fixture(autouse=True)
    def import_kernel(self):
        pytest.importorskip("vasudha.kernels.linear_attn", reason="Kernels not built")

    def test_output_shape(self, device):
        from vasudha.kernels.linear_attn import gla_forward
        B, H, Hkv, L, d = 2, 8, 2, 64, 32
        q = torch.randn(B, H, L, d, device=device)
        k = torch.randn(B, Hkv, L, d, device=device)
        v = torch.randn(B, Hkv, L, d, device=device)
        g = torch.sigmoid(torch.randn(B, Hkv, L, d, device=device))

        output, state = gla_forward(q, k, v, g)
        assert output.shape == (B, H, L, d)

    def test_no_nan(self, device):
        from vasudha.kernels.linear_attn import gla_forward
        B, H, Hkv, L, d = 1, 4, 1, 32, 16
        q = torch.randn(B, H, L, d, device=device)
        k = torch.randn(B, Hkv, L, d, device=device)
        v = torch.randn(B, Hkv, L, d, device=device)
        g = torch.sigmoid(torch.randn(B, Hkv, L, d, device=device))

        output, state = gla_forward(q, k, v, g)
        assert not output.isnan().any(), "GLA kernel produced NaN!"
