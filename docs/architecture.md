"""
Architecture documentation for Vasudha.

This document describes the key architectural decisions in Vasudha
and the research rationale behind each choice.
"""

# Vasudha Architecture

## Overview

Vasudha is a modular research LLM framework built on three pillars:

1. **Hybrid Attention** — Gated Linear Attention (GLA) interleaved with full GQA
2. **Sparse MoE** — Fine-grained experts in alternating layers
3. **Colab-First Engineering** — Every component designed for 15GB VRAM

---

## Attention Architecture

### Why Hybrid? Why GLA?

The pure transformer (all softmax attention) has quadratic memory in sequence length.
For long reasoning chains, this becomes the bottleneck.

We surveyed the linear attention landscape:

| Method | O(n) | Content Gates | Parallel Training | HW Efficiency |
|--------|------|---------------|-------------------|--------------|
| RetNet | ✅ | ❌ (fixed decay) | ✅ | ✅ |
| RWKV-6 | ✅ | ✅ (channel) | ✅ | Good |
| Mamba-2 | ✅ | ✅ (SSM) | ✅ | Excellent |
| HGRN2 | ✅ | ✅ (outer product) | ✅ | Good |
| **GLA** | ✅ | ✅ (2D gate) | ✅ | Excellent |

**GLA wins** because:
- Its 2D forget gate operates over (head_dim × head_dim) state, giving it
  content-based gating with quadratic state capacity
- Chunkwise parallel form: O(chunk_size²) within chunk, O(n) across chunks
- Directly compatible with GQA (different Q vs KV head counts)
- The flash-linear-attention library provides production Triton kernels

### Hybrid Pattern: 3:1

Every 4th layer uses full GQA attention. The 3 GLA layers preceding it provide:
- Efficient long-range context aggregation (O(n))
- Running recurrent state compression

The GQA layer provides:
- Precise content-based retrieval (impossible in pure GLA)
- "Resets" the recurrent state compression periodically

Pattern: `[GLA, GLA, GLA, GQA, GLA, GLA, GLA, GQA, ...]`

This matches Qwen3-Next and Kimi Linear designs.

---

## MoE Architecture

### Fine-Grained Experts

Vasudha uses 64 experts with Top-2 routing (2 experts activated per token).
This follows the DeepSeek-MoE and Qwen3-MoE finding that fine-grained experts
(smaller, more numerous) outperform coarse experts at the same FLOP budget.

Effective parameters per forward pass:
- Total experts: 64
- Active per token: 2 (3.1% of experts)
- Routing: Top-k with load balancing + Z-loss

### Router Losses

**Load Balancing Loss**: Penalizes imbalanced routing across experts.
```
L_balance = num_experts × Σᵢ (f_i × P_i)
```
where f_i = fraction of tokens routed to expert i,
      P_i = mean routing probability for expert i.

**Z-Loss**: Prevents router logit collapse (all logits go to one value).
```
L_z = mean(log(Σⱼ exp(logits_j))²)
```

### MoE Layer Placement

MoE replaces the dense FFN in every other layer (odd indices: 1, 3, 5, ...).
This matches the DeepSeek-V3 design philosophy:
- Even layers: Dense SwiGLU FFN (lower computational cost)
- Odd layers: Sparse MoE FFN (higher capacity, same FLOP cost)

---

## Normalization

### RMSNorm
Standard RMSNorm (no mean subtraction, no bias) with learned scale parameter.
```
RMSNorm(x) = x / RMS(x) × γ
```

### QK-Norm (Qwen3 innovation)
Per-head RMSNorm applied to Q and K before attention scores:
```
attention = softmax(QK-Norm(Q) × QK-Norm(K)ᵀ / √d) × V
```
This stabilizes training by preventing attention logit explosion at large scale.
Without QK-Norm, large models can develop extreme attention patterns that destabilize training.

---

## RoPE Position Encoding

Rotary Position Embedding with:
- `rope_theta = 1,000,000` (Qwen3 setting, enables long context)
- Applied only to Q and K (not V or gate in GLA)
- Standard 2D rotation on last two dimensions of each head

For context >32K: YaRN or dynamic NTK scaling (future work, configurable via rope_scaling).

---

## Training Design

### Colab T4 Constraints

The Tesla T4 has 15GB VRAM. Key accommodations:

1. **QLoRA**: 4-bit base + fp16 LoRA adapters → ~3.5GB for 4B model
2. **Gradient Checkpointing**: Recompute activations backward → 50% memory reduction
3. **Sequence Packing**: Pack short sequences together → 0% padding waste
4. **Streaming Datasets**: Never load full dataset into RAM
5. **Chunked Cross-Entropy**: Avoid materializing (B×L, V=151936) softmax tensor
6. **paged_adamw_32bit**: Optimizer states paged to CPU when not in use

### Checkpoint Resume

Colab disconnects every ~3-6 hours. The CheckpointManager:
- Saves every N steps (configurable, default 200)
- Saves: model weights + optimizer state + scheduler state + RNG state
- Auto-detects latest checkpoint on restart
- Preserves dataset streaming position (via HF datasets state_dict)

---

## Triton Kernels

All hot-path operations have Triton-optimized implementations with PyTorch fallbacks:

| Kernel | Memory Saved | Speedup (T4) |
|--------|-------------|-------------|
| RMSNorm | 2× HBM reads | 1.3-1.8× |
| SwiGLU | 1 HBM write | 1.2-1.5× |
| Cross-Entropy | ~4× (chunked) | 1.1-1.3× |
| GLA chunkwise | No L×L matrix | ∞ for L>8K |
| MoE Dispatch | Contiguous layout | 1.5-2× |
| MoE Gather | Atomic adds | 1.3-1.8× |

All kernels fall back to PyTorch if:
- Triton is not installed
- CUDA is not available
- Kernel compilation fails

---

## HuggingFace Compatibility

Vasudha extends PreTrainedModel, making it compatible with:
- `model.generate()` — full generation API
- `trainer = SFTTrainer(model=vasudha_model)` — standard training
- `AutoModel.from_pretrained("./vasudha_model")` — auto-loading
- `model.push_to_hub("username/vasudha-4b")` — hub upload
- PEFT/LoRA: `get_peft_model(vasudha_model, lora_config)` — standard PEFT

The only non-standard parts are:
- MoE auxiliary loss (handled transparently inside forward())
- GLA recurrent state in KV cache (stored as (B, Hkv, d, d) instead of (B, Hkv, L, d))
