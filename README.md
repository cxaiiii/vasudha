<div align="center">

# 🌐 Vasudha

### Research-Grade Efficient Reasoning Language Model Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Colab Compatible](https://img.shields.io/badge/Colab-T4%20Compatible-F9AB00.svg)](https://colab.research.google.com/)

*Maximize reasoning per FLOP — not model size.*

</div>

---

## Vision

Vasudha is a research-first language model framework designed to push the frontier of **compute-efficient reasoning**. The goal is not to build the largest model, but to achieve maximum:

- 🧠 Reasoning quality per FLOP
- ⚡ Throughput and memory efficiency  
- 🔬 Research velocity (rapid experimentation)
- 💻 Accessibility (runs on Google Colab Free Tier)

## Architecture Highlights

| Component | Design Choice | Rationale |
|-----------|--------------|-----------|
| **Attention** | Hybrid GLA + Full GQA (3:1) | O(n) linear attention 75% of layers, precise full attention every 4th layer |
| **FFN** | Sparse MoE (every other layer) | DeepSeek-V3 style; fine-grained experts with Top-k routing |
| **Linear Attention** | Gated Linear Attention (GLA) | Data-dependent 2D gates; chunkwise parallel training; O(1) inference |
| **Normalization** | RMSNorm + QK-Norm | Qwen3-style training stability |
| **Position** | RoPE with configurable theta | Long context ready |
| **Routing** | Adaptive difficulty-aware Top-k | Easy→k=1, Medium→k=2, Hard→k=4 |

## Features

### Core
- 🔀 **Hybrid Attention** — Interleaved GLA + Full GQA with configurable ratio
- 🎯 **Sparse MoE** — Top-1/2/4 routing, load balancing, Z-loss, capacity factors
- 🧮 **Adaptive Routing** — Difficulty-based compute allocation (differentiable)
- 🤔 **Reasoning Controller** — Lightweight difficulty predictor
- 🗄️ **Paged KV Cache** — vLLM-style memory management

### Training (Colab-First Design)
- ✅ QLoRA (4-bit) — Finetune 8B models on T4
- ✅ Gradient Checkpointing — Reduced activation memory
- ✅ Sequence Packing — No wasted padding tokens
- ✅ Streaming Datasets — Never run out of memory loading data
- ✅ Checkpoint Resume — Survives Colab disconnects
- ✅ SFT / DPO / ORPO / GRPO

### Backends
- ⚡ **Triton Kernels** — RMSNorm, SwiGLU, Cross-Entropy, GLA, MoE Dispatch/Gather
- 🔥 **FlashAttention-2** — Graceful fallback to SDPA if not available
- 🤗 **HuggingFace Compatible** — `from_pretrained`, tokenizers, datasets

### Evaluation
- 📊 GSM8K, MATH500, AIME, HumanEval, MBPP
- 📏 LongBench, Needle-in-Haystack
- 🔬 Throughput, VRAM, Latency benchmarks

## Quick Start

### Install

```bash
git clone https://github.com/vasudha-ai/vasudha
cd vasudha
pip install -e .
```

For GPU kernels (CUDA machine):
```bash
pip install -e ".[triton,flash]"
```

### Google Colab (T4 Free Tier)

```python
# One-click setup
!pip install -q git+https://github.com/vasudha-ai/vasudha.git
```

Open: `notebooks/00_quickstart.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Load a Model

```python
from vasudha import VasudhaForCausalLM, VasudhaConfig
from transformers import AutoTokenizer

# Load Qwen3-4B weights into Vasudha
model = VasudhaForCausalLM.from_qwen3_pretrained(
    "Qwen/Qwen3-4B",
    attention_type="hybrid",   # GLA + GQA hybrid
    use_moe=True,              # Enable sparse MoE
    load_in_4bit=True,         # QLoRA-ready
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Generate with streaming
from vasudha.inference import StreamingGenerator

gen = StreamingGenerator(model, tokenizer)
for token in gen.stream("Solve step by step: What is 15% of 240?"):
    print(token, end="", flush=True)
```

### Fine-tune on Colab (QLoRA)

Training is a **two-step** process. Vasudha's architecture differs from Qwen3
(hybrid GLA attention, sparse MoE), so the pretrained weights must first be
converted into a Vasudha checkpoint on disk. Conversion runs on CPU with a
bounded memory footprint; quantization then happens at load time, so the fp32
model is never materialized.

```bash
# Step 1 — convert once (~10 min, CPU only, writes a 7.4GB fp16 checkpoint)
python scripts/convert_qwen3_to_vasudha.py \
    --model Qwen/Qwen3-4B --out ./vasudha-4b-init \
    --num-experts 8

# Step 2 — train (loads in 4-bit NF4, ~2GB VRAM for the base weights)
python scripts/train_sft.py model.vasudha_path=./vasudha-4b-init
```

Conversion is loss-preserving: attention and norm weights are copied directly,
each dense FFN is upcycled into `num_experts` contiguous slices (the expert
width is derived from the source model, so it always partitions evenly), and routers are
zero-initialized so every MoE layer initially reproduces the dense layer it
replaced. Only the GLA gate projections start random.

```python
from vasudha.training import VasudhaTrainer, QLoRAConfig, SFTConfig
from vasudha.datasets import build_mixture

# Build a streaming dataset mixture
dataset = build_mixture(
    sources={"open-thoughts/OpenThoughts3": 0.5, "AI-MO/NuminaMath-TIR": 0.5},
    streaming=True,
)

# Configure QLoRA
qlora_config = QLoRAConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    load_in_4bit=True,
)

# Train
trainer = VasudhaTrainer(
    model=model,
    train_dataset=dataset,
    qlora_config=qlora_config,
    sft_config=SFTConfig(
        max_seq_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=1,
        output_dir="./checkpoints",
    ),
)
trainer.train()
```

## Project Structure

```
vasudha/
├── vasudha/
│   ├── models/          # Model backbone (VasudhaForCausalLM)
│   ├── attention/       # GLA, GQA, Hybrid, Sliding Window
│   ├── moe/             # Router, Experts, MoE layer, Adaptive routing
│   ├── kernels/         # Triton: RMSNorm, SwiGLU, CrossEntropy, GLA, MoE
│   ├── reasoning/       # Difficulty controller, Thinking mode
│   ├── memory/          # Paged KV, Compressed KV, Latent compression
│   ├── training/        # SFT, DPO, ORPO, GRPO, QLoRA, Packing
│   ├── datasets/        # Streaming loaders, Mixtures, Chat formats
│   ├── inference/       # Streaming gen, Dynamic batching, Speculative
│   └── utils/           # Logging, VRAM monitor, dtype utils
├── evaluation/          # GSM8K, MATH500, HumanEval, LongBench...
├── configs/             # Hydra YAML configs
├── scripts/             # Training & eval launch scripts
├── notebooks/           # Colab-ready notebooks
├── tests/               # Unit tests
└── docs/                # Architecture docs, research notes
```

## Supported Backbones

| Model | Parameters | Colab T4 | Quantization |
|-------|-----------|----------|-------------|
| Qwen3-4B | 4B | ✅ Full FP16 | ✅ 4-bit / 8-bit |
| Qwen3-8B | 8B | ✅ QLoRA 4-bit | ✅ 4-bit |
| Llama *(future)* | - | ✅ | ✅ |
| Gemma *(future)* | - | ✅ | ✅ |

## Design Philosophy

1. **Correctness first** — Every module has a PyTorch reference implementation before Triton optimization
2. **Colab-first** — Every design decision considers 15GB VRAM ceiling
3. **No placeholders** — Every file is functional before the next is written
4. **Research friendly** — Configuration-driven everything; swap attention/routing/FFN from YAML
5. **HF compatible** — Works with `transformers` ecosystem out of the box

## Benchmarks

*(Coming soon — will be populated after training runs)*

| Task | Score | Tokens/s | VRAM |
|------|-------|---------|------|
| GSM8K | - | - | - |
| MATH500 | - | - | - |
| HumanEval | - | - | - |

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{vasudha2025,
  title  = {Vasudha: Research-Grade Efficient Reasoning Language Model Framework},
  year   = {2025},
  url    = {https://github.com/vasudha-ai/vasudha},
}
```
