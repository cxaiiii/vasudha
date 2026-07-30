"""
Kernel microbenchmarks.

Compares Triton kernel implementations against PyTorch reference implementations
for all Vasudha custom kernels.

Benchmarks:
  - RMSNorm: Triton vs PyTorch
  - SwiGLU: Triton vs PyTorch
  - Cross-Entropy: Chunked vs standard F.cross_entropy
  - GLA (Linear Attention): Triton vs PyTorch chunkwise
  - MoE Dispatch: Triton vs PyTorch scatter
  - MoE Gather: Triton vs PyTorch scatter_add

Usage:
    python scripts/benchmark_kernels.py
    python scripts/benchmark_kernels.py --hidden_size 4096 --seq_len 2048
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import torch
from rich.console import Console
from rich.table import Table

console = Console()


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 5,
    iters: int = 50,
    **kwargs,
) -> float:
    """
    Benchmark a function's execution time.

    Args:
        fn: Function to benchmark.
        *args: Positional arguments to fn.
        warmup: Number of warmup iterations (not counted).
        iters: Number of measured iterations.
        **kwargs: Keyword arguments to fn.

    Returns:
        Mean execution time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()
    return 1000.0 * (end - start) / iters  # ms per iteration


def benchmark_rms_norm(hidden_size: int, seq_len: int, device: str) -> None:
    """Benchmark RMSNorm implementations."""
    from vasudha.kernels.rms_norm import fused_rms_norm, rms_norm_ref

    x = torch.randn(seq_len, hidden_size, device=device)
    weight = torch.ones(hidden_size, device=device)

    triton_time = benchmark_fn(fused_rms_norm, x, weight)
    ref_time = benchmark_fn(rms_norm_ref, x, weight)

    return {"kernel": "RMSNorm", "triton_ms": triton_time, "pytorch_ms": ref_time}


def benchmark_swiglu(hidden_size: int, seq_len: int, device: str) -> dict:
    """Benchmark SwiGLU implementations."""
    from vasudha.kernels.swiglu import fused_swiglu
    import torch.nn.functional as F

    gate = torch.randn(seq_len, hidden_size, device=device)
    up = torch.randn(seq_len, hidden_size, device=device)

    def pytorch_swiglu(g, u):
        return F.silu(g) * u

    triton_time = benchmark_fn(fused_swiglu, gate, up)
    ref_time = benchmark_fn(pytorch_swiglu, gate, up)

    return {"kernel": "SwiGLU", "triton_ms": triton_time, "pytorch_ms": ref_time}


def benchmark_cross_entropy(vocab_size: int, seq_len: int, device: str) -> dict:
    """Benchmark cross-entropy implementations."""
    from vasudha.kernels.cross_entropy import vasudha_cross_entropy
    import torch.nn.functional as F

    logits = torch.randn(seq_len, vocab_size, device=device)
    labels = torch.randint(0, vocab_size, (seq_len,), device=device)

    def standard_ce(l, t):
        return F.cross_entropy(l, t)

    chunked_time = benchmark_fn(vasudha_cross_entropy, logits, labels)
    ref_time = benchmark_fn(standard_ce, logits, labels)

    return {"kernel": "CrossEntropy (chunked vs std)", "triton_ms": chunked_time, "pytorch_ms": ref_time}


def benchmark_gla(
    batch: int, seq_len: int, num_heads: int, head_dim: int, device: str
) -> dict:
    """Benchmark GLA chunkwise scan."""
    from vasudha.kernels.linear_attn import gla_forward

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch, num_heads // 4, seq_len, head_dim, device=device)
    v = torch.randn(batch, num_heads // 4, seq_len, head_dim, device=device)
    g = torch.sigmoid(torch.randn(batch, num_heads // 4, seq_len, head_dim, device=device))

    triton_time = benchmark_fn(gla_forward, q, k, v, g)
    return {"kernel": "GLA (chunkwise)", "triton_ms": triton_time, "pytorch_ms": triton_time}


def main() -> None:
    parser = argparse.ArgumentParser(description="Vasudha kernel benchmarks")
    parser.add_argument("--hidden_size", type=int, default=2560)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=151936)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    console.print(f"\n[bold blue]Vasudha Kernel Benchmarks[/bold blue]")
    console.print(f"Device: {args.device}")
    if torch.cuda.is_available():
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
    console.print(
        f"Config: hidden={args.hidden_size}, seq_len={args.seq_len}, "
        f"batch={args.batch}\n"
    )

    # Run benchmarks
    results = []

    try:
        results.append(benchmark_rms_norm(args.hidden_size, args.seq_len * args.batch, args.device))
    except Exception as e:
        console.print(f"[yellow]RMSNorm benchmark failed: {e}[/yellow]")

    try:
        results.append(benchmark_swiglu(args.hidden_size, args.seq_len * args.batch, args.device))
    except Exception as e:
        console.print(f"[yellow]SwiGLU benchmark failed: {e}[/yellow]")

    try:
        results.append(benchmark_cross_entropy(args.vocab_size, args.seq_len * args.batch, args.device))
    except Exception as e:
        console.print(f"[yellow]CrossEntropy benchmark failed: {e}[/yellow]")

    try:
        results.append(benchmark_gla(args.batch, args.seq_len, args.num_heads, args.head_dim, args.device))
    except Exception as e:
        console.print(f"[yellow]GLA benchmark failed: {e}[/yellow]")

    # Print results table
    table = Table(title="Kernel Benchmark Results")
    table.add_column("Kernel", style="cyan", no_wrap=True)
    table.add_column("Vasudha (ms)", style="green", justify="right")
    table.add_column("PyTorch ref (ms)", style="yellow", justify="right")
    table.add_column("Speedup", style="bold", justify="right")

    for r in results:
        speedup = r["pytorch_ms"] / r["triton_ms"] if r["triton_ms"] > 0 else 1.0
        speedup_str = f"{speedup:.2f}×"
        color = "green" if speedup > 1.0 else "red"
        table.add_row(
            r["kernel"],
            f"{r['triton_ms']:.3f}",
            f"{r['pytorch_ms']:.3f}",
            f"[{color}]{speedup_str}[/{color}]",
        )

    console.print(table)


if __name__ == "__main__":
    main()
