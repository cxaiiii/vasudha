"""Is the engine fast, and is it still right? Roughly a minute.

Fast and correct are separate questions and a GPU build can regress the second
while winning the first — Vulkan does its arithmetic in a different order than
the CPU kernels, so "it feels quick now" is not evidence the answers survived.
This checks both and prints one verdict.

Run it:
    python scripts/check_engine.py --gguf ollama/vasudha-4b-v3.gguf
    python scripts/check_engine.py --model vasudha-v3        # via ollama

If it reports a CPU-only wheel, that is your Python environment rather than the
shipped app — the release bundles its own. To match what the app ships:

    pip install llama-cpp-python --force-reinstall \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan

For the full six-problem gate use scripts/bench_numeric.py; this is the quick
one you run after changing an engine setting.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench_numeric import SYSTEM, TESTS, ModelSpec, build_backend, grade, run_suite  # noqa: E402

#: Measured in this repo on the shipped 4B Q4_K_M, CPU, 6 physical cores, with
#: the tuned settings (n_batch=1024). The reference the speedup is quoted
#: against, so a number here means something rather than floating free.
CPU_BASELINE_DECODE = 6.97
CPU_BASELINE_PREFILL = 55.0

#: Three of the six gate problems: the ones whose failures would be the engine's
#: fault rather than the model's. The two the model gets wrong for knowledge
#: reasons (beam_deflection, beam_stress) are deliberately left out — they fail
#: on CPU too, and a smoke test that always shows red teaches you to ignore it.
QUICK = ("rc_time_constant", "projectile", "heat_conduction")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default=None)
    parser.add_argument("--model", default="vasudha")
    parser.add_argument("--num-ctx", type=int, default=4096)
    args = parser.parse_args()

    spec = ModelSpec(label=args.gguf or args.model, model=args.model,
                     gguf=args.gguf, num_ctx=args.num_ctx)

    print("engine")
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as _c
        gpu = bool(_c.llama_supports_gpu_offload())
        print(f"  wheel             llama_cpp {llama_cpp.__version__}")
        print(f"  gpu offload       {'SUPPORTED' if gpu else 'NOT SUPPORTED (CPU-only wheel)'}")
    except ImportError:
        gpu = False
        print("  wheel             llama-cpp-python not installed")

    started = time.time()
    backend = build_backend(spec, quiet=True)
    print(f"  backend           {backend.display_name}")
    print(f"  load              {time.time() - started:.1f}s")

    # ── speed ────────────────────────────────────────────────────────────────
    print("\nspeed")
    prefill_tps = None
    if hasattr(backend, "count_tokens") and args.gguf:
        filler = ("You are a careful engineering assistant. " * 200)
        n = backend.count_tokens(filler)
        t = time.time()
        backend._llm.reset()
        backend._llm.eval(backend._llm.tokenize(filler.encode(), special=True))
        dt = time.time() - t
        prefill_tps = n / dt
        print(f"  prefill           {prefill_tps:7.1f} tok/s"
              f"   ({prefill_tps / CPU_BASELINE_PREFILL:.1f}x the CPU baseline)")

    # ── accuracy ─────────────────────────────────────────────────────────────
    print("\naccuracy   (greedy; ground truth computed in Python, not recalled)")
    tests = [t for t in TESTS if t["id"] in QUICK]
    correct = fired = 0
    ttfts = []
    import bench_numeric
    original, bench_numeric.TESTS = bench_numeric.TESTS, tests
    try:
        for r in run_suite(spec, {"num_predict": 512, "num_ctx": args.num_ctx,
                                  "temperature": 0.0, "top_p": 1, "top_k": 1, "seed": 0},
                           backend=backend, quiet=True):
            good, got = grade(r.final, r.test["answer"], r.test["tol"])
            correct += good
            fired += r.calls > 0
            if r.ttft is not None:
                ttfts.append(r.ttft)
            print(f"  [{'PASS' if good else 'FAIL'}] {r.test['id']:<18} "
                  f"got={got!s:<20} want={r.test['answer']}  {r.elapsed:.0f}s")
    finally:
        bench_numeric.TESTS = original

    decode_tps = getattr(backend, "observed_tps", None)
    if decode_tps:
        print(f"\n  decode            {decode_tps:7.1f} tok/s"
              f"   ({decode_tps / CPU_BASELINE_DECODE:.1f}x the CPU baseline)")
    if ttfts:
        print(f"  first token       {min(ttfts):7.1f}s")

    # ── verdict ──────────────────────────────────────────────────────────────
    total = len(tests)
    print(f"\n  --> {correct}/{total} correct, tool fired {fired}/{total}")
    ok = correct == total and fired == total

    print("\nVERDICT")
    if not gpu:
        print("  Running on CPU. Expect ~7 tok/s. See the docstring for the Vulkan wheel.")
    elif decode_tps and decode_tps > CPU_BASELINE_DECODE * 2:
        print(f"  GPU active and {decode_tps / CPU_BASELINE_DECODE:.1f}x faster than CPU.")
    elif gpu:
        print("  GPU wheel present but the speed looks like CPU — check that layers were "
              "actually offloaded (n_gpu_layers) and that the model fits in VRAM.")
    if ok:
        print("  Answers correct and the tool fired every time. Engine is good.")
    elif fired < total:
        print(f"  TOOL PATH REGRESSED: fired {fired}/{total}. This is a harness bug, "
              "not a model one — fix before shipping.")
    else:
        print(f"  ACCURACY REGRESSED: {correct}/{total}. If this passes on CPU and fails "
              "here, the GPU kernels are the difference — do not ship it.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
