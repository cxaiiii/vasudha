"""Run the numeric gate over several models and print one comparison table.

The point of making app/backends.py model-agnostic was to be able to ask
whether Vasudha is actually better than the alternatives on the work it claims
to be good at. This is the thing that answers that, on the loop the desktop app
really ships rather than on a notebook.

Run it:
    # two ollama models against each other
    python scripts/compare_models.py --model vasudha-v3 --model qwen3:4b

    # a local GGUF against an ollama model
    python scripts/compare_models.py --gguf ollama/vasudha-4b-v3.gguf --model qwen3:4b

    # with and without tools, for the same model
    python scripts/compare_models.py --model vasudha-v3 --both-modes

    # from a file, one entry per line ('name' or 'path/to.gguf', # for comments)
    python scripts/compare_models.py --models-file models.txt

Every model sees an identical prompt, an identical tool schema (python_tool
only) and greedy decoding, so the only variable is the model. Results go to
--out as JSON if you want to diff two runs later.

What the columns mean:

    correct   answers within tolerance of a Python-computed ground truth
    tool      how many of the six problems triggered a tool call at all
    ttft      median seconds to the first visible token
    tok/s     decode rate the backend observed
    total     wall clock for all six problems

`tool` is the column to trust. Correctness moves by one item between runs even
at temperature 0 because llama.cpp is not bit-deterministic across model loads,
whereas tool firing was 6/6 on every run once the schema was right. A model
that scores 6/6 correct but 2/6 tool is guessing, and will not stay at 6/6.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench_numeric import (  # noqa: E402
    TESTS,
    ModelSpec,
    build_backend,
    grade,
    run_suite,
)


def _specs_from_args(args) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for name in args.model or []:
        specs.append(ModelSpec(label=name, model=name, num_ctx=args.num_ctx))
    for path in args.gguf or []:
        specs.append(ModelSpec(label=os.path.basename(path), gguf=path,
                               num_ctx=args.num_ctx))
    if args.models_file:
        with open(args.models_file, encoding="utf-8") as handle:
            for line in handle:
                entry = line.split("#", 1)[0].strip()
                if not entry:
                    continue
                # A path that exists is a GGUF; anything else is an ollama tag.
                # Guessing from the .gguf suffix instead would silently treat a
                # typo'd path as a model name and fail much later.
                if entry.lower().endswith(".gguf") or os.path.exists(entry):
                    specs.append(ModelSpec(label=os.path.basename(entry), gguf=entry,
                                           num_ctx=args.num_ctx))
                else:
                    specs.append(ModelSpec(label=entry, model=entry,
                                           num_ctx=args.num_ctx))
    return specs


def evaluate(spec: ModelSpec, options: dict, mode: str) -> dict:
    """One model, one mode, six problems. Loads and frees the model itself."""
    started = time.time()
    backend = build_backend(spec, quiet=True)
    engine = getattr(backend, "display_name", "?")

    correct = fired = 0
    ttfts, records = [], []
    try:
        for r in run_suite(spec, options, mode=mode, backend=backend, quiet=True):
            good, got = grade(r.final, r.test["answer"], r.test["tol"])
            correct += good
            fired += r.calls > 0
            if r.ttft is not None:
                ttfts.append(r.ttft)
            mark = "PASS" if good else "FAIL"
            print(f"    [{mark}] {r.test['id']:<18} got={got}  calls={r.calls}  "
                  f"{r.elapsed:.0f}s", flush=True)
            records.append(dict(id=r.test["id"], ok=bool(good), want=r.test["answer"],
                                got=got, calls=r.calls, seconds=round(r.elapsed, 1),
                                ttft=round(r.ttft, 2) if r.ttft is not None else None,
                                final=r.final))
    finally:
        # A 2-3 GB model per entry: without this a three-model comparison holds
        # all three resident and the last one loads into whatever is left.
        try:
            backend.close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass
        del backend
        gc.collect()

    return dict(
        label=spec.label, mode=mode, engine=engine,
        correct=correct, fired=fired, total=len(TESTS),
        ttft=round(statistics.median(ttfts), 1) if ttfts else None,
        tps=None, seconds=round(time.time() - started, 1), results=records,
    )


def print_table(rows: list[dict]) -> None:
    width = max([len(r["label"]) for r in rows] + [5])
    print(f"\n{'=' * (width + 52)}")
    print(f"  {'model':<{width}}  {'mode':<8} {'correct':>8} {'tool':>7} "
          f"{'ttft':>7} {'total':>8}")
    print(f"{'=' * (width + 52)}")
    for r in rows:
        total = r["total"]
        ttft = f"{r['ttft']:.0f}s" if r["ttft"] is not None else "-"
        print(f"  {r['label']:<{width}}  {r['mode']:<8} "
              f"{str(r['correct']) + '/' + str(total):>8} "
              f"{str(r['fired']) + '/' + str(total):>7} {ttft:>7} "
              f"{r['seconds']:>7.0f}s")
    print(f"{'=' * (width + 52)}")

    scored = [r for r in rows if r["mode"] == "tools"]
    if len(scored) > 1:
        best = max(scored, key=lambda r: (r["fired"], r["correct"]))
        print(f"\n  Best tool reliability: {best['label']} "
              f"({best['fired']}/{best['total']} fired, "
              f"{best['correct']}/{best['total']} correct)")
        if any(r["fired"] < r["total"] for r in scored):
            print("  A model below 6/6 on `tool` is answering from memory on at least "
                  "one problem —\n  treat its correctness score as luck, not skill.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare models on the numeric gate, through the shipping loop.")
    parser.add_argument("--model", action="append",
                        help="ollama model name; repeatable")
    parser.add_argument("--gguf", action="append",
                        help="path to a GGUF for the built-in engine; repeatable")
    parser.add_argument("--models-file", default=None,
                        help="file with one model name or GGUF path per line")
    parser.add_argument("--both-modes", action="store_true",
                        help="also run each model with no tools, for comparison")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--out", default=None, help="write full results as JSON")
    args = parser.parse_args()

    specs = _specs_from_args(args)
    if not specs:
        parser.error("give at least one --model, --gguf or --models-file")

    options = {"num_predict": 1024, "num_ctx": args.num_ctx,
               "temperature": args.temperature}
    if args.temperature == 0:
        options.update({"top_p": 1, "top_k": 1, "seed": 0})

    modes = ["tools", "notools"] if args.both_modes else ["tools"]
    estimate = len(specs) * len(modes) * len(TESTS)
    print(f"{len(specs)} model(s) x {len(modes)} mode(s) x {len(TESTS)} problems "
          f"= {estimate} generations, greedy.")
    print("On CPU that is roughly a minute each — leave it running.\n")

    rows = []
    for spec in specs:
        for mode in modes:
            print(f"  {spec.label}  [{mode}]", flush=True)
            try:
                rows.append(evaluate(spec, options, mode))
            except SystemExit as exc:
                # One unavailable model must not discard the results already
                # collected for the others.
                print(f"    skipped: {exc}", flush=True)
            print(flush=True)

    if not rows:
        print("nothing ran.")
        return 1

    print_table(rows)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
