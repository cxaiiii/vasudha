"""Run evaluation suites on a model and write a report.

    python -m neo.evaluate --model /vol/neo/models/neo-final-text --out /vol/neo/evals/final \
        --suites fast --modes nothink,think

Writes per-episode JSONL (for reading actual replies, not just numbers),
summary.json, and a markdown table. compare() merges several summaries into
the side-by-side table used in the model card.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from typing import Optional

from neo.evalsets import FAST_SUITES, THINKING_LIMITS, THINKING_SUITES, load_suite
from neo.rollout import RolloutEngine, expand
from neo.verify import score_episode_safe

MODE_SETTINGS = {
    # Greedy for the fast mode (deterministic); Qwen's recommended sampling for
    # thinking, where greedy decoding loops.
    "nothink": dict(thinking=False, temperature=0.0, top_p=1.0, top_k=-1, max_tokens=2048),
    "think": dict(thinking=True, temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384),
}


def score_all(episodes) -> list[dict]:
    out = []
    for ep in episodes:
        s = score_episode_safe(ep.task, ep.completion)
        out.append({"id": ep.row["id"], "source": ep.row["source"], "env": ep.row.get("environment", "text"),
                    "correct": bool(s.correct), "reward": float(s.reward), "metrics": s.metrics,
                    "tokens": ep.tokens, "thinking_tokens": ep.thinking_tokens, "turns": ep.turns,
                    "finish": ep.finish, "completion": ep.completion})
    return out


def summarize(records: list[dict]) -> dict:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_source[r["source"].split("/")[0]].append(r)
    groups = {}
    for key, rs in sorted(by_source.items()):
        groups[key] = {
            "n": len(rs),
            "accuracy": sum(r["correct"] for r in rs) / len(rs),
            "mean_tokens": statistics.fmean(r["tokens"] for r in rs),
            "tool_use": statistics.fmean(r["metrics"].get("tool_used", 0.0) for r in rs),
            "empty": statistics.fmean(r["metrics"].get("empty_reply", 0.0) for r in rs),
        }
    return {"n": len(records), "accuracy": sum(r["correct"] for r in records) / max(1, len(records)),
            "mean_tokens": statistics.fmean(r["tokens"] for r in records) if records else 0.0, "groups": groups}


def run(model: str, out_dir: str, suites: list[str], modes: list[str], cache_dir: Optional[str] = None,
        limit: Optional[int] = None, gpu_memory_utilization: float = 0.85, max_model_len: int = 20480,
        sandbox_workers: int = 8, log=print) -> dict:
    from vllm import LLM

    from neo.rollout import VLLMBackend
    from neo.sandbox import SandboxPool

    os.makedirs(out_dir, exist_ok=True)
    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              enable_prefix_caching=True, seed=0, **llm_kwargs(model))
    tokenizer = llm.get_tokenizer()
    pool = SandboxPool(workers=sandbox_workers, timeout=15)
    summary: dict = {"model": model, "results": {}}
    # Resume: suites finished by an earlier, interrupted run are kept, not re-run.
    done_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as fh:
            previous = json.load(fh)
        summary["results"] = {k: v for k, v in previous.get("results", {}).items()
                              if os.path.exists(os.path.join(out_dir, k.replace("/", "-") + ".jsonl"))}
        if summary["results"]:
            log(f"resuming: {len(summary['results'])} suites already scored")
    try:
        for mode in modes:
            cfg = MODE_SETTINGS[mode]
            mode_suites = suites if mode == "nothink" else [s for s in suites if s in THINKING_SUITES]
            engine = RolloutEngine(VLLMBackend(llm), tokenizer, pool, thinking=cfg["thinking"], max_turns=4,
                                   max_tokens=cfg["max_tokens"], max_model_len=max_model_len,
                                   temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"], seed=0)
            for suite in mode_suites:
                if f"{mode}/{suite}" in summary["results"]:
                    continue
                suite_limit = limit or (THINKING_LIMITS.get(suite) if mode == "think" else None)
                rows = load_suite(suite, suite_limit, cache_dir)
                started = time.time()
                episodes = engine.run(expand(rows, 1))
                records = score_all(episodes)
                summ = summarize(records)
                summ["seconds"] = time.time() - started
                summary["results"][f"{mode}/{suite}"] = summ
                with open(os.path.join(out_dir, f"{mode}-{suite}.jsonl"), "w", encoding="utf-8") as fh:
                    for rec in records:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log(f"[{mode}] {suite:14s} acc={summ['accuracy']:.4f} n={summ['n']} "
                    f"tok={summ['mean_tokens']:.0f} ({summ['seconds']:.0f}s)")
                with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
                    json.dump(summary, fh, indent=2)
    finally:
        pool.close()
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(markdown([("model", summary)]))
    return summary


def markdown(named: list[tuple[str, dict]]) -> str:
    """Side-by-side accuracy table for one or more summaries."""
    keys = sorted({k for _, s in named for k in s["results"]})
    lines = ["| suite | " + " | ".join(n for n, _ in named) + " |", "|---|" + "---|" * len(named)]
    for k in keys:
        cells = []
        for _, s in named:
            r = s["results"].get(k)
            cells.append(f"{100 * r['accuracy']:.1f} ({r['mean_tokens']:.0f} tok)" if r else "—")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def compare(paths: dict[str, str], out_path: str) -> str:
    named = []
    for name, path in paths.items():
        p = os.path.join(path, "summary.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                named.append((name, json.load(fh)))
    text = markdown(named) if named else "no results\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def calibrate(model: str, rows: list[dict], k: int = 4, max_tokens: int = 1536, max_model_len: int = 6144,
              gpu_memory_utilization: float = 0.85, log=print) -> list[float]:
    """Pass rate of each RL task under the current policy (k samples, T=1).
    Tasks the model always or never solves carry no GRPO signal."""
    from vllm import LLM

    from neo.rollout import VLLMBackend
    from neo.sandbox import SandboxPool

    llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              enable_prefix_caching=True, seed=0, **llm_kwargs(model))
    pool = SandboxPool(workers=8, timeout=15)
    try:
        engine = RolloutEngine(VLLMBackend(llm), llm.get_tokenizer(), pool, thinking=False, max_turns=4,
                               max_tokens=max_tokens, max_model_len=max_model_len, temperature=1.0, top_p=1.0,
                               top_k=-1, seed=1234)
        started = time.time()
        episodes = engine.run(expand(rows, k))
        log(f"calibration rollouts: {len(episodes)} in {time.time() - started:.0f}s")
        scores = [score_episode_safe(ep.task, ep.completion).correct for ep in episodes]
    finally:
        pool.close()
    return [sum(scores[i * k:(i + 1) * k]) / k for i in range(len(rows))]


def select_by_pass_rate(rows: list[dict], rates: list[float], n: int, seed: int = 0) -> list[dict]:
    """Prefer tasks with 0 < p < 1. If those run short, top up with some
    never-solved tasks (they may become solvable as training proceeds) and a
    few always-solved ones (they keep the easy skills anchored)."""
    import random

    rng = random.Random(seed)
    mid = [r for r, p in zip(rows, rates) if 0 < p < 1]
    zero = [r for r, p in zip(rows, rates) if p == 0]
    one = [r for r, p in zip(rows, rates) if p == 1]
    for group in (mid, zero, one):
        rng.shuffle(group)
    picked = mid[:n]
    take_zero = min(len(zero), n - len(picked), max(1, int(0.12 * n)))
    picked += zero[:take_zero]
    take_one = min(len(one), n - len(picked), max(1, int(0.08 * n)))
    picked += one[:take_one]
    picked += (zero[take_zero:] + one[take_one:])[: n - len(picked)]
    rng.shuffle(picked)
    return picked


def llm_kwargs(model: str) -> dict:
    """vLLM rejects multimodal limits on text-only checkpoints."""
    try:
        with open(os.path.join(model, "config.json"), encoding="utf-8") as fh:
            multimodal = "vision_config" in json.load(fh)
    except OSError:
        multimodal = True        # a hub id such as Qwen/Qwen3.5-4B
    return {"limit_mm_per_prompt": {"image": 0, "video": 0}} if multimodal else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="fast", help="'fast' or a comma list")
    ap.add_argument("--modes", default="nothink,think")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    suites = FAST_SUITES + ["aime25"] if args.suites == "fast" else args.suites.split(",")
    run(args.model, args.out, suites, args.modes.split(","), args.cache, args.limit)


if __name__ == "__main__":
    main()
