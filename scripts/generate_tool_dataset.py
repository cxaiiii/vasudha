"""
Resumable tool-use / research trajectory generator — teaches Vasudha to
actually call python_tool/search_tool/fetch_tool well, using web/app.py's
real SYSTEM_PROMPT and web/tools.py's real tool implementations so training
data matches production behavior exactly (see
vasudha/datasets/tool_trajectory.py for why this is a simpler, cheaper
pipeline than the reasoning-trace one: no candidates/judge needed when every
tool call is a real execution, not a guess).

DeepSeek V4 Flash only, deliberately — this doesn't need the reasoning tier,
and Flash's non-reasoning billing is what makes this affordable. Real cost
tracking (from the API's own `usage` field) and a hard --max-cost cutoff are
on by default this time, learned the hard way from the reasoning pipeline's
first run: a naive per-example cost estimate is not something to trust until
it's been checked against actual `usage` numbers from a tiny live batch.

Usage:
    python scripts/generate_tool_dataset.py --out ./data/tool-pilot --max-samples 10 --max-cost 0.50
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from vasudha.utils.logging import get_logger, log_banner, setup_logging
from vasudha.datasets.tool_trajectory import generate_trajectory, flatten_trajectory, Trajectory

logger = get_logger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL_FLASH = "deepseek-v4-flash"

# $ per 1M tokens (input, output) — same figures generate_reasoning_dataset.py
# uses. Flash-only here by design, so there's no hidden-reasoning-token
# surprise the way deepseek-v4-pro had, but the usage field is still tracked
# for real rather than assumed, after that exact assumption cost real money
# on the other pipeline.
PRICING = {MODEL_FLASH: (0.14, 0.28)}

_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))

_cost_lock = threading.Lock()
_cost_total = 0.0


def _track_cost(model: str, usage: dict) -> None:
    global _cost_total
    prices = PRICING.get(model)
    if not prices or not usage:
        return
    in_price, out_price = prices
    cost = (usage.get("prompt_tokens", 0) * in_price + usage.get("completion_tokens", 0) * out_price) / 1e6
    with _cost_lock:
        _cost_total += cost


def total_cost() -> float:
    with _cost_lock:
        return _cost_total


def _api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set.")
    return key


def _call_deepseek(messages: list[dict], temperature: float, retries: int = 5) -> str:
    """Retries on 429/5xx/timeout only. A 4xx other than 429 (bad request,
    bad auth, insufficient balance) will never succeed no matter how many
    times it's retried — the reasoning pipeline's first real run burned
    through a huge, confusing request count doing exactly that blindly."""
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    payload = {"model": MODEL_FLASH, "messages": messages, "temperature": temperature, "stream": False}
    attempt = 0
    while True:
        try:
            resp = _session.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=120)
        except requests.exceptions.RequestException as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"DeepSeek call failed after {retries} retries: {exc}") from exc
            delay = min(60, 2 ** attempt) + random.random()
            logger.warning("Network error (%s); retry %d/%d in %.1fs", type(exc).__name__, attempt, retries, delay)
            time.sleep(delay)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"DeepSeek call failed after {retries} retries: HTTP {resp.status_code}")
            delay = min(60, 2 ** attempt) + random.random()
            logger.warning("HTTP %d; retry %d/%d in %.1fs", resp.status_code, attempt, retries, delay)
            time.sleep(delay)
            continue

        if not resp.ok:
            # Non-retryable: bad request, bad auth, insufficient balance, etc.
            raise RuntimeError(f"DeepSeek call failed with non-retryable HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        _track_cost(MODEL_FLASH, data.get("usage") or {})
        return data["choices"][0]["message"]["content"]


def _extract_json_array(raw: str) -> list:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


QUESTION_CATEGORIES: dict[str, float] = {
    "computation": 0.35,
    "current_facts": 0.35,
    "multi_hop": 0.30,
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "computation": "a question requiring a nontrivial numeric calculation, algorithm, or simulation to "
                   "answer correctly (unit conversions, combinatorics, physics/engineering calculations, "
                   "statistics, algorithmic verification) — NOT a question needing outside facts",
    "current_facts": "a question about a specific, verifiable real-world fact that requires looking it up "
                      "rather than recalling from memory (recent events, exact figures, specific dates, "
                      "current records, prices, version numbers) — NOT a question needing calculation",
    "multi_hop": "a question requiring combining a web search with a computation, or looking up two "
                 "separate real facts and comparing/combining them (e.g. a ratio, difference, or sum of "
                 "two real-world figures that must each be looked up)",
}

QUESTION_PROMPT_TEMPLATE = """Generate {n} realistic, diverse questions of this kind: {description}

Requirements:
- Concrete and specific, not generic (e.g. not "tell me about a country's economy" but "what was Japan's GDP growth rate in the most recently reported quarter").
- A real user would plausibly ask this.
- No two questions should be about the same topic.
Return ONLY a JSON array of {n} question strings, no markdown fences, no commentary, no numbering."""


def _generate_questions(category: str, n: int) -> list[str]:
    prompt = QUESTION_PROMPT_TEMPLATE.format(n=n, description=CATEGORY_DESCRIPTIONS[category])
    raw = _call_deepseek([{"role": "user", "content": prompt}], temperature=1.0)
    questions = _extract_json_array(raw)
    return [str(q).strip() for q in questions if str(q).strip()]


def _question_pool(quotas: dict[str, int], pool_path: Path) -> dict[str, list[str]]:
    """Durable, resumable question pool — a separate checkpoint from
    trajectories so a rerun never re-pays to regenerate questions it
    already has, even if trajectory generation itself was interrupted."""
    pool: dict[str, list[str]] = {cat: [] for cat in quotas}
    if pool_path.exists():
        with pool_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                pool.setdefault(row["category"], []).append(row["question"])

    with pool_path.open("a", encoding="utf-8", buffering=1) as handle:
        for category, target in quotas.items():
            have = len(pool[category])
            while have < target:
                batch_n = min(10, target - have)
                try:
                    new_questions = _generate_questions(category, batch_n)
                except Exception as exc:
                    logger.warning("Question generation failed for %s, stopping this category: %s", category, exc)
                    break
                for q in new_questions:
                    handle.write(json.dumps({"category": category, "question": q}, ensure_ascii=False) + "\n")
                    handle.flush()
                    pool[category].append(q)
                    have += 1
                if not new_questions:
                    break  # avoid an infinite loop if the model returns nothing usable
    return pool


def generate_one_trajectory(question: str, category: str, max_iterations: int) -> dict | None:
    def call_fn(messages: list[dict]) -> str:
        return _call_deepseek(messages, temperature=0.7)

    trajectory: Trajectory | None = generate_trajectory(question, call_fn, max_iterations=max_iterations)
    if trajectory is None:
        return None
    flat = flatten_trajectory(trajectory)
    flat["category"] = category
    flat["question"] = question
    return flat


def _quotas(profiles: dict[str, float], total: int) -> dict[str, int]:
    weights = sum(profiles.values())
    raw = {name: total * weight / weights for name, weight in profiles.items()}
    result = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(result.values())
    for name in sorted(profiles, key=lambda n: raw[n] - result[n], reverse=True)[:remainder]:
        result[name] += 1
    return result


def _load_existing(rows_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not rows_path.exists():
        return counts
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                counts[json.loads(line)["category"]] += 1
            except (json.JSONDecodeError, KeyError):
                logger.warning("Ignoring malformed checkpoint line in %s", rows_path)
    return counts


def _chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            return
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=10, help="Total trajectories to generate")
    parser.add_argument("--max-tool-iterations", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=10,
                         help="Trajectories generated in parallel (each trajectory's own turns are "
                              "inherently sequential, so this only parallelizes across trajectories).")
    parser.add_argument("--max-cost", type=float, default=0.50,
                         help="Hard stop once observed spend reaches this many dollars. On by default "
                              "after the reasoning pipeline's first run went uncontrolled.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(level="INFO")
    _api_key()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    pool_path = out / "questions.jsonl"

    quotas = _quotas(QUESTION_CATEGORIES, args.max_samples)
    log_banner("Tool-Use Trajectory Generation", f"{args.max_samples} samples -> {out}")
    logger.info("Cost cutoff: $%.2f", args.max_cost)

    logger.info("Building/resuming question pool: %s", quotas)
    pool = _question_pool(quotas, pool_path)
    logger.info("Question pool ready. Cost so far (question gen): $%.4f", total_cost())

    counts = _load_existing(rows_path)
    logger.info("Existing durable trajectories: %d", sum(counts.values()))

    tasks: list[tuple[str, str]] = []  # (category, question)
    for category, target in quotas.items():
        remaining = target - counts[category]
        if remaining <= 0:
            continue
        available = pool.get(category, [])[counts[category]:counts[category] + remaining]
        tasks.extend((category, q) for q in available)

    if tasks:
        logger.info("Dispatching %d trajectories across %d concurrent workers", len(tasks), args.concurrency)

    completed = 0
    discarded = 0
    with rows_path.open("a", encoding="utf-8", buffering=1) as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
            for chunk in _chunked(tasks, args.concurrency):
                if total_cost() >= args.max_cost:
                    logger.warning(
                        "Stopping: observed cost $%.4f has reached --max-cost $%.2f (%d/%d done)",
                        total_cost(), args.max_cost, completed, len(tasks),
                    )
                    break
                futures = {
                    pool_exec.submit(generate_one_trajectory, question, category, args.max_tool_iterations): category
                    for category, question in chunk
                }
                for fut in as_completed(futures):
                    category = futures[fut]
                    try:
                        flat = fut.result()
                    except Exception as exc:
                        logger.warning("Skipping a %s trajectory after failure: %s", category, exc)
                        continue
                    if flat is None:
                        discarded += 1
                        logger.info("  Discarded an unresolved %s trajectory (hit iteration cap)", category)
                        continue
                    handle.write(json.dumps(flat, ensure_ascii=False) + "\n")
                    handle.flush()
                    counts[category] += 1
                    completed += 1
                    logger.info(
                        "  [%d/%d] %s: %d tool calls | cost so far: $%.4f",
                        completed, len(tasks), category, len(flat["tool_tags_used"]), total_cost(),
                    )

    from datasets import Dataset

    rows = []
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))

    if rows:
        dataset = Dataset.from_list(rows).shuffle(seed=args.seed)
        dataset.save_to_disk(str(out / "arrow"))

    with (out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "samples": len(rows),
                "discarded_unresolved": discarded,
                "counts": dict(counts),
                "quotas": quotas,
                "observed_cost_usd": round(total_cost(), 4),
                "seed": args.seed,
            },
            handle,
            indent=2,
        )
    log_banner("Generation Complete", str(out / "arrow"))
    logger.info("Total observed cost: $%.4f", total_cost())


if __name__ == "__main__":
    main()
