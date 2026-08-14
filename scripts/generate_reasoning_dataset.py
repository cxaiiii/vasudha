"""
Resumable synthetic reasoning-trace dataset generator — distills expert
reasoning traces from DeepSeek instead of pulling QA pairs from HF Hub.

Mirrors scripts/prepare_dataset.py's proven pattern (durable append-only
JSONL checkpoint, resume-on-rerun, retry/backoff, largest-remainder domain
quotas) but calls the DeepSeek API instead of streaming HF datasets. Runs
locally — this is pure API orchestration with no GPU need, so it doesn't go
through Modal.

Per example: 3 independently-generated candidate solutions (two from the
reasoning-tier model at different temperatures, one deliberately from the
cheaper/weaker model so its flaws are real rather than fabricated), then one
blind judge pass that critiques all three. See
C:\\Users\\saxen\\.claude\\plans\\hidden-sleeping-petal.md for full design
rationale.

Usage:
    python scripts/generate_reasoning_dataset.py --out ./data/reasoning-pilot --max-samples 20
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from vasudha.utils.logging import get_logger, log_banner, setup_logging
from vasudha.datasets.reasoning_schema import (
    Candidate,
    CandidateCritique,
    JudgeVerdict,
    ReasoningRecord,
    DOMAIN_PROFILES,
    PROBLEM_PROMPT_TEMPLATE,
    SOLUTION_PROMPT_TEMPLATE,
    STRATEGY_HINTS,
    JUDGE_PROMPT_TEMPLATE,
    domain_description,
)
from vasudha.datasets.reasoning_verify import verify_record
from vasudha.datasets.reasoning_format import flatten_record

logger = get_logger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL_PRO = "deepseek-v4-pro"
MODEL_FLASH = "deepseek-v4-flash"

# Kimi K3 is optional diversification for candidate B — real cross-provider,
# cross-architecture reasoning diversity (different training lineage than
# DeepSeek) rather than just the same model at a different temperature.
# Deliberately NOT used for the cheap utility calls (problem generation,
# judging): every K3 response bills its full output rate as reasoning
# tokens (no cheap non-reasoning tier), so routing lightweight calls through
# it would be pure waste. Falls back to MODEL_PRO for candidate B when no
# MOONSHOT_API_KEY is set — the pipeline works with just a DeepSeek key.
KIMI_URL = "https://api.moonshot.ai/v1/chat/completions"
MODEL_KIMI = "kimi-k3"

# Domains where the final answer is a single checkable number vs. executable
# code vs. neither — decides reference_check_method and, downstream, which
# reasoning_verify function actually runs.
NUMERIC_DOMAINS = {
    "physics_derivation", "mathematics", "simulation", "electronics", "chemistry",
    "bioengineering", "metallurgy", "cad_design",
}
CODE_DOMAINS = {"programming_reasoning", "debugging"}

# A single record makes calls concurrently at a fan-out of 3 (candidates
# A/B/C), and main() runs several records concurrently on top of that — the
# pool needs enough spare connections that concurrent requests reuse a
# warm connection instead of opening a fresh TLS handshake per call.
_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))

# $ per 1M tokens, (input, output). A real pilot run showed actual spend
# ~25x the original naive estimate ($0.05/record observed vs. $0.002
# projected) — the projection only counted visible JSON-answer tokens and
# never looked at the API's own `usage` field, so it completely missed that
# deepseek-v4-pro is a reasoning-tier model billing substantial hidden
# chain-of-thought tokens as output. Tracking real usage from here on so
# spend is never invisible again.
PRICING: dict[str, tuple[float, float]] = {
    MODEL_PRO: (0.435, 0.87),
    MODEL_FLASH: (0.14, 0.28),
    MODEL_KIMI: (3.0, 15.0),
}

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
        raise SystemExit(
            "DEEPSEEK_API_KEY is not set. Create a key at https://platform.deepseek.com "
            "then set it before running this script, e.g.:\n\n"
            "    $env:DEEPSEEK_API_KEY = 'sk-...'   (PowerShell)\n"
            "    export DEEPSEEK_API_KEY=sk-...      (bash)\n"
        )
    return key


def _kimi_api_key() -> Optional[str]:
    """Optional — unlike _api_key(), returns None instead of raising when
    unset, since Kimi diversification is an enhancement, not a requirement."""
    return os.environ.get("MOONSHOT_API_KEY") or None


def _call_chat_completion(
    url: str, api_key: str, model: str, messages: list[dict], temperature: float, retries: int = 5,
) -> str:
    """POSTs one OpenAI-compatible chat completion, retrying on
    429/5xx/timeout with backoff — same retry shape as prepare_dataset.py's
    _remote_rows, applied to a paid API instead of a free Hub stream so a
    transient failure costs a retry, not the whole run (or worse, a
    re-billed duplicate call). Shared across providers since both DeepSeek
    and Moonshot expose the same OpenAI-compatible request/response shape."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "stream": False}
    attempt = 0
    while True:
        try:
            resp = _session.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.exceptions.RequestException(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            _track_cost(model, data.get("usage") or {})
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"{model} call failed after {retries} retries: {exc}") from exc
            delay = min(60, 2 ** attempt) + random.random()
            logger.warning(
                "%s call failed (%s); retry %d/%d in %.1fs",
                model, type(exc).__name__, attempt, retries, delay,
            )
            time.sleep(delay)


def _call_deepseek(model: str, messages: list[dict], temperature: float, retries: int = 5) -> str:
    return _call_chat_completion(DEEPSEEK_URL, _api_key(), model, messages, temperature, retries)


def _call_kimi(messages: list[dict], temperature: float, retries: int = 5) -> str:
    key = _kimi_api_key()
    if not key:
        raise RuntimeError("MOONSHOT_API_KEY not set — _call_kimi should not be reached without it.")
    return _call_chat_completion(KIMI_URL, key, MODEL_KIMI, messages, temperature, retries)


def _extract_json(raw: str) -> dict:
    """Teacher models sometimes wrap JSON in markdown fences despite being
    told not to — strip those rather than fail the row over formatting."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        if text.lower().startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def _generate_problem(domain: str) -> str:
    prompt = PROBLEM_PROMPT_TEMPLATE.format(domain_description=domain_description(domain))
    return _call_deepseek(MODEL_FLASH, [{"role": "user", "content": prompt}], temperature=1.0).strip()


_CANDIDATE_TEMPERATURES = {"pro_a": 0.7, "pro_b": 1.0, "flash_c": 0.9}
# Kimi K3 has no low-temperature/deterministic guidance in its docs the way
# DeepSeek does — 0.8 keeps it in the same "independent sample, no strategy
# nudge" spirit as pro_b's 1.0 without assuming DeepSeek's temperature scale
# transfers directly to a different provider.
_KIMI_TEMPERATURE = 0.8


def _generate_candidate(instruction: str, source: str, flash_only: bool = False) -> Candidate:
    prompt = SOLUTION_PROMPT_TEMPLATE.format(instruction=instruction, strategy_hint=STRATEGY_HINTS[source])
    messages = [{"role": "user", "content": prompt}]

    if flash_only:
        # Kimi is even more expensive than Pro per token ($3/$15 vs
        # $0.435/$0.87) with no cheap tier of its own, so flash_only
        # overrides Kimi routing too — there's no cheap-mode reading of
        # "use the pricier provider."
        model = MODEL_FLASH
        raw = _call_deepseek(model, messages, temperature=_CANDIDATE_TEMPERATURES[source])
    elif source == "pro_b" and _kimi_api_key():
        model = MODEL_KIMI
        raw = _call_kimi(messages, temperature=_KIMI_TEMPERATURE)
    else:
        model = MODEL_FLASH if source == "flash_c" else MODEL_PRO
        raw = _call_deepseek(model, messages, temperature=_CANDIDATE_TEMPERATURES[source])

    data = _extract_json(raw)
    return Candidate(
        source=source,
        model=model,
        concepts=data.get("concepts") or [],
        constraints=data.get("constraints") or [],
        governing_relations=data.get("governing_relations") or [],
        tradeoffs=data.get("tradeoffs") or [],
        final_answer_text=data.get("final_answer_text") or "",
        final_answer_value=data.get("final_answer_value"),
        reference_calculation=data.get("reference_calculation"),
        code=data.get("code"),
    )


def _judge_candidates(instruction: str, candidates: list[Candidate]) -> JudgeVerdict:
    order = list(candidates)
    random.shuffle(order)
    order_map = [c.source for c in order]
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction,
        candidate_1_json=json.dumps(order[0].to_dict()),
        candidate_2_json=json.dumps(order[1].to_dict()),
        candidate_3_json=json.dumps(order[2].to_dict()),
    )
    raw = _call_deepseek(MODEL_FLASH, [{"role": "user", "content": prompt}], temperature=0.0)
    data = _extract_json(raw)
    critiques = [
        CandidateCritique(
            position=int(c["position"]),
            summary=c.get("summary", ""),
            strengths=c.get("strengths") or [],
            flaws=c.get("flaws") or [],
        )
        for c in data.get("critiques", [])
    ]
    return JudgeVerdict(
        winner_position=int(data["winner"]),
        order_map=order_map,
        justification=data.get("justification", ""),
        critiques=critiques,
    )


def generate_one_record(domain: str, flash_only: bool = False) -> ReasoningRecord:
    instruction = _generate_problem(domain)

    # The 3 candidates don't depend on each other — only on `instruction` —
    # so they run concurrently instead of one after another. This is nested
    # inside main()'s own record-level pool, so peak API concurrency is
    # roughly 3x whatever --concurrency is set to.
    sources = ("pro_a", "pro_b", "flash_c")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_generate_candidate, instruction, source, flash_only): source for source in sources}
        results = {futures[fut]: fut.result() for fut in as_completed(futures)}
    candidates = [results[source] for source in sources]

    judge = _judge_candidates(instruction, candidates)

    if domain in NUMERIC_DOMAINS:
        reference_check_method = "numeric"
    elif domain in CODE_DOMAINS:
        reference_check_method = "code_tests"
    else:
        reference_check_method = "none"

    record = ReasoningRecord(
        domain=domain,
        instruction=instruction,
        candidates=candidates,
        judge=judge,
        reference_check_method=reference_check_method,
        judge_model=MODEL_FLASH,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    # code_tests verification needs test cases the teacher wasn't asked to
    # emit in this pass, so only numeric gets a real mechanical check here —
    # code_tests support is a follow-up, not silently faked as verified.
    if reference_check_method == "numeric":
        record.verification = verify_record(record)
    return record


def _quotas(profiles: dict[str, float], total: int) -> dict[str, int]:
    """Largest-remainder allocation — same logic as
    scripts/prepare_dataset.py's _quotas(), duplicated rather than imported
    since that module pulls in HF-Hub-streaming-specific dependencies this
    script doesn't need."""
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
                counts[json.loads(line)["domain"]] += 1
            except (json.JSONDecodeError, KeyError):
                logger.warning("Ignoring malformed checkpoint line in %s", rows_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory for the durable JSONL checkpoint and final Arrow dataset")
    parser.add_argument("--max-samples", type=int, default=20, help="Total records to generate across all domains")
    parser.add_argument("--self-critique-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--concurrency", type=int, default=80,
        help="Records generated in parallel. Each record briefly holds ~2 concurrent Pro calls "
             "(candidates A/B) and ~1 Flash call (candidate C, problem-gen, or judge) at a time. "
             "DeepSeek's account limits are concurrency-based, not time-window: 500 concurrent for "
             "deepseek-v4-pro, 2,500 for deepseek-v4-flash. Default of 80 keeps peak usage (~160 Pro, "
             "~80 Flash) comfortably under both ceilings (~32%% / ~3%%) while cutting wall-clock time "
             "roughly proportionally; raise it further if a run isn't hitting rate limits.",
    )
    parser.add_argument(
        "--flash-only", action="store_true",
        help="Use deepseek-v4-flash for every call, including candidates A/B (normally deepseek-v4-pro "
             "or Kimi K3). Removes the reasoning-tier candidate entirely — cheaper and much faster, "
             "but the whole point of A/B is distilling from a model that reasons better than the "
             "student; weigh that before using this for the real training corpus.",
    )
    parser.add_argument(
        "--max-cost", type=float, default=None,
        help="Stop dispatching new records once observed spend (from the API's own usage field) "
             "reaches this many dollars. Checked between batches of --concurrency records, so it's a "
             "best-effort ceiling, not a hard real-time cutoff — a batch already in flight finishes.",
    )
    args = parser.parse_args()

    setup_logging(level="INFO")
    _api_key()  # fail fast before doing any work, not after the first call
    if _kimi_api_key():
        logger.info("MOONSHOT_API_KEY detected — candidate B will use Kimi K3 for cross-provider diversity.")
    else:
        logger.info("MOONSHOT_API_KEY not set — candidate B falls back to DeepSeek V4 Pro.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"

    quotas = _quotas(DOMAIN_PROFILES, args.max_samples)
    counts = _load_existing(rows_path)
    log_banner("Reasoning Trace Generation", f"{args.max_samples} samples -> {out}")
    logger.info("Existing durable rows: %d", sum(counts.values()))

    rng = random.Random(args.seed)

    # Round-robin across domains that still need work, rather than one
    # domain's whole block at a time — a --max-cost cutoff should spread
    # thin across every domain still needing work, not exhaust the budget on
    # whichever domain happens to sit first in DOMAIN_PROFILES before ever
    # reaching the rest. (Exactly what happened once: bioengineering/
    # metallurgy/cad_design sit last in dict order, so a cap hit mid-run on
    # an earlier domain starved them completely, despite being the reason
    # for the run.)
    remaining_by_domain: dict[str, int] = {}
    for domain, target in quotas.items():
        remaining = target - counts[domain]
        if remaining <= 0:
            logger.info("%s already complete (%d/%d)", domain, counts[domain], target)
        else:
            remaining_by_domain[domain] = remaining

    tasks: list[str] = []
    domain_cycle = list(remaining_by_domain.keys())
    while domain_cycle:
        for domain in list(domain_cycle):
            tasks.append(domain)
            remaining_by_domain[domain] -= 1
            if remaining_by_domain[domain] == 0:
                domain_cycle.remove(domain)

    if tasks:
        logger.info(
            "Dispatching %d records across %d concurrent workers (peak ~%d concurrent API calls)%s",
            len(tasks), args.concurrency, args.concurrency * 3,
            " [flash-only]" if args.flash_only else "",
        )
        if args.max_cost is not None:
            logger.info("Cost cutoff: will stop dispatching new batches at $%.2f", args.max_cost)

    def _chunked(iterable, size):
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, size))
            if not chunk:
                return
            yield chunk

    completed = 0
    with rows_path.open("a", encoding="utf-8", buffering=1) as handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for chunk in _chunked(tasks, args.concurrency):
                if args.max_cost is not None and total_cost() >= args.max_cost:
                    logger.warning(
                        "Stopping: observed cost $%.4f has reached --max-cost $%.2f (%d/%d records done)",
                        total_cost(), args.max_cost, completed, len(tasks),
                    )
                    break
                futures = {pool.submit(generate_one_record, domain, args.flash_only): domain for domain in chunk}
                for fut in as_completed(futures):
                    domain = futures[fut]
                    try:
                        record = fut.result()
                    except Exception as exc:
                        logger.warning("Skipping a %s example after generation failure: %s", domain, exc)
                        continue
                    # Writes only ever happen here, on the main thread, as
                    # each future resolves — no lock needed despite
                    # concurrent generation, since worker threads never
                    # touch the file.
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                    handle.flush()
                    counts[domain] += 1
                    completed += 1
                    logger.info(
                        "  [%d/%d] %s: %d/%d | cost so far: $%.4f",
                        completed, len(tasks), domain, counts[domain], quotas[domain], total_cost(),
                    )

    # Re-read the durable checkpoint to flatten, same pattern as
    # prepare_dataset.py's tail: a process interrupt during flattening is
    # safe, the next invocation just resumes generation then re-flattens
    # from the now-complete JSONL.
    from datasets import Dataset

    flattened = []
    dropped = 0
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = ReasoningRecord.from_dict(json.loads(line))
            flat = flatten_record(record, self_critique_ratio=args.self_critique_ratio, rng=rng)
            if flat is None:
                dropped += 1
                continue
            flattened.append(flat)

    if dropped:
        logger.warning("Dropped %d record(s) that exceeded the length cap", dropped)

    dataset = Dataset.from_list(flattened).shuffle(seed=args.seed)
    dataset.save_to_disk(str(out / "arrow"))
    with (out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "samples": len(flattened),
                "dropped": dropped,
                "counts": dict(counts),
                "quotas": quotas,
                "seed": args.seed,
                "flash_only": args.flash_only,
                "observed_cost_usd": round(total_cost(), 4),
            },
            handle,
            indent=2,
        )
    log_banner("Generation Complete", str(out / "arrow"))
    logger.info("Combine into training mix with modal_app.py::combine_data, prepared_path=%s", out / "arrow")


if __name__ == "__main__":
    main()
