"""
Builds a frontend/render_tool training addition from HuggingFaceM4/WebSight —
real, browser-rendered HTML (each sample was actually rendered to produce its
paired screenshot in the source dataset, so it's real verified structure, not
another LLM's untested guess), reformatted to match this project's actual
production behavior exactly.

Two concrete fixes applied to every sample, not just a raw dump:
1. WebSight loads Tailwind via an old versioned <link> to a CSS file
   (tailwindcss@2.2.19/dist/tailwind.min.css). web/app.py's SYSTEM_PROMPT
   mandates a specific different convention: <script src="https://
   cdn.tailwindcss.com"></script>. Training on WebSight's raw HTML verbatim
   would teach an inconsistent loading pattern from what's actually deployed
   — normalized here instead.
2. Every sample is validated to actually parse as well-formed HTML
   (BeautifulSoup) before being kept — this project's whole distillation
   philosophy has been "verify, don't just trust the source," and that
   applies to a public dataset's samples exactly as much as to a generated
   candidate.

Deliberately spans the full length distribution WebSight offers (not
filtered down to only short/simple pages) — the model needs exposure to
real layout complexity, not just toy examples.

No API cost: WebSight is a public HF dataset, and formatting is local only.

Usage:
    python scripts/prepare_frontend_dataset.py --out ./data/frontend-v1 --max-samples 1500
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vasudha.utils.logging import get_logger, log_banner, setup_logging

logger = get_logger(__name__)

_OLD_TAILWIND_LINK_RE = re.compile(
    r'<link[^>]*tailwindcss[^>]*\.css[^>]*rel="stylesheet"[^>]*>', re.IGNORECASE
)
_TAILWIND_SCRIPT_TAG = '<script src="https://cdn.tailwindcss.com"></script>'

# WebSight's synthetic image URLs point at Unsplash's old "Source" API
# (source.unsplash.com/random/WxH/?query) — verified dead (HTTP 503) before
# building this dataset, not assumed. Training on it verbatim would just
# teach the model a new, different flavor of "confidently reference a
# broken image," the same failure class already found in live testing.
# placehold.co is a real, purpose-built, currently-working placeholder
# service (verified HTTP 200) — swapped in with the same WxH the original
# request asked for.
_UNSPLASH_SOURCE_RE = re.compile(
    r"https://source\.unsplash\.com/random/(\d+)x(\d+)/?\?[^\s\"'>]*"
)

# Matches web/app.py's own worked example in SYSTEM_PROMPT exactly, so this
# training data reinforces the same pattern the model is already told to
# follow — not a competing convention.
_RENDER_TEMPLATE = """Plan: {plan}
<render_tool>
```html
{html}
```
</render_tool>
Here's your webpage."""


def _normalize_html(raw_html: str) -> str:
    html = raw_html.strip()
    html = _OLD_TAILWIND_LINK_RE.sub(_TAILWIND_SCRIPT_TAG, html)
    html = _UNSPLASH_SOURCE_RE.sub(lambda m: f"https://placehold.co/{m.group(1)}x{m.group(2)}", html)
    if "cdn.tailwindcss.com" not in html and "<head>" in html:
        html = html.replace("<head>", f"<head>\n{_TAILWIND_SCRIPT_TAG}", 1)
    elif "cdn.tailwindcss.com" not in html:
        # No <head> at all in this sample — WebSight sometimes omits it.
        # Prepend the script tag right after <html...> so it still loads.
        html = re.sub(r"(<html[^>]*>)", rf"\1\n{_TAILWIND_SCRIPT_TAG}", html, count=1)
    if not html.lstrip().lower().startswith("<!doctype"):
        html = "<!DOCTYPE html>\n" + html
    return html


def _short_plan(idea: str) -> str:
    """A short, section-oriented plan line derived from WebSight's own
    design-brief field, matching the "2-4 lines listing actual structure"
    rule in SYSTEM_PROMPT rather than restating the whole brief verbatim."""
    first_sentence = idea.split(". ")[0].strip().rstrip(".")
    return first_sentence[:180]


def _to_instruction(idea: str) -> str:
    """WebSight's `llm_generated_idea` reads as a design narrative, not a
    user request — light templating turns it into a plausible ask without
    an extra LLM call (this whole prep step stays free)."""
    return f"Build me a webpage: {idea}"


def _valid_html(html: str) -> bool:
    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return False
    # A real page needs at least a body with some content — an empty parse
    # (e.g. the parser silently gave up) isn't usable training data.
    return soup.find("body") is not None and len(soup.get_text(strip=True)) > 0


def _load_existing(rows_path: Path) -> list[dict]:
    if not rows_path.exists():
        return []
    rows = []
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed checkpoint line in %s", rows_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=6500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream-retries", type=int, default=8,
                         help="HF Hub streaming for this dataset is genuinely flaky "
                              "('client has been closed' mid-shard) — restarts the "
                              "stream from the last scanned position rather than "
                              "losing all progress, like this project's other data scripts.")
    args = parser.parse_args()

    setup_logging(level="INFO")
    log_banner("Frontend Dataset Prep", f"{args.max_samples} samples -> {args.out}")

    from datasets import load_dataset
    from vasudha.datasets.chat_format import ChatFormatter

    formatter = ChatFormatter(tokenizer=None, format="qwen3")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"

    rows = _load_existing(rows_path)
    skipped_invalid = 0
    skipped_length = 0
    seen = 0
    logger.info("Resuming with %d rows already collected", len(rows))

    rows_file = rows_path.open("a", encoding="utf-8", buffering=1)
    try:
        attempt = 0
        while len(rows) < args.max_samples and attempt <= args.stream_retries:
            ds = load_dataset("HuggingFaceM4/WebSight", split="train", streaming=True)
            skip_to = seen  # deterministic shard order — re-scan up to where we left off
            try:
                for idx, row in enumerate(ds):
                    if idx < skip_to:
                        continue
                    seen = idx + 1
                    html = row.get("text") or ""
                    idea = (row.get("llm_generated_idea") or "").strip()
                    if not html or not idea:
                        continue
                    if not (args.min_chars <= len(html) <= args.max_chars):
                        skipped_length += 1
                        continue

                    normalized = _normalize_html(html)
                    if not _valid_html(normalized):
                        skipped_invalid += 1
                        continue

                    instruction = _to_instruction(idea)
                    assistant_text = _RENDER_TEMPLATE.format(plan=_short_plan(idea), html=normalized)
                    messages = [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": assistant_text},
                    ]
                    text = formatter.format_messages(messages)
                    record = {"text": text, "source": "frontend", "tag": "websight_render",
                              "detail": f"{len(html)}chars"}
                    rows_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rows_file.flush()
                    rows.append(record)

                    if len(rows) % 200 == 0:
                        logger.info("  collected %d/%d (scanned %d, skipped: %d invalid, %d length)",
                                    len(rows), args.max_samples, seen, skipped_invalid, skipped_length)
                    if len(rows) >= args.max_samples:
                        break
                break  # completed the full pass without a stream error
            except Exception as exc:
                attempt += 1
                logger.warning(
                    "Stream interrupted at scanned=%d (%s) — durable rows are safe, "
                    "restarting stream and resuming from there (attempt %d/%d)",
                    seen, type(exc).__name__, attempt, args.stream_retries,
                )
                if attempt > args.stream_retries:
                    logger.warning("Retry budget exhausted — keeping the %d rows collected so far.", len(rows))
    finally:
        rows_file.close()

    logger.info(
        "Final: %d kept, %d skipped (invalid HTML), %d skipped (length range), %d total scanned",
        len(rows), skipped_invalid, skipped_length, seen,
    )

    from datasets import Dataset

    dataset = Dataset.from_list(rows).shuffle(seed=args.seed)
    dataset.save_to_disk(str(out / "arrow"))

    length_dist = sorted(int(r["detail"].replace("chars", "")) for r in rows)
    with (out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "samples": len(rows),
                "skipped_invalid_html": skipped_invalid,
                "skipped_length_range": skipped_length,
                "scanned": seen,
                "length_min": length_dist[0] if length_dist else None,
                "length_median": length_dist[len(length_dist) // 2] if length_dist else None,
                "length_max": length_dist[-1] if length_dist else None,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )
    log_banner("Frontend Dataset Ready", str(out / "arrow"))


if __name__ == "__main__":
    main()
