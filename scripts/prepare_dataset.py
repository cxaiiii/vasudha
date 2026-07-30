"""
Materialize a fixed slice of the streaming mixture to disk, once.

Streaming keeps a shuffle buffer of raw rows in host RAM, and OpenThoughts-style
reasoning chains are tens of KB each — that buffer is what OOM-kills a 12GB
Colab container. Draining a bounded number of samples to an Arrow dataset makes
training memory-flat and restarts instant, at the cost of one upfront pass.

Usage:
    python scripts/prepare_dataset.py --out ./data/sft-20k --max-samples 20000
    python scripts/train_sft.py ... data.prepared_path=./data/sft-20k
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vasudha.utils.logging import get_logger, log_banner, setup_logging

logger = get_logger(__name__)

DEFAULT_SOURCES = {
    "open-thoughts/OpenThoughts3-1.2M": 0.40,
    "AI-MO/NuminaMath-TIR": 0.35,
    "open-r1/OpenR1-Math-220k": 0.25,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory for the Arrow dataset")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=500,
        help="Shuffle buffer during the drain. Small on purpose — this is the "
             "knob that OOMs; the on-disk result is shuffled again at train time.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=24000,
        help="Skip rows longer than this. At ~4 chars/token they exceed any "
             "max_seq_length we train at and would only be truncated anyway.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(level="INFO")
    log_banner("Dataset Preparation", f"{args.max_samples} samples → {args.out}")

    from datasets import Dataset

    from vasudha.datasets import build_mixture

    mixture = build_mixture(
        sources=DEFAULT_SOURCES,
        streaming=True,
        seed=args.seed,
        buffer_size=args.buffer_size,
    )

    rows: list[dict[str, str]] = []
    skipped_long = 0
    skipped_empty = 0

    # islice over a generous window so filtering can't leave us short.
    for row in itertools.islice(mixture, args.max_samples * 3):
        text = row.get("text")
        if not text or not text.strip():
            skipped_empty += 1
            continue
        if len(text) > args.max_chars:
            skipped_long += 1
            continue
        rows.append({"text": text})
        if len(rows) >= args.max_samples:
            break
        if len(rows) % 2000 == 0:
            logger.info(f"  {len(rows)}/{args.max_samples} collected")

    if not rows:
        raise SystemExit("No usable rows collected — check the mixture config.")

    ds = Dataset.from_list(rows)
    ds = ds.shuffle(seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))

    avg = sum(len(r["text"]) for r in rows) / len(rows)
    log_banner("Preparation Complete", str(out))
    logger.info(
        f"{len(rows)} rows | avg {avg:,.0f} chars | "
        f"skipped {skipped_long} too-long, {skipped_empty} empty"
    )
    logger.info(f"Train with: data.prepared_path={out}")


if __name__ == "__main__":
    main()
