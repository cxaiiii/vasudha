"""
Merges the reasoning-trace and tool-use datasets into one training corpus.

Deliberately combining both is the point, not an afterthought: pure
reasoning-trace examples never call a tool (teaches "answer directly from
knowledge/computation when you can"), pure tool-use trajectories always call
at least one (teaches "use search_tool/python_tool/fetch_tool well when you
actually need them"). Training on only one shape would teach either
"never touch a tool" or "always reach for one" — mixing both in the same
corpus is what keeps tool use situational rather than reflexive.

Both source datasets already share a "text" field holding the final,
correctly-formatted (Qwen3 <|im_start|>/<|im_end|>, <think> tags where
applicable) training string — nothing here re-renders anything, it only
tags provenance and concatenates.

Usage:
    python scripts/merge_datasets.py --out ./data/vasudha-v2-corpus \\
        --reasoning ./data/reasoning-pilot ./data/reasoning-500 \\
        --tool-use ./data/tool-pilot
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_reasoning(path: Path) -> list[dict]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path / "arrow"))
    return [
        {"text": row["text"], "source": "reasoning", "tag": row["domain"], "detail": row["example_type"]}
        for row in ds
    ]


def _load_tool_use(path: Path) -> list[dict]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path / "arrow"))
    return [
        {
            "text": row["text"], "source": "tool_use", "tag": row["category"],
            "detail": ",".join(row["tool_tags_used"]) or "none",
        }
        for row in ds
    ]


def _load_frontend(path: Path) -> list[dict]:
    from datasets import load_from_disk

    # scripts/prepare_frontend_dataset.py already writes rows in the exact
    # {"text","source","tag","detail"} shape used here — no field renaming
    # needed, just pass through.
    ds = load_from_disk(str(path / "arrow"))
    return [{"text": row["text"], "source": row["source"], "tag": row["tag"], "detail": row["detail"]} for row in ds]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reasoning", nargs="*", default=[], help="Reasoning-trace dataset dirs (each must have an arrow/ subdir)")
    parser.add_argument("--tool-use", nargs="*", default=[], help="Tool-use dataset dirs (each must have an arrow/ subdir)")
    parser.add_argument("--frontend", nargs="*", default=[], help="Frontend/render_tool dataset dirs (each must have an arrow/ subdir)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows: list[dict] = []
    per_source_dir: dict[str, int] = {}

    for p in args.reasoning:
        loaded = _load_reasoning(Path(p))
        per_source_dir[f"reasoning:{p}"] = len(loaded)
        rows.extend(loaded)

    for p in args.tool_use:
        loaded = _load_tool_use(Path(p))
        per_source_dir[f"tool_use:{p}"] = len(loaded)
        rows.extend(loaded)

    for p in args.frontend:
        loaded = _load_frontend(Path(p))
        per_source_dir[f"frontend:{p}"] = len(loaded)
        rows.extend(loaded)

    if not rows:
        raise SystemExit("No input rows found — check --reasoning/--tool-use paths point at finalized (arrow/-containing) dataset dirs.")

    from datasets import Dataset

    dataset = Dataset.from_list(rows).shuffle(seed=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out / "arrow"))

    source_counts = Counter(r["source"] for r in rows)
    tag_counts = Counter((r["source"], r["tag"]) for r in rows)

    stats = {
        "total": len(rows),
        "per_input_dir": per_source_dir,
        "by_source": dict(source_counts),
        "by_source_and_tag": {f"{src}:{tag}": n for (src, tag), n in sorted(tag_counts.items())},
        "seed": args.seed,
    }
    with (out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
