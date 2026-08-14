"""
Flattens a ReasoningRecord down to training text, reusing the existing
ChatFormatter (vasudha/datasets/chat_format.py) unmodified — that class is
the single source of truth for how <think> tags get wrapped, and it's what
the Ollama Modelfile's pre-closed <think></think>-by-default behavior
depends on staying consistent with. This module only builds the 2-message
list ChatFormatter already knows how to render; it never touches tag
wrapping itself.

Two example "flavors" come out of the same record, decided at format time
rather than via separate generation calls:
- "main": instruction -> winner's compact reasoning + clean final answer.
- "self_critique": instruction -> an initial flawed attempt (a real losing
  candidate, not a fabricated strawman) with its actual recorded flaws, then
  the corrected answer.

The `thinking` field is deliberately kept to a handful of labeled lines, not
a transcript of all three candidates and the full judge deliberation.
Rendering everything would make these <think> blocks far longer than
anything currently in the training corpus and risks teaching "always
produce a huge think block" — a direct regression of this project's central
stage-2 fix (empty <think></think> = answer-directly-by-default). A length
cap is enforced here for the same reason: max_seq_length was bumped
2048->4096 earlier specifically because truncating long CoT taught the
model to stop reasoning mid-thought — oversized records should be flagged
and dropped at generation time, not silently truncated into that failure
mode.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vasudha.datasets.chat_format import ChatFormatter  # noqa: E402
from vasudha.datasets.reasoning_schema import ReasoningRecord  # noqa: E402

# Rough chars-per-token heuristic — consistent with the pragmatic char-based
# caps already used elsewhere in this codebase (web/tools.py's PageFetcher
# truncates by character count, not a real tokenizer, for the same reason:
# good enough for a budget check, no tokenizer dependency needed here).
_CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 3800

_formatter = ChatFormatter(tokenizer=None, format="qwen3")


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _find_losing_candidate_with_flaws(record: ReasoningRecord):
    """Returns (candidate, flaws) for the first losing candidate the judge
    actually recorded flaws for, or (None, []) if none exist. Never
    fabricates a flaw — if the judge didn't note one, there's no self-critique
    example for this record."""
    winner_source = record.judge.winner_source()
    for critique in record.judge.critiques:
        if critique.position - 1 >= len(record.judge.order_map):
            continue
        pos_source = record.judge.order_map[critique.position - 1]
        if pos_source != winner_source and critique.flaws:
            candidate = next((c for c in record.candidates if c.source == pos_source), None)
            if candidate is not None:
                return candidate, critique.flaws
    return None, []


def _compact_reasoning_summary(record: ReasoningRecord) -> str:
    winner = record.winning_candidate()
    lines: list[str] = []
    if winner.concepts:
        lines.append("Concepts: " + "; ".join(winner.concepts))
    if winner.constraints:
        lines.append("Constraints: " + "; ".join(winner.constraints))
    if winner.governing_relations:
        lines.append("Governing relations: " + "; ".join(winner.governing_relations))
    if winner.tradeoffs:
        lines.append("Tradeoffs: " + "; ".join(winner.tradeoffs))

    _, flaws = _find_losing_candidate_with_flaws(record)
    if flaws:
        lines.append(f"Considered and rejected an alternative approach: {flaws[0]}")

    return "\n".join(lines)


def format_main_example(record: ReasoningRecord) -> list[dict]:
    winner = record.winning_candidate()
    return [
        {"role": "user", "content": record.instruction},
        {"role": "assistant", "content": winner.final_answer_text,
         "thinking": _compact_reasoning_summary(record)},
    ]


def format_self_critique_example(record: ReasoningRecord) -> Optional[list[dict]]:
    """None if no losing candidate has recorded flaws to build this from —
    caller should fall back to format_main_example in that case."""
    flawed, flaws = _find_losing_candidate_with_flaws(record)
    if flawed is None:
        return None
    winner = record.winning_candidate()
    correction = "; ".join(winner.governing_relations) if winner.governing_relations else winner.final_answer_text
    thinking = (
        f"Initial attempt: {flawed.final_answer_text}\n"
        f"Flaws identified: {'; '.join(flaws)}\n"
        f"Corrected approach: {correction}"
    )
    return [
        {"role": "user", "content": record.instruction},
        {"role": "assistant", "content": winner.final_answer_text, "thinking": thinking},
    ]


def flatten_record(
    record: ReasoningRecord,
    self_critique_ratio: float = 0.25,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    rng: Optional[random.Random] = None,
) -> Optional[dict]:
    """Returns {"text", "domain", "example_type"} ready for
    datasets.Dataset.from_list(), or None if the record produces text over
    the length cap (flagged/dropped, never silently truncated)."""
    rng = rng or random
    example_type = "main"
    messages = format_main_example(record)

    if rng.random() < self_critique_ratio:
        critique_messages = format_self_critique_example(record)
        if critique_messages is not None:
            messages = critique_messages
            example_type = "self_critique"

    text = _formatter.format_messages(messages)
    if estimate_tokens(text) > max_tokens:
        return None

    return {"text": text, "domain": record.domain, "example_type": example_type}
