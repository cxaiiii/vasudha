"""Remembering what the tools already found, past the end of the window.

The failure this exists for: a long session fitted a mixture model, printed
real parameters, and twenty turns later reported statistics that contradicted
them. The model had not become less capable — the output it was reasoning about
had been compacted out of the context and it was filling the gap from memory.

Compaction is still necessary; the window is 8k. What was missing is that the
dropped text went nowhere. Now every tool result is kept here in full, and
whichever ones bear on the current question are retrieved back into the prompt.
The conversation stops being a sliding window and starts being a window over a
store.

Lexical, not embedding-based, deliberately:

  * The things these sessions lose are *named*: a column, a variable, an
    exception type, a printed figure. Keyword matching is good at exactly that
    and semantic similarity adds little.
  * It needs no model, no VRAM on a machine already short of it, and no
    download — the product runs offline and this must too.
  * It is instant, so it can run on every turn without a latency budget.

BM25 rather than raw overlap because tool output is repetitive — every
traceback contains "File", "line", "in" — and BM25's term saturation and length
normalisation are what stop a long stack trace outranking the one short line
that actually holds the answer.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

#: Tokens are lowercased alphanumerics plus the punctuation that carries meaning
#: in code — a dot, an underscore. "df.groupby" should not become two tokens
#: that match every dataframe in the session.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*|\d+\.?\d*")

#: Words too common in this corpus to discriminate. Not a general stop list:
#: these are the ones that appear in nearly every tool result.
_NOISE = frozenset("""
    the a an and or of in to is are was for on with it this that from at by be
    file line stdout stderr finished traceback most recent call last python
    error true false none null print return def import self
""".split())

_K1 = 1.5      # term-frequency saturation
_B = 0.75      # length normalisation


def tokenize(text: str) -> list[str]:
    return [t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
            if t not in _NOISE and len(t) > 1]


@dataclass
class Record:
    """One tool result, kept whole."""
    turn: int
    tool: str
    args: dict
    result: str
    when: float = field(default_factory=time.time)
    tokens: list[str] = field(default_factory=list)

    def summary(self, width: int = 90) -> str:
        detail = (self.args.get("code") or self.args.get("query")
                  or self.args.get("url") or self.args.get("path") or "")
        head = f"{self.tool}({str(detail)[:50]!r})" if detail else self.tool
        return f"[turn {self.turn}] {head}"


class ToolMemory:
    """Every tool result of one conversation, searchable.

    Deliberately per-conversation and in-memory. This is not the memory book —
    that holds lessons meant to outlive the session. This holds evidence, which
    is only meaningful inside the conversation that produced it, and keeping it
    on disk would mean a stale GMM fit from yesterday competing with today's.
    """

    def __init__(self, max_records: int = 400) -> None:
        self.records: list[Record] = []
        self.max_records = max_records
        self._df: Counter[str] = Counter()      # document frequency

    def add(self, turn: int, tool: str, args: dict, result: str) -> None:
        if not result or not result.strip():
            return
        record = Record(turn=turn, tool=tool, args=dict(args or {}),
                        result=result, tokens=tokenize(result))
        self.records.append(record)
        self._df.update(set(record.tokens))
        # Oldest first, because a session long enough to hit this has already
        # moved on from its opening turns.
        while len(self.records) > self.max_records:
            dropped = self.records.pop(0)
            self._df.subtract(set(dropped.tokens))
            self._df += Counter()               # drop zero/negative entries

    def search(self, query: str, limit: int = 3,
               exclude_turns: Optional[set] = None) -> list[Record]:
        """The stored results that best match a question, best first."""
        terms = tokenize(query)
        if not terms or not self.records:
            return []

        candidates = [r for r in self.records
                      if not exclude_turns or r.turn not in exclude_turns]
        if not candidates:
            return []

        total = len(candidates)
        avg_len = sum(len(r.tokens) for r in candidates) / total or 1.0

        scored: list[tuple[float, Record]] = []
        for record in candidates:
            counts = Counter(record.tokens)
            length = len(record.tokens) or 1
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                # +0.5/+0.5 is BM25's smoothed IDF; without it a term present in
                # every record scores negative and drags a good match down.
                idf = math.log(1 + (total - self._df.get(term, 0) + 0.5)
                               / (self._df.get(term, 0) + 0.5))
                score += idf * (freq * (_K1 + 1)) / (
                    freq + _K1 * (1 - _B + _B * length / avg_len))
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda pair: (-pair[0], -pair[1].turn))
        return [record for _, record in scored[:limit]]

    def recall_block(self, query: str, budget: int = 1400,
                     exclude_turns: Optional[set] = None) -> str:
        """Matching earlier results, formatted for the prompt.

        Trimmed per record so several can appear rather than one long one
        crowding the rest out — the case this serves usually needs a number
        from turn 3 *and* a column name from turn 9.
        """
        hits = self.search(query, limit=3, exclude_turns=exclude_turns)
        if not hits:
            return ""

        per_record = max(budget // len(hits), 200)
        parts = ["\n\nEARLIER IN THIS CONVERSATION (retrieved because it matches "
                 "the question — this is what the tools actually returned, and it "
                 "outranks your recollection of it):"]
        for record in hits:
            body = record.result.strip()
            if len(body) > per_record:
                body = body[:per_record].rstrip() + " […]"
            parts.append(f"{record.summary()}\n{body}")
        return "\n\n".join(parts)

    def __len__(self) -> int:
        return len(self.records)
