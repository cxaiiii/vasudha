"""What the assistant has learned, and what the user actually asked.

Three things live here, all of them on disk and all of them local:

  MemoryBook       lessons the model wrote for itself, keyed by topic so a
                   better answer replaces a worse one instead of accumulating
                   beside it. Injected into every system prompt.
  lessons.jsonl    every version of every lesson, append-only. The book is the
                   current best answer; this is the history, and it is the part
                   that becomes training data.
  InteractionLog   what was asked, how it was asked, which tools ran and what
                   came back. Also training data, and the only honest record of
                   how the product is really used.

Why a book rather than a longer prompt: the failure this exists to stop is the
model rediscovering the same dead end. It reached for textblob, found it
missing, and hand-rolled a worse sentiment lexicon — and would have done the
same thing the next day. A lesson written once ("this sandbox has no textblob
until pip_tool installs it") costs about twenty tokens and removes the loop.

Privacy: everything here is a plain file under the app's own data directory,
nothing is sent anywhere, and clear() empties it. The interaction log records
the user's own words verbatim, which is the point — tone and phrasing are the
part a synthesised dataset never gets right — so it is easy to inspect and easy
to delete.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: How much of the book may enter the system prompt. The book is worthless if
#: it crowds out the conversation: at 8k context with ~1.2k of tool schemas
#: already spent, this is about as much as can be justified.
MAX_BOOK_CHARS = 1200

#: A topic key. Normalised hard, because "textblob missing" and "TextBlob
#: Missing!" are the same lesson and two entries for it defeats the point.
_KEY_RE = re.compile(r"[^a-z0-9]+")


def _key(topic: str) -> str:
    return _KEY_RE.sub("-", (topic or "").strip().lower()).strip("-")[:60]


@dataclass
class Lesson:
    topic: str
    lesson: str
    source: str = ""          # url, tool name, or "" for the model's own finding
    updated: float = 0.0
    revision: int = 1

    def as_line(self) -> str:
        tail = f"  [{self.source}]" if self.source else ""
        return f"- **{self.topic}**: {self.lesson}{tail}"


class MemoryBook:
    """Lessons the model keeps between sessions."""

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            from app.paths import app_data_dir
            root = app_data_dir() / "memory"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.book_path = self.root / "MEMORY.md"
        self.history_path = self.root / "lessons.jsonl"
        self._entries: dict[str, Lesson] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        """Rebuild from history rather than from the rendered book.

        The .md is for humans to read and edit; the .jsonl is the record. If
        both exist the history wins, because a later revision of a lesson is
        the whole mechanism — replaying it in order leaves the newest on top.
        """
        if not self.history_path.exists():
            return
        try:
            for line in self.history_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                lesson = Lesson(**{k: data[k] for k in
                                   ("topic", "lesson", "source", "updated", "revision")
                                   if k in data})
                self._entries[_key(lesson.topic)] = lesson
        except (json.JSONDecodeError, OSError, TypeError):
            logger.warning("memory history unreadable; starting fresh", exc_info=True)

    def _write_book(self) -> None:
        lines = ["# What Vasudha has learned", "",
                 "Edited freely — this file is read at the start of every chat.", ""]
        for lesson in sorted(self._entries.values(), key=lambda l: -l.updated):
            lines.append(lesson.as_line())
        try:
            self.book_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            logger.warning("could not write MEMORY.md", exc_info=True)

    # -- api ---------------------------------------------------------------

    def remember(self, topic: str, lesson: str, source: str = "") -> str:
        topic, lesson = (topic or "").strip(), (lesson or "").strip()
        if not topic or not lesson:
            return "[error] remember_tool needs both a topic and a lesson"
        if len(lesson) > 400:
            lesson = lesson[:400].rstrip() + "…"

        key = _key(topic)
        previous = self._entries.get(key)
        entry = Lesson(topic=topic, lesson=lesson, source=source.strip(),
                       updated=time.time(),
                       revision=(previous.revision + 1) if previous else 1)
        self._entries[key] = entry

        try:
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("could not append to lessons.jsonl", exc_info=True)
        self._write_book()

        if previous:
            return (f"[updated '{topic}' — revision {entry.revision}] "
                    f"Replaced: {previous.lesson[:80]}")
        return f"[remembered '{topic}']"

    def as_prompt_section(self, budget: int = MAX_BOOK_CHARS) -> str:
        """The book, compact enough to sit in every system prompt.

        Newest first and truncated at a whole entry: half a lesson is worse
        than none, because the model acts on the half it can see.
        """
        if not self._entries:
            return ""
        lines: list[str] = []
        used = 0
        for lesson in sorted(self._entries.values(), key=lambda l: -l.updated):
            line = lesson.as_line()
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line) + 1
        if not lines:
            return ""
        return ("\n\nWHAT YOU ALREADY LEARNED (from earlier sessions — trust these "
                "over your own recollection, and call remember_tool to correct one "
                "that turns out to be wrong):\n" + "\n".join(lines))

    def clear(self) -> None:
        self._entries.clear()
        for path in (self.book_path, self.history_path):
            try:
                path.unlink()
            except OSError:
                pass

    def __len__(self) -> int:
        return len(self._entries)


class InteractionLog:
    """Every turn, as it really happened: the user's own words included.

    Appended as JSONL so it can be read straight into a dataset. The user's
    phrasing is kept verbatim and deliberately — tone, terseness, the way a
    real question is actually asked is exactly what a synthesised training set
    gets wrong.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            from app.paths import app_data_dir
            root = app_data_dir() / "memory"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "interactions.jsonl"

    def record(self, *, chat_id: str, question: str, answer: str,
               tools: list[dict], sources: list[str], persona: str = "",
               model: str = "", seconds: float = 0.0) -> None:
        row = {
            "ts": time.time(),
            "chat_id": chat_id,
            "persona": persona,
            "model": model,
            "question": question,
            "answer": answer,
            # name + arguments + result for each call, in order: enough to
            # replay the turn, which is what makes it trainable rather than
            # merely readable.
            "tools": tools,
            "sources": sources,
            "seconds": round(seconds, 2),
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("could not append to interactions.jsonl", exc_info=True)


# ── source primacy ────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def unsourced_figures(answer: str, tool_outputs: list[str],
                      max_report: int = 5) -> list[str]:
    """Numbers in the answer that appear in no tool output.

    The rule this serves is "what was observed beats what was recalled", and
    the observable violation of it is a figure that came from nowhere. Purely
    mechanical, like the citation stripper: it does not ask the model whether
    it made something up.

    Deliberately advisory rather than destructive. A number can legitimately be
    derived — a mean of figures that are each present, a rounded version of a
    tool's output — so removing them would break correct answers. Naming them
    lets the reader check the ones that matter.
    """
    if not tool_outputs:
        return []
    haystack = " ".join(tool_outputs)
    present_text = {n.replace(",", "") for n in _NUMBER_RE.findall(haystack)}
    present_values: list[float] = []
    for text in present_text:
        try:
            present_values.append(float(text))
        except ValueError:
            pass

    missing: list[str] = []
    for raw in _NUMBER_RE.findall(answer or ""):
        value = raw.replace(",", "")
        if value in present_text or value in missing:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        # Small integers are almost always list positions, years, or counts the
        # model wrote as prose ("all 4 reviews"), and flagging them buries the
        # figures that matter under noise.
        if abs(number) < 10 and "." not in value:
            continue

        # A rounded form of a sourced figure is sourced. Compared numerically
        # rather than as text: 14.0696 rounds to 14.07 and shares no prefix
        # with it, so string matching flags a correct answer — which trains the
        # reader to ignore the warning, the one outcome worth avoiding.
        decimals = len(value.split(".")[1]) if "." in value else 0
        if any(round(p, decimals) == number for p in present_values):
            continue

        missing.append(value)
        if len(missing) >= max_report:
            break
    return missing
