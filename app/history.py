"""Conversation persistence.

One JSON file per chat under %LOCALAPPDATA%\\Vasudha\\chats, rather than a
single combined file: a 200-turn conversation is megabytes, and rewriting every
chat on every turn to save one of them is the kind of thing that quietly
corrupts everything when the process is killed mid-write.

Writes go through a temp file + os.replace, which is atomic on Windows, so a
crash during a save leaves the previous version intact rather than a truncated
file that fails to parse on next launch.

No database: SQLite would be one more thing to bundle and migrate, and the
access pattern here is "load one chat, append to it" — files are the right
shape for that.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _chats_dir() -> Path:
    from app.paths import app_data_dir
    path = app_data_dir() / "chats"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _title_from(text: str, limit: int = 48) -> str:
    """First line of the opening question, trimmed. Good enough to recognise a
    conversation in a sidebar, and it costs no model call to produce."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return "New chat"
    return cleaned[:limit].rstrip() + ("…" if len(cleaned) > limit else "")


@dataclass
class Chat:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New chat"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    #: role/content dicts, exactly as the session replays them
    messages: list[dict] = field(default_factory=list)
    #: rendered UI events, so reopening a chat restores tool cards and
    #: reasoning blocks instead of flat text that loses the evidence trail
    events: list[dict] = field(default_factory=list)


class ChatStore:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or _chats_dir()

    def _path(self, chat_id: str) -> Path:
        # chat ids are generated hex, but never build a path from unvalidated
        # input without checking it.
        if not re.fullmatch(r"[0-9a-f]{6,32}", chat_id):
            raise ValueError(f"bad chat id: {chat_id!r}")
        return self.root / f"{chat_id}.json"

    def save(self, chat: Chat) -> None:
        chat.updated = time.time()
        target = self._path(chat.id)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(asdict(chat), indent=2), encoding="utf-8")
        os.replace(temp, target)

    def load(self, chat_id: str) -> Optional[Chat]:
        path = self._path(chat_id)
        if not path.exists():
            return None
        try:
            return Chat(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, OSError):
            logger.warning("could not read chat %s; skipping", chat_id, exc_info=True)
            return None

    def delete(self, chat_id: str) -> None:
        self._path(chat_id).unlink(missing_ok=True)

    def list_summaries(self, limit: int = 60) -> list[dict]:
        """Newest first. Reads each file, but a corrupt one is skipped rather
        than taking the whole sidebar down with it."""
        rows: list[dict] = []
        for path in self.root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "id": data.get("id", path.stem),
                    "title": data.get("title") or "Untitled",
                    "updated": data.get("updated", 0),
                })
            except (json.JSONDecodeError, OSError):
                logger.warning("skipping unreadable chat file %s", path.name)
        rows.sort(key=lambda r: r["updated"], reverse=True)
        return rows[:limit]

    def title_for(self, chat: Chat, first_user_message: str) -> None:
        if chat.title in ("", "New chat"):
            chat.title = _title_from(first_user_message)
