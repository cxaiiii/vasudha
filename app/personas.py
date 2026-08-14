"""Personality presets, loaded from editable JSON rather than compiled in.

Voice is the thing most likely to need tuning after release, and rebuilding a
212 MB bundle to reword one sentence is absurd. So the presets live as JSON
files that ship with the app and are copied to

    %LOCALAPPDATA%\\Vasudha\\personas\\

on first run. Edit one, restart, done. Drop in a fifth file and it appears in
the picker.

What the files may NOT change
-----------------------------
Only tone: `voice` and `engagement`. The identity block and session.CORE_RULES
are compiled in and appended AFTER the persona text, so no edit to these files
can talk the model out of using the calculator, or let it claim a source it
never opened. A persona that sounded confident and skipped python_tool would be
the worst thing this app could ship, so that is not left to a data file.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

IDENTITY = (
    "You are Vasudha, an AI assistant created by Chaitanya, an independent developer. "
    "You run entirely on the user's own machine. If asked who made you, say Chaitanya, "
    "an independent developer. Do not claim to be made by any company, and do not "
    "invent an organisation behind you."
)

#: Shared across every persona — how to be present in a conversation. Kept in
#: code rather than duplicated into each file so improving it improves all of
#: them, and so a persona cannot delete it.
ENGAGEMENT_BASE = (
    "Talk like someone who is actually in the conversation. Read what the user is really "
    "asking, not just the literal words, and respond to the person in front of you. "
    "Remember what they have told you in this chat and use it instead of asking twice. "
    "React to results rather than only reporting them — notice when a number is surprising, "
    "when an answer settles something they were stuck on, or when a question is a good one. "
    "When something is genuinely ambiguous and the answer would differ, ask one short "
    "question instead of guessing; when it is not, just answer. "
    "Never open with flattery, never pad with filler, and never ask a question you do not "
    "need the answer to. Warmth is attention and usefulness, not compliments."
)

DEFAULT_PERSONA = "engineer"


@dataclass(frozen=True)
class Persona:
    key: str
    name: str
    tagline: str
    glyph: str
    blurb: str
    voice: str
    engagement: str = ""
    order: int = 99

    @property
    def prompt(self) -> str:
        parts = [self.voice, ENGAGEMENT_BASE]
        if self.engagement:
            parts.insert(1, self.engagement)
        return "\n\n".join(p.strip() for p in parts if p.strip())


# A last-resort persona so the app still runs if the folder is emptied.
_FALLBACK = Persona(
    key="engineer", name="The Engineer", tagline="Sharp, quick, shows the working",
    glyph="📐", blurb="Straight to the equation and the number.",
    voice="Answer like a working engineer: the result first, then the governing equation "
          "and the assumptions. Short sentences, no padding.",
    order=1,
)


def bundled_dir() -> Path:
    """Where the shipped defaults live, in source or inside a frozen bundle."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "personas"
        if candidate.is_dir():
            return candidate
    return Path(__file__).resolve().parent.parent / "personas"


def user_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(base) / "Vasudha" / "personas"


#: Written after the first seed. Without it, seeding runs on every load and a
#: persona the user deliberately deleted silently reappears next launch.
_SEED_MARKER = ".seeded"


def ensure_installed(force: bool = False) -> Path:
    """Seed the editable copies once. Never overwrites an existing file, and
    never re-creates one the user removed on purpose."""
    target = user_dir()
    target.mkdir(parents=True, exist_ok=True)
    marker = target / _SEED_MARKER

    if marker.exists() and not force:
        return target

    source = bundled_dir()
    if source.is_dir():
        for path in source.glob("*.json"):
            destination = target / path.name
            if not destination.exists():
                try:
                    shutil.copyfile(path, destination)
                except OSError:
                    logger.warning("could not seed persona %s", path.name, exc_info=True)
    try:
        marker.write_text("Vasudha seeded the default personas here once.\n"
                          "Delete this file to restore any you have removed.\n",
                          encoding="utf-8")
    except OSError:
        logger.debug("could not write seed marker", exc_info=True)
    return target


def _folder_fingerprint(folder: Path) -> tuple:
    """Names and mtimes, so an edit is noticed without restarting the app."""
    try:
        return tuple(sorted((p.name, p.stat().st_mtime_ns)
                            for p in folder.glob("*.json")))
    except OSError:
        return ()


_REQUIRED = ("key", "name", "voice")


def _load_one(path: Path) -> Optional[Persona]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("persona %s is not readable JSON; skipping", path.name, exc_info=True)
        return None
    missing = [f for f in _REQUIRED if not str(data.get(f, "")).strip()]
    if missing:
        logger.warning("persona %s missing %s; skipping", path.name, ", ".join(missing))
        return None
    return Persona(
        key=str(data["key"]).strip(),
        name=str(data["name"]).strip(),
        tagline=str(data.get("tagline", "")).strip(),
        glyph=str(data.get("glyph", "✦")).strip() or "✦",
        blurb=str(data.get("blurb", "")).strip(),
        voice=str(data["voice"]).strip(),
        engagement=str(data.get("engagement", "")).strip(),
        order=int(data.get("order", 99)),
    )


def load_all(refresh: bool = False) -> list[Persona]:
    """Every valid persona, ordered.

    The cache is keyed on the folder's filenames and mtimes, so editing a file
    takes effect on the next message rather than requiring a restart — which is
    the whole point of having these outside the bundle. A broken file is
    skipped rather than fatal: one bad edit must not leave the user with an app
    that will not start.
    """
    global _CACHE, _CACHE_KEY

    folder = ensure_installed()
    key = _folder_fingerprint(folder)
    if _CACHE is not None and not refresh and key == _CACHE_KEY:
        return _CACHE

    found = [p for p in (_load_one(f) for f in sorted(folder.glob("*.json"))) if p]

    if not found:  # emptied or all invalid
        logger.warning("no usable personas in %s; using built-in fallback", folder)
        found = [_FALLBACK]

    seen, unique = set(), []
    for persona in sorted(found, key=lambda p: (p.order, p.name)):
        if persona.key not in seen:
            seen.add(persona.key)
            unique.append(persona)
    _CACHE, _CACHE_KEY = unique, key
    return unique


_CACHE: Optional[list[Persona]] = None
_CACHE_KEY: tuple = ()


def get(key: str) -> Persona:
    presets = load_all()
    for persona in presets:
        if persona.key == key:
            return persona
    for persona in presets:
        if persona.key == DEFAULT_PERSONA:
            return persona
    return presets[0]


def as_cards() -> list[dict]:
    return [{"key": p.key, "name": p.name, "tagline": p.tagline,
             "glyph": p.glyph, "blurb": p.blurb} for p in load_all()]


def build_system_prompt(persona_key: str, core_rules: str) -> str:
    """Identity, then voice, then the rules that cannot be overridden.

    core_rules goes last on purpose: it is the most recent instruction in the
    prompt, not something a persona's tone guidance sits on top of.
    """
    return f"{IDENTITY}\n\n{get(persona_key).prompt}\n\n{core_rules}"
