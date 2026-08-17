"""Is there a newer Vasudha? Asked once a day, of GitHub, and of nobody else.

This is the only part of the application that contacts a server without the
user asking a question, so it is built to be defensible rather than convenient:

  * It is a plain unauthenticated GET of the public releases API. No account,
    no identifier, no version reported, no telemetry — the request carries
    nothing that distinguishes this install from any other, and the answer is
    the same for everyone who asks.
  * It can be turned off in Settings, and turning it off is honoured
    completely: no request is made at all.
  * It never installs anything. It says a version exists and offers to open
    the release page. A program that can silently replace its own executable is
    a different security proposition from one that cannot, and this product's
    argument is that you can see what it does.

The README claims search queries are the only thing that leaves the machine.
That claim now has a second clause, and the honest response is to say so there
rather than to make the check quieter.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

RELEASES_API = "https://api.github.com/repos/cxaiiii/vasudha/releases/latest"
RELEASES_PAGE = "https://github.com/cxaiiii/vasudha/releases/latest"

#: Once a day. A desktop app that checks on every launch is a nuisance on a
#: machine that gets restarted often, and the answer changes far less often.
CHECK_INTERVAL_SECONDS = 24 * 60 * 60

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_version(text: str) -> Optional[tuple[int, int, int]]:
    """The numeric part of a tag, or None.

    Tolerant of the shapes a tag takes in practice — v0.5.0, 0.5.0,
    v0.3.0-vulkan-test — because a release named slightly differently should
    not silently stop the check working.
    """
    match = _VERSION_RE.search(text or "")
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def is_newer(candidate: str, current: str) -> bool:
    new, old = parse_version(candidate), parse_version(current)
    if not new or not old:
        return False
    return new > old


def check(current_version: str, timeout: float = 6.0) -> Optional[dict]:
    """Ask GitHub for the latest release. None on any failure, including
    offline — an update check is not something to interrupt a user about."""
    import requests

    try:
        response = requests.get(
            RELEASES_API, timeout=timeout,
            headers={"Accept": "application/vnd.github+json",
                     # Identifies the software, not the installation. No version,
                     # no machine id: the server learns that some Vasudha asked,
                     # which it would learn from the request existing anyway.
                     "User-Agent": "Vasudha"})
        if response.status_code != 200:
            return None
        data = response.json()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("update check failed: %s", exc)
        return None

    tag = str(data.get("tag_name") or "")
    if not tag or not is_newer(tag, current_version):
        return None

    # Prereleases and drafts are not offers. v0.3.0-vulkan-test exists and
    # crashes on launch; nobody should be pointed at it.
    if data.get("prerelease") or data.get("draft"):
        return None

    notes = (data.get("body") or "").strip()
    return {
        "version": tag.lstrip("v"),
        "url": data.get("html_url") or RELEASES_PAGE,
        # First paragraph only. The release notes are long by design and the
        # notification is one line in a chat window.
        "summary": notes.split("\n\n")[0][:400],
        "published": data.get("published_at", ""),
    }


def due(last_check: float, now: Optional[float] = None) -> bool:
    return (now or time.time()) - (last_check or 0) >= CHECK_INTERVAL_SECONDS
