"""
Tool-use / research trajectory generation — teaches Vasudha to actually call
python_tool/search_tool/fetch_tool well, not just answer from memory.

Deliberately a much simpler (and cheaper) shape than reasoning_schema.py's
3-candidate + judge pipeline: there's no ambiguous "which reasoning path is
better" question here, because every tool call in a trajectory is REAL —
python_tool genuinely executes, search_tool hits DuckDuckGo for real,
fetch_tool actually reads the page. That removes the main reason the
reasoning pipeline needed a judge (candidates guessing at answers that might
be wrong) — there's nothing to adjudicate about a real execution result. One
clean trajectory per question is enough.

Reuses web/app.py's SYSTEM_PROMPT and web/tools.py's real tool
implementations directly (not reimplemented) so generated training data is
driven by, and stays in sync with, the exact protocol the live app already
uses: plain <tag>...</tag> calls in visible assistant text (not OpenAI-style
function calling), tool results fed back as a synthetic next "user" turn.
No <think> tags anywhere in this format — the app's own "one-line plan
before every tool call" convention is the visible reasoning, not a hidden
block, so flattening never sets ChatFormatter's `thinking` field.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from web.app import SYSTEM_PROMPT  # noqa: E402
from web.tools import SandboxedCodeExecutor, WebSearcher, PageFetcher  # noqa: E402
from vasudha.datasets.chat_format import ChatFormatter  # noqa: E402

# render_tool deliberately excluded — it's a UI-generation skill, not
# "tool calling and research" (what this module is scoped to), and it ends
# the turn immediately rather than chaining, so it wouldn't exercise the
# multi-step research behavior this dataset exists to teach.
TOOL_TAGS = ("python_tool", "search_tool", "fetch_tool")

_executor = SandboxedCodeExecutor(timeout=15)
_searcher = WebSearcher(max_results=5)
_fetcher = PageFetcher(timeout=10, max_chars=4000)

_formatter = ChatFormatter(tokenizer=None, format="qwen3")


def extract_tool_call(text: str) -> Optional[tuple[str, str]]:
    """Same first-match-wins scan web/app.py's generate() loop uses —
    replicated exactly (not just "similar") so a trajectory only ever
    records a tool call the live app would actually have recognized."""
    for tag in TOOL_TAGS:
        if f"</{tag}>" in text:
            match = re.search(rf"<{tag}>([\s\S]*?)<\/{tag}>", text)
            if match:
                return tag, match.group(1)
            break
    return None


def run_tool_and_followup(tag: str, content: str) -> tuple[str, str]:
    """Executes the REAL tool (genuine subprocess / genuine DuckDuckGo
    query / genuine HTTP fetch — never simulated by the teacher model) and
    builds the exact same follow-up wrapper text web/app.py's generate()
    sends back, word for word, so the trajectory is indistinguishable from
    a real logged session."""
    if tag == "python_tool":
        tool_res = _executor.execute_python(content)
        follow_up = (
            f"Here is the real execution output of your Python code:\n{tool_res}\n\n"
            "Use this if needed, then give your final answer (or call another tool if more work is needed)."
        )
    elif tag == "search_tool":
        query = content.strip()
        tool_res = _searcher.search(query)
        follow_up = (
            f'Here are the real search results for "{query}":\n{tool_res}\n\n'
            "Judge whether this is conclusive. If a snippet looks promising but too short, "
            "fetch_tool that URL. If nothing is promising, refine the query and search again. "
            "Otherwise use this for your final answer, or call render_tool if the user asked for a "
            "webpage/UI, informed by what you found here."
        )
    else:  # fetch_tool
        url = content.strip()
        tool_res = _fetcher.fetch(url)
        follow_up = (
            f"Here is the real extracted text from {url}:\n{tool_res}\n\n"
            "Use this if it has the fact you needed, then give your final answer (or search_tool "
            "again with a refined query if it didn't)."
        )
    return tool_res, follow_up


@dataclass
class Trajectory:
    question: str
    messages: list[dict] = field(default_factory=list)  # system, user, assistant, user, ...
    tool_tags_used: list[str] = field(default_factory=list)
    resolved: bool = False  # True if it ended in a real final answer, not the iteration cap

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "messages": self.messages,
            "tool_tags_used": self.tool_tags_used,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(
            question=d["question"],
            messages=d["messages"],
            tool_tags_used=d.get("tool_tags_used", []),
            resolved=d.get("resolved", False),
        )


def generate_trajectory(
    question: str, call_fn: Callable[[list[dict]], str], max_iterations: int = 5,
) -> Optional[Trajectory]:
    """Drives the same request/tool-call/follow-up loop web/app.py runs at
    inference time, but with call_fn hitting a real teacher API instead of
    the local Ollama model. Returns None (never a fabricated/truncated
    ending) if max_iterations is hit without a real final answer — an
    unresolved trajectory teaches "give up after N tool calls," which is
    not the behavior this dataset is for."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tags_used: list[str] = []

    for _ in range(max_iterations):
        assistant_text = call_fn(messages)
        call = extract_tool_call(assistant_text)

        if call is None:
            messages.append({"role": "assistant", "content": assistant_text})
            return Trajectory(question=question, messages=messages, tool_tags_used=tags_used, resolved=True)

        tag, content = call
        tags_used.append(tag)
        _tool_res, follow_up = run_tool_and_followup(tag, content)
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": follow_up})

    return None


def flatten_trajectory(trajectory: Trajectory) -> dict:
    """No `thinking` field is ever set here — ChatFormatter renders plain
    turns with no <think> block, matching the live app's actual format
    where the plan-line is ordinary visible assistant text."""
    text = _formatter.format_messages(trajectory.messages)
    return {
        "text": text,
        "tool_tags_used": trajectory.tool_tags_used,
        "num_turns": sum(1 for m in trajectory.messages if m["role"] == "assistant"),
    }
