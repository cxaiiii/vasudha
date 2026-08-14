"""UI-agnostic agent loop over Ollama's NATIVE tool-calling API.

Why this exists
---------------
web/app.py's original loop asked the model to type `<python_tool>` tags into
its prose and then regex-scraped them back out. Measured on the six-problem
numeric set in scripts/bench_numeric.py, greedy decoding, vasudha-v3:

    harness                                   correct   tool fired
    regex tags + app.py's system prompt         0/6         1/6
    native tool_calls (this module)             4/6         6/6

Tool invocation is the whole difference. The model reports capability "tools"
(`ollama show vasudha-v3`), so a JSON tool schema gets structured `tool_calls`
back instead of depending on the model to follow a formatting instruction it
ignored five times out of six.

This module deliberately knows nothing about Flask, HTML or SSE. It takes
messages in and yields events out, so the same loop backs the existing web UI
and any other front end without a second implementation drifting out of sync.

It lives in web/ next to tools.py — which is equally UI-agnostic — rather than
under vasudha/, because `vasudha/__init__.py` eagerly imports the model stack:
`from vasudha.inference.agent import ...` measured 88.7s and pulled torch into
the process. This loop only speaks HTTP to Ollama, and app.py's own comments
explain why a second in-process copy of the model is exactly what to avoid.

Unit discipline
---------------
The tool descriptions below insist the generated code print the value already
converted to the unit the question asked for. That is not decoration: the one
non-knowledge failure left in the 4/6 run was the model computing 1.2e8 Pa
correctly and then writing "approximately 120,000 MPa" — a Pa->MPa conversion
done in its head, wrong, after the tool had already given it the right number.
Converting inside the sandbox keeps that arithmetic out of the model.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# Ceiling on tool calls per user turn. Generous enough for scaffolding a small
# multi-file project (write several files, list, run, read a traceback, fix,
# re-run) plus a final answer; low enough that a model stuck in a loop stops.
DEFAULT_MAX_ITERATIONS = 18


def _numeric_unit_rule(what: str) -> str:
    return (f"Compute {what} in a real Python sandbox and return its stdout. "
            "The code MUST print the final value already converted to the unit the "
            "question asks for, together with that unit — e.g. print(f'{sigma/1e6:.4g} MPa'), "
            "not print(sigma). Never leave a unit conversion for after the tool call.")


def build_tool_schemas(include: Optional[set[str]] = None) -> list[dict]:
    """JSON schemas for the seven tools, in ollama/OpenAI function format.

    `include` filters by name so a caller can expose a subset (the benchmark
    only wants python_tool; the web UI wants everything).
    """
    schemas: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "python_tool",
                "description": _numeric_unit_rule("a numeric or algorithmic result"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source. Must print the final answer and its unit.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_tool",
                "description": ("Search the web and return real result snippets (title + short "
                                "excerpt). Snippets only — use fetch_tool to read a full page."),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_tool",
                "description": ("Read the actual text of one webpage. Only pass a URL that a "
                                "search_tool result returned in this conversation; never invent "
                                "or recall a URL."),
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "render_tool",
                "description": ("Render a complete, self-contained HTML document in a live "
                                "sandboxed preview. Use for ANY webpage, UI or visual component. "
                                "Never write raw HTML into your reply — it will not render there."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "html": {"type": "string",
                                 "description": "One complete HTML document, Tailwind via CDN."}
                    },
                    "required": ["html"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": ("Write a file into this conversation's persistent workspace. "
                                "Files survive across turns and are visible to later python_tool "
                                "calls."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to the workspace."},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file back from the persistent workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List what actually exists in the persistent workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "default": "."}},
                    "required": [],
                },
            },
        },
    ]
    if include is None:
        return schemas
    return [s for s in schemas if s["function"]["name"] in include]


@dataclass
class AgentEvent:
    """One thing that happened. `kind` is one of:

    text        - assistant prose (delta or full, see `text`)
    tool_call   - the model asked for a tool; `name`, `args`
    tool_result - the tool really ran; `name`, `result`
    render      - render_tool HTML the front end should preview; `result`
    done        - the turn finished; `reason` says why
    error       - something failed; `text` explains
    """
    kind: str
    text: str = ""
    name: str = ""
    args: dict = field(default_factory=dict)
    result: str = ""
    reason: str = ""


ToolFn = Callable[..., str]


class VasudhaAgent:
    """Drives one user turn to completion, yielding AgentEvents.

    `tools` maps a tool name to a callable taking the schema's parameters as
    keyword arguments and returning a string (what the model will see). The
    caller owns the callables, so the workspace-backed tools can be bound to
    whichever session directory this conversation belongs to.
    """

    def __init__(
        self,
        model: str,
        tools: dict[str, ToolFn],
        system_prompt: str,
        url: str = OLLAMA_CHAT_URL,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        options: Optional[dict] = None,
        timeout: int = 600,
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.url = url
        self.max_iterations = max_iterations
        self.options = options or {"temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.05,
                                   "num_predict": 2048, "num_ctx": 32768}
        self.timeout = timeout
        self.schemas = build_tool_schemas(include=set(tools))

    # -- transport ---------------------------------------------------------

    def _call_model(self, messages: list[dict]) -> dict:
        response = requests.post(
            self.url,
            json={"model": self.model, "messages": messages, "tools": self.schemas,
                  "stream": False, "options": self.options},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("message", {}) or {}

    # -- tool dispatch -----------------------------------------------------

    @staticmethod
    def _parse_args(raw: Any) -> dict:
        """ollama returns arguments as an object, but some builds send a JSON
        string. Accept both rather than crashing the whole turn on a quirk."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {"value": parsed}
            except json.JSONDecodeError:
                return {"value": raw}
        return {}

    def _run_tool(self, name: str, args: dict) -> str:
        fn = self.tools.get(name)
        if fn is None:
            # Tell the model rather than raising: an unknown tool is something
            # it can recover from on the next iteration.
            return f"[error] no such tool: {name}"
        try:
            return fn(**args)
        except TypeError as exc:
            return f"[error] bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface anything to the model
            logger.exception("tool %s failed", name)
            return f"[error] {name} raised: {exc}"

    # -- main loop ---------------------------------------------------------

    def run(self, messages: list[dict]) -> Iterator[AgentEvent]:
        """Yield events until the model answers without calling a tool.

        `messages` is the conversation so far (user/assistant dicts). The
        system prompt is prepended here, so callers never have to remember to.
        """
        convo: list[dict] = [{"role": "system", "content": self.system_prompt}]
        convo += [m for m in messages if m.get("role") != "system"]

        last_tool_result = ""
        nudged = False

        for _ in range(self.max_iterations):
            try:
                message = self._call_model(convo)
            except requests.RequestException as exc:
                yield AgentEvent("error", text=f"model call failed: {exc}")
                return

            content = (message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") or []

            # Observed on vasudha-v3: after a tool result the model sometimes
            # returns an empty message with no tool calls, which would end the
            # turn with nothing shown to the user — worse than a wrong answer,
            # since the tool had already produced the right number. Ask once
            # for the final answer before giving up.
            if not content and not tool_calls and last_tool_result and not nudged:
                nudged = True
                convo.append({
                    "role": "user",
                    "content": ("State the final answer now, in one line, using the tool output "
                                "above verbatim. Do not call another tool."),
                })
                continue

            if content:
                yield AgentEvent("text", text=content)

            if not tool_calls:
                if not content and last_tool_result:
                    # Still nothing after the nudge. The tool output is real and
                    # already verified, so surface it rather than a blank reply.
                    yield AgentEvent("text", text=last_tool_result.strip())
                    yield AgentEvent("done", reason="tool_result_fallback")
                    return
                yield AgentEvent("done", reason="stop")
                return

            convo.append(message)

            for call in tool_calls:
                fn = call.get("function", {}) or {}
                name = fn.get("name", "")
                args = self._parse_args(fn.get("arguments"))
                yield AgentEvent("tool_call", name=name, args=args)

                result = self._run_tool(name, args)

                # render_tool's payload is for the front end to preview, not
                # something the model needs echoed back in full.
                if name == "render_tool":
                    yield AgentEvent("render", name=name, result=result)
                    observation = "[render_tool] preview rendered for the user."
                else:
                    yield AgentEvent("tool_result", name=name, result=result)
                    observation = result
                    last_tool_result = result

                convo.append({"role": "tool", "content": observation, "tool_name": name})

        yield AgentEvent("done", reason="max_iterations")
