"""Inference backends for the desktop app, and the logic that picks one.

The app must run on a machine with nothing installed, but it must not feel
broken on a machine that has a GPU. Those pull in opposite directions, so the
backend is chosen at runtime rather than baked in:

    OllamaBackend    - if an ollama server is already running locally. Uses the
                       user's GPU, handles its own model lifecycle. Measured on
                       the dev box: full tool-call turns in 6-20s.
    LlamaCppBackend  - in-process llama-cpp-python against a GGUF file. Needs
                       no install and no server, which is the whole point of
                       shipping a binary.

Both expose the same `stream()` / `generate()` and both return the same ToolCall
objects, so app/session.py's loop does not care which one it got.

Model independence
------------------
This harness is used to evaluate models other than Vasudha's own checkpoint, so
nothing here may assume Qwen3. Two things used to:

  * The prompt was hand-built as ChatML with a hardcoded `<tools>` preamble and
    a hardcoded `<think>` opener. Now the GGUF's own chat template is rendered
    (every modern GGUF carries one), with `tools` and `enable_thinking` passed
    in as template variables. A model whose template knows about tools gets its
    native tool syntax; ChatML is only the fallback for a GGUF with no template.
  * Tool calls were parsed with one regex for Qwen3's XML form. Now a registry
    of parsers runs in turn — Qwen3 XML, the JSON-in-`<tool_call>` form used by
    Hermes/Qwen2.5/Mistral and most others, fenced JSON, and the structured
    `tool_calls` some servers return. First match wins.

Adding a model means adding a parser at worst, and usually nothing at all.

Thinking control
----------------
`options["enable_thinking"]` is passed to the chat template rather than being
faked by pre-opening or pre-closing a `<think>` block. Measured cost of getting
this wrong on the shipped 4B: a forced think block on a post-tool-result turn
was 10.05s vs 5.90s for the same answer with thinking off — 1.7x, on every one
of up to 12 loop iterations. app/session.py turns it on to plan and off to
restate a number a tool already computed.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE = os.environ.get("VASUDHA_OLLAMA_URL", "http://127.0.0.1:11434")


class BackendError(RuntimeError):
    """A backend failure with a message fit to show a user.

    Raw requests exceptions leak URLs and status codes into the UI ("500 Server
    Error ... for url: http://127.0.0.1:11434/api/chat"), which tells the user
    nothing they can act on.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    #: Correlates a result with the call that produced it. Models that emit
    #: several calls in one turn otherwise give the loop no way to say which
    #: result answers which call, and the transcript becomes positional.
    id: str = ""


@dataclass
class Completion:
    """One assistant turn: prose plus any tool calls it asked for."""
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Tool-call parsing — one registry, several model families
# ══════════════════════════════════════════════════════════════════════════════

#: Wrappers a model may put around a tool call. `<|python_tag|>` is Llama 3.x,
#: the rest are the ChatML-family conventions.
_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>"
    r"|<\|tool_call\|>\s*(.*?)\s*(?:<\|/tool_call\|>|$)"
    r"|<function_calls>\s*(.*?)\s*</function_calls>",
    re.S,
)

# Qwen3 / Hermes XML form:  <function=NAME><parameter=KEY>value</parameter>
_XML_FN = re.compile(r"<function=([A-Za-z_]\w*)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([A-Za-z_]\w*)>\n?(.*?)\n?</parameter>", re.S)

#: Reasoning-block wrappers, in the order they are tried. Kept separate from the
#: tool-call patterns because a model may use one, both, or neither.
_THINK_TAGS = ("think", "thinking", "reasoning", "reason")
_THINK_RE = re.compile(
    r"<(" + "|".join(_THINK_TAGS) + r")>(.*?)</\1>", re.S | re.I)
#: An unclosed opener, which is what a truncated or budget-capped reply leaves.
_THINK_OPEN_RE = re.compile(
    r"<(" + "|".join(_THINK_TAGS) + r")>(.*)$", re.S | re.I)

_FENCE = re.compile(r"^```(?:json|tool_code|python)?\s*|\s*```$", re.M)

_ANY_THINK_TAG = re.compile(
    r"<(/?)(?:" + "|".join(_THINK_TAGS) + r")>", re.I)


def prompt_opens_thinking(prompt: str) -> bool:
    """Does this rendered prompt hand the model an already-open think block?

    Determined from the prompt rather than from the model's name, because it is
    a property of whatever chat template the GGUF happens to carry — which is
    exactly the thing that varies when a different model is dropped into this
    harness.
    """
    last = None
    for match in _ANY_THINK_TAG.finditer(prompt):
        last = match.group(1)      # "" for an opener, "/" for a closer
    return last == ""


def _normalise_call(obj: Any) -> Optional[ToolCall]:
    """Turn one decoded JSON object into a ToolCall, whatever it calls its keys.

    The JSON tool-call convention is near-universal but the key names are not:
    `arguments` (OpenAI, Hermes), `parameters` (Gemini-style), `args` (various
    fine-tunes). Accepting all three costs nothing and saves a per-model parser.
    """
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function") or obj.get("tool")
    if isinstance(name, dict):                      # {"function": {"name": ...}}
        nested = name
        name = nested.get("name")
        obj = {**obj, **nested}
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if args is None:
        args = obj.get("args")
    return ToolCall(name=name, args=_coerce_args(args), id=str(obj.get("id") or ""))


def _parse_json_calls(blob: str) -> list[ToolCall]:
    """Decode one tool-call payload that may be an object, a list, or several
    concatenated objects (some models emit two calls with no separator)."""
    blob = _FENCE.sub("", blob).strip()
    if not blob:
        return []

    try:
        decoded = json.loads(blob)
    except json.JSONDecodeError:
        # Concatenated or trailing-comma'd objects: walk them with raw_decode
        # rather than giving up on the whole block.
        calls, idx, decoder = [], 0, json.JSONDecoder()
        while idx < len(blob):
            try:
                obj, end = decoder.raw_decode(blob, idx)
            except json.JSONDecodeError:
                break
            call = _normalise_call(obj)
            if call:
                calls.append(call)
            idx = end
            while idx < len(blob) and blob[idx] in " \t\r\n,":
                idx += 1
        return calls

    if isinstance(decoded, list):
        return [c for c in (_normalise_call(o) for o in decoded) if c]
    call = _normalise_call(decoded)
    return [call] if call else []


def _parse_xml_calls(blob: str) -> list[ToolCall]:
    """Qwen3's `<function=name><parameter=key>value</parameter>` form."""
    calls = []
    for match in _XML_FN.finditer(blob):
        args = {k: v.strip() for k, v in _XML_PARAM.findall(match.group(2))}
        calls.append(ToolCall(name=match.group(1), args=args))
    return calls


def split_thinking(text: str) -> tuple[str, str]:
    """Separate a reasoning block from the visible reply.

    Handles both the closed form and the unclosed one a truncated reply leaves
    behind — an unclosed `<think>` means everything after it is reasoning, and
    showing that as the answer is how a user ends up reading the model's
    working as if it were a conclusion.
    """
    thinking_parts = []

    def _take(match: re.Match) -> str:
        thinking_parts.append(match.group(2).strip())
        return ""

    text = _THINK_RE.sub(_take, text)

    open_match = _THINK_OPEN_RE.search(text)
    if open_match:
        thinking_parts.append(open_match.group(2).strip())
        text = text[: open_match.start()]

    return text.strip(), "\n\n".join(p for p in thinking_parts if p).strip()


def parse_tool_calls(text: str) -> tuple[str, str, list[ToolCall]]:
    """Split raw model output into (visible_text, thinking, tool_calls).

    Model-agnostic: tries each known wrapper, and inside a wrapper tries JSON
    before XML because JSON is the more common convention by a wide margin.
    """
    text, thinking = split_thinking(text)

    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []

    for match in _CALL_BLOCK.finditer(text):
        blob = next((g for g in match.groups() if g), "")
        found = _parse_json_calls(blob) or _parse_xml_calls(blob)
        if found:
            calls.extend(found)
            spans.append(match.span())

    # A bare `<function=...>` with no `<tool_call>` wrapper around it.
    if not calls:
        for match in _XML_FN.finditer(text):
            args = {k: v.strip() for k, v in _XML_PARAM.findall(match.group(2))}
            calls.append(ToolCall(name=match.group(1), args=args))
            spans.append(match.span())

    # Last resort: the whole reply is a bare JSON tool call with no wrapper at
    # all. Guarded on the presence of a "name" key so an ordinary reply that
    # happens to be JSON (a model answering "give me JSON") is not eaten.
    if not calls:
        stripped = _FENCE.sub("", text).strip()
        if stripped.startswith(("{", "[")) and '"name"' in stripped:
            calls = _parse_json_calls(stripped)
            if calls:
                spans.append((0, len(text)))

    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]

    for i, call in enumerate(calls):
        if not call.id:
            call.id = f"call_{i}"

    return text.strip(), thinking, calls


# ══════════════════════════════════════════════════════════════════════════════
# Incremental stream filter
# ══════════════════════════════════════════════════════════════════════════════

#: Everything a delta might be the beginning of. The filter refuses to emit a
#: tail that could still turn into one of these, so a half-arrived `<tool_c`
#: never flashes on screen before we know it was markup.
_SENTINELS = tuple(
    ["<tool_call>", "</tool_call>", "<|tool_call|>", "<function=",
     "<function_calls>", "<|python_tag|>", "<|"]
    + [f"<{t}>" for t in _THINK_TAGS] + [f"</{t}>" for t in _THINK_TAGS]
)
_MAX_SENTINEL = max(len(s) for s in _SENTINELS)


class StreamFilter:
    """Turns a raw token stream into visible / thinking deltas.

    The model's output is one stream containing prose, an optional reasoning
    block, and possibly a tool call. Only the prose should reach the chat as it
    arrives; reasoning belongs in the collapsible panel and a tool call should
    not be shown as text at all. Doing that incrementally means never emitting
    a fragment that a later token might reveal to be markup, which is what the
    holdback below is for.
    """

    def __init__(self, in_think: bool = False) -> None:
        self.raw = ""
        self._pending = ""
        #: Several chat templates (Qwen3's among them) end their generation
        #: prompt with an already-open `<think>`, so the model's very first
        #: token is *inside* the reasoning block and the only tag it ever emits
        #: is the closing one. A filter that assumed it started outside saw a
        #: stray `</think>`, matched nothing, and published the entire chain of
        #: thought as the answer.
        self._in_think = in_think
        self._in_call = False

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """Consume one delta, return [(channel, text), ...] ready to display."""
        self.raw += delta
        self._pending += delta
        return self._drain()

    def close(self) -> list[tuple[str, str]]:
        """Flush whatever is left once the stream ends."""
        out = self._drain(final=True)
        if self._pending and not self._in_call:
            channel = "thinking" if self._in_think else "text"
            out.append((channel, self._pending))
        self._pending = ""
        return out

    def _safe_len(self) -> int:
        """How much of the buffer cannot still be the start of a sentinel."""
        buf = self._pending
        tail_start = max(0, len(buf) - _MAX_SENTINEL)
        idx = buf.rfind("<", tail_start)
        if idx == -1:
            return len(buf)
        tail = buf[idx:]
        if any(s.startswith(tail) for s in _SENTINELS):
            return idx
        return len(buf)

    def _drain(self, final: bool = False) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []

        while True:
            buf = self._pending

            if self._in_call:
                # Suppress until the call block closes; if it never does, the
                # non-streaming parse at the end still recovers it.
                end = re.search(r"</tool_call>|</function_calls>|<\|/tool_call\|>", buf)
                if not end:
                    return out
                self._pending = buf[end.end():]
                self._in_call = False
                continue

            if self._in_think:
                end = re.search(r"</(?:" + "|".join(_THINK_TAGS) + r")>", buf, re.I)
                if not end:
                    cut = self._safe_len()
                    if cut > 0:
                        out.append(("thinking", buf[:cut]))
                        self._pending = buf[cut:]
                    return out
                if end.start():
                    out.append(("thinking", buf[: end.start()]))
                self._pending = buf[end.end():]
                self._in_think = False
                continue

            opener = re.search(
                r"<tool_call>|<\|tool_call\|>|<function_calls>|<function=|<\|python_tag\|>"
                r"|<(?:" + "|".join(_THINK_TAGS) + r")>",
                buf, re.I)
            if opener:
                if opener.start():
                    out.append(("text", buf[: opener.start()]))
                matched = opener.group(0).lower()
                self._in_think = matched.startswith("<") and matched.strip("<>/") in _THINK_TAGS
                self._in_call = not self._in_think
                self._pending = buf[opener.end():]
                continue

            cut = len(buf) if final else self._safe_len()
            if cut > 0:
                out.append(("text", buf[:cut]))
                self._pending = buf[cut:]
            return out


# ══════════════════════════════════════════════════════════════════════════════
# Prompt rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_tools_block(schemas: list[dict]) -> str:
    """A `<tools>` preamble for a GGUF whose chat template has no tool support.

    Only used as a fallback. When the template does know about tools, passing
    them as a template variable produces that model's own syntax instead, which
    is what it was trained on and what it parses back reliably.
    """
    lines = ["# Tools", "", "You have access to the following functions:", "", "<tools>"]
    for schema in schemas:
        lines.append(json.dumps(schema.get("function", schema)))
    lines += [
        "</tools>",
        "",
        "To call one, reply with ONLY a tool_call block and no other text:",
        "",
        "<tool_call>",
        '{"name": "example_function", "arguments": {"example_key": "value"}}',
        "</tool_call>",
    ]
    return "\n".join(lines)


def _template_env():
    """A jinja environment with the globals chat templates expect.

    Templates in the wild call `raise_exception`, `strftime_now` (Llama 3.x) and
    the `tojson` filter. A missing global is an UndefinedError at render time,
    i.e. a model that simply refuses to load, so they are all provided.
    """
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError

    def _raise(message: str = "") -> None:
        raise TemplateError(message)

    def _strftime_now(fmt: str) -> str:
        import datetime
        return datetime.datetime.now().strftime(fmt)

    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = _strftime_now
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, **kw)
    return env


def render_chat_template(
    template: str,
    messages: list[dict],
    schemas: list[dict],
    bos: str = "",
    eos: str = "",
    enable_thinking: bool = True,
) -> str:
    """Render a GGUF's own chat template. Raises on any template problem.

    `tools` and `enable_thinking` are passed unconditionally: a template that
    does not use them ignores them silently, and a template that does gets the
    model's native tool syntax rather than our approximation of it.
    """
    env = _template_env()
    tools = [s.get("function", s) for s in schemas] if schemas else None
    return env.from_string(template).render(
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        bos_token=bos,
        eos_token=eos,
    )


def render_chatml(messages: list[dict], schemas: list[dict],
                  enable_thinking: bool = True) -> str:
    """Fallback for a GGUF with no chat template at all.

    ChatML because it is the most widely trained-on format, not because this
    harness assumes it.
    """
    parts: list[str] = []
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    preamble = (render_tools_block(schemas) + "\n\n" + system) if schemas else system
    if preamble.strip():
        parts.append(f"<|im_start|>system\n{preamble.strip()}<|im_end|>\n")

    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "tool":
            # Role name preserved rather than rewritten to `user`: the two
            # backends used to disagree here, which meant the same conversation
            # produced two different prompts depending on which one had loaded.
            parts.append("<|im_start|>tool\n<tool_response>\n"
                         f"{message.get('content', '')}\n</tool_response><|im_end|>\n")
        else:
            content = message.get("content", "") or ""
            for call in message.get("tool_calls") or []:
                fn = call.get("function", call)
                content += ("\n<tool_call>\n"
                            + json.dumps({"name": fn.get("name"),
                                          "arguments": fn.get("arguments", {})})
                            + "\n</tool_call>")
            parts.append(f"<|im_start|>{role}\n{content.strip()}<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    if not enable_thinking:
        # Pre-closing an empty block is how the ChatML family disables thinking.
        parts.append("<think>\n\n</think>\n\n")
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Backends
# ══════════════════════════════════════════════════════════════════════════════

class Backend:
    name = "base"
    display_name = "Backend"
    #: rough tokens/sec, filled in after first generation so the UI can warn
    observed_tps: Optional[float] = None

    def stream(self, messages: list[dict], schemas: list[dict],
               options: dict) -> Iterator[tuple[str, str]]:
        """Yield (channel, text) deltas; return the Completion via StopIteration.

        Callers use `completion = yield from backend.stream(...)`, which lets
        app/session.py stay a single generator all the way to the UI instead of
        threading a callback through a queue.
        """
        raise NotImplementedError

    def generate(self, messages: list[dict], schemas: list[dict],
                 options: dict) -> Completion:
        """Non-streaming convenience — drains stream(). Kept because
        scripts/bench_numeric.py and web/agent.py call it."""
        gen = self.stream(messages, schemas, options)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            return stop.value or Completion()

    def warm(self, messages: list[dict], schemas: list[dict],
             options: dict) -> None:
        """Optionally pre-compute whatever does not change between turns.

        A no-op for servers that cache prefixes themselves (ollama does).
        """

    def count_tokens(self, text: str) -> int:
        """Rough token count, used for the context budget.

        The heuristic is deliberately pessimistic (3.5 chars/token rather than
        the ~4 typical of English prose) because the thing being counted is
        mostly tool output — JSON, tracebacks, tables — which tokenizes far
        worse than prose. Undercounting here means discovering the overflow as
        a hard failure mid-turn instead of compacting a turn early.
        """
        return int(len(text) / 3.5) + 1

    #: Tokens this backend can actually accept. None = unknown, which the
    #: caller reads as "no budget enforcement possible".
    context_limit: Optional[int] = None

    def close(self) -> None:
        pass


class OllamaBackend(Backend):
    """Uses a running ollama server. Fast path: it owns GPU offload."""

    name = "ollama"
    display_name = "Ollama (GPU)"

    def __init__(self, model: str, base_url: str = OLLAMA_BASE) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def probe(base_url: str = OLLAMA_BASE, timeout: float = 1.5) -> list[str]:
        """Model names ollama is serving, or [] if it is not running.

        Short timeout on purpose: this runs during startup and must not stall
        the splash screen on a machine that has never heard of ollama.
        """
        try:
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
            response.raise_for_status()
            return [m.get("name", "") for m in response.json().get("models", [])]
        except requests.RequestException:
            return []

    def _body(self, messages, schemas, options) -> dict:
        # think is ollama's own switch for reasoning models; models without a
        # thinking mode ignore it.
        opts = {k: v for k, v in options.items() if k != "enable_thinking"}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": opts,
            "think": bool(options.get("enable_thinking", True)),
        }
        if schemas:
            body["tools"] = schemas
        return body

    def stream(self, messages, schemas, options):
        try:
            response = requests.post(f"{self.base_url}/api/chat",
                                     json=self._body(messages, schemas, options),
                                     stream=True, timeout=900)
            if response.status_code >= 400:
                raise BackendError(
                    f"the model server returned HTTP {response.status_code}",
                    detail=response.text[:600])
        except requests.RequestException as exc:
            raise BackendError(
                "Could not reach the model server. Is Ollama still running?",
                detail=str(exc)) from exc

        filt = StreamFilter()
        native_calls: list[ToolCall] = []
        thinking_direct = ""
        eval_count = eval_ns = 0

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise BackendError("the model server reported an error",
                                       detail=str(chunk["error"])[:600])

                message = chunk.get("message") or {}

                # Ollama separates reasoning into its own field for models it
                # knows; that path never reaches the filter.
                delta_think = message.get("thinking") or ""
                if delta_think:
                    thinking_direct += delta_think
                    yield ("thinking", delta_think)

                for tc in message.get("tool_calls") or []:
                    fn = tc.get("function", {}) or {}
                    native_calls.append(ToolCall(
                        name=fn.get("name", ""),
                        args=_coerce_args(fn.get("arguments")),
                        id=str(tc.get("id") or f"call_{len(native_calls)}")))

                delta = message.get("content") or ""
                if delta:
                    for event in filt.feed(delta):
                        yield event

                if chunk.get("done"):
                    eval_count = chunk.get("eval_count") or 0
                    eval_ns = chunk.get("eval_duration") or 0
        finally:
            response.close()

        for event in filt.close():
            yield event

        if eval_count and eval_ns:
            self.observed_tps = eval_count / (eval_ns / 1e9)

        text, parsed_think, parsed_calls = parse_tool_calls(filt.raw)
        return Completion(
            text=text,
            thinking=thinking_direct or parsed_think,
            # Structured calls win: the server already decoded them, and the
            # raw text it echoed alongside may be a partial rendering.
            tool_calls=native_calls or parsed_calls,
        )


class LlamaCppBackend(Backend):
    """In-process llama-cpp-python. No server, no install, ships in the binary.

    Engine settings are not left at their defaults. Measured on the shipped 4B
    Q4_K_M, CPU, 6 physical cores, 200-token generations:

        defaults (what this used to do)   5.77 tok/s
        n_threads=12 (all logical)        6.15 tok/s   1.07x
        flash_attn=True                   6.35 tok/s   1.10x
        n_batch=1024, n_threads=6         6.97 tok/s   1.21x   <- chosen
        n_batch=1024 + flash_attn         6.70 tok/s   1.16x

    n_batch and flash_attn do not stack, and threads past the physical core
    count cost more than they return, so the pairing is not the obvious one.
    """

    name = "llamacpp"
    display_name = "Built-in (CPU)"

    def __init__(self, model_path: str, n_ctx: int = 8192,
                 n_gpu_layers: int = 0, n_batch: int = 1024,
                 n_threads: Optional[int] = None, verbose: bool = False) -> None:
        from llama_cpp import Llama  # imported lazily: heavy, and optional

        if n_threads is None:
            n_threads = _physical_cores()

        self.model_path = model_path
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx,
                          n_gpu_layers=n_gpu_layers, n_batch=n_batch,
                          n_threads=n_threads, n_threads_batch=n_threads,
                          verbose=verbose)
        metadata = self._llm.metadata or {}
        self._template = metadata.get("tokenizer.chat_template", "")
        self._bos = metadata.get("tokenizer.ggml.bos_token", "") or ""
        self._eos = metadata.get("tokenizer.ggml.eos_token", "") or ""
        #: Set once if the GGUF's template raises, so a broken template degrades
        #: to ChatML for the rest of the session instead of on every single turn.
        self._template_failed = False
        self.n_ctx = n_ctx
        self.context_limit = n_ctx
        if n_gpu_layers:
            self.display_name = "Built-in (GPU)"

    def count_tokens(self, text: str) -> int:
        """The model's real tokenizer, so the budget matches the engine's own
        accounting rather than a guess about it."""
        try:
            return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False,
                                          special=True))
        except Exception:  # noqa: BLE001 - fall back rather than fail a turn
            return super().count_tokens(text)

    @staticmethod
    def gpu_available() -> bool:
        try:
            from llama_cpp import llama_cpp as _c
            return bool(_c.llama_supports_gpu_offload())
        except Exception:  # noqa: BLE001 - absence is a normal answer here
            return False

    def _render_prompt(self, messages: list[dict], schemas: list[dict],
                       enable_thinking: bool) -> str:
        if self._template and not self._template_failed:
            try:
                return render_chat_template(
                    self._template, messages, schemas,
                    bos=self._bos, eos=self._eos,
                    enable_thinking=enable_thinking)
            except Exception as exc:  # noqa: BLE001 - any template bug at all
                self._template_failed = True
                logger.warning(
                    "this GGUF's chat template failed to render (%s) — falling "
                    "back to ChatML for the rest of this session", exc)
        return render_chatml(messages, schemas, enable_thinking)

    def warm(self, messages, schemas, options) -> None:
        """Prefill the system-prompt-and-tools prefix, for the first turn only.

        Measured on the shipped 4B: prefill runs at ~55 tok/s on CPU and the
        static prefix is ~1745 tokens (805 of system prompt, 940 of tool
        schemas), so the opening question of a conversation pays ~32s before
        its first token — longer than writing the answer costs. Warming behind
        the splash takes that to 0.7s.

        It helps once and only once, and that is a llama-cpp-python limitation
        rather than a design choice. Its prefix-reuse path calls
        `kv_cache_seq_rm` to rewind the cache, and in this build that returns
        False after any completion has run (verified: a manual trim-and-eval of
        the suffix fails with `llama_decode returned -1` from the same state).
        The library detects the prefix match, logs "partial kv removal not
        supported", and re-evaluates the whole prompt. So every turn after the
        first pays full prefill again, and every iteration within a tool-using
        turn pays it too.

        Two things actually fix that, neither of them here: a llama-cpp-python
        build whose KV cache can be rewound, or the GPU path, where prefill is
        fast enough that the question stops mattering. Ollama is unaffected —
        it caches prefixes server-side and does not use this code.
        """
        system = [m for m in messages if m.get("role") == "system"] or messages[:1]
        if not system:
            return
        try:
            # Rendered with a throwaway user turn rather than system-only:
            # several templates (Qwen3's included) raise "No user query found
            # in messages" on a system-only render, and that exception would
            # otherwise trip _template_failed and silently downgrade the whole
            # session to ChatML — a warm-up must never change how real turns
            # are rendered.
            probes = [
                self._render_prompt(system + [{"role": "user", "content": marker}],
                                    schemas, True)
                for marker in ("aaaa", "bbbb")
            ]
            token_lists = [self._llm.tokenize(p.encode("utf-8"), special=True)
                           for p in probes]
            # The shared prefix of two prompts that differ only in the user
            # message is exactly the static part — found by comparison rather
            # than by assuming where the template puts its turn boundaries.
            shared = 0
            for a, b in zip(*token_lists):
                if a != b:
                    break
                shared += 1
            if shared < 16:
                return
            self._llm.reset()
            self._llm.eval(token_lists[0][:shared])
            logger.info("warmed %d shared prefix tokens", shared)
        except Exception:  # noqa: BLE001 - warming is an optimisation, never a
            # precondition; a failure here must not stop the app from starting.
            logger.debug("prefix warm failed", exc_info=True)

    def stream(self, messages, schemas, options):
        import time

        prompt = self._render_prompt(
            messages, schemas, bool(options.get("enable_thinking", True)))

        stops = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>"]
        if self._eos and self._eos not in stops:
            stops.append(self._eos)

        started = time.time()
        produced = 0
        #: Decode is timed from the first token, not from the call. Timing the
        #: whole span folds prefill into the rate and reports something that is
        #: neither: on this machine a 30-token reply behind a 29s prefill came
        #: out as "0.9 tok/s" when decode was really about 7. That number is
        #: what a GPU build gets judged on, so it has to mean one thing.
        first_token_at = None
        pre_opened = prompt_opens_thinking(prompt)
        filt = StreamFilter(in_think=pre_opened)

        try:
            chunks = self._llm(
                prompt,
                max_tokens=int(options.get("num_predict", 1024)),
                temperature=float(options.get("temperature", 0.6)),
                top_p=float(options.get("top_p", 0.9)),
                top_k=int(options.get("top_k", 40)),
                repeat_penalty=float(options.get("repeat_penalty", 1.05)),
                stop=stops,
                stream=True,
            )
            for chunk in chunks:
                delta = chunk["choices"][0].get("text", "")
                if not delta:
                    continue
                if first_token_at is None:
                    first_token_at = time.time()
                produced += 1
                for event in filt.feed(delta):
                    yield event
        except ValueError as exc:
            # llama.cpp raises this when the rendered prompt is longer than
            # n_ctx. Naming the real limit matters: the user's setting and the
            # window the engine was built with have been different before.
            raise BackendError(
                "This conversation no longer fits in the model's context "
                f"window ({self.n_ctx} tokens). Start a new chat, or raise the "
                "context size in Settings.", detail=str(exc)) from exc

        for event in filt.close():
            yield event

        # produced - 1: the first token is the one prefill produced, so counting
        # it against the decode span inflates the rate on short replies.
        decode_span = (time.time() - first_token_at) if first_token_at else 0.0
        if produced > 1 and decode_span > 0:
            self.observed_tps = (produced - 1) / decode_span
        self.last_prefill_seconds = (
            (first_token_at - started) if first_token_at else None)

        # Re-attach the opener the template supplied, so the non-streaming
        # parse sees the same balanced text the filter did.
        raw = ("<think>" + filt.raw) if pre_opened else filt.raw
        text, thinking, calls = parse_tool_calls(raw)
        return Completion(text=text, thinking=thinking, tool_calls=calls)

    def close(self) -> None:
        self._llm = None


def set_gpu_device(index: int) -> None:
    """Restrict llama.cpp to one GPU, before its shared library loads.

    A laptop with switchable graphics presents two Vulkan devices and llama.cpp
    takes device 0, which on this class of machine is the *integrated* GPU:

        0 = AMD Radeon 740M Graphics   uma: 1
        1 = NVIDIA GeForce RTX 4050    uma: 0

    Device 0 is the wrong one and nothing says so — it is fast enough (46.7
    tok/s measured, 6.7x CPU) to look like the discrete card is working. The
    integrated part shares system RAM, so it is also bandwidth-starved exactly
    where decode is bandwidth-bound.

    Filtering by environment variable rather than by main_gpu because it has to
    happen before the ggml backends register at library load; main_gpu is read
    afterwards and cannot un-choose a device that is already initialised. The
    variables filter and reindex like CUDA_VISIBLE_DEVICES, so the survivor
    becomes device 0 and main_gpu stays at its default.

    index < 0 leaves everything alone, which is the default: guessing wrong
    here means no GPU at all rather than a slower one.
    """
    if index < 0:
        return
    for name in ("GGML_VK_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                 "HIP_VISIBLE_DEVICES"):
        os.environ.setdefault(name, str(index))
    logger.info("restricted GPU selection to device %d", index)


def _physical_cores() -> int:
    """Physical cores, falling back to half the logical count.

    Measured: 6 threads on a 6-core/12-thread box beat 12 by 13%. Hyperthreads
    contend for the same vector units, which is the whole bottleneck here.
    """
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        if cores:
            return int(cores)
    except Exception:  # noqa: BLE001 - psutil is optional
        pass
    return max((os.cpu_count() or 4) // 2, 1)


def _coerce_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": raw}
    if raw is None:
        return {}
    return {"value": raw}


def select_backend(model_path: Optional[str], ollama_model_hint: str = "vasudha",
                   prefer: Optional[str] = None, n_ctx: int = 8192,
                   n_batch: int = 1024, n_threads: int = 0) -> Backend:
    """Pick the fastest backend that will actually work on this machine.

    Order: explicit preference, then a running ollama (GPU, ~5-10x faster in
    measurement), then the bundled GGUF. Raises only if nothing is usable, so
    the caller can send the user to the first-run download screen.
    """
    def _builtin() -> Backend:
        gpu = LlamaCppBackend.gpu_available()
        logger.info("using built-in llama.cpp backend (gpu=%s, n_ctx=%s, n_batch=%s)",
                    gpu, n_ctx, n_batch)
        return LlamaCppBackend(model_path, n_ctx=n_ctx,
                               n_gpu_layers=-1 if gpu else 0,
                               n_batch=n_batch,
                               n_threads=n_threads or None)

    if prefer == "llamacpp" and model_path and os.path.exists(model_path):
        return _builtin()

    if prefer in (None, "ollama"):
        served = OllamaBackend.probe()
        if served:
            match = next((n for n in served if ollama_model_hint in n), None)
            # Any served model, not only a Vasudha one: this harness is used to
            # compare other models, and refusing to attach to a running ollama
            # because the name did not match would make that impossible.
            if match is None and prefer == "ollama":
                match = served[0]
            if match:
                logger.info("using ollama backend with model %s", match)
                return OllamaBackend(match)

    if model_path and os.path.exists(model_path):
        return _builtin()

    raise RuntimeError("no model available: download one on first run")
