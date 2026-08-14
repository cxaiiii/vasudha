"""Inference backends for the desktop app, and the logic that picks one.

The app must run on a machine with nothing installed, but it must not feel
broken on a machine that has a GPU. Those pull in opposite directions, so the
backend is chosen at runtime rather than baked in:

    OllamaBackend    - if an ollama server is already running locally. Uses the
                       user's GPU, handles its own model lifecycle. Measured on
                       the dev box: full tool-call turns in 6-20s.
    LlamaCppBackend  - in-process llama-cpp-python against a GGUF file. Needs
                       no install and no server, which is the whole point of
                       shipping a binary. Measured with the CPU-only wheel:
                       5.5 tok/s, i.e. 229 tokens in 41.7s. Correct, but slow.

Both expose the same `generate()` and both return the same ToolCall objects, so
web/agent.py's loop does not care which one it got.

Tool calling
------------
We do NOT depend on a backend to parse tool calls for us. The vasudha GGUF
carries the Qwen3 chat template in its metadata (verified: 7756 chars, contains
'tool_call'), and that template emits a specific, stable format:

    <tool_call>
    <function=python_tool>
    <parameter=code>
    ...source...
    </parameter>
    </function>
    </tool_call>

parse_tool_calls() below reads exactly that. Ollama happens to parse it into
structured `tool_calls` for us; llama-cpp-python does not, so the same parser
runs over raw text there. One format, one parser, two transports.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

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


@dataclass
class Completion:
    """One assistant turn: prose plus any tool calls it asked for."""
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


# --- Qwen3 tool-call format -------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([A-Za-z_]\w*)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([A-Za-z_]\w*)>\n?(.*?)\n?</parameter>", re.S)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def parse_tool_calls(text: str) -> tuple[str, str, list[ToolCall]]:
    """Split raw model output into (visible_text, thinking, tool_calls)."""
    thinking = ""
    think_match = _THINK_RE.search(text)
    if think_match:
        thinking = think_match.group(1).strip()
        text = _THINK_RE.sub("", text, count=1)

    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text):
        args = {name: value.strip() for name, value in _PARAM_RE.findall(match.group(2))}
        calls.append(ToolCall(name=match.group(1), args=args))

    visible = _TOOL_CALL_RE.sub("", text).strip()
    return visible, thinking, calls


def render_tools_block(schemas: list[dict]) -> str:
    """The <tools> preamble the Qwen3 template expects, built by hand so it is
    identical whichever backend is in use."""
    lines = ["# Tools", "", "You have access to the following functions:", "", "<tools>"]
    for schema in schemas:
        lines.append(json.dumps(schema.get("function", schema)))
    lines += [
        "</tools>",
        "",
        "If you choose to call a function ONLY reply in the following format with NO suffix:",
        "",
        "<tool_call>",
        "<function=example_function_name>",
        "<parameter=example_parameter_1>",
        "value_1",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ]
    return "\n".join(lines)


# --- backends ---------------------------------------------------------------

class Backend:
    name = "base"
    display_name = "Backend"
    #: rough tokens/sec, filled in after first generation so the UI can warn
    observed_tps: Optional[float] = None

    def generate(self, messages: list[dict], schemas: list[dict],
                 options: dict) -> Completion:
        raise NotImplementedError

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

    #: How much room to give a retry after a truncated tool call, and the
    #: ceiling past which we stop growing and admit defeat.
    RETRY_MULTIPLIER = 3
    RETRY_CEILING = 6144

    def _post(self, messages, schemas, options) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if schemas:
            body["tools"] = schemas
        response = requests.post(f"{self.base_url}/api/chat", json=body, timeout=900)
        if response.status_code >= 400:
            raise BackendError(
                f"the model server returned HTTP {response.status_code}",
                detail=response.text[:600])
        return response.json()

    def generate(self, messages, schemas, options) -> Completion:
        try:
            payload = self._post(messages, schemas, options)
        except BackendError:
            # Observed failure mode: the model runs to the num_predict ceiling
            # mid-<tool_call>, ollama's PEG tool parser cannot parse the
            # truncated block, and the whole request 500s. Measured: exactly
            # 1024 generated tokens against num_predict=1024, then HTTP 500.
            # Retrying with a bigger budget lets the call finish and parse.
            budget = int(options.get("num_predict", 1024))
            if budget >= self.RETRY_CEILING:
                raise
            roomier = dict(options)
            roomier["num_predict"] = min(budget * self.RETRY_MULTIPLIER, self.RETRY_CEILING)
            logger.warning("retrying with num_predict=%s after a truncated response",
                           roomier["num_predict"])
            try:
                payload = self._post(messages, schemas, roomier)
            except BackendError as exc:
                raise BackendError(
                    "The model produced a reply this app could not read, twice. "
                    "Try rephrasing, or start a new chat.", detail=exc.detail) from exc
        except requests.RequestException as exc:
            raise BackendError(
                "Could not reach the model server. Is Ollama still running?",
                detail=str(exc)) from exc

        message = payload.get("message", {}) or {}

        calls = [
            ToolCall(name=(tc.get("function", {}) or {}).get("name", ""),
                     args=_coerce_args((tc.get("function", {}) or {}).get("arguments")))
            for tc in (message.get("tool_calls") or [])
        ]
        text = message.get("content") or ""
        thinking = message.get("thinking") or ""

        # Older/edge builds sometimes leave the raw block in content instead of
        # parsing it; fall back to our own parser rather than losing the call.
        if not calls and "<tool_call>" in text:
            text, extra_think, calls = parse_tool_calls(text)
            thinking = thinking or extra_think

        eval_count = payload.get("eval_count") or 0
        eval_ns = payload.get("eval_duration") or 0
        if eval_count and eval_ns:
            self.observed_tps = eval_count / (eval_ns / 1e9)

        return Completion(text=text, thinking=thinking, tool_calls=calls)


class LlamaCppBackend(Backend):
    """In-process llama-cpp-python. No server, no install, ships in the binary.

    Applies the GGUF's own chat template via llama-cpp's tokenizer metadata
    rather than hardcoding ChatML here, so a future checkpoint that changes its
    template keeps working.
    """

    name = "llamacpp"
    display_name = "Built-in (CPU)"

    def __init__(self, model_path: str, n_ctx: int = 8192,
                 n_gpu_layers: int = 0, verbose: bool = False) -> None:
        from llama_cpp import Llama  # imported lazily: heavy, and optional

        self.model_path = model_path
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx,
                          n_gpu_layers=n_gpu_layers, verbose=verbose)
        metadata = self._llm.metadata or {}
        self._template = metadata.get("tokenizer.chat_template", "")
        if n_gpu_layers:
            self.display_name = "Built-in (GPU)"

    @staticmethod
    def gpu_available() -> bool:
        try:
            from llama_cpp import llama_cpp as _c
            return bool(_c.llama_supports_gpu_offload())
        except Exception:  # noqa: BLE001 - absence is a normal answer here
            return False

    def _render_prompt(self, messages: list[dict], schemas: list[dict]) -> str:
        """Build the ChatML prompt the Qwen3 template defines.

        Rendered directly instead of running the GGUF's jinja: the template's
        tool branch expects ollama-shaped inputs, and reproducing it by hand
        keeps the <tools> preamble byte-identical to what OllamaBackend sends.
        """
        parts: list[str] = []
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        preamble = render_tools_block(schemas) + "\n\n" + system if schemas else system
        if preamble.strip():
            parts.append(f"<|im_start|>system\n{preamble.strip()}<|im_end|>\n")

        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role == "tool":
                parts.append("<|im_start|>user\n<tool_response>\n"
                             f"{message.get('content', '')}\n</tool_response><|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{role}\n{message.get('content', '')}<|im_end|>\n")

        parts.append("<|im_start|>assistant\n<think>\n")
        return "".join(parts)

    def generate(self, messages, schemas, options) -> Completion:
        import time

        prompt = self._render_prompt(messages, schemas)
        started = time.time()
        result = self._llm(
            prompt,
            max_tokens=int(options.get("num_predict", 1024)),
            temperature=float(options.get("temperature", 0.6)),
            top_p=float(options.get("top_p", 0.9)),
            top_k=int(options.get("top_k", 40)),
            repeat_penalty=float(options.get("repeat_penalty", 1.05)),
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        elapsed = time.time() - started
        produced = (result.get("usage", {}) or {}).get("completion_tokens", 0)
        if produced and elapsed:
            self.observed_tps = produced / elapsed

        raw = result["choices"][0]["text"]
        # The prompt opened <think>, so the model's output closes it. Re-open it
        # here so one parser handles both backends.
        if "</think>" in raw and "<think>" not in raw:
            raw = "<think>\n" + raw
        text, thinking, calls = parse_tool_calls(raw)
        return Completion(text=text, thinking=thinking, tool_calls=calls)

    def close(self) -> None:
        self._llm = None


def _coerce_args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": raw}
    return {}


def select_backend(model_path: Optional[str], ollama_model_hint: str = "vasudha",
                   prefer: Optional[str] = None) -> Backend:
    """Pick the fastest backend that will actually work on this machine.

    Order: explicit preference, then a running ollama (GPU, ~5-10x faster in
    measurement), then the bundled GGUF. Raises only if nothing is usable, so
    the caller can send the user to the first-run download screen.
    """
    if prefer == "llamacpp" and model_path and os.path.exists(model_path):
        return LlamaCppBackend(model_path, n_gpu_layers=-1 if LlamaCppBackend.gpu_available() else 0)

    if prefer in (None, "ollama"):
        served = OllamaBackend.probe()
        if served:
            match = next((n for n in served if ollama_model_hint in n), None)
            if match:
                logger.info("using ollama backend with model %s", match)
                return OllamaBackend(match)

    if model_path and os.path.exists(model_path):
        gpu = LlamaCppBackend.gpu_available()
        logger.info("using built-in llama.cpp backend (gpu=%s)", gpu)
        return LlamaCppBackend(model_path, n_gpu_layers=-1 if gpu else 0)

    raise RuntimeError("no model available: download one on first run")
