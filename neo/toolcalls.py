"""Tool calls in Qwen3.5's native format: parse, render, and check.

Qwen3.5 (and therefore Vasudha Neo) calls tools by emitting

    <tool_call>
    <function=name>
    <parameter=arg>
    value
    </parameter>
    </function>
    </tool_call>

This module is deliberately independent of any application harness: it reads
that format from raw text, renders tool schemas into a system message exactly
the way the model's own chat template does, and grades calls against a gold
answer the way the Berkeley Function Calling Leaderboard does (AST match:
right function, every required argument present with an acceptable value, no
unknown arguments).
"""
from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

_CALL_RE = re.compile(r"<tool_call>\s*<function=([^\s>]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM_RE = re.compile(r"<parameter=([^\s>]+)>\n?(.*?)\n?</parameter>", re.S)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)


@dataclass
class ParsedCall:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class ParsedReply:
    content: str                     # visible text with tool-call blocks removed
    reasoning: str                   # text inside <think>...</think>, if any
    calls: list[ParsedCall]
    malformed: bool                  # tool-call markup that did not parse


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, rest). Handles a completion that starts inside a think block
    (the generation prompt already opened it) as well as a full block."""
    text = text or ""
    if "</think>" in text:
        head, rest = text.split("</think>", 1)
        return head.replace("<think>", "").strip(), rest.lstrip("\n")
    return "", text


def parse_reply(text: str, schemas: Optional[list[dict]] = None) -> ParsedReply:
    reasoning, body = split_thinking(text)
    types = schema_types(schemas or [])
    calls = []
    for name, inner in _CALL_RE.findall(body):
        args = {}
        for pname, raw in _PARAM_RE.findall(inner):
            args[pname] = coerce_value(raw, types.get(name, {}).get(pname))
        calls.append(ParsedCall(name.strip(), args))
    content = _CALL_RE.sub("", body).strip()
    malformed = bool(re.search(r"<tool_call>|<function=|</tool_call>", content))
    return ParsedReply(content, reasoning, calls, malformed)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "dict": "object", "object": "object", "float": "number", "number": "number", "double": "number",
    "int": "integer", "integer": "integer", "str": "string", "string": "string", "list": "array",
    "array": "array", "tuple": "array", "bool": "boolean", "boolean": "boolean", "any": "string",
}


def _json_type(t: Any) -> str:
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "string")
    t = str(t or "string").lower()
    base = re.split(r"[,\[ ]", t)[0]
    if base.startswith("list"):
        return "array"
    if base.startswith("dict"):
        return "object"
    return _TYPE_MAP.get(base, "string")


def _convert_property(spec: dict) -> dict:
    out = {"type": _json_type(spec.get("type", "string"))}
    if spec.get("description"):
        out["description"] = str(spec["description"])
    if "enum" in spec:
        out["enum"] = spec["enum"]
    if out["type"] == "array":
        items = spec.get("items")
        if isinstance(items, dict):
            out["items"] = _convert_property(items)
    if out["type"] == "object" and isinstance(spec.get("properties"), dict):
        out["properties"] = {k: _convert_property(v) for k, v in spec["properties"].items() if isinstance(v, dict)}
    if "default" in spec and spec["default"] not in ("", None):
        out["default"] = spec["default"]
    return out


def to_openai_tool(tool: dict) -> dict:
    """Normalise BFCL / xLAM / OpenAI-style tool specs into
    {"type": "function", "function": {name, description, parameters}}."""
    fn = tool.get("function", tool)
    params = fn.get("parameters") or {}
    if "properties" in params:                      # JSON-schema style (BFCL, OpenAI)
        props = {k: _convert_property(v) for k, v in (params.get("properties") or {}).items() if isinstance(v, dict)}
        required = [r for r in (params.get("required") or []) if r in props]
    else:                                           # xLAM style: {param: {type, description, default}}
        props, required = {}, []
        for k, v in params.items():
            if not isinstance(v, dict):
                continue
            props[k] = _convert_property(v)
            optional = "optional" in str(v.get("type", "")).lower() or ("default" in v and v["default"] not in ("", None))
            if not optional:
                required.append(k)
    return {"type": "function", "function": {
        "name": str(fn.get("name", "")),
        "description": str(fn.get("description", "")),
        "parameters": {"type": "object", "properties": props, "required": required},
    }}


def schema_types(schemas: Iterable[dict]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for s in schemas:
        fn = s.get("function", s)
        props = (fn.get("parameters") or {}).get("properties") or {}
        out[fn.get("name", "")] = {k: _json_type(v.get("type", "string")) for k, v in props.items() if isinstance(v, dict)}
    return out


def coerce_value(raw: str, json_type: Optional[str]) -> Any:
    """Turn a <parameter> body into a Python value, guided by the schema type
    (vLLM's qwen3_coder parser does the same)."""
    text = raw.strip("\n")
    stripped = text.strip()
    if json_type == "string":
        return text
    if json_type in ("integer", "number"):
        try:
            value = float(stripped)
            if json_type == "integer" and value.is_integer():
                return int(value)
            return value
        except ValueError:
            return text
    if json_type == "boolean":
        if stripped.lower() in ("true", "false"):
            return stripped.lower() == "true"
        return text
    if json_type in ("array", "object") or json_type is None:
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(stripped)
            except (ValueError, SyntaxError, TypeError, MemoryError):
                continue
        if json_type is None:
            low = stripped.lower()
            if low in ("true", "false"):
                return low == "true"
        return text
    return text


# ---------------------------------------------------------------------------
# rendering — identical to the Qwen3.5 chat template's tools block
# ---------------------------------------------------------------------------

_TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_FOOTER = (
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
    "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
    "the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)


def render_tools_system(tools: list[dict], system: Optional[str] = None) -> str:
    """The system-message text the Qwen3.5 template produces for `tools`.

    Putting this text in the system message and rendering the conversation
    WITHOUT tools yields the same tokens as rendering it WITH tools, which lets
    a dataset carry per-example tool sets while the trainer uses one template.
    tests/test_neo_toolcalls.py checks the equivalence against the real
    template when it is available.
    """
    parts = [_TOOLS_HEADER]
    for tool in tools:
        parts.append("\n" + json.dumps(tool, ensure_ascii=False))
    parts.append(_TOOLS_FOOTER)
    text = "".join(parts)
    if system and system.strip():
        text += "\n\n" + system.strip()
    return text


# ---------------------------------------------------------------------------
# checking calls against gold
# ---------------------------------------------------------------------------

def _std_string(s: str) -> str:
    """BFCL's string standardisation: case, whitespace and some punctuation do
    not matter ("San Francisco, CA" == "san francisco ca")."""
    s = str(s).lower()
    s = re.sub(r"[\s,./\-_*^]", "", s)
    return s.replace("'", '"')


def values_equal(pred: Any, gold: Any) -> bool:
    if isinstance(gold, bool) or isinstance(pred, bool):
        if isinstance(pred, str):
            pred = pred.strip().lower() == "true" if pred.strip().lower() in ("true", "false") else pred
        return pred == gold
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        try:
            p = float(pred)
        except (TypeError, ValueError):
            return False
        return math.isclose(p, float(gold), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(gold, str):
        if isinstance(pred, (int, float)) and not isinstance(pred, bool):
            try:
                return math.isclose(float(gold), float(pred), rel_tol=1e-6, abs_tol=1e-9)
            except ValueError:
                return False
        return _std_string(pred) == _std_string(gold)
    if isinstance(gold, (list, tuple)):
        if isinstance(pred, str):
            pred = coerce_value(pred, "array")
        if not isinstance(pred, (list, tuple)) or len(pred) != len(gold):
            return False
        return all(values_equal(p, g) for p, g in zip(pred, gold))
    if isinstance(gold, dict):
        if isinstance(pred, str):
            pred = coerce_value(pred, "object")
        if not isinstance(pred, dict) or set(pred) != set(gold):
            return False
        return all(values_equal(pred[k], gold[k]) for k in gold)
    return pred == gold


def _bfcl_value_ok(pred: Any, acceptable: list) -> bool:
    for option in acceptable:
        if option == "":
            continue
        # BFCL nests dict options as lists of acceptable values per key
        if isinstance(option, dict) and isinstance(pred, dict):
            if set(pred) - set(option):
                continue
            if all(k in pred and any(values_equal(pred[k], o) for o in (v if isinstance(v, list) else [v]))
                   or (k not in pred and "" in (v if isinstance(v, list) else [v]))
                   for k, v in option.items()):
                return True
            continue
        if values_equal(pred, option):
            return True
    return False


def bfcl_call_ok(call: ParsedCall, gold: dict, schema_props: Optional[dict] = None) -> bool:
    """`gold` is one BFCL possible-answer entry: {func_name: {param: [acceptable...]}}."""
    (gname, gparams), = gold.items()
    if call.name != gname:
        return False
    if schema_props is not None and set(call.arguments) - set(schema_props):
        return False                               # hallucinated argument
    for param, acceptable in gparams.items():
        optional = "" in acceptable
        if param not in call.arguments:
            if not optional:
                return False
            continue
        if not _bfcl_value_ok(call.arguments[param], acceptable):
            return False
    for param in call.arguments:
        if param not in gparams and schema_props is None:
            return False
    return True


def bfcl_match(calls: list[ParsedCall], ground_truth: list[dict], schemas: list[dict]) -> bool:
    """All gold calls matched one-to-one (order-insensitive), nothing extra."""
    if ground_truth is None:            # unscorable item: never count it as correct
        return False
    if len(calls) != len(ground_truth):
        return False
    props = {}
    for s in schemas:
        fn = s.get("function", s)
        props[fn.get("name")] = ((fn.get("parameters") or {}).get("properties") or {})
    remaining = list(range(len(calls)))
    for gold in ground_truth:
        (gname, _), = gold.items()
        hit = next((i for i in remaining if bfcl_call_ok(calls[i], gold, props.get(gname))), None)
        if hit is None:
            return False
        remaining.remove(hit)
    return True


def call_match_score(calls: list[ParsedCall], gold: list[dict]) -> float:
    """Training reward for function calling with exact gold calls
    ({"name", "arguments"} each): fraction of gold calls reproduced exactly,
    divided by the larger of the two counts so extra calls cost too."""
    if not gold:
        return 1.0 if not calls else 0.0
    remaining = list(range(len(calls)))
    hits = 0
    for g in gold:
        for i in remaining:
            c = calls[i]
            if c.name == g["name"] and set(c.arguments) == set(g.get("arguments") or {}) and all(
                    values_equal(c.arguments[k], v) for k, v in (g.get("arguments") or {}).items()):
                hits += 1
                remaining.remove(i)
                break
    return hits / max(len(gold), len(calls))
