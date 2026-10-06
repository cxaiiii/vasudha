"""Score one finished episode: the task, the model's turns, and any tool calls.

The same function is the RL reward (neo.rewards wraps it for TRL) and the
evaluation metric (neo.evaluate calls it on vLLM rollouts), so a number in a
report means exactly what the training signal meant.

Task types:

numeric   value + unit, graded by neo.grading (unit-aware, no hedging).
math      math-verify equivalence of the \\boxed{} answer.
code      the reply's code passes the task's hidden tests in the sandbox.
fc        function calling: the first turn's tool calls vs. gold calls
          (exact args) or BFCL possible answers; "no call" when no tool fits.
ifeval    verifiable formatting constraints (IFEval checkers from lm-eval).
choice    multiple choice, letter answer.

General shaping, applied to every type:
  * an empty final reply is penalised, more so right after a tool result
    (a 4B model going silent after a correct tool output is a known failure);
  * tool-call markup left in the text (a call that does not parse) is
    penalised — any harness would silently drop it;
  * failed sandbox runs inside an otherwise correct episode cost a little:
    clean tool use beats trial and error (cf. GRPO-RoC, rStar2-Agent).
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Optional, Sequence

from neo.grading import grade_numeric
from neo.toolcalls import ParsedCall, bfcl_match, call_match_score, coerce_value, schema_types

EMPTY_AFTER_TOOL = -0.5
EMPTY = -0.3
MALFORMED_TOOL_TEXT = -0.3
FAILED_CALL = 0.03
FAILED_CALL_CAP = 0.1


@dataclass
class ToolRecord:
    name: str
    result: str = ""
    ok: bool = True


@dataclass
class EpisodeScore:
    reward: float
    correct: bool
    metrics: dict = field(default_factory=dict)


_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_TOOL_MARKUP = re.compile(r"<tool_call>|<function=|</tool_call>")


def strip_think(text: str) -> str:
    text = _THINK.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


def _content(message: dict) -> str:
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return strip_think(content).strip()


def final_reply(messages: Sequence[dict]) -> str:
    """Text of the last assistant turn; "" if the episode ended on a tool call."""
    for m in reversed(list(messages or [])):
        if m.get("role") == "assistant":
            if m.get("tool_calls"):
                return ""
            return _content(m)
    return ""


def first_turn(messages: Sequence[dict]) -> tuple[str, list[ParsedCall]]:
    """Content and tool calls of the first assistant turn (function calling)."""
    for m in messages or []:
        if m.get("role") == "assistant":
            calls = []
            for call in m.get("tool_calls") or []:
                fn = call.get("function", call)
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"__raw__": args}
                calls.append(ParsedCall(fn.get("name", ""), dict(args)))
            return _content(m), calls
    return "", []


def tool_records(messages: Sequence[dict]) -> list[ToolRecord]:
    records: list[ToolRecord] = []
    pending: list[str] = []
    for m in messages or []:
        if m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                pending.append(call.get("function", call).get("name", ""))
        elif m.get("role") == "tool":
            name = pending.pop(0) if pending else m.get("name", "")
            content = m.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content)
            failed = ("Traceback (most recent call last)" in content or "TimeoutError" in content
                      or content.startswith(("{'error'", '{"error"', "SystemError")))
            records.append(ToolRecord(name, content, not failed))
    return records


# ---------------------------------------------------------------------------
# per-type scorers: (task, messages, tools) -> EpisodeScore, reply non-empty
# ---------------------------------------------------------------------------

def score_numeric(task: dict, reply: str, tools: list[ToolRecord]) -> EpisodeScore:
    unit = task.get("unit", "")
    g = grade_numeric(reply, float(task["answer"]), None if unit == "*" else unit, float(task.get("tol", 0.015)),
                      givens=task.get("givens") or (), absolute=bool(task.get("absolute")))
    failed = sum(1 for t in tools if not t.ok)
    reward = g.score - min(FAILED_CALL_CAP, FAILED_CALL * failed) if g.correct else 0.0
    strict = g.correct and g.score == 1.0
    return EpisodeScore(reward, strict, {"answer_correct": float(g.correct), "strict_correct": float(strict),
                                         "hedged": float(g.hedged), "missing_unit": float(g.missing_unit),
                                         "tool_used": float(bool(tools)), "tool_errors": float(failed)})


def math_equal(gold: str, reply: str) -> bool:
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    except ImportError:  # pragma: no cover - math-verify ships in every image
        return False
    # math-verify's timeouts use signals, which only work on the main thread.
    timeout = 5 if threading.current_thread() is threading.main_thread() else None
    gold_text = gold if ("\\boxed" in gold or gold.strip().startswith("$")) else f"\\boxed{{{gold}}}"
    try:
        g = parse(gold_text, parsing_timeout=timeout)
        p = parse(reply, extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
                  parsing_timeout=timeout)
        return bool(g) and bool(p) and bool(verify(g, p, timeout_seconds=timeout))
    except Exception:  # noqa: BLE001
        return False


def score_math(task: dict, reply: str, tools: list[ToolRecord]) -> EpisodeScore:
    ok = math_equal(str(task["answer"]), reply)
    failed = sum(1 for t in tools if not t.ok)
    reward = (1.0 - min(FAILED_CALL_CAP, FAILED_CALL * failed)) if ok else 0.0
    return EpisodeScore(reward, ok, {"answer_correct": float(ok), "tool_used": float(bool(tools))})


_FENCE = re.compile(r"```(?:python|py|Python|python3)?[ \t]*\n(.*?)```", re.S)


def extract_code(reply: str) -> str:
    blocks = _FENCE.findall(reply or "")
    if blocks:
        for block in reversed(blocks):
            if "def " in block or "class " in block:
                return block
        return blocks[-1]
    return reply if "def " in (reply or "") else ""


_KODCODE_RUNNER = r'''
import sys, types, inspect, traceback
_src = {src!r}
_mod = types.ModuleType("solution")
_mod.__file__ = "solution.py"
exec(compile(_src, "solution.py", "exec"), _mod.__dict__)
sys.modules["solution"] = _mod
_ns = {{"__name__": "test_solution"}}
exec(compile({tests!r}, "test_solution.py", "exec"), _ns)
_passed = _failed = _skipped = 0
for _name, _fn in list(_ns.items()):
    if _name.startswith("test") and inspect.isfunction(_fn):
        if inspect.signature(_fn).parameters:
            _skipped += 1
            continue
        try:
            _fn()
            _passed += 1
        except Exception:
            _failed += 1
            traceback.print_exc(limit=2)
print(f"__NEO_TESTS__ passed={{_passed}} failed={{_failed}} skipped={{_skipped}}")
'''

_HUMANEVAL_RUNNER = r'''
{program}

{tests}

check({entry_point})
print("__NEO_TESTS__ passed=1 failed=0 skipped=0")
'''

_ASSERT_RUNNER = r'''
{program}

{setup}
{asserts}
print("__NEO_TESTS__ passed=1 failed=0 skipped=0")
'''

_TESTS_LINE = re.compile(r"__NEO_TESTS__ passed=(\d+) failed=(\d+) skipped=(\d+)")


def build_test_program(task: dict, code: str) -> str:
    style = task.get("test_style", "kodcode")
    if style == "kodcode":
        return _KODCODE_RUNNER.format(src=code, tests=task["tests"])
    if style == "humaneval":
        entry = task["entry_point"]
        prompt = task.get("prompt_code", "")
        program = code if re.search(rf"def\s+{re.escape(entry)}\s*\(", code) else prompt + code
        imports = "\n".join(line for line in prompt.splitlines() if line.startswith(("import ", "from ")))
        return _HUMANEVAL_RUNNER.format(program=imports + "\n" + program, tests=task["tests"], entry_point=entry)
    if style == "asserts":
        return _ASSERT_RUNNER.format(program=code, setup=task.get("setup", ""), asserts="\n".join(task["tests"]))
    raise ValueError(f"unknown test_style {style!r}")


def run_tests(task: dict, code: str, pool=None) -> tuple[bool, str]:
    if not (code or "").strip():
        return False, "no code"
    if pool is None:
        from neo.sandbox import default_pool
        pool = default_pool()
    out = pool.run_detailed(build_test_program(task, code), timeout=float(task.get("timeout", 10)))
    m = _TESTS_LINE.search(out.get("stdout", ""))
    if not m or out.get("timed_out"):
        return False, out.get("result", "")[-400:]
    passed, failed = int(m.group(1)), int(m.group(2))
    return passed > 0 and failed == 0, out.get("result", "")[-400:]


def score_code(task: dict, reply: str, tools: list[ToolRecord]) -> EpisodeScore:
    ok, _ = run_tests(task, extract_code(reply))
    return EpisodeScore(1.0 if ok else 0.0, ok, {"answer_correct": float(ok)})


def _coerce_calls(calls: list[ParsedCall], schemas: list[dict]) -> list[ParsedCall]:
    """Values arrive as strings from some parsers; type them by the schema."""
    types = schema_types(schemas)
    out = []
    for c in calls:
        args = {}
        for k, v in c.arguments.items():
            args[k] = coerce_value(v, types.get(c.name, {}).get(k)) if isinstance(v, str) else v
        out.append(ParsedCall(c.name, args))
    return out


def score_fc(task: dict, messages: Sequence[dict]) -> EpisodeScore:
    content, calls = first_turn(messages)
    schemas = task.get("tools") or []
    calls = _coerce_calls(calls, schemas)
    if task.get("bfcl_ground_truth") is not None or task.get("bfcl_category"):
        cat = task.get("bfcl_category", "")
        if cat.endswith("irrelevance"):
            ok = not calls
        elif cat.endswith("relevance"):
            ok = bool(calls)
        else:
            ok = bfcl_match(calls, task["bfcl_ground_truth"], schemas)
        return EpisodeScore(1.0 if ok else 0.0, ok, {"answer_correct": float(ok), "n_calls": float(len(calls))})
    gold = task.get("gold_calls") or []
    if not gold and not calls and not content:
        return EpisodeScore(EMPTY, False, {"answer_correct": 0.0, "empty_reply": 1.0})
    score = call_match_score(calls, gold)
    ok = score == 1.0
    return EpisodeScore(score, ok, {"answer_correct": float(ok), "n_calls": float(len(calls)),
                                    "should_call": float(bool(gold))})


def ifeval_check(instruction_ids: list[str], kwargs_list: list[dict], prompt: str, reply: str) -> list[bool]:
    """Strict per-instruction results with lm-eval's reference IFEval checkers."""
    from lm_eval.tasks.ifeval import instructions_registry as registry

    results = []
    for iid, kwargs in zip(instruction_ids, kwargs_list):
        inst = registry.INSTRUCTION_DICT[iid](iid)
        kw = {k: v for k, v in (kwargs or {}).items() if v}   # exactly lm-eval's filter
        inst.build_description(**kw)
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=prompt)
        results.append(bool(reply.strip()) and bool(inst.check_following(reply)))
    return results


def score_ifeval(task: dict, reply: str, tools: list[ToolRecord]) -> EpisodeScore:
    try:
        followed = ifeval_check(task["instruction_id_list"], task["kwargs"], task.get("prompt_text", ""), reply)
    except ImportError:  # pragma: no cover
        return EpisodeScore(0.0, False, {"unavailable": 1.0})
    frac = sum(followed) / max(1, len(followed))
    ok = all(followed)
    return EpisodeScore(frac, ok, {"answer_correct": float(ok), "instruction_frac": frac})


_LETTER = [
    re.compile(r"answer\s*(?:is|:)\s*\**\(?([A-J])\)?\**", re.I),
    re.compile(r"^\s*\**\(?([A-J])\)?\**\s*$", re.M),
    re.compile(r"\*\*\(?([A-J])\)?\*\*"),
    re.compile(r"\\boxed\{\(?([A-J])\)?\}"),
]


def score_choice(task: dict, reply: str, tools: list[ToolRecord]) -> EpisodeScore:
    letter = None
    for pattern in _LETTER:
        found = pattern.findall(reply or "")
        if found:
            letter = found[-1].upper()
            break
    ok = letter == str(task["answer"]).upper()
    return EpisodeScore(1.0 if ok else 0.0, ok, {"answer_correct": float(ok)})


_SCORERS = {"numeric": score_numeric, "math": score_math, "code": score_code, "ifeval": score_ifeval,
            "choice": score_choice}


def score_episode(task: dict, messages: Sequence[dict], tools: Optional[list[ToolRecord]] = None) -> EpisodeScore:
    """Score a finished episode. `messages` are the model's completion messages
    (assistant and tool turns, not the prompt)."""
    if task["type"] == "fc":
        score = score_fc(task, messages)
        content, _ = first_turn(messages)
        if _TOOL_MARKUP.search(content):
            score.reward += MALFORMED_TOOL_TEXT
            score.metrics["malformed_tool_text"] = 1.0
        return score
    if tools is None:
        tools = tool_records(messages)
    reply = final_reply(messages)
    if not reply:
        return EpisodeScore(EMPTY_AFTER_TOOL if tools else EMPTY, False,
                            {"empty_reply": 1.0, "tool_used": float(bool(tools))})
    score = _SCORERS[task["type"]](task, reply, tools)
    if _TOOL_MARKUP.search(reply):
        score.reward += MALFORMED_TOOL_TEXT
        score.metrics["malformed_tool_text"] = 1.0
    score.metrics.setdefault("empty_reply", 0.0)
    return score
