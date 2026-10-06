"""Episode scoring: the reward for every task type."""
from __future__ import annotations

import pytest

from neo.sandbox import SandboxPool
from neo.verify import EMPTY, EMPTY_AFTER_TOOL, MALFORMED_TOOL_TEXT, ToolRecord, extract_code, run_tests, \
    score_episode


def assistant(text, calls=None):
    m = {"role": "assistant", "content": text}
    if calls:
        m["tool_calls"] = [{"type": "function", "function": {"name": n, "arguments": a}} for n, a in calls]
    return m


NUM = {"type": "numeric", "answer": 16.0, "unit": "mm", "tol": 0.015, "givens": [[2, "m"], [5, "kN"]]}


def test_numeric_with_and_without_tools():
    assert score_episode(NUM, [assistant("Tip deflection: **16 mm**.")]).reward == 1.0
    tool_run = [assistant("", [("python", {"code": "print(16)"})]), {"role": "tool", "content": "16.0 mm"},
                assistant("The tip deflects **16.0 mm**.")]
    s = score_episode(NUM, tool_run)
    assert s.correct and s.reward == 1.0 and s.metrics["tool_used"] == 1.0


def test_failed_tool_calls_cost_a_little():
    tools = [ToolRecord("python", "Traceback (most recent call last):\nNameError", ok=False),
             ToolRecord("python", "16.0 mm", ok=True)]
    s = score_episode(NUM, [assistant("**16 mm**")], tools)
    assert s.correct and s.reward == pytest.approx(0.97)


def test_empty_replies():
    assert score_episode(NUM, [assistant("")]).reward == EMPTY
    run = [assistant("", [("python", {"code": "print(16)"})]), {"role": "tool", "content": "16"}, assistant("")]
    assert score_episode(NUM, run).reward == EMPTY_AFTER_TOOL
    # ended on a tool call (iteration cap): no final answer
    assert score_episode(NUM, [assistant("", [("python", {"code": "x"})])]).reward == EMPTY


def test_malformed_tool_text_penalised():
    s = score_episode(NUM, [assistant("**16 mm** <tool_call><function=python>")])
    assert s.reward == pytest.approx(1.0 + MALFORMED_TOOL_TEXT)


def test_math():
    task = {"type": "math", "answer": "\\frac{1}{2}"}
    assert score_episode(task, [assistant("So the answer is $\\boxed{0.5}$.")]).correct
    assert not score_episode(task, [assistant("$\\boxed{2}$")]).correct


def test_fc_gold_and_irrelevance():
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}}}}}]
    task = {"type": "fc", "tools": tools, "gold_calls": [{"name": "get_weather", "arguments": {"city": "Pune"}}]}
    assert score_episode(task, [assistant("", [("get_weather", {"city": "Pune"})])]).reward == 1.0
    assert score_episode(task, [assistant("It is sunny.")]).reward == 0.0
    irrelevant = {"type": "fc", "tools": tools, "gold_calls": []}
    assert score_episode(irrelevant, [assistant("Here is a joke...")]).reward == 1.0
    assert score_episode(irrelevant, [assistant("", [("get_weather", {"city": "x"})])]).reward == 0.0


def test_fc_bfcl():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "x": {"type": "integer"}}}}}]
    task = {"type": "fc", "tools": tools, "bfcl_category": "simple", "bfcl_ground_truth": [{"f": {"x": [3]}}]}
    assert score_episode(task, [assistant("", [("f", {"x": "3"})])]).correct      # string coerced by schema
    irr = {"type": "fc", "tools": tools, "bfcl_category": "irrelevance", "bfcl_ground_truth": None}
    assert score_episode(irr, [assistant("I can't help with that using these tools.")]).correct


def test_choice():
    task = {"type": "choice", "answer": "C"}
    assert score_episode(task, [assistant("... so the answer is (C)")]).correct
    assert not score_episode(task, [assistant("the answer is (B)")]).correct


@pytest.fixture(scope="module")
def pool():
    p = SandboxPool(workers=2, timeout=10)
    yield p
    p.close()


def test_code_runners(pool):
    kod = {"type": "code", "test_style": "kodcode",
           "tests": "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"}
    assert run_tests(kod, "def add(a, b):\n    return a + b\n", pool)[0]
    assert not run_tests(kod, "def add(a, b):\n    return a - b\n", pool)[0]
    he = {"type": "code", "test_style": "humaneval", "entry_point": "inc", "prompt_code": "def inc(x):\n",
          "tests": "def check(candidate):\n    assert candidate(1) == 2\n"}
    assert run_tests(he, "    return x + 1\n", pool)[0]
    assert run_tests(he, "def inc(x):\n    return x + 1\n", pool)[0]
    mbpp = {"type": "code", "test_style": "asserts", "tests": ["assert sq(3) == 9"], "setup": ""}
    assert run_tests(mbpp, "def sq(x):\n    return x * x\n", pool)[0]


def test_extract_code():
    assert extract_code("Here:\n```python\ndef f():\n    return 1\n```\n") == "def f():\n    return 1\n"
    assert extract_code("no code here") == ""


def test_ifeval_scoring():
    pytest.importorskip("lm_eval")
    task = {"type": "ifeval", "prompt_text": "Write about cats.", "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}]}
    assert score_episode(task, [assistant("Cats are great pets. They purr.")]).correct
    assert not score_episode(task, [assistant("Cats, dogs, and birds.")]).correct
