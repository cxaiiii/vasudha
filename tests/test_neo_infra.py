"""Sandbox, budget, data helpers and the GRPO tool-loop override."""
from __future__ import annotations

import json
import os
import time

import pytest

from neo.budget import Ledger, BudgetExceeded, hourly_rate
from neo.data import drop_classic, last_boxed, make_prompt, parse_reference, rl_row
from neo.sandbox import SandboxPool, format_result


def test_sandbox_basics():
    pool = SandboxPool(workers=1, timeout=2)
    try:
        assert pool.run("print(2 + 2)").strip() == "4"
        assert "ZeroDivisionError" in pool.run("1/0")
        assert pool.run("x = 1") == "(no output)"
        assert pool.run("```python\nprint('fenced')\n```").strip() == "fenced"
        started = time.time()
        assert "TimeoutError" in pool.run("while True:\n    pass")
        assert time.time() - started < 6
        assert pool.run("import numpy as np\nprint(np.sqrt(16))").strip() == "4.0"
    finally:
        pool.close()


def test_format_result_truncates():
    out = format_result("x" * 10000, "", 0.1, False, 15, max_chars=1000)
    assert len(out) < 1200 and "truncated" in out


def test_ledger(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.jsonl"), cap=5.0)
    ledger.record("a", "H100", 3600, hourly_rate("H100", 8, 48), True)
    assert ledger.spent() == pytest.approx(4.7116, rel=1e-3)
    with pytest.raises(BudgetExceeded):
        ledger.ensure("b", 1.0)
    ledger.ensure("c", 0.2)


def test_reference_parsing_and_boxed():
    assert parse_reference("$12,672.96") == (12672.96, "money")
    assert parse_reference("30.5 kPa") == (30.5, "kPa")
    assert parse_reference("3630") == (3630.0, "*")
    assert parse_reference("1.4 T in the negative x-direction") is None
    assert parse_reference("3.2e-6 C") is None          # coulomb vs Celsius: skip
    assert last_boxed("so \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"
    assert last_boxed("\\boxed{a} then \\boxed{b{c}}") == "b{c}"


def test_classic_sheet_held_out_of_training():
    rows = [{"prompt": make_prompt("Population standard deviation of 2, 4, 4, 4, 5, 5, 7, 9?")},
            {"prompt": make_prompt("A 12 mm diameter steel rod carries 25 kN in tension. Stress?", "Be precise.")},
            {"prompt": make_prompt("Mean of 1, 2, 3?", "You are a helpful assistant.")}]
    assert [r["prompt"][-1]["content"] for r in drop_classic(rows)] == ["Mean of 1, 2, 3?"]


def test_rl_row_shape():
    row = rl_row("x", "s", "q?", {"type": "math", "answer": "1"}, "python", "sys")
    assert row["prompt"][0]["role"] == "system" and json.loads(row["task"])["answer"] == "1"


def test_text_env_tool_calls_are_not_executed(monkeypatch):
    pytest.importorskip("trl")
    from trl import GRPOTrainer

    from neo.trl_ext import NeoGRPOTrainer

    seen = {}

    def fake_loop(self, prompts, prompt_ids, completion_ids, completions, *args, **kwargs):
        seen["calls"] = [c[0].get("tool_calls") for c in completions]
        return (None, completions, completion_ids, None, 0, 0, [])

    monkeypatch.setattr(GRPOTrainer, "_tool_call_loop", fake_loop)
    trainer = object.__new__(NeoGRPOTrainer)
    trainer._batch_environments = ["text", "python"]
    call = [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Pune"}}}]
    completions = [[{"role": "assistant", "content": "", "tool_calls": list(call)}],
                   [{"role": "assistant", "content": "", "tool_calls": list(call)}]]
    out = trainer._tool_call_loop([], [], [[1], [2]], completions, None, None, {})
    assert seen["calls"][0] is None and seen["calls"][1] == call        # text env hidden from execution
    assert out[1][0][0]["tool_calls"] == call                            # restored for the reward


def test_python_env_schema_matches_trl_rendering():
    pytest.importorskip("transformers")
    from transformers.utils import get_json_schema

    from neo.envs import ENVIRONMENTS, PYTHON_ENVS, python_tool_schema

    for name in PYTHON_ENVS:
        method = getattr(ENVIRONMENTS[name](), name)
        assert get_json_schema(method) == python_tool_schema(name)
