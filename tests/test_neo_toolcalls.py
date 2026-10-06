"""Native Qwen3.5 tool calls: parsing, rendering, and BFCL-style checking."""
from __future__ import annotations

import os

import pytest

from neo.toolcalls import (ParsedCall, bfcl_match, call_match_score, coerce_value, parse_reply, render_tools_system,
                           to_openai_tool)

TRIANGLE = to_openai_tool({
    "name": "calculate_triangle_area", "description": "Area of a triangle.",
    "parameters": {"type": "dict", "properties": {
        "base": {"type": "integer", "description": "base"}, "height": {"type": "integer", "description": "height"},
        "unit": {"type": "string", "description": "unit"}}, "required": ["base", "height"]}})


def test_parse_single_call_with_types():
    text = ("I'll compute it.\n\n<tool_call>\n<function=calculate_triangle_area>\n<parameter=base>\n10\n</parameter>\n"
            "<parameter=height>\n5\n</parameter>\n</function>\n</tool_call>")
    r = parse_reply(text, [TRIANGLE])
    assert r.content == "I'll compute it."
    assert r.calls == [ParsedCall("calculate_triangle_area", {"base": 10, "height": 5})]
    assert not r.malformed


def test_parse_parallel_calls_and_thinking():
    one = "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
    r = parse_reply("<think>\nplan\n</think>\n\n" + one + "\n" + one.replace("=1", "=2"))
    assert r.reasoning == "plan" and len(r.calls) == 2


def test_malformed_markup_detected():
    r = parse_reply("<tool_call>\n<function=f>\n<parameter=x>\n1\n")
    assert r.calls == [] and r.malformed


def test_python_code_parameter_keeps_text():
    code = "import math\nprint(math.pi)"
    text = f"<tool_call>\n<function=python>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"
    schema = {"type": "function", "function": {"name": "python", "parameters": {"type": "object", "properties": {
        "code": {"type": "string"}}}}}
    assert parse_reply(text, [schema]).calls[0].arguments["code"] == code


@pytest.mark.parametrize("raw, t, want", [
    ("10", "integer", 10), ("2.5", "number", 2.5), ("true", "boolean", True), ("[3, 5]", "array", [3, 5]),
    ('{"a": 1}', "object", {"a": 1}), ("hello", None, "hello"), ("[1,2]", None, [1, 2]),
])
def test_coerce(raw, t, want):
    assert coerce_value(raw, t) == want


def test_xlam_schema_conversion():
    tool = to_openai_tool({"name": "live_giveaways_by_type", "description": "d",
                           "parameters": {"type": {"description": "kind", "type": "str", "default": "game"}}})
    params = tool["function"]["parameters"]
    assert params["properties"]["type"]["type"] == "string" and params["required"] == []


def test_bfcl_match_rules():
    gold = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    assert bfcl_match([ParsedCall("calculate_triangle_area", {"base": 10, "height": 5})], gold, [TRIANGLE])
    assert bfcl_match([ParsedCall("calculate_triangle_area", {"base": 10, "height": 5, "unit": "Units"})], gold, [TRIANGLE])
    assert not bfcl_match([ParsedCall("calculate_triangle_area", {"base": 10})], gold, [TRIANGLE])
    assert not bfcl_match([ParsedCall("calculate_triangle_area", {"base": 10, "height": 5, "colour": "red"})], gold, [TRIANGLE])
    assert not bfcl_match([ParsedCall("f", {})], None, [])           # item shipped without an answer
    par = [{"f": {"x": [1]}}, {"f": {"x": [2]}}]
    assert bfcl_match([ParsedCall("f", {"x": 2}), ParsedCall("f", {"x": 1})], par, [])


def test_call_match_score():
    gold = [{"name": "get_weather", "arguments": {"city": "Pune"}}]
    assert call_match_score([ParsedCall("get_weather", {"city": "pune"})], gold) == 1.0
    assert call_match_score([ParsedCall("get_weather", {"city": "Delhi"})], gold) == 0.0
    assert call_match_score([ParsedCall("get_weather", {"city": "Pune"}), ParsedCall("x", {})], gold) == 0.5
    assert call_match_score([], []) == 1.0 and call_match_score([ParsedCall("x", {})], []) == 0.0


def _real_template():
    try:
        import trl
        from transformers.utils.chat_template_utils import _compile_jinja_template
    except ImportError:
        return None
    path = os.path.join(os.path.dirname(trl.__file__), "chat_templates", "qwen3_5_think.jinja")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return _compile_jinja_template(fh.read())


@pytest.mark.parametrize("system", [None, "You are a helpful assistant."])
def test_render_matches_real_template(system):
    template = _real_template()
    if template is None:
        pytest.skip("Qwen3.5 template not available (needs trl)")
    user = {"role": "user", "content": "Area of a triangle with base 10 and height 5?"}
    msgs = ([{"role": "system", "content": system}] if system else []) + [user]
    native = template.render(messages=msgs, tools=[TRIANGLE], add_generation_prompt=True, enable_thinking=False)
    prerendered = template.render(messages=[{"role": "system", "content": render_tools_system([TRIANGLE], system)}, user],
                                  tools=None, add_generation_prompt=True, enable_thinking=False)
    assert native == prerendered
