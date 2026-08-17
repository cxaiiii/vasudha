"""Tests for the model-agnostic parts of the desktop harness.

These exist because the harness is used to compare models, not only to run the
bundled one. Every case below is a format some real model family actually
emits; a new model that fails here fails visibly at test time rather than by
quietly never calling a tool.

No model is loaded: everything here is string handling, which is where the
model-specific assumptions used to live.
"""
from __future__ import annotations

import pytest

from app.backends import (
    Completion,
    StreamFilter,
    ToolCall,
    parse_tool_calls,
    prompt_opens_thinking,
    render_chatml,
    split_thinking,
)


# ── tool-call formats ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("label,raw,expected_name,expected_args", [
    ("qwen3 xml",
     "<tool_call>\n<function=python_tool>\n<parameter=code>\nprint(1)\n"
     "</parameter>\n</function>\n</tool_call>",
     "python_tool", {"code": "print(1)"}),
    ("hermes / qwen2.5 json",
     '<tool_call>\n{"name": "python_tool", "arguments": {"code": "print(1)"}}\n</tool_call>',
     "python_tool", {"code": "print(1)"}),
    ("parameters instead of arguments",
     '<tool_call>{"name":"search_tool","parameters":{"query":"x"}}</tool_call>',
     "search_tool", {"query": "x"}),
    ("llama pipe tag",
     '<|tool_call|>{"name":"python_tool","arguments":{"code":"1"}}',
     "python_tool", {"code": "1"}),
    ("fenced json",
     '```json\n{"name":"fetch_tool","arguments":{"url":"http://x"}}\n```',
     "fetch_tool", {"url": "http://x"}),
    ("bare json, no wrapper",
     '{"name": "python_tool", "arguments": {"code": "print(2)"}}',
     "python_tool", {"code": "print(2)"}),
    ("nested function object",
     '<tool_call>{"function":{"name":"read_file","arguments":{"path":"a.py"}}}</tool_call>',
     "read_file", {"path": "a.py"}),
])
def test_tool_formats(label, raw, expected_name, expected_args):
    _, _, calls = parse_tool_calls(raw)
    assert len(calls) == 1, f"{label}: expected one call, got {calls}"
    assert calls[0].name == expected_name
    assert calls[0].args == expected_args


def test_multiple_calls_in_one_turn():
    raw = ('<tool_call>{"name":"a","arguments":{}}</tool_call>'
           '<tool_call>{"name":"b","arguments":{}}</tool_call>')
    _, _, calls = parse_tool_calls(raw)
    assert [c.name for c in calls] == ["a", "b"]
    # Distinct ids, or the loop cannot say which result answers which call.
    assert len({c.id for c in calls}) == 2


def test_concatenated_calls_without_separator():
    """Some fine-tunes emit two objects back to back inside one block."""
    raw = '<tool_call>{"name":"a","arguments":{}}{"name":"b","arguments":{}}</tool_call>'
    _, _, calls = parse_tool_calls(raw)
    assert [c.name for c in calls] == ["a", "b"]


def test_json_answer_is_not_mistaken_for_a_tool_call():
    """A model asked to reply in JSON must not have its answer eaten.

    This is the failure mode that makes a permissive bare-JSON parser
    dangerous, so the parser requires a "name" key before it will treat an
    unwrapped object as a call.
    """
    raw = '{"result": 42, "unit": "mm"}'
    text, _, calls = parse_tool_calls(raw)
    assert calls == []
    assert text == raw


def test_prose_is_left_alone():
    text, thinking, calls = parse_tool_calls("The deflection is 14.07 mm.")
    assert (text, thinking, calls) == ("The deflection is 14.07 mm.", "", [])


# ── reasoning blocks ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("tag", ["think", "thinking", "reasoning", "reason"])
def test_reasoning_tags(tag):
    text, thinking = split_thinking(f"<{tag}>working it out</{tag}>The answer is 4.")
    assert text == "The answer is 4."
    assert thinking == "working it out"


def test_unclosed_reasoning_is_not_shown_as_the_answer():
    """A reply truncated inside its reasoning leaves an unclosed tag. Treating
    the remainder as the answer publishes the model's working as a conclusion.
    """
    text, thinking = split_thinking("<think>I am still working through the")
    assert text == ""
    assert thinking.startswith("I am still working")


def test_prompt_opens_thinking():
    # Qwen3-style: template hands the model an already-open block, so the only
    # tag it ever emits is the closing one.
    assert prompt_opens_thinking("<|im_start|>assistant\n<think>\n") is True
    assert prompt_opens_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n") is False
    assert prompt_opens_thinking("<|im_start|>assistant\n") is False
    # A think block belonging to an earlier turn must not count as open.
    assert prompt_opens_thinking(
        "<think>a</think>ans<|im_end|><|im_start|>assistant\n") is False


# ── incremental streaming ─────────────────────────────────────────────────────

def _drive(raw: str, chunk: int, in_think: bool = False) -> tuple[str, str]:
    filt = StreamFilter(in_think=in_think)
    events = []
    for i in range(0, len(raw), chunk):
        events += filt.feed(raw[i:i + chunk])
    events += filt.close()
    visible = "".join(t for c, t in events if c == "text")
    thinking = "".join(t for c, t in events if c == "thinking")
    return visible, thinking


@pytest.mark.parametrize("chunk", [1, 3, 7, 1000])
def test_stream_never_leaks_markup(chunk):
    """Whatever the delta boundaries, no markup reaches the visible channel.

    Parameterised down to one character per delta because that is the case
    that breaks naive buffering: a sentinel arrives split across deltas and a
    filter that checks each delta in isolation emits half a tag.
    """
    raw = ('<think>plan the calculation</think>'
           'Computing now.'
           '<tool_call>{"name":"python_tool","arguments":{"code":"1+1"}}</tool_call>')
    visible, thinking = _drive(raw, chunk)
    assert visible.strip() == "Computing now."
    assert thinking.strip() == "plan the calculation"
    for marker in ("<think", "</think", "<tool_call", "</tool_call", '"name"'):
        assert marker not in visible


@pytest.mark.parametrize("chunk", [1, 5])
def test_stream_with_preopened_think(chunk):
    """The template opened the block, so only `</think>` is ever emitted."""
    visible, thinking = _drive("Reasoning.</think>The answer is 14.07 mm.",
                               chunk, in_think=True)
    assert visible.strip() == "The answer is 14.07 mm."
    assert thinking.strip() == "Reasoning."


def test_stream_flushes_an_unclosed_tool_call():
    """A reply cut off mid-call still shows the prose that preceded it."""
    visible, _ = _drive('Working on it.<tool_call>{"name":"a"', 4)
    assert visible.strip() == "Working on it."


def test_stream_raw_is_complete():
    """The filter must retain everything for the non-streaming parse."""
    raw = '<think>a</think>b<tool_call>{"name":"c","arguments":{}}</tool_call>'
    filt = StreamFilter()
    for ch in raw:
        filt.feed(ch)
    filt.close()
    assert filt.raw == raw


# ── prompt rendering ──────────────────────────────────────────────────────────

def test_chatml_fallback_preserves_tool_role():
    """The two backends must render the same conversation identically.

    They did not: one mapped role=tool onto a user turn and the other passed it
    through, so a benchmark number depended on which backend had loaded.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "python_tool",
                                      "arguments": {"code": "1"}}}]},
        {"role": "tool", "content": "2"},
    ]
    rendered = render_chatml(messages, [], enable_thinking=True)
    assert "<|im_start|>tool" in rendered
    # The call the assistant made has to survive into the prompt, or the model
    # sees a result it has no record of requesting.
    assert "python_tool" in rendered


def test_chatml_thinking_switch():
    on = render_chatml([{"role": "user", "content": "q"}], [], enable_thinking=True)
    off = render_chatml([{"role": "user", "content": "q"}], [], enable_thinking=False)
    assert not on.rstrip().endswith("</think>")
    assert off.rstrip().endswith("</think>")


# ── session loop ──────────────────────────────────────────────────────────────

class FakeBackend:
    """Replays a fixed script of completions, so the loop can be tested without
    a model. Records what it was asked, which is where the history bugs showed."""

    context_limit = 4096
    observed_tps = None

    def __init__(self, script: list[Completion]) -> None:
        self.script = list(script)
        self.seen: list[list[dict]] = []
        self.thinking_flags: list[bool] = []

    def stream(self, messages, schemas, options):
        self.seen.append([dict(m) for m in messages])
        self.thinking_flags.append(bool(options.get("enable_thinking", True)))
        completion = self.script.pop(0)
        for word in (completion.text or "").split():
            yield ("text", word + " ")
        return completion

    def generate(self, messages, schemas, options):
        gen = self.stream(messages, schemas, options)
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            return stop.value

    def count_tokens(self, text: str) -> int:
        return max(len(text) // 4, 1)

    def warm(self, *a, **k) -> None:
        pass


def _session(script, tmp_path):
    from app.session import ChatSession
    return ChatSession(FakeBackend(script), workspace=str(tmp_path))


def test_assistant_message_keeps_its_tool_calls(tmp_path):
    """Without this the model is shown a tool result it never asked for."""
    script = [
        Completion(text="", tool_calls=[ToolCall(name="list_files", args={}, id="c0")]),
        Completion(text="Nothing there."),
    ]
    session = _session(script, tmp_path)
    list(session.ask("what files?"))

    assistant = next(m for m in session.history
                     if m["role"] == "assistant" and m.get("tool_calls"))
    assert assistant["tool_calls"][0]["function"]["name"] == "list_files"


def test_tool_result_is_persisted_and_linked(tmp_path):
    """Tool output used to live only in the working transcript, so a follow-up
    question arrived with no memory of what had been computed."""
    script = [
        Completion(text="", tool_calls=[ToolCall(name="list_files", args={}, id="c0")]),
        Completion(text="Empty."),
    ]
    session = _session(script, tmp_path)
    list(session.ask("what files?"))

    tool_messages = [m for m in session.history if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "c0"
    assert tool_messages[0]["name"] == "list_files"


def test_thinking_is_off_after_a_tool_result(tmp_path):
    """Measured 1.7x on the shipped 4B: reasoning again before restating a
    number a tool already computed is the loop's most expensive habit."""
    script = [
        Completion(text="", tool_calls=[ToolCall(name="list_files", args={}, id="c0")]),
        Completion(text="Empty."),
    ]
    session = _session(script, tmp_path)
    list(session.ask("what files?"))

    assert session.backend.thinking_flags == [True, False]


def test_streaming_events_reach_the_caller(tmp_path):
    session = _session([Completion(text="hello there")], tmp_path)
    kinds = [e["kind"] for e in session.ask("hi")]
    assert "token" in kinds
    assert kinds[-1] == "text" or "text" in kinds


def test_long_tool_result_is_clipped(tmp_path):
    from app.session import MAX_TOOL_RESULT_CHARS, _clip_tool_result
    clipped = _clip_tool_result("x" * (MAX_TOOL_RESULT_CHARS * 2))
    assert len(clipped) < MAX_TOOL_RESULT_CHARS + 200
    # Both ends survive: a traceback ends with the exception, a page starts
    # with the summary.
    assert clipped.startswith("x")
    assert clipped.endswith("x")


def test_compaction_keeps_system_and_recent_turns(tmp_path):
    session = _session([], tmp_path)
    convo = [{"role": "system", "content": "SYSTEM RULES"}]
    for i in range(10):
        convo.append({"role": "user", "content": f"q{i}"})
        convo.append({"role": "tool", "content": "z" * 4000})
    convo.append({"role": "user", "content": "the current question"})

    compacted, changed = session._compact(convo, budget=500)
    assert changed
    assert compacted[0]["content"] == "SYSTEM RULES"
    assert compacted[-1]["content"] == "the current question"


# ── file tools ────────────────────────────────────────────────────────────────

def test_edit_file_refuses_an_ambiguous_match(tmp_path):
    session = _session([], tmp_path)
    session._write_file("a.py", "x = 1\nx = 1\n")
    result = session._edit_file("a.py", "x = 1", "x = 2")
    assert "appears 2 times" in result
    # Unchanged: a silent first-occurrence replacement is how a file ends up
    # subtly wrong somewhere nobody looks. Checked against the raw contents,
    # since _read_file adds line numbers for the model's benefit.
    assert session._files().read_file("a.py") == "x = 1\nx = 1\n"


def test_edit_file_replaces_a_unique_match(tmp_path):
    session = _session([], tmp_path)
    session._write_file("a.py", "alpha = 1\nbeta = 2\n")
    assert "one replacement" in session._edit_file("a.py", "beta = 2", "beta = 3")
    assert session._files().read_file("a.py") == "alpha = 1\nbeta = 3\n"


def test_edit_file_reports_a_missing_anchor(tmp_path):
    session = _session([], tmp_path)
    session._write_file("a.py", "alpha = 1\n")
    assert "not in" in session._edit_file("a.py", "gamma = 9", "gamma = 8")


# ── grounding: figures must come from a tool ──────────────────────────────────

def test_a_computed_turn_can_still_fabricate(tmp_path):
    """python_tool used to be excluded from the grounding check on the theory
    that a run computing its own numbers cannot invent any. A real session
    disproved it: legitimate code ran, and the write-up then quoted
    "BIC = 1234.56, AIC = 1256.78" — placeholder digits in no stdout at all.
    """
    session = _session([], tmp_path)
    tools = [{"name": "python_tool", "args": {},
              "result": "[Stdout]:\nmeans: [2.014, 7.982]\nlog-likelihood: -1043.27"}]
    missing = session._ungrounded_figures(
        "BIC = 1234.56 and AIC = 1256.78, means 2.014 and 7.982.", tools)
    assert missing == ["1234.56", "1256.78"]


def test_values_printed_by_the_run_are_not_flagged(tmp_path):
    session = _session([], tmp_path)
    tools = [{"name": "python_tool", "args": {},
              "result": "[Stdout]:\nmeans: [2.014, 7.982]\nlog-likelihood: -1043.27"}]
    assert session._ungrounded_figures(
        "Means were 2.014 and 7.982, log-likelihood -1043.27.", tools) == []


def test_the_repair_prompt_names_the_exact_figures(tmp_path):
    """Told only that something is ungrounded, the model rewrites the prose
    around the same invented number. Told which number, it computes or drops."""
    session = _session([], tmp_path)
    prompt = session._repair_prompt(["1234.56", "1256.78"])
    assert "1234.56" in prompt and "1256.78" in prompt
    assert "python_tool" in prompt


def test_a_fabricated_figure_survives_to_the_footer(tmp_path):
    session = _session([], tmp_path)
    tools = [{"name": "python_tool", "args": {}, "result": "[Stdout]:\nx = 5.0"}]
    note = session._unsourced_note("The result is 9999.99.", tools)
    assert "9999.99" in note
    assert "unverified" in note.lower()


# ── grounding: a diverged run is not an answer ────────────────────────────────

@pytest.mark.parametrize("output", [
    "[Stderr]: RuntimeWarning: overflow encountered in double_scalars",
    "[Stdout]: position: nan",
    "[Stdout]: energy: inf",
    "[Stderr]: RuntimeWarning: invalid value encountered in sqrt",
])
def test_numerical_failure_is_flagged(output):
    """The model explains that Euler can go unstable and then fails to notice
    that its own run just did. A spring simulation reached 1e306 and a period
    was reported from it."""
    from app.session import _flag_numerical_failure
    flagged = _flag_numerical_failure(output)
    assert "numerical failure detected" in flagged
    assert "not usable" in flagged


def test_a_healthy_run_is_left_alone():
    from app.session import _flag_numerical_failure
    clean = "[Stdout]:\nperiod: 1.2566 s\n[Finished in 0.1s]"
    assert _flag_numerical_failure(clean) == clean


# ── grounding: edits land on the real source ──────────────────────────────────

def test_read_file_numbers_its_lines(tmp_path):
    """A traceback says "line 27"; an unnumbered listing gives the model no way
    to act on that."""
    session = _session([], tmp_path)
    session._write_file("a.py", "alpha = 1\nbeta = 2\ngamma = 3\n")
    listing = session._read_file("a.py")
    assert "1| alpha = 1" in listing
    assert "2| beta = 2" in listing


def test_a_failed_edit_shows_the_real_source(tmp_path):
    """The observed loop: the model diagnosed the error correctly every time
    and re-emitted the same broken line, because a bare "not found" left it
    editing from memory. Handing back the actual text ends that."""
    session = _session([], tmp_path)
    session._write_file("astar.py",
                        "def f():\n    for n in neighbors(grid, (ny, nx), w, h):\n"
                        "        pass\n")
    # Same intent, wrong whitespace — the classic near-miss.
    result = session._edit_file("astar.py",
                                "for n in neighbors(grid, (ny,nx), w, h):",
                                "for n in neighbors(grid, current, w, h):")
    assert "not in astar.py" in result
    assert "closest match is line 2" in result
    assert "(ny, nx)" in result          # the real text, to copy from


def test_the_edit_lands_once_the_anchor_is_right(tmp_path):
    session = _session([], tmp_path)
    session._write_file("a.py", "x = 1\ny = 2\n")
    assert "one replacement" in session._edit_file("a.py", "y = 2", "y = 3")
    assert "y = 3" in session._read_file("a.py")


# ── effort modes ──────────────────────────────────────────────────────────────

def test_every_mode_survives_clamping():
    """A mode must not promise a bound the app then quietly reduces.

    YOLO declared 300s and 40 iterations while clamp() capped them at 120 and
    32, so the UI advertised limits that were never applied. Settings claiming
    something the engine will not honour is the exact failure this file exists
    to prevent.
    """
    from app.settings import EFFORT_MODES, Settings
    for name, mode in EFFORT_MODES.items():
        settings = Settings().apply_effort(name)
        for field in ("num_predict", "num_ctx", "max_iterations", "tool_timeout"):
            assert getattr(settings, field) == mode[field], (
                f"{name}.{field} was clamped from {mode[field]} "
                f"to {getattr(settings, field)}")


def test_modes_increase_monotonically():
    from app.settings import EFFORT_MODES, EFFORT_ORDER
    for field in ("num_predict", "num_ctx", "max_iterations", "tool_timeout"):
        values = [EFFORT_MODES[n][field] for n in EFFORT_ORDER]
        assert values == sorted(values), f"{field} is not monotonic: {values}"


def test_an_unknown_mode_falls_back_rather_than_breaking():
    from app.settings import Settings
    assert Settings(effort="turbo").clamp().effort == "medium"
    # ...and an unknown name applied is simply ignored.
    settings = Settings().apply_effort("medium")
    assert settings.apply_effort("nonsense").effort == "medium"


# ── attachments ───────────────────────────────────────────────────────────────

def _big_csv(tmp_path, rows=5000):
    path = tmp_path / "data.csv"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("id,rating,text\n")
        for i in range(rows):
            handle.write(f"{i},{i % 5 + 1},review number {i}\n")
    return path


def test_a_large_file_is_described_not_inlined(tmp_path):
    """The whole design: a file the model must work on never enters the prompt.

    A description costs a few hundred tokens and answers the same questions
    inlining would — how big, what columns, what the rows look like — except it
    also works when the file is a gigabyte.
    """
    from web.tools import describe_file
    source = _big_csv(tmp_path, rows=5000)
    described = describe_file(source)

    assert len(described) < 1500
    assert len(described) < source.stat().st_size / 50
    assert "5,001 lines" in described
    assert "3 columns" in described
    assert "review number 0" in described      # a real sample
    assert "review number 4999" not in described   # but not the whole file


def test_read_file_refuses_a_large_file_instead_of_truncating(tmp_path):
    """Truncation looks like success: the model reads the first slice, sees
    plausible rows, and answers about a fraction of the data without saying so.
    """
    session = _session([], tmp_path)
    big = _big_csv(tmp_path / "ws" if (tmp_path / "ws").exists() else tmp_path, rows=4000)
    session._write_file("data.csv", big.read_text(encoding="utf-8"))

    result = session._read_file("data.csv")
    assert "too large to read into the conversation" in result
    assert "python_tool" in result
    # The description is offered in its place, so the turn is not wasted.
    assert "lines" in result


def test_a_small_file_is_still_read_normally(tmp_path):
    session = _session([], tmp_path)
    session._write_file("notes.txt", "a short note")
    # Numbered for display; the raw contents are still exactly what was written.
    assert session._read_file("notes.txt") == "1| a short note"
    assert session._files().read_file("notes.txt") == "a short note"


def test_attaching_puts_the_file_in_the_workspace(tmp_path):
    source = tmp_path / "outside.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    session = _session([], workspace)
    record = session.attach(str(source))

    assert record["ok"]
    assert (workspace / "outside.csv").exists()   # copied, so the sandbox can open it
    assert "outside.csv" in session._attachment_section()


def test_attaching_the_same_name_twice_does_not_duplicate(tmp_path):
    source = tmp_path / "x.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    session = _session([], tmp_path / "ws2")
    session.attach(str(source))
    session.attach(str(source))
    assert len(session.attachments) == 1


def test_attaching_a_missing_file_reports_it(tmp_path):
    session = _session([], tmp_path)
    assert session.attach(str(tmp_path / "nope.csv"))["ok"] is False


# ── generated images ──────────────────────────────────────────────────────────

_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
    "00000049454e44ae426082")


def test_a_new_image_is_surfaced_and_an_old_one_is_not(tmp_path):
    """A chart is the answer, not a side effect: matplotlib writes a PNG and
    prints nothing, so without this the model reports success and the user sees
    no plot. Re-showing an unchanged image on every later call is the opposite
    failure."""
    session = _session([], tmp_path)
    root = session._files().workspace

    (root / "plot.png").write_bytes(_PNG)          # pre-existing
    session._python_tool(code="print('hello')")
    assert session.new_images == []                 # untouched, so not shown

    session._python_tool(
        code="open('new.png','wb').write(bytes.fromhex('%s'))" % _PNG.hex())
    assert [p for p in session.new_images if p.endswith("new.png")]


# ── memory book ───────────────────────────────────────────────────────────────

def test_a_lesson_survives_a_restart(tmp_path):
    from app.memory import MemoryBook
    MemoryBook(tmp_path).remember("sandbox packages",
                                  "textblob is not installed; pip_tool first.")
    assert "textblob is not installed" in MemoryBook(tmp_path).as_prompt_section()


def test_rewriting_a_topic_replaces_it(tmp_path):
    """The whole point: a better answer overwrites a worse one rather than
    accumulating beside it, so the book does not fill with contradictions."""
    from app.memory import MemoryBook
    book = MemoryBook(tmp_path)
    book.remember("sentiment", "hand-roll a word list")
    result = book.remember("sentiment", "use textblob after pip_tool",
                           source="https://textblob.readthedocs.io")

    assert "revision 2" in result
    assert len(book) == 1
    section = book.as_prompt_section()
    assert "textblob" in section
    assert "hand-roll" not in section
    # ...but the superseded version survives in the history, which is the part
    # that becomes training data.
    history = (tmp_path / "lessons.jsonl").read_text(encoding="utf-8")
    assert "hand-roll" in history


@pytest.mark.parametrize("a,b", [
    ("Sandbox Packages", "sandbox packages"),
    ("sandbox-packages", "sandbox packages!"),
])
def test_topic_keys_are_normalised(tmp_path, a, b):
    """Two spellings of one topic must not become two lessons."""
    from app.memory import MemoryBook
    book = MemoryBook(tmp_path)
    book.remember(a, "first")
    book.remember(b, "second")
    assert len(book) == 1


def test_book_is_truncated_at_a_whole_lesson(tmp_path):
    """Half a lesson is worse than none — the model acts on the half it sees."""
    from app.memory import MemoryBook
    book = MemoryBook(tmp_path)
    for i in range(60):
        book.remember(f"topic {i}", "x" * 200)
    section = book.as_prompt_section(budget=500)
    assert len(section) < 900
    assert not section.rstrip().endswith("x" * 50 + "…")


def test_empty_book_adds_nothing_to_the_prompt(tmp_path):
    from app.memory import MemoryBook
    assert MemoryBook(tmp_path).as_prompt_section() == ""


def test_interaction_log_keeps_the_users_own_words(tmp_path):
    """Verbatim on purpose: phrasing and tone are what a synthesised dataset
    gets wrong, and this log exists to become a real one."""
    import json
    from app.memory import InteractionLog
    log = InteractionLog(tmp_path)
    log.record(chat_id="c1", question="yo whats the deflection dawg",
               answer="14.07 mm", tools=[{"name": "python_tool", "args": {},
                                          "result": "14.07"}],
               sources=["https://example.com"], seconds=3.2)
    row = json.loads((tmp_path / "interactions.jsonl").read_text(encoding="utf-8"))
    assert row["question"] == "yo whats the deflection dawg"
    assert row["tools"][0]["name"] == "python_tool"
    assert row["sources"] == ["https://example.com"]


# ── sources outrank recall ────────────────────────────────────────────────────

def test_figures_absent_from_every_source_are_flagged():
    from app.memory import unsourced_figures
    tools = ["The bat weighs 1180 grams and costs 14500 rupees."]
    missing = unsourced_figures("It weighs 1180 g, costs 14500, and 3200 were sold.",
                                tools)
    assert missing == ["3200"]


def test_a_figure_present_in_a_source_is_not_flagged():
    from app.memory import unsourced_figures
    assert unsourced_figures("The price is 14,500 rupees.",
                             ["costs 14500 rupees"]) == []


def test_rounding_a_sourced_figure_is_not_flagged():
    """A derived or rounded number is legitimate; flagging it would train the
    reader to ignore the warning."""
    from app.memory import unsourced_figures
    assert unsourced_figures("about 14.07 mm", ["delta = 14.0696 mm"]) == []


def test_nothing_is_flagged_when_no_tool_ran():
    from app.memory import unsourced_figures
    assert unsourced_figures("The answer is 42000.", []) == []


def test_workspace_paths_cannot_escape(tmp_path):
    session = _session([], tmp_path)
    assert "escapes the workspace" in session._write_file("../../evil.txt", "x")


# ── browsing ──────────────────────────────────────────────────────────────────
# No browser is launched here: format_digest is a pure function of the scraped
# page, which is where the budgeting decisions live and where they can regress.

def _page(text="body text", n_headings=0, n_items=0):
    return {
        "title": "A Page", "url": "https://example.com/", "text": text,
        "headings": [f"h2: Section {i}" for i in range(n_headings)],
        "items": [f'[ref{i}] link "Item {i}" -> /item/{i}' for i in range(n_items)],
    }


def test_digest_fits_the_tool_result_budget():
    """A digest over the session's clip limit gets cut down the middle, which
    truncates the outline and the ref list at once."""
    from app.session import MAX_TOOL_RESULT_CHARS
    from web.browser import format_digest

    digest = format_digest(_page(text="x " * 40000, n_headings=40, n_items=200))
    assert len(digest) <= MAX_TOOL_RESULT_CHARS


def test_digest_keeps_structure_and_compresses_prose():
    """Headings and refs survive; the page text is what gives way."""
    from web.browser import format_digest

    digest = format_digest(_page(text="prose " * 5000, n_headings=10, n_items=12))
    for i in range(10):
        assert f"Section {i}" in digest
    for i in range(12):
        assert f"[ref{i}]" in digest
    assert "more characters on this page" in digest


def test_digest_never_squeezes_the_text_away_entirely():
    """Even a page with a huge control list must still show some prose."""
    from web.browser import format_digest

    digest = format_digest(_page(text="the answer is 42. " * 200, n_items=200))
    assert "the answer is 42" in digest


def test_digest_survives_an_empty_page():
    from web.browser import format_digest
    digest = format_digest({"title": "", "url": "", "text": "", "headings": [], "items": []})
    assert "(untitled)" in digest
    assert "still be rendering" in digest


@pytest.mark.parametrize("given,expected", [
    ("ref3", 3), ("ref_3", 3), ("3", 3), ("REF12", 12), ("ref 7", 7),
    ("", None), ("nope", None), (None, None),
])
def test_ref_parsing_is_forgiving(given, expected):
    """Models are inconsistent about the prefix, and rejecting a valid intent
    over punctuation costs a whole turn."""
    from web.browser import _ref_index
    assert _ref_index(given) == expected


def test_browse_tool_rejects_bad_input_without_launching(tmp_path):
    session = _session([], tmp_path)
    assert "http(s) url" in session._browse_tool(action="open", url="notaurl")
    assert "unknown action" in session._browse_tool(action="teleport")
    assert "needs a ref" in session._browse_tool(action="click")
    # A malformed call must not be the reason a headless Chromium gets launched.
    assert session._browser is None


def test_browser_schema_is_dropped_when_playwright_is_absent():
    """An unusable schema costs ~200 tokens of every prompt and invites the
    model to spend a turn discovering it does not work."""
    from app.session import default_schemas
    names = [s["function"]["name"] for s in default_schemas(include_browser=False)]
    assert "browse_tool" not in names
    assert "fetch_tool" in names          # the static fallback stays
    assert "browse_tool" in [s["function"]["name"]
                             for s in default_schemas(include_browser=True)]
