import os
import sys
import json
import re
import requests
from flask import Flask, render_template, request, Response, stream_with_context

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from web.tools import SandboxedCodeExecutor, WebSearcher, PageFetcher, WorkspaceManager, WorkspaceFileTools
# Reused, not reimplemented — the exact same stdout/number extraction
# reasoning_verify.py already uses to mechanically check candidate answers
# during dataset generation. Applying the same logic live, at inference
# time, to catch the model stating a different number than the real tool
# output it just received: live testing found it confidently reverting to
# a wrong prior answer even after running python_tool correctly.
from vasudha.datasets.reasoning_verify import _extract_stdout, _extract_number

app = Flask(__name__)

# Ollama already owns the model (GGUF, quantized, with the correct chat
# template + stop tokens baked into the "vasudha" Modelfile) — loading a
# second full copy here via transformers/bitsandbytes would need the whole
# quantized model to fit in VRAM at once (no CPU fallback), and would have
# to re-apply enable_thinking=False by hand or reproduce the exact same
# "forced-open <think> tag" bug that Modelfile was built to fix.
OLLAMA_URL = "http://localhost:11434/api/chat"
# Overridable so a new checkpoint (e.g. "vasudha-v2") can be tried without
# touching the deployed default. Set the env var in the SAME shell before
# launching — PowerShell: $env:VASUDHA_OLLAMA_MODEL="vasudha-v2"
# cmd.exe: set VASUDHA_OLLAMA_MODEL=vasudha-v2
# (bash-style `VAR=val python app.py` on one line does NOT work in either.)
OLLAMA_MODEL = os.environ.get("VASUDHA_OLLAMA_MODEL", "vasudha")
print(f"\n{'='*60}\n  ACTIVE OLLAMA MODEL: {OLLAMA_MODEL}\n{'='*60}\n", flush=True)
executor = SandboxedCodeExecutor(timeout=15)
searcher = WebSearcher(max_results=5)
fetcher = PageFetcher(timeout=10, max_chars=4000)
# One persistent directory per chat session — write_file/python_tool both
# operate inside it, so a file written in one turn is still there (and
# importable/runnable) in the next. See web/tools.py's WorkspaceManager
# docstring for why this didn't work before (SandboxedCodeExecutor used to
# wipe its tempdir after every single call).
workspace_manager = WorkspaceManager()
# Room for a real "search, judge, refine, search again" loop (up to 3
# attempts per the prompt) PLUS multiple independent python_tool checks when
# a request has several verifiable sub-claims, PLUS scaffolding a small
# multi-file project (write several files, list to confirm, run/test,
# fix an error, re-run), plus a final answer/render_tool. 9 was already
# tight for verification-heavy questions alone; project creation needs
# meaningfully more calls than a single computed answer does.
MAX_TOOL_ITERATIONS = 18

# Three distinct tools, three distinct tags. render_tool is deliberately not
# a variant of python_tool: it's never executed server-side at all (the
# frontend renders it client-side in a sandboxed iframe), so it doesn't need
# — and shouldn't get — a round trip through the execution sandbox.
#
# Tools chain across multiple turns within one user request (see
# MAX_TOOL_ITERATIONS below) — e.g. search_tool (possibly more than once) to
# find real reference material, then render_tool informed by what was
# actually found, instead of guessing from memory. PLAN-FIRST is universal,
# not render-specific: jumping straight to a tool call with no stated plan is
# exactly what produced rambling, self-contradicting output before this
# rewrite (a python_tool re-run instead of trusting its own first result, a
# duplicated render_tool nobody asked for).
SYSTEM_PROMPT = """You are Vasudha, an intelligent and extremely concise AI assistant with access to four tools.

UNIVERSAL RULE — PLAN BEFORE EVERY TOOL CALL: before any python_tool, search_tool, fetch_tool, or render_tool, write one short line stating what you're about to do and why (e.g. "Checking the current record holder before answering." or "Need the CSS class names for this component before building the page."). Then make exactly one tool call. Never call a tool with no stated reason, and never call the same tool twice in a row without a new one-line reason for the second call.

1. PYTHON EXECUTION — for computation, algorithms, data processing, or testing logic. Code actually runs in a real sandboxed interpreter; you will see the genuine output before giving your final answer. Never claim a numeric or computed result without this tool backing it. Trust a clean result — don't re-run the same computation again without a real reason (e.g. you spot a bug in the first attempt).
<python_tool>
```python
# code here
```
</python_tool>

2. LIVE WEBPAGE RENDER — for ANY request involving HTML, CSS, JS, a webpage, a UI mockup, or a visual component. This is MANDATORY: you must NEVER write raw HTML/CSS/JS directly in your reply — it will not render there. It always goes inside <render_tool> tags, rendered live in a preview canvas, not executed as a script. Write one complete, self-contained HTML document.

STYLING RULE: always include Tailwind CSS via CDN in <head> and style using Tailwind utility classes, not hand-written CSS:
<script src="https://cdn.tailwindcss.com"></script>
Use it deliberately: a real layout (e.g. a hero section, centered content, a card grid — not just stacked default elements), generous spacing (padding/margin utilities), a coherent color palette (2-3 colors, not every Tailwind color at once), rounded corners, and shadows on interactive elements. Prefer a Google Font (link it in <head>) over the default sans-serif.

For any non-trivial page (more than one section, or based on research), the plan (per the universal rule above) must list the actual sections/structure you will build, 2-4 lines, before the render_tool call — not just a one-line reason. Write <render_tool> matching that plan exactly. After </render_tool> closes, stop completely — do not write a second render_tool, do not repeat yourself, do not re-describe the page in prose afterward beyond one short closing line.

Example — user asks "write a webpage that says Hi":
Plan: single centered heading, dark background, one accent color.
<render_tool>
```html
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen flex items-center justify-center bg-slate-900">
<h1 class="text-5xl font-bold text-teal-400">Hi</h1>
</body>
</html>
```
</render_tool>
Here's your webpage.

3. WEB SEARCH — for current events, facts you are not certain of, anything time-sensitive, or real reference material for a render_tool request. Returns real search RESULT SNIPPETS ONLY (title + short excerpt per result), not full page content.
<search_tool>your search query</search_tool>

4. READ A PAGE — search_tool's snippets are often too short or too generic to contain a precise fact (an exact version number, a specific figure, a precise date). When that happens, fetch the actual content of the single most promising result URL instead of guessing from the snippet.
<fetch_tool>https://exact-url-copied-from-a-search_tool-result</fetch_tool>
Only ever fetch a URL that a search_tool call actually returned this turn — never invent, guess, or recall a URL from memory.

SEARCH IS A PROCESS, NOT A LOOKUP. Raw results are not an answer — after each search:
1. Read the actual snippets. Name, in one line, which specific result(s) contain the fact you needed, or note that none do.
2. Judge honestly: is this conclusive, partial, or off-target? A page that merely mentions your topic without stating the fact you need is not conclusive. If a specific result looks promising but its snippet doesn't spell out the fact, use fetch_tool on it before giving up on that result.
3. If still not conclusive after that, refine the query based on what you now know is missing (more specific terms, a different angle, a name/date you just learned) and search again.
4. Repeat up to three total searches (each may include a fetch_tool call on its best result). If still inconclusive after that, say plainly what remains uncertain rather than presenting a guess as fact.
Never treat the first result as the answer just because it was first — synthesize across multiple results when they matter, and prefer results that directly state the fact over ones that merely reference the topic.

VERIFY EVERY VERIFIABLE PART OF THE REQUEST, not just the first one. If the user asks about multiple claims, options, or sub-questions (e.g. "which of these statements are true"), each one that can be checked with python_tool, search_tool, or fetch_tool gets its own check before you conclude — do not verify the first claim and then assert the rest from memory. Use tools freely and as often as the question actually requires; a slower, fully-verified answer beats a fast, partly-guessed one. If a specific part genuinely cannot be verified with the tools available (e.g. an image you cannot see), say that plainly instead of guessing.

5. PROJECT FILES — for building an actual multi-file project (an app, a script with multiple modules, anything beyond a single snippet). Unlike a one-off python_tool call, these three tools share ONE persistent workspace for this entire conversation: a file you write now is still there — and importable, runnable, editable — in every later turn and every later python_tool call, until you start a new chat.

Write a file (creates parent folders automatically):
<write_file>
relative/path/to/file.py
file content starts on this next line, can span as many lines as needed
</write_file>
The very first line after the opening tag is the path — nothing else on that line. Every line after that, verbatim, is the file's content.

Read a file back (to check what's actually there before editing, never assume):
<read_file>relative/path/to/file.py</read_file>

List what exists in the project so far:
<list_directory>.</list_directory>

WORKFLOW FOR BUILDING A PROJECT: state a short plan (which files, what each does), write each file, list_directory to confirm what actually landed, then python_tool to actually run/test it — real execution, in the same workspace, so it sees the files you just wrote. If it errors, read the real traceback, fix the specific file, and re-run — don't rewrite files you have no evidence are wrong. Only give your final answer once you've actually run the project and seen it work, not once you've merely written code that looks right.

CRITICAL BEHAVIOR RULES:
- CONCISENESS: Reply only with what is asked, beyond the required one-line plan before each tool call. Keep answers brief, direct, and factual. Avoid conversational filler, introductory chatter, and unsolicited over-explanation.
- CONTEXT ISOLATION: Treat each new question as completely independent unless the user explicitly references an earlier message.
- Never claim to have executed code, rendered a page, searched the web, read a page, or written/read/listed a project file unless the corresponding tool tag was actually used this turn.
- REAL EXECUTION OUTRANKS YOUR OWN MATH, ALWAYS: once python_tool has actually run, its printed result is the answer — not a suggestion, not one input among several. If your own mental calculation (before or after running the tool) disagrees with what the tool printed, your mental calculation is wrong and the tool is right, with no exceptions. Never state a final numeric answer that differs from the tool's real output when the tool was used for that exact calculation."""

# Four power levels trading generation length / context capacity for wall-
# clock time and RAM — this runs entirely on local hardware the user owns,
# so unlike a hosted API there's no fixed "right" answer for how much to
# spend per turn; a quick question doesn't need "ultra", a multi-fetch
# research report might need more than "balanced" gives it. 24 of this
# model's 32 layers are linear-attention (O(1) memory vs. context length),
# so even "ultra"'s 131072 (half the model's real 262144 max) is cheaper
# here than the same number would be for an ordinary dense transformer.
# timeout scales with num_predict since local generation speed is fixed —
# 8192 tokens genuinely can take much longer than 180s to finish.
POWER_PRESETS = {
    "low":      {"num_predict": 512,  "num_ctx": 8192,   "timeout": 120},
    "balanced": {"num_predict": 2048, "num_ctx": 32768,  "timeout": 300},
    "high":     {"num_predict": 4096, "num_ctx": 65536,  "timeout": 600},
    "ultra":    {"num_predict": 8192, "num_ctx": 131072, "timeout": 1200},
}
DEFAULT_POWER = "balanced"


def run_single_generation(messages_to_generate, power=DEFAULT_POWER):
    """Stream one assistant turn from Ollama. Ollama applies this model's
    real chat template and stop tokens itself (baked into the "vasudha"
    Modelfile, including the pre-closed <think></think> that makes direct
    answers the default) — messages are passed through as-is, no manual
    template application needed on this side."""
    preset = POWER_PRESETS.get(power, POWER_PRESETS[DEFAULT_POWER])
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages_to_generate,
            "stream": True,
            "options": {
                "temperature": 0.6, "top_p": 0.9, "repeat_penalty": 1.05,
                "num_predict": preset["num_predict"], "num_ctx": preset["num_ctx"],
            },
        },
        stream=True,
        timeout=preset["timeout"],
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content
        if chunk.get("done"):
            break

@app.route("/")
def index():
    return render_template(
        "index.html",
        system_prompt=SYSTEM_PROMPT,
        power_levels=list(POWER_PRESETS.keys()),
        default_power=DEFAULT_POWER,
    )

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])
    power = data.get("power", DEFAULT_POWER)
    if power not in POWER_PRESETS:
        power = DEFAULT_POWER

    if not messages:
        return {"error": "No messages provided"}, 400

    # Guarantee our system prompt is always present as the very first message
    if messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    else:
        messages[0]["content"] = SYSTEM_PROMPT

    # The frontend generates one session_id per chat (regenerated on
    # Reset/New Chat) and sends it with every /chat call — that's what lets
    # write_file/python_tool share one real, persistent workspace across an
    # entire conversation instead of each tool call getting a wiped-clean
    # sandbox. A missing session_id (an old/other client) falls back to a
    # single shared "default" workspace rather than failing the request —
    # this is a single-user local dev server, not a multi-tenant service.
    session_id = data.get("session_id") or "default"
    workspace_dir = workspace_manager.get(session_id)
    file_tools = WorkspaceFileTools(workspace_dir)

    def generate():
        # Genuinely multi-step now, not one-shot: python_tool/search_tool
        # feed a real result back and the loop continues, so a turn can
        # chain e.g. search_tool (find real reference material) into
        # render_tool (build the page informed by it) before the model gives
        # its final answer. render_tool ends the loop — nothing to feed back
        # for it, and it's the deliverable itself, not an intermediate step.
        # Capped at MAX_TOOL_ITERATIONS so a confused model can't loop
        # forever burning local generation time.
        # write_file carries a path AND content, so it needs its own regex
        # (first line = path, "---" separator, rest = content) rather than
        # the single-capture-group pattern the plain-content tags share.
        SIMPLE_TAGS = ("python_tool", "search_tool", "fetch_tool", "render_tool", "read_file", "list_directory")
        WRITE_FILE_RE = re.compile(r"<write_file>\s*([^\n]+)\n([\s\S]*?)</write_file>")

        for _ in range(MAX_TOOL_ITERATIONS):
            accumulated_text = ""
            handled_tag = None
            handled_content = None
            handled_path = None

            generation = run_single_generation(messages, power)
            for new_text in generation:
                accumulated_text += new_text
                yield new_text

                for tag in SIMPLE_TAGS:
                    if f"</{tag}>" in accumulated_text:
                        match = re.search(rf"<{tag}>([\s\S]*?)<\/{tag}>", accumulated_text)
                        if match:
                            handled_tag = tag
                            handled_content = match.group(1)
                        break

                if handled_tag is None and "</write_file>" in accumulated_text:
                    match = WRITE_FILE_RE.search(accumulated_text)
                    if match:
                        handled_tag = "write_file"
                        handled_path = match.group(1).strip()
                        handled_content = match.group(2)

                if handled_tag is not None:
                    # Stop consuming the moment a tool call closes — anything
                    # the model writes after that in this same pass is not
                    # something we asked for. Without this, a closed
                    # render_tool (which ends the turn) would still let the
                    # model ramble on and possibly write a second, unhandled
                    # tool tag that leaks through as raw unprocessed text.
                    generation.close()
                    break

            if handled_tag is None:
                return  # No tool this pass — that was the final answer.

            if handled_tag == "render_tool":
                return  # Rendered client-side already; nothing to feed back.

            if handled_tag == "python_tool":
                tool_res = executor.execute_python(handled_content, cwd=str(workspace_dir))
                computed = _extract_number(_extract_stdout(tool_res))
                if computed is not None:
                    # Stated as a direct, unambiguous instruction — not "use
                    # this if needed" — because the failure mode observed
                    # live is the model running the tool correctly, then
                    # reverting to a different number from its own prior
                    # anyway. Naming the exact real value removes any room
                    # for it to substitute a different one "from memory."
                    follow_up = (
                        f"Here is the real execution output of your Python code:\n{tool_res}\n\n"
                        f"The real computed value is {computed!r}. Your final answer's number MUST be exactly "
                        f"this value — not a number from your own mental math, not a recalled/remembered value, "
                        f"not a rounded or adjusted version. If {computed!r} conflicts with something you believed "
                        f"before running this code, the code is correct and your prior belief was wrong. State "
                        f"your final answer now using {computed!r}, or call another tool if more work is needed."
                    )
                else:
                    follow_up = (
                        f"Here is the real execution output of your Python code:\n{tool_res}\n\n"
                        "Use this if needed, then give your final answer (or call another tool if more work is needed)."
                    )
            elif handled_tag == "search_tool":
                query = handled_content.strip()
                tool_res = searcher.search(query)
                follow_up = (
                    f'Here are the real search results for "{query}":\n{tool_res}\n\n'
                    "Judge whether this is conclusive. If a snippet looks promising but too short, "
                    "fetch_tool that URL. If nothing is promising, refine the query and search again. "
                    "Otherwise use this for your final answer, or call render_tool if the user asked for a "
                    "webpage/UI, informed by what you found here."
                )
            elif handled_tag == "fetch_tool":
                url = handled_content.strip()
                tool_res = fetcher.fetch(url)
                follow_up = (
                    f"Here is the real extracted text from {url}:\n{tool_res}\n\n"
                    "Use this if it has the fact you needed, then give your final answer (or search_tool "
                    "again with a refined query if it didn't)."
                )
            elif handled_tag == "write_file":
                tool_res = file_tools.write_file(handled_path, handled_content)
                follow_up = (
                    f"Here is the real result of writing {handled_path!r}:\n{tool_res}\n\n"
                    "Continue building the project (write more files, list_directory to confirm what's "
                    "there, or python_tool to actually run/test it), or give your final answer once "
                    "you've verified it works."
                )
            elif handled_tag == "read_file":
                path = handled_content.strip()
                tool_res = file_tools.read_file(path)
                follow_up = (
                    f"Here is the real current content of {path!r}:\n{tool_res}\n\n"
                    "Use this if needed, then continue building or give your final answer."
                )
            else:  # list_directory
                path = handled_content.strip() or "."
                tool_res = file_tools.list_directory(path)
                follow_up = (
                    f"Here is the real current file listing for {path!r}:\n{tool_res}\n\n"
                    "Use this to confirm what actually exists, then continue or give your final answer."
                )

            yield f"\n\n<tool_output>\n{tool_res}\n</tool_output>\n\n"
            messages.append({"role": "assistant", "content": accumulated_text})
            messages.append({"role": "user", "content": follow_up})

        yield "\n\n[Reached the max tool steps for this turn.]"

    return Response(stream_with_context(generate()), mimetype='text/plain')

if __name__ == "__main__":
    try:
        requests.get("http://localhost:11434", timeout=2)
    except requests.exceptions.ConnectionError:
        print(
            "Ollama doesn't appear to be running at localhost:11434 — "
            "start it (or `ollama serve`) before using this app."
        )
    app.run(host="0.0.0.0", port=5000, threaded=True)
