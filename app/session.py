"""One conversation: the agent loop, bound to a backend and a workspace.

This is web/agent.py's loop rewritten against app/backends.py so it works with
or without ollama, and yielding plain dicts the UI can render directly.

The repairs measured in scripts/bench_numeric.py are kept, because both were
real failures on this model and neither is hypothetical:

  * Unit conversion happens inside the sandbox. The model computed 1.2e8 Pa
    correctly and then wrote "120,000 MPa" — dividing by 1e3 instead of 1e6 in
    its head. Making the tool print the converted value fixed it.
  * An empty turn is repaired rather than shown. After a correct tool result
    the model sometimes returns nothing at all, which would leave the user
    staring at a blank reply when the right number was already in hand.
"""
from __future__ import annotations

import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Iterator, Optional

from app import personas
from app.backends import Backend, BackendError, ToolCall

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 12

#: Fraction of the context window at which older tool output starts being
#: summarised away. Not higher: the compaction pass itself has to fit, and so
#: does the reply the model is about to write.
COMPACT_AT = 0.70

#: A tool result longer than this is truncated in the middle before it ever
#: enters the transcript. A 40 KB page fetch is not more informative than its
#: first and last few thousand characters, and it will evict the actual
#: question from the window.
MAX_TOOL_RESULT_CHARS = 6000

#: The rules a persona must never be able to soften. personas.build_system_prompt
#: appends this AFTER the voice guidance for exactly that reason.
CORE_RULES = """Answer briefly and directly. Do not restate the question or pad the reply.

NUMBERS — python_tool
For ANY question with a numeric answer, call python_tool and take the number from its real output. Never state a computed value you did not get from the tool. Have the code print the value already converted to the unit asked for.
Before using a standard formula, name the exact case it belongs to — the support condition, the boundary conditions, the assumptions — and check your formula is the one for THAT case, not a similar one. A coefficient recalled from a neighbouring case gives a confidently wrong answer.

SOURCES OUTRANK YOUR MEMORY — always
When a tool has returned something, that is the fact. Your own recollection is not a second opinion about it. If a page says a figure and you remember a different one, the page wins and you say what the page says. If a tool result contradicts what you were about to write, discard what you were about to write. Never revise a fetched number toward the one you expected, and never fill a gap in a search result with a remembered figure without saying that is what you did.

FACTS — search_tool, fetch_tool
For anything current, time-sensitive, or that you are not certain of, search before answering. Search is a process, not a lookup:
1. Read the actual snippets. Say in one line which result contains the fact, or that none do.
2. If a promising result's snippet does not spell the fact out, fetch_tool that URL.
3. If still inconclusive, refine the query and search again — up to three searches.
4. If sources disagree, say so explicitly and give both figures with their sources. Never present a contested fact as settled.
Name your sources in the final answer. If you could not establish something, say that plainly rather than guessing.

DELIVERABLES — document_tool
If the user asks for a report, summary, comparison, table, spreadsheet or page, build it with document_tool rather than pasting it into the chat. Keep your chat reply to one or two lines saying what you made. Research it first, compute any figures with python_tool, then write the document from what you actually found — never from memory alone.

If a question needs no calculation, no search and no document, just answer it in a sentence.

NEVER claim you consulted a source you did not actually open with fetch_tool. Do not write "Source: ..." under a table unless you read that page this turn. A citation you did not earn is worse than none, because it is what makes a wrong figure look checked."""

SYSTEM_PROMPT = personas.build_system_prompt(personas.DEFAULT_PERSONA, CORE_RULES)


def _unit_rule(what: str) -> str:
    return (f"Compute {what} in a real Python sandbox and return its stdout. "
            "The code MUST print the final value already converted to the unit the "
            "question asks for, together with that unit — e.g. print(f'{sigma/1e6:.4g} MPa'), "
            "not print(sigma). Never leave a unit conversion to be done afterwards. "
            # Watched failure: a model defined `reviews` in one call, referred to
            # it in the next, and spent four turns on the NameError. Nothing in
            # the schema told it the process does not persist, so it reasonably
            # assumed a REPL.
            "EACH CALL RUNS IN A FRESH PROCESS: variables, imports and state from "
            "an earlier call are gone. Every script must be self-contained. Files "
            "you wrote to the workspace do persist — read them back with open().")


def default_schemas(include_browser: Optional[bool] = None) -> list[dict]:
    """Tool schemas for one session.

    browse_tool is dropped when Playwright is not installed rather than left in
    to fail: an unusable schema costs ~200 tokens of every prompt and invites
    the model to spend a turn discovering it does not work.
    """
    if include_browser is None:
        from web.browser import playwright_available
        include_browser = playwright_available()

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "python_tool",
                "description": _unit_rule("a numeric or algorithmic result"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string",
                                 "description": "Python source. Must print the final answer and its unit."},
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_tool",
                "description": (
                    "Search the web and return real result snippets (title + short excerpt). "
                    "Use for current events, prices, releases, anything time-sensitive, or any "
                    "fact you are not certain of. Returns SNIPPETS ONLY — if a snippet does not "
                    "actually state the fact you need, follow up with fetch_tool on that URL."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Search terms. Be specific; refine and search "
                                                 "again if the first pass is inconclusive."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_tool",
                "description": (
                    "Read the real text of one webpage. Only pass a URL that a search_tool "
                    "result returned in this conversation — never invent or recall a URL."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Exact URL from a search result."},
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "document_tool",
                "description": (
                    "Produce a finished document and show it in the preview canvas beside the "
                    "chat. Use this whenever the user asks for a report, summary, table, "
                    "comparison, spreadsheet, webpage or any deliverable they will keep — do "
                    "NOT paste a long document into the chat instead. The file is saved to the "
                    "workspace and the user can export it."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short document title."},
                        "format": {"type": "string", "enum": ["markdown", "html", "csv"],
                                   "description": "markdown for reports and notes; csv for "
                                                  "spreadsheet data (first row = headers); "
                                                  "html for a styled page or complex layout."},
                        "content": {"type": "string",
                                    "description": "The complete document body in that format."},
                    },
                    "required": ["title", "format", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_in_file",
                "description": (
                    "Search a workspace file and get back the matching lines with "
                    "their line numbers and surrounding context. Use this BEFORE "
                    "edit_file to locate exactly what to change, and after a "
                    "traceback to see the line it named. Far cheaper than reading a "
                    "whole file, and it gives you the verbatim text to use as an "
                    "edit anchor."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File to search."},
                        "pattern": {"type": "string",
                                    "description": "Text or regular expression to find, "
                                                   "e.g. a variable name or 'def solve'."},
                        "context": {"type": "integer",
                                    "description": "Lines of context each side. Default 3."},
                    },
                    "required": ["path", "pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "debug_tool",
                "description": (
                    "Look up an error message that others have hit before. Paste the "
                    "exception line and this searches developer sites (Stack Overflow, "
                    "GitHub issues, the docs) with the machine-specific parts — your "
                    "file paths, line numbers, memory addresses — stripped out, which "
                    "is what makes the search actually match. Use it when a traceback "
                    "is unfamiliar, not for one you already understand."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string",
                                  "description": "The error or traceback, pasted as-is."},
                        "context": {"type": "string",
                                    "description": "Optional: the library or thing you "
                                                   "are doing, e.g. 'numpy integration'."},
                    },
                    "required": ["error"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember_tool",
                "description": (
                    "Write a lesson into your memory book, which you are shown at "
                    "the start of every chat. Use it when you discover something "
                    "worth not rediscovering: a tool that behaves unexpectedly, a "
                    "package this sandbox lacks, a fact you got wrong and then "
                    "corrected from a source, a technique that worked. Writing the "
                    "same topic again REPLACES the old lesson, so correct yourself "
                    "here whenever a source proves an earlier note wrong."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string",
                                  "description": "Short key, e.g. 'sentiment analysis' "
                                                 "or 'sandbox packages'. Reusing a topic "
                                                 "overwrites it."},
                        "lesson": {"type": "string",
                                   "description": "One or two sentences, written to be "
                                                  "useful to yourself later."},
                        "source": {"type": "string",
                                   "description": "URL or tool name it came from, if any. "
                                                  "A lesson with a source outranks one without."},
                    },
                    "required": ["topic", "lesson"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pip_tool",
                "description": (
                    "Install Python packages so python_tool can import them. The "
                    "sandbox has the standard library and little else, so install "
                    "before importing anything third-party rather than guessing "
                    "whether it is there. Installed once, available in every later "
                    "call and every later chat."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "packages": {
                            "type": "string",
                            "description": "Space- or comma-separated names, "
                                           "optionally pinned: 'textblob' or "
                                           "'pandas numpy' or 'requests==2.31.0'.",
                        },
                    },
                    "required": ["packages"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "shell_tool",
                "description": (
                    "Run one shell command in the workspace directory. Use for "
                    "things the other tools do not cover — inspecting files, "
                    "running a build or a test command, checking what is installed. "
                    "Not for interactive programs: stdin is closed, so anything "
                    "that waits for input is killed on timeout."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string",
                                    "description": "The command line to run."},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "browse_tool",
                "description": (
                    "Open a page in a real browser and read it AFTER its JavaScript "
                    "has run, then click links and fill in fields. Use this instead of "
                    "fetch_tool when a page needs a browser: documentation sites, "
                    "anything interactive, search results you want to follow, or when "
                    "fetch_tool came back empty or looked like a shell. Returns the "
                    "page's text plus a numbered list of things you can interact with; "
                    "pass one of those refs back to click or type."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["open", "read", "click", "type", "back"],
                            "description": "open a url; read the current page again; "
                                           "click a ref; type into a ref; go back.",
                        },
                        "url": {"type": "string", "description": "For action=open."},
                        "ref": {"type": "string",
                                "description": "For click/type: a ref from the last "
                                               "digest, e.g. 'ref3'."},
                        "text": {"type": "string", "description": "For action=type."},
                        "submit": {"type": "boolean",
                                   "description": "For action=type: press Enter after "
                                                  "typing. Use for search boxes."},
                    },
                    "required": ["action"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Write a text file into the workspace. Use for source code, "
                    "configs and data the user asked you to build — not for reports, "
                    "which belong in document_tool. Overwrites an existing file."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Path relative to the workspace, e.g. 'src/main.py'."},
                        "content": {"type": "string", "description": "Full file contents."},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read a text file from the workspace. Output is prefixed with "
                    "line numbers as 'NN| ' so you can match a traceback's line "
                    "number to the source. Those prefixes are NOT part of the file "
                    "— strip them before passing text to edit_file or write_file. "
                    "Read before you edit: an edit written from memory usually "
                    "misses on whitespace and changes nothing."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to the workspace."},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List what is currently in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Subdirectory to list. Omit for the whole workspace."},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": (
                    "Replace an exact string in a workspace file. Prefer this over "
                    "write_file when changing part of an existing file — rewriting a "
                    "whole file to alter one line is how details get silently dropped. "
                    "old_text must appear EXACTLY once."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to the workspace."},
                        "old_text": {"type": "string",
                                     "description": "Exact text to replace, including indentation."},
                        "new_text": {"type": "string", "description": "Replacement text."},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
    ]

    if not include_browser:
        schemas = [s for s in schemas if s["function"]["name"] != "browse_tool"]
    return schemas


_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$", re.M)
_HEADING = re.compile(r"^#{1,4}\s+\S", re.M)
_BULLET = re.compile(r"^\s*([-*+]|\d+[.)])\s+\S", re.M)


def looks_like_document(text: str) -> Optional[str]:
    """Is this reply a deliverable rather than a chat message?

    The model is told to call document_tool for reports and tables and simply
    does not — it pastes them into the chat instead. That is the same
    instruction-following gap that made tool use fire 1/6 before the tool
    schema forced it. So rather than trusting the prompt, document-shaped
    replies are detected and promoted to the canvas.

    Returns a title, or None to leave the reply in the chat. Thresholds are set
    so an ordinary two-paragraph answer stays a chat message: promoting normal
    prose into a document panel would be far more annoying than missing one.
    """
    if not text or len(text) < 220:
        return None

    has_table = bool(_TABLE_SEP.search(text)) and len(_TABLE_ROW.findall(text)) >= 3
    headings = len(_HEADING.findall(text))
    bullets = len(_BULLET.findall(text))
    lines = text.count("\n") + 1

    # A heading plus four or more bullets is a briefing, not a chat reply. The
    # length floor stays low because the negative cases that matter — a short
    # answer, a three-item list with no heading, a code snippet — all miss on
    # structure rather than on length.
    if not (has_table or (headings >= 2 and lines >= 10) or
            (headings >= 1 and bullets >= 4 and len(text) > 400)):
        return None

    heading = _HEADING.search(text)
    if heading:
        line = text[heading.start():text.find("\n", heading.start()) % (len(text) + 1)]
        title = line.lstrip("#").strip()
        if title:
            return title[:70]
    return "Document"


class ChatSession:
    def __init__(self, backend: Backend, system_prompt: str = SYSTEM_PROMPT,
                 workspace: Optional[str] = None, memory=None,
                 interaction_log=None) -> None:
        from web.tools import PageFetcher, SandboxedCodeExecutor, WebSearcher

        self.backend = backend
        self.system_prompt = system_prompt
        #: Lessons from earlier sessions, and the log the next dataset is built
        #: from. Optional so the benchmark harness can run without either.
        self.memory = memory
        self.interaction_log = interaction_log
        self.history: list[dict] = []
        self._executor = SandboxedCodeExecutor(timeout=15)
        self._searcher = WebSearcher(max_results=5)
        self._fetcher = PageFetcher(timeout=10, max_chars=4000)
        self._workspace = workspace
        self._file_tools = None
        self._installer = None
        self._shell = None
        #: Every tool result of this conversation, kept whole and
        #: searchable. Compaction removes text from the prompt; this
        #: is where it goes instead of nowhere.
        from app.retrieval import ToolMemory
        self.tool_memory = ToolMemory()
        self._turn_no = 0
        #: Files the user attached this session, as {name, path, size,
        #: description}. Only the description ever reaches the prompt.
        self.attachments: list[dict] = []
        #: Images written by the last python_tool call, drained by ask().
        self.new_images: list[str] = []
        #: Lazily launched on the first browse_tool call, then kept for the rest
        #: of the chat — research is several steps and relaunching would drop
        #: the cookies that make a multi-page flow work.
        self._browser = None
        #: filled by document_tool so the UI can render the canvas
        self.last_document: Optional[dict] = None
        #: what this turn really consulted — the basis for provenance, since
        #: the model's own account of its sources cannot be trusted
        self._searches: list[str] = []
        self._fetches: list[str] = []
        #: Figures in the reply that no source this turn contained.
        self._unsourced_line = ""
        self.tools = {
            "python_tool": self._python_tool,
            "search_tool": self._search_tool,
            "fetch_tool": self._fetch_tool,
            "document_tool": self._document_tool,
            "browse_tool": self._browse_tool,
            "pip_tool": self._pip_tool,
            "shell_tool": self._shell_tool,
            "remember_tool": self._remember_tool,
            "find_in_file": self._find_in_file,
            "debug_tool": self._debug_tool,
            "write_file": self._write_file,
            "read_file": self._read_file,
            "list_files": self._list_files,
            "edit_file": self._edit_file,
        }
        self.schemas = default_schemas()
        self.options: dict = {"temperature": 0.6, "top_p": 0.9,
                              "num_predict": 2048, "num_ctx": 16384}
        self.max_iterations = MAX_ITERATIONS

    def set_workspace(self, path: str) -> None:
        """Point the file tools at a different directory.

        Resets the cached WorkspaceFileTools rather than mutating it, so a call
        already running keeps writing where it started instead of having the
        root swapped underneath it mid-write.
        """
        self._workspace = path
        self._file_tools = None

    def set_tool_timeout(self, seconds: int) -> None:
        """Rebuild the executor rather than mutating it: a run already in
        flight keeps the timeout it started with."""
        from web.tools import SandboxedCodeExecutor
        self._executor = SandboxedCodeExecutor(timeout=int(seconds))

    # -- tools -------------------------------------------------------------

    # -- attachments -------------------------------------------------------

    def attach(self, source: str) -> dict:
        """Copy a file into the workspace and describe it to the model.

        Copied rather than referenced so the sandbox — which has no access to
        the rest of the disk by design — can actually open it, and so the file
        is still there if the user moves the original.

        The description, not the contents, is what enters the conversation.
        See web.tools.describe_file: a 40 MB CSV becomes a row count, a column
        list and twelve sample lines, which is everything the model needs in
        order to write code that processes the whole thing.
        """
        import shutil
        from web.tools import describe_file

        origin = Path(source)
        if not origin.is_file():
            return {"ok": False, "error": f"not a file: {source}"}

        root = self._files().workspace
        target = root / re.sub(r"[^\w\-. ]+", "_", origin.name)
        try:
            if origin.resolve() != target.resolve():
                shutil.copy2(origin, target)
        except OSError as exc:
            return {"ok": False, "error": f"could not copy: {exc}"}

        record = {"name": target.name, "path": str(target),
                  "size": target.stat().st_size,
                  "description": describe_file(target)}
        self.attachments = [a for a in self.attachments if a["name"] != record["name"]]
        self.attachments.append(record)
        return {"ok": True, **record}

    def _attachment_section(self) -> str:
        """The note appended to the system prompt while files are attached."""
        if not self.attachments:
            return ""
        parts = ["\n\nFILES THE USER ATTACHED (already in your workspace — open "
                 "them by name with python_tool; do NOT ask the user to paste "
                 "their contents):"]
        parts += [a["description"] for a in self.attachments]
        parts.append(
            "For anything larger than a few hundred lines, do not read the file "
            "into the conversation. Write code that streams or aggregates it and "
            "prints only the answer.")
        return "\n\n".join(parts)

    # -- tools -------------------------------------------------------------

    def _python_tool(self, code: str = "", **_: object) -> str:
        """Run code, then surface any picture it drew.

        Charts are the case where 'print the answer' is not enough: matplotlib
        writes a PNG and says nothing, so without this the model reports success
        and the user sees no plot. Files created during the call are compared
        against a snapshot taken before it, so an image that was already in the
        workspace is not re-shown on every later call.
        """
        from web.tools import IMAGE_SUFFIXES

        root = self._files().workspace
        before = {p: p.stat().st_mtime for p in root.rglob("*")
                  if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()}

        result = self._executor.execute_python(code, cwd=self._workspace)
        result = _flag_numerical_failure(result)

        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if before.get(path) == path.stat().st_mtime:
                continue
            self.new_images.append(str(path))
        return result

    def _find_in_file(self, path: str = "", pattern: str = "",
                      context: int = 3, **_: object) -> str:
        """Locate text in a file and return it with line numbers.

        The point is to make an edit anchor obtainable without reading the whole
        file. The reported loop had the model patching from memory because a
        failed edit told it nothing about the source; this lets it ask a narrow
        question and get verbatim text back to copy.
        """
        raw = self._files().read_file(path)
        if raw.startswith("["):
            return raw
        if not pattern:
            return "[error] find_in_file needs a pattern"

        try:
            matcher = re.compile(pattern)
        except re.error:
            # A model searching for `df.groupby(` is writing text, not a regex.
            matcher = re.compile(re.escape(pattern))

        lines = raw.split("\n")
        hits = [i for i, line in enumerate(lines) if matcher.search(line)]
        if not hits:
            return (f"[no match for {pattern!r} in {path}. The file has "
                    f"{len(lines)} lines — try a shorter or different pattern.]")

        window = max(int(context or 3), 0)
        shown: list[str] = []
        last_end = -1
        for index in hits[:20]:
            start, end = max(0, index - window), min(len(lines), index + window + 1)
            if start > last_end + 1 and shown:
                shown.append("   ⋮")
            for i in range(max(start, last_end + 1), end):
                marker = ">" if i == index else " "
                shown.append(f"{marker}{i + 1:>5}| {lines[i]}")
            last_end = end - 1

        more = f"\n[{len(hits) - 20} more matches not shown]" if len(hits) > 20 else ""
        return (f"[{len(hits)} match(es) for {pattern!r} in {path}; "
                f"'>' marks the matching line]\n" + "\n".join(shown) + more)

    #: Stripped from an error before searching. A path, a line number or a
    #: memory address is unique to this machine and is exactly what stops a
    #: search matching anyone else's report of the same problem.
    _ERROR_NOISE = (
        (re.compile(r'File "[^"]+", line \d+(?:, in \S+)?'), ""),
        (re.compile(r"0x[0-9a-fA-F]+"), ""),
        (re.compile(r"[A-Za-z]:\\[^\s'\"]+"), ""),
        (re.compile(r"/(?:home|users|tmp|var)/[^\s'\"]+", re.I), ""),
        (re.compile(r"\bline \d+\b"), ""),
        (re.compile(r"\s+"), " "),
    )

    def _debug_tool(self, error: str = "", context: str = "", **_: object) -> str:
        """Search developer sites for an error, with the local noise removed.

        Not a scraper: this is search_tool with a query built properly. The
        reason it needs its own tool is that the query is the hard part — a
        traceback pasted verbatim contains this machine's paths and line
        numbers, matches nothing, and the model concludes nobody has hit the
        problem before.
        """
        if not error.strip():
            return "[error] debug_tool needs an error message"

        # The last non-empty line of a traceback is the exception itself, which
        # is the part worth searching; everything above it is this run's stack.
        lines = [ln.strip() for ln in error.strip().split("\n") if ln.strip()]
        subject = next((ln for ln in reversed(lines)
                        if re.match(r"^[A-Za-z_.]*(Error|Exception|Warning)\b", ln)),
                       lines[-1])
        for pattern, replacement in self._ERROR_NOISE:
            subject = pattern.sub(replacement, subject)
        subject = subject.strip(" :")[:180]

        query = " ".join(filter(None, [subject, context.strip()]))
        results = self._searcher.search(f"{query} site:stackoverflow.com OR site:github.com")
        self._searches.append(query)

        if results.startswith("[No results") or results.startswith("[Search failed"):
            # The site filter is worth one retry without it: an error from a
            # small library may only be discussed in its own docs.
            results = self._searcher.search(query)

        return (f"[searched for: {query!r}]\n\n{results}\n\n"
                "These are snippets. If one looks like your problem, open it with "
                "fetch_tool before acting on it.")

    def _remember_tool(self, topic: str = "", lesson: str = "",
                       source: str = "", **_: object) -> str:
        if self.memory is None:
            return "[error] memory is not available in this session"
        return self.memory.remember(topic, lesson, source)

    def _pip_tool(self, packages: str = "", **_: object) -> str:
        from web.tools import PackageInstaller
        if self._installer is None:
            self._installer = PackageInstaller()
        return self._installer.install(packages)

    def _shell_tool(self, command: str = "", **_: object) -> str:
        from web.tools import ShellRunner
        if self._shell is None:
            self._shell = ShellRunner(timeout=max(self._executor.timeout * 4, 60))
        # Runs in the workspace, so `dir`/`ls` shows the model its own files and
        # a relative path means the same thing here as in write_file.
        return self._shell.run(command, cwd=self._files().workspace.as_posix())

    def _search_tool(self, query: str = "", **_: object) -> str:
        """Real DuckDuckGo search. This is the one tool that sends anything off
        the machine, which is why the UI shows the query it used."""
        if not query.strip():
            return "[error] empty search query"
        result = self._searcher.search(query)
        self._searches.append(query.strip())
        return result

    def _fetch_tool(self, url: str = "", **_: object) -> str:
        if not url.lower().startswith(("http://", "https://")):
            return f"[error] not a fetchable URL: {url!r}"
        result = self._fetcher.fetch(url)
        self._fetches.append(url.strip())
        return result

    # -- browsing ----------------------------------------------------------

    def _browse_tool(self, action: str = "open", url: str = "", ref: str = "",
                     text: str = "", submit: bool = False, **_: object) -> str:
        """Drive a real browser. Every branch returns a page digest or an error
        the model can act on — never a bare exception, which would end the turn.
        """
        from web.browser import BrowserUnavailable, PageBrowser

        action = (action or "open").strip().lower()

        # Validate before touching the browser at all: a malformed call should
        # not be the reason a headless Chromium gets launched.
        if action not in ("open", "read", "click", "type", "back"):
            return (f"[error] unknown action {action!r}. "
                    "Use open, read, click, type or back.")
        if action == "open" and not url.lower().startswith(("http://", "https://")):
            return f"[error] browse_tool needs an http(s) url, got {url!r}"
        if action in ("click", "type") and not ref:
            return f"[error] action={action} needs a ref from the last page digest"

        try:
            if self._browser is None:
                self._browser = PageBrowser()

            if action == "open":
                result = self._browser.navigate(url)
            elif action == "read":
                result = self._browser.digest()
            elif action == "click":
                result = self._browser.click(ref)
            elif action == "type":
                result = self._browser.type_text(ref, text, submit=bool(submit))
            else:   # back — the action set was validated above
                result = self._browser.back()
        except BrowserUnavailable as exc:
            return f"[error] {exc}"
        except ValueError as exc:
            return f"[error] {exc}"
        except Exception as exc:  # noqa: BLE001 - a dead page must not kill the turn
            logger.exception("browse_tool failed")
            return (f"[error] the browser could not complete that: {exc}. "
                    "Try fetch_tool for a static page, or open a different url.")

        # Provenance counts a browsed page the same as a fetched one: it was
        # genuinely read, which is the only thing the footer claims.
        current = self._browser.current_url
        if current and current not in self._fetches:
            self._fetches.append(current)
        return result

    def close(self) -> None:
        """Release the browser. A headless Chromium outlives the process that
        forgot it, so this is not merely tidy."""
        if self._browser is not None:
            self._browser.close()
            self._browser = None

    # -- workspace files ---------------------------------------------------
    # WorkspaceFileTools already resolves every path against the workspace root
    # and rejects anything that escapes it, so the traversal check lives there
    # rather than being re-implemented per tool here.

    def _files(self):
        """The file tool bound to this session's workspace, or None.

        A session with no workspace (the one-shot benchmark path) gets a
        temporary directory rather than an error: refusing to write a file is
        a worse failure than writing it somewhere ephemeral.
        """
        from web.tools import WorkspaceFileTools
        if self._file_tools is None:
            root = (Path(self._workspace) if self._workspace
                    else Path(tempfile.mkdtemp(prefix="vasudha_ws_")))
            root.mkdir(parents=True, exist_ok=True)
            self._workspace = self._workspace or str(root)
            self._file_tools = WorkspaceFileTools(root)
        return self._file_tools

    def _write_file(self, path: str = "", content: str = "", **_: object) -> str:
        return self._files().write_file(path, content)

    def _read_file(self, path: str = "", **_: object) -> str:
        """Read a file with line numbers.

        Numbered because a traceback says "line 27" and an unnumbered listing
        gives the model no way to act on that. The observed failure: a
        traceback named an undefined variable, the model diagnosed it correctly
        every time, and then re-emitted the same broken line — it never
        established *where* the line was, so its "fix" was a rewrite from memory
        rather than an edit to the source.
        """
        content = self._files().read_file(path)
        if content.startswith("["):          # an error or a size refusal
            return content
        lines = content.split("\n")
        width = len(str(len(lines)))
        return "\n".join(f"{i:>{width}}| {line}"
                         for i, line in enumerate(lines, 1))

    def _list_files(self, path: str = ".", **_: object) -> str:
        return self._files().list_directory(path or ".")

    @staticmethod
    def _nearest_context(source: str, wanted: str, window: int = 4) -> str:
        """The real lines around the closest thing to what was asked for.

        Matched on the most distinctive line of the attempted anchor rather
        than on the whole block, because a failed edit usually has one line
        almost right and the rest reconstructed. Falls back to the head of the
        file, which still beats saying nothing.
        """
        import difflib

        lines = source.split("\n")
        probes = [ln.strip() for ln in wanted.split("\n") if ln.strip()]
        best_index, best_score = None, 0.0
        for probe in probes:
            for i, line in enumerate(lines):
                score = difflib.SequenceMatcher(None, probe, line.strip()).ratio()
                if score > best_score:
                    best_index, best_score = i, score

        if best_index is None or best_score < 0.45:
            head = lines[: window * 3]
            return ("The file currently begins:\n"
                    + "\n".join(f"{i:>4}| {ln}" for i, ln in enumerate(head, 1)))

        start = max(0, best_index - window)
        end = min(len(lines), best_index + window + 1)
        body = "\n".join(f"{i:>4}| {lines[i - 1]}" for i in range(start + 1, end + 1))
        return (f"The closest match is line {best_index + 1}. "
                f"What is actually there:\n{body}")

    def _edit_file(self, path: str = "", old_text: str = "",
                   new_text: str = "", **_: object) -> str:
        """Exact-string replacement, refusing anything ambiguous.

        A small model will happily pass an `old_text` that occurs three times
        and expect the one it meant. Replacing the first occurrence silently is
        how a file ends up subtly wrong in a place nobody looks; the count is
        reported back instead so the model can widen its anchor.
        """
        files = self._files()
        current = files.read_file(path)
        if current.startswith("[Error"):
            return current
        if not old_text:
            return "[error] edit_file needs old_text; use write_file to create a file"
        occurrences = current.count(old_text)
        if occurrences == 0:
            # Show the source rather than only refusing. A bare "not found"
            # leaves the model editing from its own recollection of the file,
            # which is how the same broken line gets re-emitted three times in
            # a row. Handing back the real text of the nearest match ends that
            # loop: the next attempt is against what is actually on disk.
            return (f"[error] that exact text is not in {path}.\n"
                    + self._nearest_context(current, old_text)
                    + "\nCopy the target text verbatim from above, including "
                      "indentation, and try again.")
        if occurrences > 1:
            return (f"[error] that text appears {occurrences} times in {path}. "
                    "Include more surrounding lines so it matches exactly once.")
        result = files.write_file(path, current.replace(old_text, new_text))
        if result.startswith("[Error"):
            return result
        return f"[edited {path}] one replacement made"

    # -- provenance --------------------------------------------------------

    def _stats_event(self, out_tokens: int, started: float) -> dict:
        """What the turn cost, in the engine's own units.

        Token counts come from the backend rather than from a character-count
        heuristic here, because only the engine knows what it actually
        tokenized — and the context figure is meaningless unless it is measured
        the same way the window is.
        """
        elapsed = time.time() - started
        limit = getattr(self.backend, "context_limit", None)
        return {
            "kind": "stats",
            "tps": round(getattr(self.backend, "observed_tps", 0) or 0, 1),
            "out_tokens": out_tokens,
            "context_tokens": getattr(self.backend, "last_prompt_tokens", 0),
            "context_limit": limit or 0,
            "seconds": round(elapsed, 1),
        }

    def _record_turn(self, question: str, answer: str, tools: list[dict],
                     started: float) -> None:
        """Append one turn to the training log. Never raises into the turn."""
        if self.interaction_log is None:
            return
        try:
            self.interaction_log.record(
                chat_id=getattr(self, "chat_id", ""),
                question=question, answer=answer, tools=tools,
                sources=list(dict.fromkeys(self._fetches)),
                model=getattr(self.backend, "display_name", ""),
                seconds=time.time() - started)
        except Exception:  # noqa: BLE001 - logging must not break a reply
            logger.debug("interaction log failed", exc_info=True)

    #: Tools whose output is evidence a figure can be grounded in. python_tool
    #: is the important one and used to be excluded, on the reasoning that a
    #: run that computes its own numbers cannot invent any. It can: a real
    #: session produced a statistical write-up quoting "BIC = 1234.56, AIC =
    #: 1256.78" — placeholder digits that appeared in no stdout at all — after
    #: legitimately running code. Running code is not the same as reporting
    #: what it printed, and that gap is the failure this catches.
    EVIDENCE_TOOLS = ("python_tool", "shell_tool", "search_tool",
                      "fetch_tool", "browse_tool", "read_file")

    def _ungrounded_figures(self, answer: str, tools: list[dict]) -> list[str]:
        """Figures in the answer that appear in no tool output this turn."""
        evidence = [t["result"] for t in tools if t["name"] in self.EVIDENCE_TOOLS]
        if not evidence:
            return []
        from app.memory import unsourced_figures
        return unsourced_figures(answer, evidence)

    def _repair_prompt(self, missing: list[str]) -> str:
        """Send the model back to compute what it asserted.

        Naming the exact figures matters. Told only that something is
        ungrounded, the model rewrites the prose around the same invented
        number; told that 1234.56 appears in no output, it either computes it
        or drops the claim.
        """
        return (
            "STOP. These figures appear in your reply but in no tool output this "
            "turn: " + ", ".join(missing) + ".\n"
            "You have not computed them. Either call python_tool and use the value "
            "it prints, or delete the claim entirely. Do not restate a number you "
            "have not calculated, and do not adjust it to look plausible.")

    def _unsourced_note(self, answer: str, tools: list[dict]) -> str:
        """The footer line for figures that survived the repair attempt.

        Advisory rather than destructive: a figure can legitimately be derived
        from ones that are present — a ratio, a rounded value, a sum — so
        deleting them would break correct answers. Naming them lets a reader
        check the ones that matter.
        """
        missing = self._ungrounded_figures(answer, tools)
        if not missing:
            return ""
        return ("\n\n> **Not produced by any tool this turn:** "
                + ", ".join(f"`{m}`" for m in missing)
                + ". The model asserted these rather than computing them — "
                "treat them as unverified.")

    def _provenance_block(self) -> str:
        """A footer describing what this turn ACTUALLY consulted.

        Written by the harness, never by the model. Observed failure: given a
        request for planetary data, the model ran three searches that returned
        only generic prose — one result was a children's song — then produced a
        table of figures from memory and signed it "Source: NASA Solar System
        data sheet", a page it never opened. A fabricated citation is worse
        than no citation, because it is what makes a wrong number credible.

        So the record of what was consulted comes from the tool log, and it
        says plainly when nothing was read.
        """
        lines = ["", "---", "**How this was produced**", ""]
        if self._unsourced_line:
            lines.append(self._unsourced_line.strip())
            lines.append("")
        if self._searches:
            lines.append("Searched:")
            lines += [f"- `{q}`" for q in dict.fromkeys(self._searches)]
        if self._fetches:
            lines.append("")
            lines.append("Pages actually read:")
            lines += [f"- {u}" for u in dict.fromkeys(self._fetches)]

        if not self._fetches:
            lines += [
                "",
                "> **No source page was opened.** Search returned result summaries only, "
                "and no page was read in full. Any figures above therefore come from the "
                "model's own recall, not from a checked source — verify them before use.",
            ]
        return "\n".join(lines)

    def _document_payload(self, title: str, content: str, fmt: str = "markdown") -> dict:
        """Build a canvas payload and save the file, without going through the
        tool-call path. Used when a document-shaped reply is promoted."""
        safe = re.sub(r"[^\w\-. ]+", "", title or "document").strip() or "document"
        ext = {"markdown": "md", "html": "html", "csv": "csv"}[fmt]
        filename = f"{safe[:60]}.{ext}"
        saved_to = ""
        try:
            root = (Path(self._workspace) if self._workspace
                    else Path(tempfile.gettempdir()) / "vasudha_docs")
            root.mkdir(parents=True, exist_ok=True)
            path = root / filename
            path.write_text(content, encoding="utf-8")
            saved_to = str(path)
        except OSError as exc:
            logger.warning("could not save promoted document: %s", exc)
        payload = {"title": title, "format": fmt, "content": content,
                   "filename": filename, "path": saved_to,
                   "sourced": bool(self._fetches),
                   "searches": list(dict.fromkeys(self._searches)),
                   "fetches": list(dict.fromkeys(self._fetches))}
        self.last_document = payload
        return payload

    _FAKE_CITE = re.compile(
        r"^\s*[*_>\s]*(source|sources|reference|references|data\s+from)\s*[:\-]",
        re.I | re.M)

    def _strip_unbacked_citation(self, content: str) -> tuple[str, bool]:
        """Remove a 'Source: ...' line the model wrote without reading anything.

        Only fires when zero pages were fetched this turn — if it really did
        read a page, its citation stays and the provenance footer corroborates
        it. This exists because the model signs fabricated tables with
        authoritative-sounding sources, which is precisely the thing a reader
        uses to decide whether to trust the numbers.
        """
        if self._fetches or not self._FAKE_CITE.search(content):
            return content, False
        kept = [ln for ln in content.split("\n") if not self._FAKE_CITE.match(ln)]
        return "\n".join(kept), True

    def _document_tool(self, title: str = "", format: str = "markdown",  # noqa: A002
                       content: str = "", **_: object) -> str:
        """Write a deliverable and hand it to the preview canvas.

        Returns a short confirmation rather than echoing the document back: the
        content is already in the canvas, and feeding a long document into the
        model's own context would burn the window it needs for the actual work.
        """
        fmt = (format or "markdown").lower().strip()
        if fmt not in ("markdown", "html", "csv"):
            fmt = "markdown"
        if not content.strip():
            return "[error] document_tool called with empty content"

        stripped = False
        if fmt == "markdown":
            content, stripped = self._strip_unbacked_citation(content)
            content = content.rstrip() + "\n" + self._provenance_block() + "\n"

        safe = re.sub(r"[^\w\-. ]+", "", title or "document").strip() or "document"
        ext = {"markdown": "md", "html": "html", "csv": "csv"}[fmt]
        filename = f"{safe[:60]}.{ext}"

        saved_to = ""
        try:
            root = Path(self._workspace) if self._workspace else Path(tempfile.gettempdir()) / "vasudha_docs"
            root.mkdir(parents=True, exist_ok=True)
            path = root / filename
            path.write_text(content, encoding="utf-8")
            saved_to = str(path)
        except OSError as exc:
            logger.warning("could not save document: %s", exc)

        self.last_document = {"title": title or "Document", "format": fmt,
                              "content": content, "filename": filename,
                              "path": saved_to,
                              "sourced": bool(self._fetches),
                              "searches": list(dict.fromkeys(self._searches)),
                              "fetches": list(dict.fromkeys(self._fetches))}

        lines = content.count("\n") + 1
        note = (" A citation you wrote was removed because no page was actually read this turn."
                if stripped else "")
        return (f"[document created] '{title}' ({fmt}, {len(content)} chars, {lines} lines)"
                + (f", saved to {saved_to}" if saved_to else "")
                + ". It is now shown in the preview canvas." + note)

    def _run_tool(self, call: ToolCall) -> str:
        fn = self.tools.get(call.name)
        if fn is None:
            return f"[error] no such tool: {call.name}"
        try:
            return fn(**call.args)
        except TypeError as exc:
            return f"[error] bad arguments for {call.name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - the model can recover from this
            logger.exception("tool %s failed", call.name)
            return f"[error] {call.name} raised: {exc}"

    # -- turn --------------------------------------------------------------

    def reset(self) -> None:
        self.history = []

    # -- context budget ----------------------------------------------------

    def _budget(self, options: dict) -> Optional[int]:
        """How many tokens the transcript may occupy, or None if unknowable.

        The backend's own limit wins when it has one: the built-in engine is
        built with a fixed n_ctx and exceeding it is a hard failure, whereas
        num_ctx in options is only a request. These disagreed before — the
        setting said 16384 while the engine had been built at 8192 — and the
        symptom was silent left-truncation rather than an error.
        """
        limit = getattr(self.backend, "context_limit", None) or options.get("num_ctx")
        if not limit:
            return None
        # Leave room for the reply itself, or the budget is met exactly at the
        # moment there is nowhere to put the answer.
        return max(int(limit) - int(options.get("num_predict", 1024)) - 256, 1024)

    def _measure(self, convo: list[dict]) -> int:
        return sum(self.backend.count_tokens(str(m.get("content") or "")) + 8
                   for m in convo)

    def _compact(self, convo: list[dict], budget: int) -> tuple[list[dict], bool]:
        """Shrink the transcript to fit, oldest evidence first.

        Tool results are collapsed before anything else and the user's own
        messages are never touched. That ordering is deliberate: a summarised
        page fetch still supports the answer, whereas dropping the question
        changes what is being answered. The system prompt and the last two
        exchanges are always kept whole, because those are what the current
        turn is actually reasoning about.
        """
        if self._measure(convo) <= budget:
            return convo, False

        keep_tail = 4
        head, middle, tail = convo[:1], convo[1:-keep_tail], convo[-keep_tail:]
        compacted = False

        for message in middle:
            if message.get("role") != "tool":
                continue
            body = str(message.get("content") or "")
            if len(body) <= 400:
                continue
            message["content"] = (
                body[:200].rstrip()
                + f"\n[… {len(body) - 400} characters of earlier tool output "
                  "dropped to stay inside the context window …]\n"
                + body[-200:].lstrip())
            compacted = True
            if self._measure(head + middle + tail) <= budget:
                return head + middle + tail, True

        # Still over: drop whole exchanges from the front rather than letting
        # the engine truncate from the left, which would silently amputate the
        # system prompt and with it every rule in CORE_RULES.
        while middle and self._measure(head + middle + tail) > budget:
            middle.pop(0)
            compacted = True

        return head + middle + tail, compacted

    # -- streaming ---------------------------------------------------------

    def _stream_turn(self, convo: list[dict], options: dict) -> Iterator[dict]:
        """Yield token events for one model call; return its Completion.

        Used with `yield from`, so the completion falls out as the expression
        value and app/session.py stays one generator from backend to UI.
        """
        stream = self.backend.stream(convo, self.schemas, options)
        try:
            while True:
                channel, delta = next(stream)
                if not delta:
                    continue
                yield {"kind": "token" if channel == "text" else "thinking_token",
                       "text": delta}
        except StopIteration as stop:
            from app.backends import Completion
            return stop.value or Completion()

    def ask(self, text: str, options: Optional[dict] = None) -> Iterator[dict]:
        # num_predict is deliberately roomy. At 1024 the model reliably ran out
        # of budget partway through a <tool_call>, and ollama's tool parser
        # returns HTTP 500 on a truncated block rather than a partial reply.
        options = options or dict(self.options)

        # Provenance is per-turn: what was read answering the last question
        # says nothing about this one.
        self._turn_no += 1
        self._searches, self._fetches = [], []
        self._unsourced_line = ""
        self.last_document = None

        self.history.append({"role": "user", "content": text})
        # The memory book is appended at ask() time rather than baked into
        # system_prompt, so a lesson written during this turn is visible on the
        # next one without rebuilding the session.
        system = self.system_prompt
        if self.memory is not None:
            system += self.memory.as_prompt_section()
        system += self._attachment_section()
        # Retrieved against the question, and excluding this turn,
        # which has produced nothing yet.
        system += self.tool_memory.recall_block(
            text, exclude_turns={self._turn_no})
        convo = [{"role": "system", "content": system}] + list(self.history)

        turn_started = time.time()
        turn_tools: list[dict] = []
        out_tokens = 0
        final_answer = ""

        budget = self._budget(options)
        last_tool_result = ""
        nudged = False
        repaired = False        # one figure-grounding retry per turn
        used_tools = False

        for _ in range(self.max_iterations):
            if budget:
                convo, compacted = self._compact(convo, budget)
                if compacted:
                    yield {"kind": "status",
                           "text": "Trimmed older tool output to stay in context."}

            # Thinking is worth its cost when the model is deciding what to do,
            # and not when it is restating a number a tool already computed.
            # Measured on the shipped 4B: 10.05s with thinking vs 5.90s without,
            # for the same post-tool answer — 1.7x, on every iteration.
            step_options = dict(options)
            step_options["enable_thinking"] = not used_tools

            try:
                completion = yield from self._stream_turn(convo, step_options)
            except BackendError as exc:
                logger.warning("backend error: %s (%s)", exc, exc.detail)
                yield {"kind": "error", "text": str(exc)}
                return

            out_tokens += getattr(self.backend, "last_completion_tokens", 0)

            if completion.thinking:
                yield {"kind": "thinking", "text": completion.thinking}

            visible = (completion.text or "").strip()

            if not visible and not completion.tool_calls and last_tool_result and not nudged:
                nudged = True
                convo.append({"role": "user",
                              "content": ("State the final answer now, in one line, using the "
                                          "tool output above verbatim. Do not call another tool.")})
                continue

            if visible:
                yield {"kind": "text", "text": visible}

            if not completion.tool_calls:
                if not visible and last_tool_result:
                    final_answer = last_tool_result.strip()
                    yield {"kind": "text", "text": final_answer}
                    yield self._stats_event(out_tokens, turn_started)
                    self.history.append({"role": "assistant", "content": final_answer})
                    self._record_turn(text, final_answer, turn_tools, turn_started)
                else:
                    # Detection before acceptance. A figure the model asserted
                    # rather than computed gets one chance to be computed —
                    # naming it back is far more effective than a general
                    # instruction, because the model otherwise rewrites the
                    # prose around the same invented number.
                    ungrounded = self._ungrounded_figures(visible, turn_tools)
                    if ungrounded and not repaired:
                        repaired = True
                        yield {"kind": "status",
                               "text": "Checking figures against tool output…"}
                        convo.append({"role": "user",
                                      "content": self._repair_prompt(ungrounded)})
                        continue

                    final_answer = visible
                    yield self._stats_event(out_tokens, turn_started)
                    self._unsourced_line = self._unsourced_note(visible, turn_tools)
                    if self._unsourced_line:
                        # It survived a repair attempt, so it is not a slip.
                        # Loud rather than footnoted: a fabricated statistic
                        # that reads as checked is worse than no answer.
                        yield {"kind": "ungrounded", "figures": ungrounded}
                    self._record_turn(text, final_answer, turn_tools, turn_started)
                    self.history.append({"role": "assistant", "content": visible})
                    # The model was asked to use document_tool for deliverables
                    # and routinely does not; promote what it actually produced.
                    if self.last_document is None:
                        title = looks_like_document(visible)
                        if title:
                            body, _ = self._strip_unbacked_citation(visible)
                            body = body.rstrip() + "\n" + self._provenance_block() + "\n"
                            yield {"kind": "document", **self._document_payload(title, body)}
                return

            # The assistant message must carry the calls it made, not just its
            # prose. Without them the next prompt shows the model a tool result
            # with no record that it ever asked for one, and a model that is
            # told it received an answer to a question it cannot see asking is
            # being invited to hallucinate the question.
            assistant_message = {
                "role": "assistant",
                "content": completion.text or "",
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.name, "arguments": call.args}}
                    for call in completion.tool_calls
                ],
            }
            convo.append(assistant_message)
            self.history.append(assistant_message)
            used_tools = True

            for call in completion.tool_calls:
                yield {"kind": "tool_call", "name": call.name,
                       "args": call.args, "id": call.id}
                result = _clip_tool_result(self._run_tool(call))
                last_tool_result = result
                turn_tools.append({"name": call.name, "args": call.args,
                                   "result": result})
                self.tool_memory.add(self._turn_no, call.name,
                                     call.args, result)
                yield {"kind": "tool_result", "name": call.name,
                       "result": result, "id": call.id}

                # The canvas gets the document itself, not the confirmation
                # string the model sees — so a long report never has to travel
                # back through the model's context to reach the screen.
                if call.name == "document_tool" and self.last_document:
                    yield {"kind": "document", **self.last_document}

                # A chart is a result, not a side effect: show it as soon as
                # it exists rather than at the end of the turn.
                while self.new_images:
                    yield {"kind": "image", "path": self.new_images.pop(0)}

                # tool_call_id, and recorded in history rather than only in the
                # working transcript. Tool results used to be dropped at the end
                # of the turn, so a follow-up like "redo that with L=3" arrived
                # with no memory of what had been computed.
                tool_message = {"role": "tool", "tool_call_id": call.id,
                                "name": call.name, "content": result}
                convo.append(tool_message)
                self.history.append(tool_message)

        yield {"kind": "status", "text": "Stopped after too many tool calls."}


#: Signs that a numerical run produced garbage rather than an answer. All of
#: these appeared in a real session where a spring simulation diverged to 1e306
#: and the model went on to report a period from it.
_NUMERICAL_TROUBLE = (
    ("overflow", "a value exceeded the floating-point range"),
    ("invalid value", "an operation produced NaN"),
    ("divide by zero", "a division by zero occurred"),
    ("nan", "the output contains NaN"),
    ("inf", "the output contains infinity"),
)


def _flag_numerical_failure(result: str) -> str:
    """Make a diverged computation impossible to narrate past.

    The model is good at explaining that Euler integration can go unstable and
    poor at noticing that its own run just did. RuntimeWarnings arrive on
    stderr next to plausible-looking numbers, and the numbers get reported. The
    harness cannot tell whether the physics is right, but it can refuse to let
    `inf` and `nan` slide by unremarked — which is enough to stop the model
    building an answer on top of them.
    """
    haystack = result.lower()
    hits = [why for token, why in _NUMERICAL_TROUBLE if token in haystack]
    if not hits:
        return result
    # Deduplicated and capped: one clear sentence, not a wall matching every
    # token in a long traceback.
    unique = list(dict.fromkeys(hits))[:3]
    return (result.rstrip() + "\n\n[numerical failure detected: "
            + "; ".join(unique) + ". These results are not usable. Do NOT report "
            "a value derived from this run. Fix the computation — a smaller step "
            "size, a stable integrator, or a check on the maths — and run it "
            "again before drawing any conclusion.]")


def _clip_tool_result(result: str) -> str:
    """Bound one tool result before it enters the transcript.

    Clipped in the middle, not the tail: a traceback puts the exception on the
    last line and a fetched page puts the summary on the first, so keeping both
    ends preserves whichever one mattered.
    """
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    half = MAX_TOOL_RESULT_CHARS // 2
    dropped = len(result) - MAX_TOOL_RESULT_CHARS
    return (result[:half].rstrip()
            + f"\n\n[… {dropped} characters clipped …]\n\n"
            + result[-half:].lstrip())
