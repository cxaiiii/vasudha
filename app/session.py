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
from pathlib import Path
from typing import Iterator, Optional

from app import personas
from app.backends import Backend, BackendError, ToolCall

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 12

#: The rules a persona must never be able to soften. personas.build_system_prompt
#: appends this AFTER the voice guidance for exactly that reason.
CORE_RULES = """Answer briefly and directly. Do not restate the question or pad the reply.

NUMBERS — python_tool
For ANY question with a numeric answer, call python_tool and take the number from its real output. Never state a computed value you did not get from the tool. Have the code print the value already converted to the unit asked for.
Before using a standard formula, name the exact case it belongs to — the support condition, the boundary conditions, the assumptions — and check your formula is the one for THAT case, not a similar one. A coefficient recalled from a neighbouring case gives a confidently wrong answer.

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
            "not print(sigma). Never leave a unit conversion to be done afterwards.")


def default_schemas() -> list[dict]:
    return [
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
    ]


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
                 workspace: Optional[str] = None) -> None:
        from web.tools import PageFetcher, SandboxedCodeExecutor, WebSearcher

        self.backend = backend
        self.system_prompt = system_prompt
        self.history: list[dict] = []
        self._executor = SandboxedCodeExecutor(timeout=15)
        self._searcher = WebSearcher(max_results=5)
        self._fetcher = PageFetcher(timeout=10, max_chars=4000)
        self._workspace = workspace
        #: filled by document_tool so the UI can render the canvas
        self.last_document: Optional[dict] = None
        #: what this turn really consulted — the basis for provenance, since
        #: the model's own account of its sources cannot be trusted
        self._searches: list[str] = []
        self._fetches: list[str] = []
        self.tools = {
            "python_tool": self._python_tool,
            "search_tool": self._search_tool,
            "fetch_tool": self._fetch_tool,
            "document_tool": self._document_tool,
        }
        self.schemas = default_schemas()
        self.options: dict = {"temperature": 0.6, "top_p": 0.9,
                              "num_predict": 2048, "num_ctx": 16384}
        self.max_iterations = MAX_ITERATIONS

    def set_tool_timeout(self, seconds: int) -> None:
        """Rebuild the executor rather than mutating it: a run already in
        flight keeps the timeout it started with."""
        from web.tools import SandboxedCodeExecutor
        self._executor = SandboxedCodeExecutor(timeout=int(seconds))

    # -- tools -------------------------------------------------------------

    def _python_tool(self, code: str = "", **_: object) -> str:
        return self._executor.execute_python(code, cwd=self._workspace)

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

    # -- provenance --------------------------------------------------------

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

    def ask(self, text: str, options: Optional[dict] = None) -> Iterator[dict]:
        # num_predict is deliberately roomy. At 1024 the model reliably ran out
        # of budget partway through a <tool_call>, and ollama's tool parser
        # returns HTTP 500 on a truncated block rather than a partial reply.
        options = options or dict(self.options)

        # Provenance is per-turn: what was read answering the last question
        # says nothing about this one.
        self._searches, self._fetches = [], []
        self.last_document = None

        self.history.append({"role": "user", "content": text})
        convo = [{"role": "system", "content": self.system_prompt}] + list(self.history)

        last_tool_result = ""
        nudged = False

        for _ in range(self.max_iterations):
            try:
                completion = self.backend.generate(convo, self.schemas, options)
            except BackendError as exc:
                logger.warning("backend error: %s (%s)", exc, exc.detail)
                yield {"kind": "error", "text": str(exc)}
                return

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
                    yield {"kind": "text", "text": last_tool_result.strip()}
                    self.history.append({"role": "assistant", "content": last_tool_result.strip()})
                else:
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

            convo.append({"role": "assistant", "content": completion.text or ""})

            for call in completion.tool_calls:
                yield {"kind": "tool_call", "name": call.name, "args": call.args}
                result = self._run_tool(call)
                last_tool_result = result
                yield {"kind": "tool_result", "name": call.name, "result": result}

                # The canvas gets the document itself, not the confirmation
                # string the model sees — so a long report never has to travel
                # back through the model's context to reach the screen.
                if call.name == "document_tool" and self.last_document:
                    yield {"kind": "document", **self.last_document}

                convo.append({"role": "tool", "content": result})

        yield {"kind": "status", "text": "Stopped after too many tool calls."}
