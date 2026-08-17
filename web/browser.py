"""A real browser the model can drive, for pages fetch_tool cannot read.

fetch_tool retrieves HTML over requests and strips tags. That works for a static
article and returns nothing useful for anything rendered client-side — which is
most documentation sites, dashboards, and every SPA. The README already admits
the consequence: "Research is weaker than calculation... Treat research output
as a draft." A large part of that weakness is that the page was never actually
read.

Vision is not the missing piece. A text model does not need a screenshot; it
needs the page's *structure*. What is returned here is the same thing a screen
reader consumes — headings, main text, and the interactive elements with stable
references — which a 4B handles perfectly well because it is just a list.

Optional by design
------------------
Playwright ships a browser binary of its own (~150 MB), which would contradict
the "no bundled browser" property the desktop app is built around. So this is an
optional capability: the app works without it, browse_tool is simply not offered
when it is absent, and a user who wants it runs

    pip install playwright && playwright install chromium

Budget
------
A real page's structure runs to tens of thousands of tokens and the context
window is 16k. Everything here is therefore pruned at the source rather than
truncated afterwards: keeping the first N characters of a page tends to keep the
cookie banner and drop the article.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

#: Total digest budget, deliberately under app.session.MAX_TOOL_RESULT_CHARS.
#: If a digest exceeds that cap the session clips it down the middle, which cuts
#: the structure in half and can leave the model reading a page whose outline and
#: control list are both truncated mid-line. Budgeting here instead means the
#: pruning is done by something that knows which parts matter.
MAX_DIGEST_CHARS = 5000
MAX_HEADINGS = 15
MAX_ELEMENTS = 30
DEFAULT_TIMEOUT_MS = 20_000


def playwright_available() -> bool:
    """Is the library importable? Says nothing about the browser binary, which
    is a separate download and reported separately when a launch fails."""
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return False


#: Collects the page's structure in one pass. Interactive elements are stashed
#: on `window` so a later click or type can resolve `ref3` back to the element
#: the model was actually looking at, rather than re-deriving it from a selector
#: that may have matched something else by then.
_DIGEST_JS = """
() => {
  const refs = [];
  window.__vasudha_refs = refs;

  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return false;
    const s = getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0';
  };

  const label = (el) => {
    const raw = el.getAttribute('aria-label') || el.innerText || el.value ||
                el.getAttribute('placeholder') || el.getAttribute('title') ||
                el.getAttribute('alt') || '';
    return String(raw).replace(/\\s+/g, ' ').trim().slice(0, 80);
  };

  const selector = 'a[href], button, input:not([type=hidden]), textarea, select,' +
                   '[role=button], [role=link], [role=textbox], [role=searchbox]';
  const items = [];
  for (const el of document.querySelectorAll(selector)) {
    if (!visible(el)) continue;
    const idx = refs.push(el) - 1;
    const tag = el.tagName.toLowerCase();
    let kind = tag;
    if (tag === 'a') kind = 'link';
    else if (tag === 'input') kind = (el.getAttribute('type') || 'text') + ' field';
    else if (tag === 'textarea') kind = 'text field';
    const href = (tag === 'a' && el.getAttribute('href')) ? ' -> ' + el.getAttribute('href') : '';
    items.push('[ref' + idx + '] ' + kind + ' "' + label(el) + '"' + href);
  }

  // innerText, not textContent: it respects layout, so it skips hidden nodes
  // and keeps the line breaks that make a page readable as prose. Preferring
  // <main>/<article> drops the nav and footer without a blocklist.
  const root = document.querySelector('main, article, [role=main]') || document.body;
  const text = String(root.innerText || '').replace(/\\n{3,}/g, '\\n\\n').trim();

  const headings = [];
  for (const h of document.querySelectorAll('h1, h2, h3')) {
    if (!visible(h)) continue;
    const t = String(h.innerText || '').replace(/\\s+/g, ' ').trim();
    if (t) headings.push(h.tagName.toLowerCase() + ': ' + t.slice(0, 90));
  }

  return { title: document.title, url: location.href, text, items, headings };
}
"""


def format_digest(data: dict, budget: int = MAX_DIGEST_CHARS,
                  max_headings: int = MAX_HEADINGS,
                  max_elements: int = MAX_ELEMENTS) -> str:
    """Render one page snapshot as the compact text the model reads.

    The outline and the control list are capped by count and the page text
    absorbs whatever budget is left, rather than each part having a fixed size.
    Prose is the compressible part — a heading list truncated mid-way is close
    to useless, and a ref list that stops early silently removes the model's
    ability to click something it can see mentioned in the text.

    Pure function of the scraped dict, so shaping and budgeting are testable
    without launching a browser.
    """
    title = (data.get("title") or "").strip() or "(untitled)"
    url = data.get("url") or ""
    text = (data.get("text") or "").strip()
    headings = (data.get("headings") or [])[:max_headings]
    items = data.get("items") or []
    shown = items[:max_elements]

    head = [f"# {title}", f"url: {url}", ""]
    if headings:
        head += ["## Outline", *(f"- {h}" for h in headings), ""]

    tail: list[str] = []
    if items:
        tail = ["", "## Things you can interact with", *shown]
        if len(items) > max_elements:
            tail.append(f"[… {len(items) - max_elements} more controls not listed …]")
        tail += ["", 'To act on one, call browse_tool with action="click" or "type" '
                     'and its ref, e.g. ref="ref3".']

    fixed = len("\n".join(head + ["## Page text", ""] + tail))
    room = max(budget - fixed, 400)     # never squeeze the text out entirely
    if len(text) > room:
        text = text[:room].rstrip() + f"\n[… {len(text) - room} more characters on this page. "
        text += 'Call browse_tool again with action="read" after clicking through '
        text += "to a more specific page if you need the rest …]"

    body = ["## Page text",
            text or "(no readable text — the page may still be rendering)"]
    return "\n".join(head + body + tail).strip()


class PageBrowser:
    """One headless browser, kept alive across calls within a chat.

    Kept alive because launching costs a second or two and a research task is
    several steps: navigate, read, click through, read again. Restarting per
    call would also lose the session cookies that make a multi-page flow work at
    all.
    """

    def __init__(self, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> None:
        self.timeout_ms = timeout_ms
        self._playwright = None
        self._browser = None
        self._page = None
        #: URLs actually loaded, for the provenance footer.
        self.visited: list[str] = []

    # -- lifecycle ---------------------------------------------------------

    def _ensure_page(self):
        if self._page is not None:
            return self._page

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=True)
        except Exception as exc:  # noqa: BLE001 - the message is the useful part
            self.close()
            raise BrowserUnavailable(
                "The browser binary is not installed. Run:  playwright install chromium"
            ) from exc

        context = self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36 VasudhaBot/1.0"),
        )
        self._page = context.new_page()
        self._page.set_default_timeout(self.timeout_ms)
        return self._page

    def close(self) -> None:
        for attr in ("_browser", "_playwright"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            try:
                obj.stop() if attr == "_playwright" else obj.close()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                logger.debug("closing %s failed", attr, exc_info=True)
            setattr(self, attr, None)
        self._page = None

    # -- actions -----------------------------------------------------------

    def _settle(self, page) -> None:
        """Give client-side rendering a chance to finish.

        networkidle rather than load: a React page fires load with an empty
        root and fills it in afterwards, which is exactly the case fetch_tool
        already handles badly. The wait is allowed to time out — a page that
        polls forever never goes idle, and a digest of what rendered so far
        beats an error.
        """
        try:
            page.wait_for_load_state("networkidle", timeout=min(self.timeout_ms, 8000))
        except Exception:  # noqa: BLE001
            logger.debug("networkidle not reached; digesting anyway", exc_info=True)

    def navigate(self, url: str) -> str:
        page = self._ensure_page()
        page.goto(url, wait_until="domcontentloaded")
        self._settle(page)
        self.visited.append(page.url)
        return self.digest()

    def digest(self) -> str:
        page = self._ensure_page()
        return format_digest(page.evaluate(_DIGEST_JS))

    def _resolve(self, page, ref: str):
        index = _ref_index(ref)
        if index is None:
            raise ValueError(f"{ref!r} is not a ref — use one from the page digest, like 'ref3'")
        handle = page.evaluate_handle(
            "(i) => (window.__vasudha_refs || [])[i] || null", index)
        element = handle.as_element()
        if element is None:
            raise ValueError(
                f"{ref} is no longer on the page. The page changed — read it again "
                "with action='read' and use a ref from the new digest.")
        return element

    def click(self, ref: str) -> str:
        page = self._ensure_page()
        self._resolve(page, ref).click()
        self._settle(page)
        if page.url not in self.visited:
            self.visited.append(page.url)
        return self.digest()

    def type_text(self, ref: str, text: str, submit: bool = False) -> str:
        page = self._ensure_page()
        element = self._resolve(page, ref)
        element.fill(text)
        if submit:
            element.press("Enter")
            self._settle(page)
            if page.url not in self.visited:
                self.visited.append(page.url)
        return self.digest()

    def back(self) -> str:
        page = self._ensure_page()
        page.go_back()
        self._settle(page)
        return self.digest()

    @property
    def current_url(self) -> str:
        return self._page.url if self._page else ""


class BrowserUnavailable(RuntimeError):
    """Playwright is importable but cannot actually drive a browser."""


_REF_RE = re.compile(r"^\s*(?:ref)?[_\-\s]*(\d+)\s*$", re.I)


def _ref_index(ref: str) -> Optional[int]:
    """Parse 'ref3', 'ref_3', '3' — models are inconsistent about the prefix,
    and rejecting a valid intent over punctuation costs a whole turn."""
    match = _REF_RE.match(str(ref or ""))
    return int(match.group(1)) if match else None
