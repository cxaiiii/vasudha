/* Vasudha desktop front end.
 *
 * Talks to the Python side through pywebview's js_api bridge (window.pywebview.api),
 * so there is no HTTP server, no port to collide with, and nothing listening on
 * the network — which is the point of a local-only product.
 *
 * The Python side pushes events in by calling window.vasudha.onEvent(...), so a
 * long turn renders tool calls as they happen instead of appearing all at once
 * at the end. On the CPU backend a turn can take a minute; showing the tool
 * card the moment it runs is the difference between "working" and "frozen".
 */

const $ = (sel) => document.querySelector(sel);

const thread = $('#thread');
const threadInner = $('#thread-inner');
const input = $('#input');
const sendBtn = $('#send');

/* Snapshotted before anything mutates the thread, so "New chat" can put the
   splash + suggestion chips back exactly as they shipped. */
const EMPTY_STATE_HTML = threadInner.innerHTML;

let busy = false;
let currentAssistant = null;   // the bubble being appended to this turn

/* ── helpers ─────────────────────────────────────────────────────────── */

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

/* Deliberately minimal: fenced code, inline code, bold. The model emits a lot
   of LaTeX and half-markdown; running a full parser over it produced more
   mangling than it fixed, so anything not in this list stays literal. */
function renderMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/```(\w+)?\n([\s\S]*?)```/g,
    (_, lang, code) => `<pre><code>${code.replace(/\n$/, '')}</code></pre>`);
  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  return html;
}

function atBottom() {
  return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 120;
}

function scrollDown(force) {
  if (force || atBottom()) thread.scrollTop = thread.scrollHeight;
}

function clearEmptyState() {
  const empty = $('#empty');
  if (empty) empty.remove();
}

/* ── message rendering ───────────────────────────────────────────────── */

function addUserMessage(text) {
  clearEmptyState();
  const row = document.createElement('div');
  row.className = 'msg user';
  row.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  threadInner.appendChild(row);
  scrollDown(true);
}

function beginAssistant() {
  const row = document.createElement('div');
  row.className = 'msg assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  row.appendChild(bubble);
  threadInner.appendChild(row);
  currentAssistant = bubble;
  scrollDown(true);
  return bubble;
}

/* Streaming.

   Tokens land in a plain-text element as they arrive, and the finished 'text'
   event replaces that element with properly rendered markdown. Rendering
   markdown on every delta instead would mean re-parsing a half-written code
   fence sixty times a second, and a fence that is not closed yet renders as
   garbage until the closing backticks show up. */
let streamEl = null;
let streamText = '';

function streamDelta(text) {
  if (!text) return;
  if (!currentAssistant) beginAssistant();
  if (!streamEl) {
    streamEl = document.createElement('div');
    streamEl.className = 'streaming';
    currentAssistant.appendChild(streamEl);
    streamText = '';
  }
  streamText += text;
  streamEl.textContent = streamText;
  scrollDown();
}

/* Hand the streamed run over to the rendered version. Returns true if it
   consumed the text, so the caller does not append a second copy. */
function finishStream(finalText) {
  if (!streamEl) return false;
  const el = streamEl;
  streamEl = null;
  const body = (finalText && finalText.trim()) ? finalText : streamText;
  streamText = '';
  if (!body.trim()) { el.remove(); return true; }
  el.className = '';
  el.innerHTML = renderMarkdown(body.trim());
  scrollDown();
  return true;
}

/* Reasoning arrives as deltas too, but goes into the collapsible card rather
   than the reply body. Built on first delta so a model that never emits a
   think block does not leave an empty card behind. */
let thinkStreamBody = null;

function thinkingDelta(text) {
  if (!text) return;
  if (!currentAssistant) beginAssistant();
  if (!thinkStreamBody) {
    addThinking('​');            // zero-width: builds the card structure
    const cards = currentAssistant.querySelectorAll('.think-card');
    const card = cards[cards.length - 1];
    thinkStreamBody = card.querySelector('.think-body');
    thinkStreamBody.textContent = '';
  }
  thinkStreamBody.textContent += text;
  scrollDown();
}

function addStatus(label) {
  if (!currentAssistant) beginAssistant();
  const line = document.createElement('div');
  line.className = 'status-line';
  line.innerHTML = `<span class="spinner"></span><span>${escapeHtml(label)}</span>` +
                   `<span class="tool-elapsed" data-elapsed></span>`;
  currentAssistant.appendChild(line);
  // A status line is removed rather than settled, so its timer is cleared by
  // removeStatus — a detached node with a live interval keeps ticking for the
  // life of the page.
  startElapsed(line.querySelector('[data-elapsed]'));
  scrollDown();
  return line;
}

/* Every path that drops a status line goes through here, so no timer is
   orphaned on a node that has left the document. */
function removeStatus() {
  if (!pendingStatus) return;
  stopElapsed(pendingStatus.querySelector('[data-elapsed]'));
  pendingStatus.remove();
  pendingStatus = null;
}

function addThinking(text) {
  if (!text || !text.trim()) return;
  if (!currentAssistant) beginAssistant();
  const card = document.createElement('div');
  const openByDefault = !!settings.show_reasoning;
  card.className = 'think-card' + (openByDefault ? ' open' : '');
  card.innerHTML =
    `<button class="think-toggle">${openByDefault ? 'Hide' : 'Show'} reasoning</button>
     <div class="think-body">${escapeHtml(text.trim())}</div>`;
  card.querySelector('.think-toggle').addEventListener('click', () => {
    card.classList.toggle('open');
    card.querySelector('.think-toggle').textContent =
      card.classList.contains('open') ? 'Hide reasoning' : 'Show reasoning';
  });
  currentAssistant.appendChild(card);
  scrollDown();
}

/* A tool call is evidence, so it gets a real card: the exact code that ran and
   the exact stdout it produced, both inspectable.

   Which tools are evidence rather than plumbing.

   python_tool stays expanded because the computation IS the justification for
   the number — that is the whole argument of this project. The others became
   noise once there were nine tools: a file listing does not need to be open by
   default the way a calculation does. */
const EVIDENCE_TOOLS = ['python_tool'];

const TOOL_LABELS = {
  python_tool: 'ran Python',
  // Named explicitly: these are what send something off the machine, so the
  // user should see exactly what left it.
  search_tool: 'searched the web (query sent to DuckDuckGo)',
  fetch_tool: 'downloaded and read a page',
  browse_tool: 'opened a page in a browser',
  document_tool: 'built a document',
  render_tool: 'rendered a preview',
  write_file: 'wrote a file',
  read_file: 'read a file',
  edit_file: 'edited a file',
  list_files: 'listed the workspace',
};

/* What the model asked for, in one line. browse_tool is the awkward one: its
   meaningful argument depends on the action, and showing "open" alone tells
   nobody anything. */
function toolPayload(name, args) {
  if (name === 'browse_tool') {
    const what = args.url || args.ref || '';
    return [args.action || 'open', what, args.text ? `"${args.text}"` : '']
      .filter(Boolean).join('  ');
  }
  return args.code || args.query || args.url || args.title || args.path || '';
}

function addToolCard(name, args) {
  if (!currentAssistant) beginAssistant();
  const card = document.createElement('div');
  const expand = settings.open_tool_cards !== false && EVIDENCE_TOOLS.includes(name);
  card.className = 'tool-card running' + (expand ? ' open' : '');

  const payload = toolPayload(name, args);
  const label = TOOL_LABELS[name] || name;

  card.innerHTML =
    `<div class="tool-head">
       <span class="tool-dot" aria-hidden="true"></span>
       <span class="tag">${escapeHtml(name)}</span>
       <span class="tool-label">${escapeHtml(label)}</span>
       <span class="tool-elapsed" data-elapsed></span>
       <span class="chev">&#9656;</span>
     </div>
     <div class="tool-body">
       ${payload ? `<pre><code>${escapeHtml(payload)}</code></pre>` : ''}
       <div class="tool-out" data-out>running…</div>
     </div>`;

  card.querySelector('.tool-head').addEventListener('click',
    () => card.classList.toggle('open'));
  currentAssistant.appendChild(card);
  startElapsed(card.querySelector('[data-elapsed]'));
  scrollDown();
  return card;
}

/* A counting seconds display, on anything that can run long.

   Browsing waits on a real page load and a research turn can run a minute. A
   spinner that never changes is indistinguishable from a hang, and the honest
   fix is to show that something is still happening rather than to guess at a
   percentage nobody can compute. */
function startElapsed(el) {
  if (!el) return;
  const started = Date.now();
  const tick = () => {
    const s = (Date.now() - started) / 1000;
    el.textContent = s < 1 ? '' : (s < 60 ? `${s.toFixed(0)}s`
                                          : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);
  };
  tick();
  el._timer = setInterval(tick, 500);
}

function stopElapsed(el) {
  if (el && el._timer) { clearInterval(el._timer); el._timer = null; }
}

/* Settle a tool card. A result beginning with [error] is a failure the model
   will now try to recover from, and showing it as a success would misrepresent
   what the transcript actually says happened. */
function finishToolCard(card, result) {
  if (!card) return;
  const text = (result || '').trim();
  const failed = text.startsWith('[error]') || text.startsWith('[Error');
  card.classList.remove('running');
  card.classList.add(failed ? 'failed' : 'done');
  stopElapsed(card.querySelector('[data-elapsed]'));
  const out = card.querySelector('[data-out]');
  if (out) out.textContent = text || '(no output)';
  if (failed) card.classList.add('open');   // a failure is worth seeing unasked
}

/* ── event bus from Python ───────────────────────────────────────────── */

let pendingTool = null;
let pendingStatus = null;

window.vasudha = {
  onEvent(evt) {
    switch (evt.kind) {
      case 'token':
        removeStatus();
        streamDelta(evt.text);
        break;

      case 'thinking_token':
        thinkingDelta(evt.text);
        break;

      case 'thinking':
        // Only build a card here if nothing was streamed into one already —
        // backends that report reasoning as a separate field (ollama) send
        // deltas, and this final event would otherwise duplicate the block.
        if (thinkStreamBody) { thinkStreamBody = null; break; }
        addThinking(evt.text);
        break;

      case 'text':
        // finishStream consumes the streamed run when there is one; only a
        // non-streaming backend reaches the append path below.
        if (finishStream(evt.text)) break;
        if (evt.text && evt.text.trim()) {
          if (!currentAssistant) beginAssistant();
          const p = document.createElement('div');
          p.innerHTML = renderMarkdown(evt.text.trim());
          currentAssistant.appendChild(p);
          scrollDown();
        }
        break;

      case 'tool_call':
        // The prose before a tool call is finished text, not an abandoned
        // stream: close it out before the card goes in, or the two interleave.
        finishStream(null);
        thinkStreamBody = null;
        removeStatus();
        pendingTool = addToolCard(evt.name, evt.args || {});
        break;

      case 'tool_result': {
        finishToolCard(pendingTool, evt.result);
        pendingTool = null;
        pendingStatus = addStatus('Reading the result…');
        break;
      }

      case 'render': {
        if (pendingTool) {
          const out = pendingTool.querySelector('[data-out]');
          if (out) out.textContent = 'Preview opened in a new window.';
          pendingTool = null;
        }
        break;
      }

      case 'document':
        recordArtifact(evt);
        showDocument(evt);
        break;

      case 'status':
        removeStatus();
        pendingStatus = addStatus(evt.text);
        break;

      case 'error':
        finishStream(null);
        thinkStreamBody = null;
        removeStatus();
        if (!currentAssistant) beginAssistant();
        const err = document.createElement('div');
        err.className = 'tool-out';
        err.style.borderLeftColor = 'var(--saffron)';
        err.textContent = evt.text;
        currentAssistant.appendChild(err);
        break;

      case 'done':
        finishStream(null);       // a turn cut short still shows what arrived
        thinkStreamBody = null;
        removeStatus();
        finishTurn();
        break;
    }
  },

  onBackend(info) {
    const badge = $('#backend-badge');
    const text = $('#backend-text');
    text.textContent = info.label;
    badge.classList.toggle('fast', !!info.fast);
    badge.classList.toggle('slow', !info.fast);
    badge.title = info.detail || '';
  },

  onDownloadProgress(p) {
    const bar = $('#dl-bar');
    bar.style.width = `${p.percent.toFixed(1)}%`;
    bar.classList.toggle('verifying', p.stage === 'verifying');

    if (p.stage === 'done') {
      $('#fr-title').textContent = 'Ready';
      $('#dl-detail').textContent = `${p.total_mb.toFixed(0)} MB installed and verified.`;
      $('#dl-start').disabled = false;
      $('#dl-start').textContent = 'Download model';
      return;
    }

    if (p.stage === 'verifying') {
      // Hashing 2.2 GB takes a few seconds. A bar parked at 100% with no
      // explanation looks like a hang, so say what is happening.
      $('#fr-title').textContent = 'Checking the download';
      $('#dl-detail').textContent =
        `Verifying ${p.percent.toFixed(0)}% — making sure the file arrived intact.`;
      return;
    }

    const parts = [`${(p.done_mb / 1000).toFixed(2)} / ${(p.total_mb / 1000).toFixed(2)} GB`];
    if (p.mbps) parts.push(`${p.mbps.toFixed(1)} MB/s`);
    if (p.eta_seconds > 1) {
      const m = Math.floor(p.eta_seconds / 60);
      const s = Math.round(p.eta_seconds % 60);
      parts.push(m ? `about ${m} min ${s}s left` : `about ${s}s left`);
    }
    $('#fr-title').textContent = 'Downloading Vasudha';
    $('#dl-detail').textContent = parts.join('   ·   ');
  },

  showFirstRun(show) {
    $('#firstrun').classList.toggle('show', !!show);
  },

  onDownloadError(payload) {
    const box = $('#dl-error');
    box.innerHTML = `<b>Download failed.</b>\n${escapeHtml(payload.message || 'Unknown error.')}`;
    box.classList.add('show');
    $('#dl-bar').style.width = '0%';
    $('#dl-detail').textContent = '';
    // Re-enable so the user can retry — a partial file resumes rather than
    // restarting, so retrying is genuinely cheap.
    const btn = $('#dl-start');
    btn.disabled = false;
    btn.textContent = 'Try again';
  },

  setChats(items) {
    const list = $('#chat-list');
    list.innerHTML = '';
    items.forEach((c) => {
      const el = document.createElement('div');
      el.className = 'chat-item' + (c.active ? ' active' : '');
      el.textContent = c.title || 'Untitled';
      el.title = c.title || 'Untitled';
      el.addEventListener('click', () => window.pywebview.api.open_chat(c.id));
      list.appendChild(el);
    });
  },

  /* Replay a stored conversation. The saved events include tool calls and
     their real stdout, so reopening a chat restores the evidence trail rather
     than flattening it to prose — the whole point is that the working is
     inspectable later, not just live. */
  loadChat(payload) {
    threadInner.innerHTML = '';
    currentAssistant = null;
    pendingTool = null;
    pendingStatus = null;

    const events = payload.events || [];
    if (!events.length) {
      threadInner.innerHTML = EMPTY_STATE_HTML;
      finishTurn();
      return;
    }

    events.forEach((evt) => {
      if (evt.kind === 'user') {
        currentAssistant = null;
        addUserMessage(evt.text);
      } else if (evt.kind === 'tool_result') {
        if (pendingTool) {
          const out = pendingTool.querySelector('[data-out]');
          if (out) out.textContent = (evt.result || '').trim() || '(no output)';
          pendingTool = null;
        }
      } else if (evt.kind !== 'done' && evt.kind !== 'status') {
        window.vasudha.onEvent(evt);
      }
    });

    removeStatus();
    finishTurn();
    scrollDown(true);
  },
};

/* ── sending ─────────────────────────────────────────────────────────── */

function finishTurn() {
  busy = false;
  sendBtn.disabled = false;
  currentAssistant = null;
  pendingTool = null;
  input.focus();
}

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;

  busy = true;
  sendBtn.disabled = true;
  input.value = '';
  autoGrow();

  addUserMessage(text);
  beginAssistant();
  pendingStatus = addStatus('Thinking…');

  try {
    await window.pywebview.api.ask(text);
  } catch (e) {
    window.vasudha.onEvent({ kind: 'error', text: String(e) });
    finishTurn();
  }
}

function autoGrow() {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 190)}px`;
}

/* ── wiring ──────────────────────────────────────────────────────────── */

input.addEventListener('input', autoGrow);
input.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  const enterSends = settings.send_on_enter !== false;
  if (enterSends ? !e.shiftKey : (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    send();
  }
});
sendBtn.addEventListener('click', send);

document.addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (chip) { input.value = chip.dataset.q; autoGrow(); send(); }
});

$('#new-chat').addEventListener('click', () => window.pywebview.api.new_chat());
$('#btn-min').addEventListener('click', () => window.pywebview.api.minimise());
$('#btn-close').addEventListener('click', () => window.pywebview.api.close());
$('#dl-start').addEventListener('click', (e) => {
  // Disabled immediately: a second click would start a competing writer on the
  // same .part file.
  e.target.disabled = true;
  e.target.textContent = 'Downloading…';
  $('#dl-detail').textContent = 'Connecting…';
  $('#dl-error').classList.remove('show');
  window.pywebview.api.start_download();
});
$('#dl-locate').addEventListener('click', () => window.pywebview.api.locate_model());

/* ── Preview canvas ──────────────────────────────────────────────────
 *
 * Documents are rendered here rather than dumped into the chat: a report is a
 * thing you keep, not a message you scroll past. The renderers below are
 * intentionally small — the goal is a faithful preview, not a full editor.
 */

let currentDoc = null;

/* CSV -> table. Handles quoted fields containing commas and escaped quotes,
   because real spreadsheet data has both and a naive split(',') mangles it. */
function parseCsv(text) {
  const rows = [];
  let row = [], field = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
    else if (c !== '\r') field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows.filter(r => r.some(cell => cell.trim() !== ''));
}

const isNumeric = (s) => s !== '' && !isNaN(Number(String(s).replace(/[, ]/g, '')));

function csvToTable(text) {
  const rows = parseCsv(text);
  if (!rows.length) return '<p>Empty spreadsheet.</p>';
  const [head, ...body] = rows;
  const th = head.map(h => `<th>${escapeHtml(h)}</th>`).join('');
  const trs = body.map(r =>
    `<tr>${r.map(c => `<td class="${isNumeric(c) ? 'num' : ''}">${escapeHtml(c)}</td>`).join('')}</tr>`
  ).join('');
  return `<table><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`;
}

/* Small markdown subset: headings, lists, tables, code, emphasis, links.
   Deliberately not a full parser — a partial render that is predictable beats
   a complete one that mangles the model's half-markdown. */
function markdownToHtml(md) {
  const lines = escapeHtml(md).split('\n');
  const out = [];
  let inList = null, inCode = false, tableBuf = [];

  const flushTable = () => {
    if (!tableBuf.length) return;
    const cells = (line) => line.trim().replace(/^\||\|$/g, '').split('|').map(s => s.trim());
    const head = cells(tableBuf[0]);
    const body = tableBuf.slice(tableBuf.length > 1 && /^[\s|:-]+$/.test(tableBuf[1]) ? 2 : 1);
    out.push(`<table><thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>`
      + body.map(r => `<tr>${cells(r).map(c =>
          `<td class="${isNumeric(c) ? 'num' : ''}">${c}</td>`).join('')}</tr>`).join('')
      + '</tbody></table>');
    tableBuf = [];
  };
  const closeList = () => { if (inList) { out.push(`</${inList}>`); inList = null; } };

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');

    if (/^```/.test(line)) {
      flushTable(); closeList();
      out.push(inCode ? '</code></pre>' : '<pre><code>');
      inCode = !inCode;
      continue;
    }
    if (inCode) { out.push(line); continue; }

    if (/^\s*\|.*\|\s*$/.test(line)) { closeList(); tableBuf.push(line); continue; }
    flushTable();

    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { closeList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }

    if (/^\s*>\s?/.test(line)) { closeList(); out.push(`<blockquote>${inline(line.replace(/^\s*>\s?/, ''))}</blockquote>`); continue; }
    if (/^\s*([-*+])\s+/.test(line)) {
      if (inList !== 'ul') { closeList(); out.push('<ul>'); inList = 'ul'; }
      out.push(`<li>${inline(line.replace(/^\s*([-*+])\s+/, ''))}</li>`); continue;
    }
    if (/^\s*\d+[.)]\s+/.test(line)) {
      if (inList !== 'ol') { closeList(); out.push('<ol>'); inList = 'ol'; }
      out.push(`<li>${inline(line.replace(/^\s*\d+[.)]\s+/, ''))}</li>`); continue;
    }
    if (/^\s*([-*_]\s*){3,}$/.test(line)) { closeList(); out.push('<hr>'); continue; }

    if (!line.trim()) { closeList(); continue; }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  flushTable(); closeList();
  if (inCode) out.push('</code></pre>');
  return out.join('\n');

  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }
}

/* Every document produced this session, newest first. Kept in memory and
   mirrored into the sidebar so a report from three questions ago is one click
   away instead of lost up the scrollback. */
const artifacts = [];

function recordArtifact(doc) {
  artifacts.unshift({ ...doc, at: Date.now() });
  renderArtifacts();
}

function renderArtifacts() {
  const list = $('#artifact-list');
  if (!artifacts.length) {
    list.innerHTML = '<div class="rail-empty">Documents and spreadsheets Vasudha '
      + 'builds will collect here.</div>';
    return;
  }
  list.innerHTML = '';
  artifacts.forEach((a, i) => {
    const el = document.createElement('div');
    el.className = 'artifact-item';
    const kind = { markdown: 'DOC', csv: 'SHEET', html: 'PAGE' }[a.format] || 'DOC';
    const mins = Math.round((Date.now() - a.at) / 60000);
    el.innerHTML =
      `<span class="kind${a.sourced ? '' : ' unsourced'}">${kind}</span>
       <span class="meta">
         <span class="name">${escapeHtml(a.title || 'Document')}</span>
         <span class="when">${mins < 1 ? 'just now' : mins + ' min ago'}` +
      `${a.sourced ? '' : ' · unverified'}</span>
       </span>`;
    el.title = a.sourced ? 'Sources were read for this' : 'No source page was opened';
    el.addEventListener('click', () => showDocument(artifacts[i]));
    list.appendChild(el);
  });
}

document.querySelectorAll('.rail-tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.rail-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const showChats = tab.dataset.rail === 'chats';
    $('#chat-list').hidden = !showChats;
    $('#artifact-list').hidden = showChats;
    if (!showChats) renderArtifacts();
  });
});

/* Model-authored HTML, made safe to inline.
 *
 * This used to render into <iframe sandbox="" srcdoc="...">. That isolates the
 * markup properly, but a fully-restricted sandbox gives the frame an opaque
 * origin and WebView2 often declines to paint srcdoc under it — the canvas
 * opened showing nothing, which reads as "the document tool did not fire".
 *
 * A report is headings, lists and tables; scripts have no legitimate role in
 * one. Stripping everything executable removes the attack surface entirely and
 * renders reliably, which is the better trade for a document preview.
 */
const _BLOCKED_TAGS = ['script', 'style', 'iframe', 'object', 'embed', 'link',
                       'meta', 'base', 'form', 'input', 'button'];

function sanitizeHtml(html) {
  const doc = new DOMParser().parseFromString(String(html), 'text/html');

  doc.body.querySelectorAll(_BLOCKED_TAGS.join(',')).forEach(el => el.remove());

  doc.body.querySelectorAll('*').forEach((el) => {
    [...el.attributes].forEach((attr) => {
      const name = attr.name.toLowerCase();
      const value = String(attr.value || '');
      // Event handlers, and any URL that could execute.
      if (name.startsWith('on')
          || (['href', 'src', 'xlink:href', 'action', 'formaction'].includes(name)
              && /^\s*(javascript|data|vbscript):/i.test(value))) {
        el.removeAttribute(attr.name);
      }
    });
    if (el.tagName === 'A') {
      el.setAttribute('target', '_blank');
      el.setAttribute('rel', 'noopener noreferrer');
    }
  });

  return doc.body.innerHTML;
}

function showDocument(doc) {
  currentDoc = doc;
  const shell = $('#shell');
  const canvas = $('#canvas');
  const render = $('#canvas-render');

  // The banner is driven by what the harness recorded, not by anything the
  // model said about its sources.
  $('#canvas-warning').classList.toggle('show', doc.sourced === false);

  /* The address strip shows where this actually came from. For a document the
     model wrote that is the file it was saved to; a bare title would look like
     a URL slot with nothing in it. */
  const kindEl = $('#canvas-kind');
  kindEl.textContent = { markdown: 'DOC', csv: 'SHEET', html: 'PAGE' }[doc.format] || 'DOC';
  kindEl.classList.toggle('web', doc.format === 'web');
  $('#canvas-title').textContent = doc.path || doc.filename || doc.title || 'Document';
  $('#canvas-title').title = doc.path || doc.title || '';
  $('#canvas-source').querySelector('code').textContent = doc.content || '';

  if (doc.format === 'csv') {
    render.innerHTML = csvToTable(doc.content || '');
  } else if (doc.format === 'html') {
    render.innerHTML = sanitizeHtml(doc.content || '');
  } else {
    render.innerHTML = markdownToHtml(doc.content || '');
  }

  $('#canvas-body').classList.remove('showing-source');
  $('#canvas-toggle').textContent = 'Source';
  canvas.setAttribute('aria-hidden', 'false');
  shell.classList.add('with-canvas');
}

function closeCanvas() {
  $('#shell').classList.remove('with-canvas');
  $('#canvas').setAttribute('aria-hidden', 'true');
}

$('#canvas-close').addEventListener('click', closeCanvas);
$('#canvas-toggle').addEventListener('click', () => {
  const body = $('#canvas-body');
  body.classList.toggle('showing-source');
  $('#canvas-toggle').textContent = body.classList.contains('showing-source') ? 'Preview' : 'Source';
});
$('#canvas-save').addEventListener('click', () => {
  if (currentDoc) window.pywebview.api.save_document(currentDoc);
});

/* ── Window resizing ─────────────────────────────────────────────────
 *
 * A frameless window has no OS resize border, so the eight grips in the DOM
 * drive it manually. Screen coordinates are used rather than client ones:
 * while dragging the west or north edge the window itself is moving, so
 * client coords shift under the cursor and the box jitters.
 *
 * Bounds are pushed at most once per frame — every set_bounds is a round trip
 * through the Python bridge into a WinForms call, and firing one per mousemove
 * makes the drag feel like it is fighting back.
 */

const MIN_W = 900;
const MIN_H = 600;

let rzState = null;
let rzQueued = false;

function beginResize(e, edge) {
  e.preventDefault();
  window.pywebview.api.get_bounds().then((b) => {
    rzState = {
      edge,
      startX: e.screenX,
      startY: e.screenY,
      x: b.x, y: b.y, w: b.width, h: b.height,
    };
    document.body.classList.add('resizing');
  });
}

function applyResize(e) {
  if (!rzState) return;
  const dx = e.screenX - rzState.startX;
  const dy = e.screenY - rzState.startY;
  const { edge } = rzState;

  let { x, y, w, h } = rzState;

  if (edge.includes('e')) w = rzState.w + dx;
  if (edge.includes('s')) h = rzState.h + dy;
  if (edge.includes('w')) { w = rzState.w - dx; x = rzState.x + dx; }
  if (edge.includes('n')) { h = rzState.h - dy; y = rzState.y + dy; }

  // Clamp before moving, so dragging a left/top edge past the minimum pins the
  // window instead of sliding it while the size refuses to shrink further.
  if (w < MIN_W) { if (edge.includes('w')) x = rzState.x + (rzState.w - MIN_W); w = MIN_W; }
  if (h < MIN_H) { if (edge.includes('n')) y = rzState.y + (rzState.h - MIN_H); h = MIN_H; }

  rzState.pending = { x: Math.round(x), y: Math.round(y),
                      width: Math.round(w), height: Math.round(h) };

  if (!rzQueued) {
    rzQueued = true;
    requestAnimationFrame(() => {
      rzQueued = false;
      if (rzState && rzState.pending) {
        const p = rzState.pending;
        window.pywebview.api.set_bounds(p.x, p.y, p.width, p.height);
      }
    });
  }
}

function endResize() {
  if (!rzState) return;
  rzState = null;
  document.body.classList.remove('resizing');
}

document.querySelectorAll('.rz').forEach((grip) => {
  grip.addEventListener('mousedown', (e) => beginResize(e, grip.dataset.edge));
});
window.addEventListener('mousemove', applyResize);
window.addEventListener('mouseup', endResize);

/* Double-clicking the title strip toggles maximise, as a normal window does. */
$('#titlebar').addEventListener('dblclick', (e) => {
  if (e.target.closest('button')) return;
  window.pywebview.api.toggle_maximise();
});

/* ── Settings ────────────────────────────────────────────────────────── */

const SETTING_FIELDS = {
  persona: '#s-persona',
  backend: '#s-backend',
  model_path: '#s-model-path',
  ollama_url: '#s-ollama-url',
  temperature: '#s-temperature',
  top_p: '#s-top-p',
  num_predict: '#s-num-predict',
  num_ctx: '#s-num-ctx',
  tool_timeout: '#s-tool-timeout',
  max_iterations: '#s-max-iterations',
  show_reasoning: '#s-show-reasoning',
  open_tool_cards: '#s-open-tool-cards',
  send_on_enter: '#s-send-on-enter',
  reduce_motion: '#s-reduce-motion',
};

let settings = {};

function openSettings() {
  $('#settings').classList.add('show');
  $('#scrim').classList.add('show');
  $('#settings').setAttribute('aria-hidden', 'false');
}

function closeSettings() {
  $('#settings').classList.remove('show');
  $('#scrim').classList.remove('show');
  $('#settings').setAttribute('aria-hidden', 'true');
}

async function fillPersonaSelect() {
  const sel = $('#s-persona');
  if (sel.options.length) return;          // populated once
  const list = await window.pywebview.api.list_personas();
  list.forEach((p) => {
    const opt = document.createElement('option');
    opt.value = p.key;
    opt.textContent = `${p.glyph}  ${p.name}`;
    opt.title = p.blurb;
    sel.appendChild(opt);
  });
}

function paintSettings(values) {
  settings = values;
  fillPersonaSelect().then(() => { $('#s-persona').value = values.persona || 'engineer'; });
  Object.entries(SETTING_FIELDS).forEach(([key, sel]) => {
    const el = $(sel);
    if (!el) return;
    if (el.type === 'checkbox') el.checked = !!values[key];
    else el.value = values[key];
  });
  $('#s-temperature-val').textContent = Number(values.temperature).toFixed(2);
  $('#s-top-p-val').textContent = Number(values.top_p).toFixed(2);
  document.body.classList.toggle('reduce-motion', !!values.reduce_motion);
}

function collectSettings() {
  const out = { ...settings };
  Object.entries(SETTING_FIELDS).forEach(([key, sel]) => {
    const el = $(sel);
    if (!el) return;
    if (el.type === 'checkbox') out[key] = el.checked;
    else if (el.type === 'number' || el.type === 'range') out[key] = Number(el.value);
    else out[key] = el.value;
  });
  return out;
}

$('#settings-btn').addEventListener('click', openSettings);
$('#settings-close').addEventListener('click', closeSettings);
$('#scrim').addEventListener('click', closeSettings);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && $('#settings').classList.contains('show')) closeSettings();
});

$('#s-temperature').addEventListener('input', (e) => {
  $('#s-temperature-val').textContent = Number(e.target.value).toFixed(2);
});
$('#s-top-p').addEventListener('input', (e) => {
  $('#s-top-p-val').textContent = Number(e.target.value).toFixed(2);
});
$('#s-reduce-motion').addEventListener('change', (e) => {
  document.body.classList.toggle('reduce-motion', e.target.checked);
});

$('#s-save').addEventListener('click', async () => {
  const values = await window.pywebview.api.save_settings(collectSettings());
  paintSettings(values);
  const status = $('#settings-status');
  status.classList.add('show');
  setTimeout(() => status.classList.remove('show'), 1600);
});

$('#s-reset').addEventListener('click', async () => {
  paintSettings(await window.pywebview.api.reset_settings());
});

$('#s-browse').addEventListener('click', async () => {
  const path = await window.pywebview.api.locate_model();
  if (path) $('#s-model-path').value = path;
});

$('#s-open-folder').addEventListener('click', () => window.pywebview.api.open_data_folder());
$('#s-open-personas').addEventListener('click', () => window.pywebview.api.open_personas_folder());

$('#s-clear-history').addEventListener('click', async () => {
  if (!confirm('Delete every saved chat on this machine? This cannot be undone.')) return;
  await window.pywebview.api.clear_history();
});

window.vasudha.onSettings = paintSettings;
window.vasudha.setDataPath = (p) => { $('#s-data-path').textContent = p; };

/* ── First-run persona picker ────────────────────────────────────────── */

let chosenPersona = null;

window.vasudha.showOnboarding = (payload) => {
  const grid = $('#persona-grid');
  grid.innerHTML = '';
  (payload.personas || []).forEach((p) => {
    const card = document.createElement('button');
    card.className = 'persona-card';
    card.type = 'button';
    card.innerHTML =
      `<span class="tick">&#10003;</span>
       <span class="glyph">${p.glyph}</span>
       <span class="pname">${escapeHtml(p.name)}</span>
       <span class="ptag">${escapeHtml(p.tagline)}</span>
       <span class="pblurb">${escapeHtml(p.blurb)}</span>`;
    card.addEventListener('click', () => {
      grid.querySelectorAll('.persona-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      chosenPersona = p.key;
      $('#onboard-go').disabled = false;
      $('#onboard-hint').textContent = `${p.name} selected`;
    });
    grid.appendChild(card);
  });
  $('#onboard').classList.add('show');
  $('#onboard').setAttribute('aria-hidden', 'false');
};

$('#onboard-go').addEventListener('click', async () => {
  if (!chosenPersona) return;
  await window.pywebview.api.choose_persona(chosenPersona);
  $('#onboard').classList.remove('show');
  $('#onboard').setAttribute('aria-hidden', 'true');
  input.focus();
});

/* A throw anywhere in event handling used to stop the turn with no trace: the
 * document tool ran, the file was written, and the screen simply did nothing.
 * Silent UI failure is the worst kind to debug, so surface it in the chat. */
const _rawOnEvent = window.vasudha.onEvent.bind(window.vasudha);
window.vasudha.onEvent = (evt) => {
  try {
    _rawOnEvent(evt);
  } catch (err) {
    console.error('vasudha: failed handling event', evt && evt.kind, err);
    try {
      if (!currentAssistant) beginAssistant();
      const box = document.createElement('div');
      box.className = 'tool-out';
      box.style.borderLeftColor = 'var(--saffron)';
      box.textContent =
        `Display error on a "${evt && evt.kind}" event: ${err && err.message}. `
        + 'The result itself was produced — this is a rendering fault.';
      currentAssistant.appendChild(box);
      scrollDown();
    } catch (_) { /* nothing further we can do */ }
  }
};

window.addEventListener('pywebviewready', () => {
  window.pywebview.api.ready();
  input.focus();
});
