<div align="center">

<img src="app/ui/assets/monogram.png" alt="Vasudha" width="180">

# Vasudha

### An offline engineering assistant that computes its answers instead of guessing them

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-2C7430.svg)](LICENSE)
[![Windows](https://img.shields.io/badge/Windows-10%20%7C%2011-F93E00.svg)](#install)
[![Model on HuggingFace](https://img.shields.io/badge/weights-HuggingFace-EFB68B.svg)](https://huggingface.co/cxaiiii/vasudha-4b-v3-gguf)

*A 4-billion-parameter model that runs entirely on your own machine — no API key,
no subscription, nothing leaving your computer — paired with a real Python sandbox
so every number in an answer is executed, not recalled.*

Built by **Chaitanya**, an independent developer.

</div>

---

## The idea

Small local models are confidently wrong about arithmetic. Ask a 4B model for a
beam deflection and it will write the right equation, substitute the right
numbers, and then state a result that is wrong by a factor of ten — because it
is predicting text, not calculating.

Vasudha treats that as an engineering problem rather than a limitation to
apologise for. The model picks the governing equation and sets the problem up.
**A real Python interpreter does the arithmetic.** You can open the working and
see the exact code that ran and the exact output it produced.

The difference is not subtle:

| Setup | Correct | Tool actually fired |
|---|---|---|
| Model asked politely to use a calculator | **0 / 6** | 1 / 6 |
| Model given a real tool interface | **4 / 6** | **6 / 6** |

Six numeric engineering problems, ground truth computed in Python, greedy
decoding. Reproduce it yourself with `python scripts/bench_numeric.py`.

## What it does

**Calculates, and shows the working.** Cantilever deflection, Reynolds numbers,
RC cutoff frequencies, heat transfer. Every result comes with the code that
produced it, expanded by default — the computation *is* the justification.

**Searches the web, and tells you what it actually read.** Search queries are the
only thing that ever leaves your machine, and the interface says so at the moment
it happens.

**Builds documents.** Reports, comparisons and spreadsheets render in a preview
canvas beside the chat, with export. Markdown, CSV and HTML.

**Tracks its own sources — honestly.** The application records which pages were
genuinely fetched. If a document was written without opening a single source, it
is labelled *unverified*, and a citation the model wrote without reading anything
is removed. That is enforced by the app, not requested of the model, because the
model will otherwise sign fabricated tables with authoritative-looking sources.

**Four personalities**, chosen on first run: Engineer, Teacher, Analyst,
Companion. They are plain JSON files you can edit or extend without rebuilding.

## Install

**Download** the release, unzip it, run `Vasudha.exe`.

On first launch it fetches the model once (2.11 GB) with a resumable, checksum-
verified download. After that it works with no internet at all.

Windows 10 or 11. No Python, no Ollama, no setup. If you already have Ollama
running, Vasudha finds it and uses your GPU automatically — noticeably faster.

<details>
<summary><b>Run from source</b></summary>

```bash
git clone <this-repo>
cd vasudha
pip install -r requirements.txt
python app/main.py
```

</details>

<details>
<summary><b>Just the weights</b></summary>

[huggingface.co/cxaiiii/vasudha-4b-v3-gguf](https://huggingface.co/cxaiiii/vasudha-4b-v3-gguf)

```
sha256  0a31e817bf8e7a3f9d1214ac1445648e734b3acd10ce019f02fa12b691a791c9
```

Works with Ollama (`ollama create vasudha -f ollama/Modelfile-v3`) or llama.cpp
(`llama-server -m <file> --jinja`). The GGUF carries its own chat template
including tool-call support.

</details>

## What it is bad at

Documented deliberately, because you will hit these.

- **It can recall a standard formula with the wrong boundary condition.** For a
  cantilever it has produced `PL³/(48EI)` — the simply-supported case — instead
  of `PL³/(3EI)`. It gets the simply-supported version right, so the coefficient
  is memorised without the support condition attached.
- **Magnitude errors of exactly 10× or 1000×** in both arithmetic and recalled
  facts. Asked for planetary masses it returned Neptune ten times too heavy while
  ordering all eight planets correctly.
- **Research is weaker than calculation.** Given thin search results it will fill
  the gaps from memory. The unverified labelling makes this visible; it does not
  make it accurate. Treat research output as a draft.

The calculation path is the trustworthy one. The research and document features
are genuinely useful and genuinely experimental, in that order.

## Architecture

Qwen3-4B converted to a hybrid — most layers Gated Linear Attention (O(1) memory
in context length), a minority kept as full attention, with MoE upcycling. See
[docs/architecture.md](docs/architecture.md).

The desktop app is a pywebview shell over a WebView2 window: no bundled browser,
no HTTP server, no open port. The UI talks to Python over an in-process bridge,
so "nothing leaves your machine" is structural rather than a promise.

```
app/
├── main.py         window, splash, JS bridge
├── backends.py     Ollama or in-process llama.cpp, chosen at runtime
├── session.py      the agent loop, tools, provenance
├── personas.py     personality presets, loaded from editable JSON
├── bootstrap.py    first-run model download, resumable + verified
└── ui/             the interface
scripts/
├── bench_numeric.py      the accuracy gate
├── publish_model.py      upload weights, wire the download link
└── simulate_first_run.py wipe local state, see what a new user sees
```

## Verifying it yourself

```bash
python scripts/bench_numeric.py
```

Six problems, verified answers, about two minutes. There is also a 28-prompt
manual sheet at [docs/EVALUATION.md](docs/EVALUATION.md) covering mechanics,
fluids, thermal, electronics, chemistry and behaviour — including cases that
*should not* trigger a tool call.

Don't take the numbers in this README on faith. That would be an odd way to use
a project about not taking numbers on faith.

## License

Apache 2.0 — see [LICENSE](LICENSE).
