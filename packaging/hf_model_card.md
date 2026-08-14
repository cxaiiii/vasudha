---
license: apache-2.0
language:
  - en
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - ollama
  - engineering
  - offline
  - local-llm
base_model: Qwen/Qwen3-4B
---

# Vasudha 4B — v3 (GGUF)

A 4B assistant that runs entirely on your own machine, built for engineering and
science questions where the answer is a number you can check.

Made by **Chaitanya**, an independent developer.

## What it is for

Vasudha is designed to be used **with a Python sandbox**, not on its own. It
picks the governing equation and sets the problem up; the arithmetic is executed
in a real interpreter and the working is shown. Used that way it is a calculator
that explains itself. Used without tools it will state numbers it has not
computed, and some of them will be wrong.

That distinction is the whole design, and the numbers below say why.

## Measured behaviour

Six numeric engineering problems (cantilever deflection and stress, Reynolds
number, RC cutoff, projectile range, wall conduction), ground truth computed in
Python, greedy decoding:

| Setup | Correct | Tool fired |
|---|---|---|
| Prompted to use a tool, prose-tag protocol | 0 / 6 | 1 / 6 |
| Native structured tool calling | ~4 / 6 | 6 / 6 |
| No tools at all | 1–2 / 6 | — |

The harness matters more than the prompt: asking the model to use a calculator
got a tool call one time in six; giving it a real tool schema got six out of six.

## Known limitations

Stated plainly because you will hit them.

- **Standard formulas can be recalled with the wrong boundary condition.** For a
  cantilever with an end load it has produced `PL³/(48EI)` — the
  simply-supported centre-load case — instead of `PL³/(3EI)`. It gets the
  simply-supported version right, so the coefficient is memorised without the
  support condition attached to it.
- **Magnitude errors of exactly 10× or 1000×** appear in both arithmetic and
  recalled facts. Asked for planetary masses it returned Neptune ten times too
  heavy while getting the ordering of all eight planets right.
- **It will cite sources it did not read.** Asked for planetary data it ran three
  searches that returned no figures at all, wrote a table from memory, and signed
  it "Source: NASA Solar System data sheet". Any application built on this model
  should track provenance itself rather than trusting the model's account of it.
- **Long replies can be cut off mid tool call**, which some servers reject
  outright. Allow at least 2048 output tokens.

None of these are subtle once you look for them, and all of them are why the
tool path is not optional.

## Files

| File | Size | Notes |
|---|---|---|
| `vasudha-leaprunning-4b-v3-Q3_K_M.gguf` | 2.11 GiB | Fits a 6 GB card with 64K context |

SHA-256 of the Q3_K_M build:

```
0a31e817bf8e7a3f9d1214ac1445648e734b3acd10ce019f02fa12b691a791c9
```

Check it after downloading — a truncated GGUF does not fail cleanly, it fails
confusingly.

## Use with Ollama

```bash
ollama create vasudha -f Modelfile
```

The GGUF carries its own Qwen3 chat template, including tool-call support, so
structured tool calling works without extra configuration.

## Use with llama.cpp

```bash
llama-server -m vasudha-leaprunning-4b-v3-Q3_K_M.gguf --jinja -c 16384
```

`--jinja` matters: it uses the chat template embedded in the file, which is what
makes tool calls parse.

## Architecture

Qwen3-4B converted to a hybrid: most layers Gated Linear Attention (O(1) memory
in context length) with a minority kept as full attention, plus MoE upcycling.
Trained with LoRA in three stages — base SFT, a behavioural correction pass, and
a reasoning-trace and tool-use distillation pass.

## License

Apache 2.0.
