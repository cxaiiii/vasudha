# Vasudha Neo

Neo is the next Vasudha model: stock **Qwen3.5-4B**, post-trained with
**on-policy methods only** — no supervised fine-tuning step anywhere — for
about **$24 of Modal compute**, with a hard cap at $25.

It is trained to stand on its own. Nothing in the training or the evaluation
depends on this repository's desktop app: no app system prompt, no app tool
schemas, no app conventions. The model is driven through Qwen3.5's own chat
template and native tool-call format, so Ollama, llama.cpp, LM Studio, vLLM or
any OpenAI-compatible agent framework can use it with their own tools.

---

## Why not SFT again

Vasudha v1–v3 were LoRA SFT on other models' reasoning traces, each version
stacked on the last. That is the setup that erases what the base model
already knew.

*RL's Razor* (Shenfeld, Pari, Agrawal — ICLR 2026) measured why. When SFT
and RL reach the same score on a new task, RL keeps far more of the old
skills, and the amount of forgetting is predicted by one number: how far the
fine-tuned model has moved from the base model's own distribution (KL on the
new task). Training on the model's **own samples** is implicitly biased
toward the smallest such move; training on someone else's text is not.

So every gradient in Neo comes from text the model itself generated:

| Stage | Signal | Why |
|---|---|---|
| 1. On-policy distillation | The 4B student writes; Qwen3.5-9B grades every token it wrote (reverse KL) | Dense, per-token signal, no labels, no off-policy text. Same tokenizer, so the comparison is exact. Lifts pass rates so stage 2 has something to learn from. |
| 2. RL with verifiable rewards | GRPO; rewards come from checks that cannot be argued with — unit-aware numeric grading, math-verify, hidden unit tests in a sandbox, exact tool-call matching, IFEval checkers | Teaches *being right*, not *sounding like* a reference answer. A real Python interpreter is available on most tasks. |

## Stage 1 — on-policy distillation (Qwen3.5-9B → 4B)

- **How:** TRL `DistillationTrainer`, student samples with vLLM (colocated),
  teacher scores, loss is the reverse KL on the student's own tokens
  (`beta=1`). LoRA r=64/α=128 on every linear layer, lr 5e-5 (LoRA wants
  roughly 10× the full fine-tuning rate), 128 sequences per step, completions
  up to 1,536 tokens, fast (non-thinking) mode.
- **Prompts (16,000 built, as many used as the time box allows):**
  math 18%, general 13%, science 13%, code 12%, engineering 12%, function
  calling 12%, instruction following 10%, "Python is available" 6%, GSM8K 4%.
  Prompts only — no answers are ever shown to the student.
- **Time-boxed** to what the budget buys (~80 minutes); it stops cleanly and
  saves, never killed mid-step.

## Stage 2 — RL with verifiable rewards and a real sandbox

- **Tasks** (6,000 built, then filtered — see calibration):

  | Share | Source | Checked by |
  |---|---|---|
  | 28% | 54 procedural engineering families (beams, pipes, heat transfer, circuits, chemistry, finance, statistics…) with exact answers. **6 families are never trained on** and only appear in evaluation | unit-aware numeric grading |
  | 12% | WebInstruct — real science/finance numeric problems | unit-aware numeric grading |
  | 20% | competition-style math | math-verify on `\boxed{}` |
  | 12% | code (KodCode, validated so the reference solution passes) | hidden unit tests in the sandbox |
  | 16% | function calling (APIGen, plus 15% "no tool fits" cases) | exact call + argument match |
  | 12% | instruction following, synthesised from IFEval's constraint registry | IFEval's reference checkers |

- **No single harness.** Each task gets either no tools, or a Python tool
  under one of several common names (`python`, `code_interpreter`,
  `run_python`, `python_tool`) with a neutral output format, or — for function
  calling — whatever arbitrary tool list the task brings. The model learns tool
  use as a skill rather than one app's schema.
- **Calibration before RL:** 4,000 candidate tasks are each attempted 4 times
  by the stage-1 model. Tasks with mixed results are kept first — a task it
  always or never solves gives GRPO no gradient — and only if those run short
  is the set of 3,000 topped up with a few never-solved tasks (they may become
  solvable) and always-solved ones (they keep easy skills anchored).
- **Rewards** (identical code scores training and evaluation, so a reported
  number means exactly what the training signal meant):

  | Case | Reward |
  |---|---|
  | correct, committed answer with the right unit | 1.0 |
  | correct but hedged between values, or unit missing (both: 0.25) | 0.5 |
  | wrong | 0.0 |
  | each failed sandbox run in a correct episode | −0.03 (max −0.1) |
  | tool-call markup left in the text (a call that did not parse) | −0.3 |
  | empty final reply | −0.3 |
  | empty final reply right after a tool result | −0.5 |

  That table is the numeric case. Math, code and IFEval tasks score 1 or 0
  with the same penalties; function calling gives partial credit for a
  partially right set of calls. Numeric grading reads the value the reply
  commits to (bold, else the last number), converts units, ignores the givens
  restated from the question, and scores the *last* committed value — so
  "1.2e8 Pa ≈ 120,000 MPa" is wrong, not half right.
- **Config:** GRPO with the DAPO loss, 8 samples per prompt, clip-higher
  (ε 0.2 / 0.28), batch-level reward scaling, truncated completions masked out,
  no reference-KL term (LoRA plus on-policy sampling already keeps it close),
  up to 4 tool rounds per episode. LoRA r=32/α=64, lr 1e-5. ~95 minutes.

Training targets the fast (non-thinking) mode. Thinking mode is not trained,
and is evaluated before and after precisely to check that it survives — that
is the RL's Razor claim, measured rather than assumed.

## Evaluation

Every suite uses standard prompts with no system prompt; tools appear only
where a suite is about tools. The same items are cached once and reused for
every model.

| Suite | Measures | Fast mode | Thinking mode |
|---|---|---|---|
| GSM8K, MATH-500, AIME 2025 | math | ✓ | MATH-500 (250), AIME 2025 |
| HumanEval, MBPP | code, sandboxed tests | ✓ | |
| IFEval | instruction following | ✓ | ✓ |
| MMLU-Pro (1,400, stratified) | knowledge + reasoning | ✓ | 560 |
| BFCL v3 (11 categories) | function calling, AST match | ✓ | |
| eng_text / eng_tool (300 each) | engineering numerics without / with a Python tool, incl. the 6 held-out families (reported separately) | ✓ | eng_text |
| classic_text / classic_tool | the 29 numeric prompts of [EVALUATION.md](EVALUATION.md) | ✓ | classic_text |
| WebInstruct (300, held out) | real science/finance numerics | ✓ | |

Fast mode is greedy; thinking mode uses temperature 0.6, top-p 0.95. Engineering
evaluation problems are excluded from training by signature, and the
classic sheet is filtered out of every training source.

The run writes a side-by-side table — stock Qwen3.5-4B, Neo after stage 1,
final Neo — to `/neo/evals/comparison.md` on the volume, plus every
individual reply as JSONL, so any number can be checked against what the
model actually said.

## Money

| Stage | Hardware | Minutes | Cost |
|---|---|---|---|
| prepare_models — text-only copies of the 4B and 9B | CPU | ~25 | $0.37 |
| build_data — prompts, RL pool, eval suites | CPU | ~30 | $0.45 |
| smoke_and_base — 2 real steps of each trainer, then the baseline eval | H100 | ~40 | $3.15 |
| opd — stage 1 | H100 | ~87 | $6.82 |
| stage1 — eval + pass-rate calibration | H100 | ~30 | $2.37 |
| grpo — stage 2 | H100 | ~104 | $8.16 |
| final_eval | H100 | ~26 | $2.05 |
| export — full model + GGUF | CPU | ~40 | $0.60 |
| orchestrator | 0.125 CPU | whole run | $0.06 |
| **total** | | **~6.5 h** | **~$24.05** |

H100 at Modal's $3.95/h plus 8 cores and 48 GiB ($4.71/h all-in). How the cap
is enforced:

1. **Ledger.** Every stage's wall-clock is priced and appended to
   `/neo/ledger.jsonl`. A stage that could push the total past the cap does
   not start.
2. **Adaptive training time.** The two training stages get whatever the
   remaining money buys after reserving every later stage plus $1. If the
   fixed stages run 25% slow, training shrinks (to ~73 + ~81 min) and the
   total is still ~$24.70.
3. **Hard stop.** Each stage is cancelled the moment it has used the money
   that is left, whatever it is doing.
4. **Fail cheap.** The smoke stage runs real optimizer steps of both trainers
   at full batch size before any long stage, so a version or memory problem
   costs ~$1, not $10.

Every stage is resumable: rerunning the same command skips finished stages
and resumes training from its last checkpoint.

## Running it

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>     # or: modal setup
modal secret create huggingface HF_TOKEN=hf_...              # read access is enough

modal deploy modal_neo.py                       # deploy (repeat after code changes)
python -c "import modal; modal.Function.from_name('vasudha-neo', 'pipeline').spawn(budget=25.0)"
modal run modal_neo.py::status                  # spend so far, stages done, results
```

`budget=20.0` lowers the cap. The pipeline runs on the deployed app, so closing the terminal (or losing the connection) does not stop it. As a second line of defence, set a workspace
budget in Modal's billing settings. Results land on the `vasudha-neo` volume:

```bash
modal volume get vasudha-neo neo/gguf ./gguf                 # GGUF: Q4_K_M, Q3_K_M, Q8_0, f16
modal volume get vasudha-neo neo/evals/comparison.md .
modal volume get vasudha-neo neo/models/vasudha-neo-4b ./vasudha-neo-4b   # full HF checkpoint
```

The full checkpoint is the original multimodal Qwen3.5 layout with the trained
language weights written back in; the vision encoder and MTP head are the
untouched originals. The GGUF drops the MTP head (`--no-mtp`) for
compatibility with older loaders; a separate `mmproj` file carries vision.

## Using the model

```bash
ollama create vasudha-neo -f ollama/Modelfile-neo
ollama run vasudha-neo --think=false
```

The Modelfile sets no template and no system prompt: Ollama's built-in
Qwen3.5 renderer/parser provide the real template, tool calls included.
With llama.cpp, `llama-server -m vasudha-neo-4b-Q4_K_M.gguf --jinja` serves
an OpenAI-compatible endpoint with tool calling from the GGUF's own template.

## What was tested before spending anything

- 70 unit tests (`pytest tests/test_neo_*.py`): numeric grading against the
  failures v1–v3 actually produced, every engineering family's answer, tool-call
  parsing, BFCL matching, tool rendering byte-for-byte against the Qwen3.5 chat
  template, the sandbox, the ledger, and the export round trip (vision and MTP
  tensors bit-identical, logits matching after reload).
- Both trainers end to end on CPU with a tiny Qwen3.5, and one real GRPO step
  with Qwen3.5-0.8B whose native tool calls were executed in the sandbox and
  rewarded correctly.
- Every data builder and eval suite against the real Hugging Face datasets.

Not tested: anything on a GPU — vLLM colocation, the Gated DeltaNet kernels on
an H100, memory headroom at full batch size. That is exactly what the
`smoke_and_base` stage is for, and why it runs before the expensive stages.

## What to expect, honestly

Likely, based on what these methods have done for models of this size:
clear gains on engineering numerics (especially with the tool), function
calling and instruction following; solid gains on fast-mode math and code
from distilling a 9B teacher; thinking-mode scores roughly where the base
model has them.

Not promised: $25 buys about three H100-hours of training. That will not turn
a 4B model into a 30B one, and it will not move thinking-mode AIME much. The
comparison table will show where it beat the stock model and where it did not
— it is written by the run, not by this document.

## Files

```
neo/            the package: data, rewards, sandbox, rollouts, evaluation, export
  engineering.py  54 problem families with exact answers
  grading.py      unit-aware numeric grading (units.py: the unit table)
  verify.py       the reward for every task type = the evaluation metric
  sandbox.py      fork-server Python sandbox, ~10 ms per call
  toolcalls.py    Qwen3.5 tool-call parsing/rendering, BFCL matching
  envs.py         tool environments for TRL
  data.py         prompt and task builders; evalsets.py the eval suites
  budget.py       ledger and time-box callback; export.py checkpoint plumbing
scripts/neo_train_opd.py    stage 1
scripts/neo_train_grpo.py   stage 2
modal_neo.py                the whole run on Modal
ollama/Modelfile-neo        harness-free Ollama import
```
