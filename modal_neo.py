"""Vasudha Neo — the whole run on Modal, under a hard dollar cap.

    modal deploy modal_neo.py                       # once (and after any code change)
    python -c "import modal; modal.Function.from_name('vasudha-neo', 'pipeline').spawn(budget=25.0)"
    modal run modal_neo.py::status                  # money spent, stages done, results so far

The pipeline is spawned on a deployed app so nothing local can cancel it:
with `modal run --detach`, a disconnected client keeps only the last function
it called alive, and the stage calls the pipeline makes get cancelled.
`modal run --detach modal_neo.py` still works if the terminal stays connected.

Stages (each resumable — rerunning skips what already finished):

  1. prepare_models  CPU  text-only copies of Qwen3.5-4B (student) and -9B (teacher)
  2. build_data      CPU  distillation prompts, RL task pool, cached eval suites
  3. smoke_and_base  GPU  1-2 real steps of both trainers at full batch size
                          (catches OOM / version breakage for ~$1), then the
                          full baseline evaluation of the stock model
  4. opd             GPU  on-policy distillation 9B -> 4B (time-boxed)
  5. stage1          GPU  evaluate stage 1, then measure each RL task's pass
                          rate and keep the ones with learning signal
  6. grpo            GPU  RL with verifiable rewards + Python sandbox (time-boxed)
  7. final_eval      GPU  full evaluation, both modes
  8. export          CPU  full multimodal checkpoint + GGUF quantisations

Money: every stage's wall-clock is written to /vol/neo/ledger.jsonl and
priced at Modal's rates (GPU + CPU + memory). A stage that could push the
total over the cap does not start; the two training stages get whatever time
the remaining budget allows after reserving the later stages.

Secrets: a Modal secret named "huggingface" with HF_TOKEN (read access is
enough; write access only if you push the result).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import modal

GPU = os.environ.get("NEO_GPU", "H100")
APP_NAME = "vasudha-neo"
STUDENT_HUB = os.environ.get("NEO_STUDENT", "Qwen/Qwen3.5-4B")
TEACHER_HUB = os.environ.get("NEO_TEACHER", "Qwen/Qwen3.5-9B")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("vasudha-neo", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface")]

VOL = "/vol"
ROOT = f"{VOL}/neo"
MODELS, DATA, RUNS, EVALS = f"{ROOT}/models", f"{ROOT}/data", f"{ROOT}/runs", f"{ROOT}/evals"
EVAL_CACHE = f"{ROOT}/evalcache"
LEDGER, STATE = f"{ROOT}/ledger.jsonl", f"{ROOT}/state.json"
REPO = "/root/vasudha"

ENV = {
    "HF_HOME": f"{VOL}/hf",
    "PYTHONPATH": REPO,
    "TOKENIZERS_PARALLELISM": "false",
    "NLTK_DATA": "/usr/local/share/nltk_data",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TRL_EXPERIMENTAL_SILENCE": "1",
    "NEO_SANDBOX_WORKERS": "4",
    "VLLM_LOGGING_LEVEL": "WARNING",
    # fla's Gated DeltaNet backward has an optional TileLang path on Hopper;
    # pin the Triton kernels, which are the ones validated with Triton 3.7.1.
    "FLA_TILELANG": "0",
}

# CUDA 13.0.3 matches the runtime the torch 2.13 wheels bundle; Modal hosts run
# driver 580 (CUDA 13.0). The devel image is for Triton's launcher build (needs
# a C compiler) and any kernel vLLM JIT-compiles. torch 2.13 pins Triton 3.7.1,
# just outside the [3.4.0, 3.7.1) range where flash-linear-attention refuses to
# run its Gated DeltaNet backward on Hopper.
train_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "vllm==0.30.0",
        "torch==2.13.0",
        "transformers==5.17.0",
        "trl==1.14.1",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "datasets==5.1.0",
        "flash-linear-attention==0.5.2",
        "math-verify[antlr4_13_2]==0.9.0",
        "lm-eval==0.4.13",
        "langdetect", "immutabledict", "nltk",
        "safetensors", "huggingface_hub", "tensorboard", "scipy", "sympy", "numpy", "pytest", "requests",
    )
    .run_commands(
        "python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab",
        # nobody must be able to read the interpreter's packages, or the
        # sandbox falls back to running model code as root.
        "chmod -R o+rX /usr/local/lib/python3.12 || true",
    )
    .env(ENV)
    .add_local_dir("neo", f"{REPO}/neo", ignore=["__pycache__", "*.pyc"])
    .add_local_dir("scripts", f"{REPO}/scripts", ignore=["__pycache__", "*.pyc"])
)

gguf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "cmake")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy", "sentencepiece", "protobuf", "gguf", "transformers==5.17.0", "safetensors",
                 "huggingface_hub", "peft==0.21.2")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /root/llama.cpp",
        "cmake -B /root/llama.cpp/build -S /root/llama.cpp -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF",
        "cmake --build /root/llama.cpp/build --target llama-quantize --parallel 8",
        "pip install -r /root/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt || true",
    )
    .env({"HF_HOME": f"{VOL}/hf", "PYTHONPATH": REPO})
    .add_local_dir("neo", f"{REPO}/neo", ignore=["__pycache__", "*.pyc"])
)

GPU_CPU, GPU_MEM_GIB = 8.0, 48
CPU_CPU, CPU_MEM_GIB = 8.0, 64
# Containers are billed while idle too; the default 60 s scale-down window
# would be paid after every stage for nothing (each stage is a single call).
IDLE_S = 10
GPU_KW = dict(image=train_image, gpu=GPU, cpu=GPU_CPU, memory=GPU_MEM_GIB * 1024, volumes={VOL: vol},
              secrets=secrets, scaledown_window=IDLE_S)
CPU_KW = dict(image=train_image, cpu=CPU_CPU, memory=CPU_MEM_GIB * 1024, volumes={VOL: vol}, secrets=secrets,
              scaledown_window=IDLE_S)


def _sh(cmd: str) -> None:
    print(f"\n$ {cmd}\n", flush=True)
    result = subprocess.run(cmd, shell=True, cwd=REPO, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {cmd}")


def _exists(path: str) -> bool:
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

@app.function(**CPU_KW, timeout=3 * 3600)
def prepare_models(student: str = STUDENT_HUB, teacher: str = TEACHER_HUB) -> None:
    from huggingface_hub import snapshot_download

    from neo.export import make_text_checkpoint

    for hub, name in ((student, "student-text"), (teacher, "teacher-text")):
        out = f"{MODELS}/{name}"
        if _exists(f"{out}/config.json"):
            print(f"{out} exists — skipping")
            continue
        make_text_checkpoint(hub, out)
        vol.commit()
    snapshot_download(student)          # the full original, needed at export time
    vol.commit()


@app.function(**CPU_KW, timeout=3 * 3600)
def build_data(opd_n: int = 16000, rl_n: int = 6000) -> None:
    from neo.data import build_opd_prompts, build_rl_pool, eng_eval_problems, write_jsonl
    from neo.evalsets import SUITES, THINKING_LIMITS, load_suite

    exclude = {p.signature for p in eng_eval_problems(400)}
    os.makedirs(DATA, exist_ok=True)
    if not _exists(f"{DATA}/opd.jsonl"):
        write_jsonl(build_opd_prompts(opd_n, exclude_eng=exclude), f"{DATA}/opd.jsonl")
        vol.commit()
    if not _exists(f"{DATA}/rl_pool.jsonl"):
        write_jsonl(build_rl_pool(rl_n, exclude_eng=exclude), f"{DATA}/rl_pool.jsonl")
        vol.commit()
    for name in SUITES:
        load_suite(name, None, EVAL_CACHE)
        if name in THINKING_LIMITS:
            load_suite(name, THINKING_LIMITS[name], EVAL_CACHE)
    vol.commit()


def _opd_cmd(minutes: float, out: str, merge_to: str, extra: str = "") -> str:
    return (f"python scripts/neo_train_opd.py --student {MODELS}/student-text --teacher {MODELS}/teacher-text "
            f"--data {DATA}/opd.jsonl --out {out} --merge-to {merge_to} --max-minutes {minutes:.1f} {extra}")


def _grpo_cmd(model: str, data: str, minutes: float, out: str, merge_to: str, extra: str = "") -> str:
    return (f"python scripts/neo_train_grpo.py --model {model} --data {data} --out {out} --merge-to {merge_to} "
            f"--max-minutes {minutes:.1f} {extra}")


def _eval_cmd(model: str, tag: str, modes: str, suites: str = "fast", limit: int = 0) -> str:
    lim = f"--limit {limit}" if limit else ""
    return (f"python -m neo.evaluate --model {model} --out {EVALS}/{tag} --suites {suites} --modes {modes} "
            f"--cache {EVAL_CACHE} {lim}")


@app.function(**GPU_KW, timeout=2 * 3600)
def smoke_and_base(skip_smoke: bool = False) -> None:
    """Fail cheap: one or two optimizer steps of each trainer at the real batch
    size and memory settings, then the full baseline evaluation."""
    if not skip_smoke:
        _sh(_opd_cmd(8, f"{RUNS}/smoke-opd", f"{RUNS}/smoke-opd/merged", "--max-steps 2 --save-steps 1000"))
        _sh(f"python - <<'PY'\nfrom neo.data import read_jsonl, write_jsonl\n"
            f"write_jsonl(read_jsonl('{DATA}/rl_pool.jsonl')[:256], '{DATA}/rl_smoke.jsonl')\nPY")
        _sh(_grpo_cmd(f"{MODELS}/student-text", f"{DATA}/rl_smoke.jsonl", 10, f"{RUNS}/smoke-grpo",
                      f"{RUNS}/smoke-grpo/merged", "--max-steps 2 --save-steps 1000"))
        _sh(_eval_cmd(f"{RUNS}/smoke-grpo/merged", "smoke", "nothink", "gsm8k,bfcl,eng_tool,ifeval,humaneval", 16))
        vol.commit()
    if not _exists(f"{EVALS}/base/report.md"):          # written only when every suite is scored
        _sh(_eval_cmd(f"{MODELS}/student-text", "base", "nothink,think"))
        vol.commit()


@app.function(**GPU_KW, timeout=2 * 3600)
def base_eval() -> None:
    """Finish the stock-model evaluation outside the pipeline (it resumes
    suite by suite) and record its own cost in the ledger."""
    from neo.budget import Ledger

    started = time.time()
    ok = False
    try:
        _sh(_eval_cmd(f"{MODELS}/student-text", "base", "nothink,think"))
        ok = True
    finally:
        vol.reload()
        Ledger(LEDGER, 25.0).record("base_eval (resumed)", GPU, time.time() - started + 60 + IDLE_S, _rate("gpu"), ok,
                                    note="ran beside the pipeline; +60 s for container start")
        vol.commit()


@app.function(**GPU_KW, timeout=4 * 3600)
def opd(minutes: float = 80.0) -> None:
    _sh(_opd_cmd(minutes, f"{RUNS}/opd", f"{MODELS}/neo-stage1-text"))
    vol.commit()


@app.function(**GPU_KW, timeout=2 * 3600)
def stage1(n_select: int = 3000, k: int = 4, pool_limit: int = 4000) -> None:
    from neo.data import read_jsonl, write_jsonl
    from neo.evaluate import calibrate, select_by_pass_rate

    if not _exists(f"{EVALS}/stage1/report.md"):
        _sh(_eval_cmd(f"{MODELS}/neo-stage1-text", "stage1", "nothink"))
        vol.commit()
    if not _exists(f"{DATA}/rl_selected.jsonl"):
        rows = read_jsonl(f"{DATA}/rl_pool.jsonl")[:pool_limit]
        rates = calibrate(f"{MODELS}/neo-stage1-text", rows, k=k)
        with open(f"{DATA}/rl_pass_rates.json", "w") as fh:
            json.dump({r["id"]: p for r, p in zip(rows, rates)}, fh)
        write_jsonl(select_by_pass_rate(rows, rates, n_select), f"{DATA}/rl_selected.jsonl")
        vol.commit()


@app.function(**GPU_KW, timeout=4 * 3600)
def grpo(minutes: float = 105.0) -> None:
    _sh(_grpo_cmd(f"{MODELS}/neo-stage1-text", f"{DATA}/rl_selected.jsonl", minutes, f"{RUNS}/grpo",
                  f"{MODELS}/neo-final-text"))
    vol.commit()


@app.function(**GPU_KW, timeout=2 * 3600)
def final_eval() -> None:
    _sh(_eval_cmd(f"{MODELS}/neo-final-text", "final", "nothink,think"))
    from neo.evaluate import compare

    print(compare({"Qwen3.5-4B": f"{EVALS}/base", "Neo stage 1 (OPD)": f"{EVALS}/stage1",
                   "Vasudha Neo": f"{EVALS}/final"}, f"{EVALS}/comparison.md"))
    vol.commit()


@app.function(image=gguf_image, cpu=CPU_CPU, memory=48 * 1024, volumes={VOL: vol}, secrets=secrets,
              timeout=3 * 3600, scaledown_window=IDLE_S)
def export(quants: str = "Q4_K_M,Q3_K_M,Q8_0", student: str = STUDENT_HUB) -> None:
    from huggingface_hub import snapshot_download

    from neo.export import export_full

    full = f"{MODELS}/vasudha-neo-4b"
    if not _exists(f"{full}/neo_export.json"):
        original = snapshot_download(student)
        print(export_full(f"{MODELS}/neo-final-text", original, full))
        vol.commit()
    out = f"{ROOT}/gguf"
    os.makedirs(out, exist_ok=True)
    f16 = f"{out}/vasudha-neo-4b-f16.gguf"
    if not _exists(f16):
        # --no-mtp: older llama.cpp/Ollama loaders read the trailing MTP block
        # as a 33rd decoder layer and refuse the file; it only serves
        # speculative decoding. Passed only if this llama.cpp still has it.
        help_text = subprocess.run(["python", "/root/llama.cpp/convert_hf_to_gguf.py", "--help"],
                                   capture_output=True, text=True).stdout
        no_mtp = "--no-mtp" if "--no-mtp" in help_text else ""
        _sh(f"python /root/llama.cpp/convert_hf_to_gguf.py {full} --outfile {f16} --outtype f16 {no_mtp}")
    if not _exists(f"{out}/mmproj-vasudha-neo-4b-f16.gguf"):
        try:
            _sh(f"python /root/llama.cpp/convert_hf_to_gguf.py {full} --outfile {out}/mmproj-vasudha-neo-4b-f16.gguf "
                f"--outtype f16 --mmproj")
        except RuntimeError as exc:      # vision projector is a bonus, not a blocker
            print(f"mmproj export skipped: {exc}")
    for q in [q.strip() for q in quants.split(",") if q.strip()]:
        dst = f"{out}/vasudha-neo-4b-{q}.gguf"
        if not _exists(dst):
            _sh(f"/root/llama.cpp/build/bin/llama-quantize {f16} {dst} {q}")
    vol.commit()


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

ORCH_CPU, ORCH_MEM_GIB = 0.125, 0.5


def _rate(kind: str) -> float:
    from neo.budget import hourly_rate

    return {"gpu": lambda: hourly_rate(GPU, GPU_CPU, GPU_MEM_GIB),
            "cpu": lambda: hourly_rate(None, CPU_CPU, CPU_MEM_GIB),
            "orchestrator": lambda: hourly_rate(None, ORCH_CPU, ORCH_MEM_GIB)}[kind]()


def _load_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as fh:
            return json.load(fh)
    return {}


def _save_state(state: dict) -> None:
    os.makedirs(ROOT, exist_ok=True)
    with open(STATE, "w") as fh:
        json.dump(state, fh, indent=2)
    vol.commit()


# Non-preemptible: a preempted orchestrator restarts and re-attaches, but
# not being preempted at all is cheaper than finding out what that breaks.
@app.function(image=train_image, cpu=ORCH_CPU, memory=int(ORCH_MEM_GIB * 1024), volumes={VOL: vol},
              secrets=secrets, timeout=24 * 3600, nonpreemptible=True)
def pipeline(budget: float = 25.0, skip_smoke: bool = False, opd_max_minutes: float = 85.0,
             grpo_max_minutes: float = 110.0, min_train_minutes: float = 20.0) -> str:
    from neo.budget import BudgetExceeded, Ledger

    ledger = Ledger(LEDGER, budget)
    gpu_rate, cpu_rate, orch_rate = _rate("gpu"), _rate("cpu"), _rate("orchestrator")
    t0 = time.time()

    def orch_cost() -> float:          # this container, billed for as long as the run lasts
        return (time.time() - t0) / 3600 * orch_rate

    # Fixed-size stages, estimated generously (container start, model load, vLLM init included).
    est_min = {"prepare_models": 25, "build_data": 30, "smoke_and_base": 40, "stage1": 30, "final_eval": 26,
               "export": 40}
    est = {k: v / 60 * (gpu_rate if k in ("smoke_and_base", "stage1", "final_eval") else cpu_rate)
           for k, v in est_min.items()}
    overhead_min = 8.0       # per training stage: container start, model load, vLLM init, merge, save

    def run(name: str, fn, gpu: bool, estimate: float, **kwargs) -> None:
        vol.reload()
        state = _load_state()
        if state.get(name) == "done":
            print(f"[pipeline] {name}: already done")
            return
        rate = gpu_rate if gpu else cpu_rate
        key_call, key_started = f"{name}:call", f"{name}:started"
        call, started = None, None
        if state.get(key_call):
            # A previous orchestrator (preempted, restarted) spawned this stage:
            # re-attach to it instead of starting it again.
            prev = modal.FunctionCall.from_id(state[key_call])
            try:
                prev.get(timeout=0)
                call, started = prev, state[key_started]                  # finished meanwhile
            except (modal.exception.TimeoutError, TimeoutError):
                call, started = prev, state[key_started]
                print(f"[pipeline] {name}: re-attaching to the running stage", flush=True)
            except Exception as exc:  # noqa: BLE001 - it failed while nobody was watching
                ledger.record(f"{name} (failed unattended)", GPU if gpu else None,
                              time.time() - state[key_started] + IDLE_S, rate, False, note=str(exc)[:200])
                vol.commit()
        if call is None:
            ledger.ensure(name, estimate + orch_cost())
            started = time.time()
            call = fn.spawn(**kwargs)
            state = _load_state()
            state[key_call], state[key_started] = call.object_id, started
            _save_state(state)
        # The hard stop: however the stage behaves (a hang, a slow eval), it is
        # cancelled once it has used the money that is left.
        limit_s = (ledger.remaining() - orch_cost() - IDLE_S / 3600 * rate) / (rate + orch_rate) * 3600
        print(f"[pipeline] {name}: running (est ${estimate:.2f}, spent ${ledger.spent() + orch_cost():.2f} of "
              f"${budget:.2f}, hard stop {limit_s / 60:.0f} min after its start)", flush=True)
        outcome = None                                  # "ok" | "failed" | "cap"
        try:
            while True:
                left = limit_s - (time.time() - started)
                if left <= 0:
                    call.cancel(terminate_containers=True)
                    outcome = "cap"
                    raise BudgetExceeded(f"{name} was cancelled at the ${budget:.2f} cap after "
                                         f"{(time.time() - started) / 60:.0f} min")
                try:
                    call.get(timeout=min(600.0, left))
                    outcome = "ok"
                    break
                except (modal.exception.TimeoutError, TimeoutError):   # "not finished yet"
                    if isinstance(sys.exc_info()[1], modal.exception.OutputExpiredError):
                        outcome = "failed"
                        raise
                    used = time.time() - started
                    print(f"[pipeline] {name}: {used / 60:.0f} min, ~${used / 3600 * rate:.2f} so far", flush=True)
                except Exception:
                    outcome = "failed"                  # the stage itself raised
                    raise
        finally:
            # outcome None means this orchestrator was interrupted (preemption):
            # the stage keeps running and the restarted orchestrator re-attaches.
            if outcome is not None:
                vol.reload()
                entry = ledger.record(name, GPU if gpu else None, time.time() - started + IDLE_S, rate, outcome == "ok")
                state = _load_state()
                state.pop(key_call, None)
                state.pop(key_started, None)
                if outcome == "ok":
                    state[name] = "done"
                _save_state(state)
                print(f"[pipeline] {name}: {outcome} in {entry.seconds / 60:.1f} min, ${entry.cost:.2f}", flush=True)

    def affordable_minutes(later: list[str]) -> float:
        """Training minutes the remaining money buys after reserving `later`."""
        reserve = sum(est[s] for s in later) + orch_cost() + 1.0          # $1 safety margin
        return (ledger.remaining() - reserve) / (gpu_rate + orch_rate) * 60

    ok = False
    try:
        run("prepare_models", prepare_models, False, est["prepare_models"])
        run("build_data", build_data, False, est["build_data"])
        run("smoke_and_base", smoke_and_base, True, est["smoke_and_base"], skip_smoke=skip_smoke)

        # Whatever remains after reserving the fixed stages is split ~45/55
        # between distillation and RL (each also pays its own start-up overhead).
        pool_minutes = affordable_minutes(["stage1", "final_eval", "export"]) - 2 * overhead_min
        m = min(opd_max_minutes, 0.45 * pool_minutes)
        if m < min_train_minutes:
            raise BudgetExceeded(f"only {m:.0f} distillation minutes affordable; stopping before spending more")
        run("opd", opd, True, (m + overhead_min) / 60 * gpu_rate, minutes=m)
        run("stage1", stage1, True, est["stage1"])
        m = min(grpo_max_minutes, affordable_minutes(["final_eval", "export"]) - overhead_min)
        if m < min_train_minutes:
            raise BudgetExceeded(f"only {m:.0f} RL minutes affordable; stopping before spending more")
        run("grpo", grpo, True, (m + overhead_min) / 60 * gpu_rate, minutes=m)
        run("final_eval", final_eval, True, est["final_eval"])
        run("export", export, False, est["export"])
        ok = True
    finally:
        vol.reload()
        ledger.record("orchestrator", None, time.time() - t0, orch_rate, ok)
        vol.commit()
    report = ledger.table()
    if os.path.exists(f"{EVALS}/comparison.md"):
        with open(f"{EVALS}/comparison.md") as fh:
            report += "\n\n" + fh.read()
    print(report)
    return report


@app.function(image=train_image, cpu=ORCH_CPU, memory=int(ORCH_MEM_GIB * 1024), volumes={VOL: vol})
def status() -> str:
    from neo.budget import Ledger

    vol.reload()
    out = [Ledger(LEDGER, 25.0).table(), "", json.dumps(_load_state(), indent=2)]
    for tag in ("base", "stage1", "final"):
        path = f"{EVALS}/{tag}/report.md"
        if os.path.exists(path):
            with open(path) as fh:
                out += ["", f"## {tag}", fh.read()]
    text = "\n".join(out)
    print(text)
    return text


@app.local_entrypoint()
def main(budget: float = 25.0, skip_smoke: bool = False) -> None:
    print(pipeline.remote(budget=budget, skip_smoke=skip_smoke))
