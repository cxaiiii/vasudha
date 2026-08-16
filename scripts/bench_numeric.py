"""Six numeric engineering problems with mechanically-verified answers.

This is the pass/fail gate for anything that touches the tool path. Every
expected value here was computed in Python first, not taken from a textbook or
from the model.

Run it:
    python scripts/bench_numeric.py                     # the shipping loop, via ollama
    python scripts/bench_numeric.py --gguf model.gguf   # the shipping loop, built-in engine
    python scripts/bench_numeric.py --mode notools      # no tools, for comparison
    python scripts/bench_numeric.py --model qwen3:4b    # any model ollama is serving
    python scripts/bench_numeric.py --legacy            # the old web/agent.py loop

This drives app/session.py — the loop the desktop app actually runs. It used to
drive web/agent.py, which the app stopped using; a gate pointed at code the
product does not execute reports on a path nobody ships. `--legacy` still runs
the old loop so the two can be compared directly rather than by memory.

Because the backend is model-agnostic, `--model` and `--gguf` accept anything,
which is what makes this usable as a comparison harness and not only a
regression gate. Only python_tool is exposed regardless of what the app offers:
letting the model search the web for a cantilever formula would measure
something else entirely, and would not be comparable to the numbers below.

Greedy decoding by default, so results are reproducible rather than a sample.

Measured on vasudha-v3, greedy:

    harness                                        correct   tool fired
    regex <python_tool> tags + old app.py prompt     0/6         1/6
    native tool_calls (web/agent.py)                 4/6         6/6

Tool firing is the solid, reproducible result: 1/6 -> 6/6, every run. The
correctness number is ~4/6 and moves by one item between runs even at
temperature 0 (llama.cpp is not bit-deterministic across separate model
loads), so treat 4/6 as "about four", not a fixed score.

Failures seen, and what they are:
  * beam_deflection - the model writes the simply-supported centre-load
    formula PL^3/(48EI) for a cantilever, which is PL^3/(3EI). A knowledge
    error; the arithmetic on its own wrong formula is correct. Not something
    the harness can fix.
  * beam_stress     - used to compute 1.2e8 Pa correctly and then report
    "120,000 MPa" (divided by 1e3, not 1e6). Fixed by the unit rule in
    web/agent.py's tool descriptions: the sandbox now does the conversion.
  * pipe_reynolds   - used to return an EMPTY reply: the model went silent
    after a correct tool result. Fixed by the nudge/fallback in
    VasudhaAgent.run, which is why that path exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from web.tools import SandboxedCodeExecutor  # noqa: E402

TESTS = [
    dict(id="beam_deflection", answer=16.0, unit="mm", tol=0.02,
         q="A steel cantilever beam is 2 m long, with a rectangular cross-section 50 mm wide "
           "and 100 mm deep. A point load of 5 kN is applied at the free end. E = 200 GPa. "
           "Compute the maximum tip deflection in mm. Give the final number in mm."),
    dict(id="beam_stress", answer=120.0, unit="MPa", tol=0.02,
         q="Same beam: 2 m cantilever, rectangular section 50 mm wide x 100 mm deep, 5 kN point "
           "load at the free end. Compute the maximum bending stress in MPa. Give the final "
           "number in MPa."),
    dict(id="pipe_reynolds", answer=49800.0, unit="", tol=0.03,
         q="Water (density 998 kg/m^3, dynamic viscosity 1.002e-3 Pa*s) flows at 2 m/s through a "
           "pipe of internal diameter 25 mm. Compute the Reynolds number. Give the final number."),
    dict(id="rc_time_constant", answer=338.6, unit="Hz", tol=0.03,
         q="An RC low-pass filter uses R = 4.7 kOhm and C = 100 nF. Compute the -3 dB cutoff "
           "frequency in Hz. Give the final number in Hz."),
    dict(id="projectile", answer=90.36, unit="m", tol=0.02,
         q="A projectile is launched at 30 m/s at 40 degrees above the horizontal from flat "
           "ground, g = 9.81 m/s^2, no air resistance. Compute the horizontal range in metres. "
           "Give the final number in m."),
    dict(id="heat_conduction", answer=1000.0, unit="W", tol=0.02,
         q="A plane wall is 0.2 m thick with thermal conductivity 0.8 W/(m*K) and area 10 m^2. "
           "The inner surface is at 30 C and the outer at 5 C, steady state. Compute the heat "
           "transfer rate through the wall in watts. Give the final number in W."),
]

SYSTEM = ("You are Vasudha, an engineering assistant. For any question with a numeric answer you "
          "must call python_tool and base your answer on its real output. State the final value "
          "with its unit, exactly as the tool printed it.")

_NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def grade(text: str, answer: float, tol: float):
    """Scan only the model's final answer text. Deliberately strict: an earlier
    version scanned the reasoning too, which let a correct intermediate value
    score a pass even when the stated answer was wrong."""
    values = []
    for match in _NUM.finditer(text or ""):
        try:
            values.append(float(match.group().replace(",", "")))
        except ValueError:
            pass
    for value in values[-8:]:
        if answer and abs(value - answer) / abs(answer) <= tol:
            return True, value
    return False, (values[-1] if values else None)


@dataclass
class ModelSpec:
    """One thing to evaluate. Either an ollama model name or a GGUF path.

    Exists so scripts/compare_models.py can drive this suite over several
    models without reconstructing an argparse namespace per model.
    """
    label: str
    model: Optional[str] = None
    gguf: Optional[str] = None
    num_ctx: int = 8192

    @classmethod
    def from_args(cls, args) -> "ModelSpec":
        return cls(label=args.gguf or args.model, model=args.model,
                   gguf=args.gguf, num_ctx=args.num_ctx)


@dataclass
class Result:
    test: dict
    final: str
    calls: int
    elapsed: float
    ttft: Optional[float]
    tokens: int


def build_backend(spec: ModelSpec, quiet: bool = False):
    """An ollama or built-in backend, by the same selection the app uses."""
    from app.backends import LlamaCppBackend, OllamaBackend

    if spec.gguf:
        if not os.path.exists(spec.gguf):
            raise SystemExit(f"no such GGUF: {spec.gguf}")
        gpu = LlamaCppBackend.gpu_available()
        if not quiet:
            print(f"  engine: built-in llama.cpp, gpu_offload={gpu}"
                  + ("" if gpu else "  <- CPU-only wheel, expect ~6 tok/s"))
        return LlamaCppBackend(spec.gguf, n_ctx=spec.num_ctx,
                               n_gpu_layers=-1 if gpu else 0)

    served = OllamaBackend.probe()
    if not served:
        raise SystemExit(
            "ollama is not running. Start it, or pass --gguf to use the built-in engine.")
    match = next((n for n in served if spec.model in n), None)
    if match is None:
        raise SystemExit(f"ollama is not serving {spec.model!r}. Available: {served}")
    if not quiet:
        print(f"  engine: ollama, model={match}")
    return OllamaBackend(match)


def run_suite(spec: ModelSpec, options: dict, mode: str = "tools",
              backend=None, quiet: bool = False):
    """Drive app/session.py — the loop the desktop app ships.

    Yields a Result per problem. `backend` may be supplied by the caller so a
    comparison run can decide when to load and free each model itself.
    """
    from app.session import ChatSession

    backend = backend or build_backend(spec, quiet=quiet)
    executor = SandboxedCodeExecutor(timeout=15)
    workspace = tempfile.mkdtemp(prefix="vasudha_bench_")

    for test in TESTS:
        # A fresh session per problem, so one failure cannot contaminate the
        # next through conversation history. The backend is reused: reloading
        # a 2.3 GB GGUF six times would dominate the timings.
        session = ChatSession(backend, system_prompt=SYSTEM, workspace=workspace)
        session.max_iterations = 4

        if mode == "notools":
            session.schemas, session.tools = [], {}
        else:
            # python_tool only. The app also offers search, fetch, documents and
            # file tools; leaving those exposed would let the model look up a
            # formula instead of computing it, which is a different experiment
            # and not comparable to the historical numbers in this docstring.
            session.schemas = [s for s in session.schemas
                               if s["function"]["name"] == "python_tool"]
            session.tools = {"python_tool": lambda code="", **_:
                             executor.execute_python(code)}

        started = time.time()
        first_token = None
        final, calls, tokens = "", 0, 0

        for event in session.ask(test["q"], dict(options)):
            kind = event["kind"]
            if kind == "token":
                tokens += 1
                if first_token is None:
                    first_token = time.time() - started
            elif kind == "text":
                final = event["text"]
            elif kind == "tool_call":
                calls += 1
            elif kind == "error":
                final = f"[error] {event['text']}"

        yield Result(test=test, final=final, calls=calls,
                     elapsed=time.time() - started, ttft=first_token, tokens=tokens)


def _run_legacy_loop(args, options):
    """The pre-rewrite web/agent.py loop, kept so the two are comparable."""
    from web.agent import VasudhaAgent

    executor = SandboxedCodeExecutor(timeout=15)
    tools = ({} if args.mode == "notools"
             else {"python_tool": lambda code: executor.execute_python(code)})
    print("  engine: ollama via web/agent.py (legacy loop)")

    for test in TESTS:
        agent = VasudhaAgent(model=args.model, tools=tools, system_prompt=SYSTEM,
                             max_iterations=4, options=options)
        started = time.time()
        final, calls = "", 0
        for event in agent.run([{"role": "user", "content": test["q"]}]):
            if event.kind == "text":
                final = event.text
            elif event.kind == "tool_call":
                calls += 1
            elif event.kind == "error":
                final = f"[error] {event.text}"
        yield Result(test=test, final=final, calls=calls,
                     elapsed=time.time() - started, ttft=None, tokens=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("VASUDHA_OLLAMA_MODEL", "vasudha-v3"),
                        help="Model name ollama is serving. Substring match.")
    parser.add_argument("--gguf", default=None,
                        help="Drive the built-in llama.cpp engine against this GGUF instead.")
    parser.add_argument("--mode", choices=["tools", "notools"], default="tools")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--legacy", action="store_true",
                        help="Run the old web/agent.py loop instead of the shipping one.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    options = {"num_predict": 1024, "num_ctx": args.num_ctx,
               "temperature": args.temperature}
    if args.temperature == 0:
        options.update({"top_p": 1, "top_k": 1, "seed": 0})

    label = args.gguf or args.model
    loop = "web/agent.py (legacy)" if args.legacy else "app/session.py (shipping)"
    print(f"{'=' * 72}\n  {label}\n  loop={loop}  mode={args.mode}  "
          f"temp={args.temperature}\n{'=' * 72}")

    if args.legacy:
        results = _run_legacy_loop(args, options)
    else:
        results = run_suite(ModelSpec.from_args(args), options, mode=args.mode)

    correct = fired = 0
    records = []
    for r in results:
        good, got = grade(r.final, r.test["answer"], r.test["tol"])
        correct += good
        fired += r.calls > 0
        timing = f"{r.elapsed:.0f}s"
        if r.ttft is not None:
            timing += f" (first token {r.ttft:.1f}s)"
        print(f"  [{'PASS' if good else 'FAIL'}] {r.test['id']:<18} "
              f"want={r.test['answer']:<9} got={got}  calls={r.calls}  {timing}", flush=True)
        records.append(dict(id=r.test["id"], ok=bool(good), want=r.test["answer"], got=got,
                            calls=r.calls, seconds=round(r.elapsed, 1),
                            ttft=round(r.ttft, 2) if r.ttft is not None else None,
                            final=r.final))

    print(f"\n  --> {correct}/{len(TESTS)} correct, tool fired {fired}/{len(TESTS)}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(dict(model=label, loop=loop, mode=args.mode,
                           correct=correct, fired=fired, total=len(TESTS),
                           results=records), handle, indent=2)
        print(f"wrote {args.out}")

    # Non-zero exit when the tool path regresses, so this can gate a build.
    # Correctness is not gated: it moves by one item between runs even at
    # temperature 0, because llama.cpp is not bit-deterministic across model
    # loads. Tool firing is the reproducible signal — it was 6/6 every run.
    if args.mode == "tools" and fired < len(TESTS):
        print(f"\n  FAIL: the tool path regressed — fired {fired}/{len(TESTS)}, expected all.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
