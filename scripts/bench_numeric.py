"""Six numeric engineering problems with mechanically-verified answers.

This is the pass/fail gate for anything that touches the tool path. Every
expected value here was computed in Python first, not taken from a textbook or
from the model.

Run it:
    python scripts/bench_numeric.py                    # native tool loop (web/agent.py)
    python scripts/bench_numeric.py --mode notools     # no tools, for comparison
    python scripts/bench_numeric.py --model vasudha-v2

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
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from web.agent import VasudhaAgent  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("VASUDHA_OLLAMA_MODEL", "vasudha-v3"))
    parser.add_argument("--mode", choices=["tools", "notools"], default="tools")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    executor = SandboxedCodeExecutor(timeout=15)
    tools = {} if args.mode == "notools" else {"python_tool": lambda code: executor.execute_python(code)}

    options = {"num_predict": 1024, "num_ctx": 8192, "temperature": args.temperature}
    if args.temperature == 0:
        options.update({"top_p": 1, "top_k": 1, "seed": 0})

    correct = fired = 0
    records = []
    print(f"{'=' * 72}\n  {args.model}  mode={args.mode}  temp={args.temperature}\n{'=' * 72}")

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

        good, got = grade(final, test["answer"], test["tol"])
        correct += good
        fired += calls > 0
        print(f"  [{'PASS' if good else 'FAIL'}] {test['id']:<18} want={test['answer']:<9} "
              f"got={got}  calls={calls}  {time.time() - started:.0f}s", flush=True)
        records.append(dict(id=test["id"], ok=bool(good), want=test["answer"], got=got,
                            calls=calls, final=final))

    print(f"\n  --> {correct}/{len(TESTS)} correct, tool fired {fired}/{len(TESTS)}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
