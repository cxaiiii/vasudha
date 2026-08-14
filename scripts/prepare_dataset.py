"""Create a restartable engineering SFT dataset on local disk or a Modal Volume.

Unlike a single streaming mixture, this collector opens one Hub dataset at a
time, writes every accepted example immediately to ``rows.jsonl``, and resumes
from it.  A 429 or a dropped connection therefore costs a retry, not the whole
collection run (and never an expensive GPU allocation).

Usage:
    python scripts/prepare_dataset.py --out ./data/vasudha-20k --max-samples 20000
    python scripts/prepare_dataset.py --out /vol/ckpt/data/sft-20k --max-samples 20000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))

from vasudha.utils.logging import get_logger, log_banner, setup_logging

logger = get_logger(__name__)

# These are SFT sources, not pre-training sources.  We deliberately do not put
# The Stack v2 here: its HF repository contains SWH identifiers, not code text;
# getting file contents requires accepting its terms and separate S3 credentials.
DEFAULT_SOURCES = {
    "open-thoughts/OpenThoughts3-1.2M": 0.20,
    "AI-MO/NuminaMath-TIR": 0.16,
    "open-r1/OpenR1-Math-220k": 0.12,
    "artificial-citizen/Evol-Instruct-Code": 0.12,
    "deepcode-ai/math_dataset": 0.05,
    "ThomasTheMaker/TextToCadQuery-2": 0.12,
    "AbijahKaj/kicad-netlist-sft-dataset": 0.10,
    "AbdulrhmanEldeeb/metallurgy-qa": 0.04,
    "camel-ai/biology": 0.03,
    "synthetic_engineering_workflows": 0.06,
}

# Add-on curriculum for a model that has already seen the reasoning-heavy
# default mix.  This deliberately excludes the three large math sources so a
# 20k expansion buys new engineering/tool-use behaviour rather than more of
# the same proof traces.
ENGINEERING_SOURCES = {
    "artificial-citizen/Evol-Instruct-Code": 0.30,
    "ThomasTheMaker/TextToCadQuery-2": 0.25,
    "AbijahKaj/kicad-netlist-sft-dataset": 0.20,
    "AbdulrhmanEldeeb/metallurgy-qa": 0.05,
    "synthetic_engineering_workflows": 0.20,
}

BIO_SIM_SOURCES = {
    # Biology rows already written by an interrupted run remain in the JSONL
    # checkpoint, but we do not reopen the 20k archive just to obtain a few
    # hundred more.  The rest is generated locally and completes immediately.
    "synthetic_diverse_engineering_workflows": 1.0,
}

# Stage-2 behavioral correction: short, direct instruction-answer pairs with
# no chain-of-thought, to counterweight a mix that is otherwise almost all
# long reasoning traces. Combine this (via modal_app.py::combine_data) into
# the existing reasoning mixture rather than training on it alone — the goal
# is teaching the model when to stop reasoning, not un-teaching it to reason.
INSTRUCTION_FOLLOWING_SOURCES = {
    "teknium/OpenHermes-2.5": 1.0,
}

# Stage-2 grounding: real tool use, not hallucinated results. code_act is the
# core of this — multi-turn traces where the model writes Python, a real
# interpreter executes it, and the model continues from the actual output
# rather than an assumed one. The hermes_* sources teach the structured
# <tool_call> syntax already baked into this model's chat template — a
# different skill (calling a named tool correctly) from code_act's (writing
# and trusting real execution). self_oss_instruct is lower-weighted on
# purpose: general code data is already well represented via
# evol_instruct_code, so this is topped up rather than leaned on.
# Referenced by alias (not HF ID) below — three of these share one HF repo
# under different configs, and alias lookup is what keeps that unambiguous.
#
# Weights below are near-exact row counts, not proportions — the four
# "precious" grounding sources are small and fixed-size (verified against
# the real Hub row counts: code_act/codeact=7139, hermes_func_calling=1893,
# hermes_func_calling_singleturn=1893, hermes_glaive_func_calling=5209), so
# each is capped just under its real supply rather than a round fraction —
# _quotas() raises instead of under-filling if a quota exceeds what a source
# actually has. self_oss_instruct_exec_filter (50k+ available) is the only
# elastic source here and tops the total up to a round number. This means
# --max-samples for this profile should stay close to the sum of these
# (~20,600) — pushing it much higher will overshoot the scarce sources' real
# supply and fail the same way the original 20k/proportional-weight version
# did.
TOOL_USE_SOURCES = {
    "code_act": 7000,
    "hermes_func_calling": 1800,
    "hermes_func_calling_singleturn": 1800,
    "hermes_glaive_func_calling": 5000,
    "self_oss_instruct_exec_filter": 5000,
}

SOURCE_PROFILES = {
    "full": DEFAULT_SOURCES,
    "engineering": ENGINEERING_SOURCES,
    "bio_sim": BIO_SIM_SOURCES,
    "instruction_following": INSTRUCTION_FOLLOWING_SOURCES,
    "tool_use": TOOL_USE_SOURCES,
}


def _quotas(sources: dict[str, float], total: int) -> dict[str, int]:
    """Allocate exactly ``total`` examples using largest-remainder rounding."""
    weights = sum(sources.values())
    raw = {name: total * weight / weights for name, weight in sources.items()}
    result = {name: math.floor(value) for name, value in raw.items()}
    for name in sorted(sources, key=lambda n: raw[n] - result[n], reverse=True)[: total - sum(result.values())]:
        result[name] += 1
    return result


def engineering_workflows(seed: int) -> Iterator[dict[str, str]]:
    """Deterministic, numerically-grounded simulation/tool-use SFT examples.

    The responses show the engineering loop (equations -> assumptions -> model
    -> code -> validation -> limitations) rather than merely stating formulas.
    Values are calculated from the supplied parameters; no model API is used.
    """
    rng = random.Random(seed)
    materials = [("6061-T6 aluminum", 69e9, 276e6), ("A36 steel", 200e9, 250e6)]
    while True:
        material, elastic_modulus, yield_strength = rng.choice(materials)
        length = rng.choice([0.25, 0.30, 0.40, 0.50])
        width = rng.choice([0.020, 0.025, 0.030])
        height = rng.choice([0.008, 0.010, 0.012])
        force = rng.choice([100.0, 150.0, 200.0, 300.0])
        inertia = width * height**3 / 12
        stress = force * length * (height / 2) / inertia
        deflection = force * length**3 / (3 * elastic_modulus * inertia)
        fos = yield_strength / stress
        prompt = (
            f"Design a {material} cantilever beam, fixed at one end, with a {force:.0f} N tip load. "
            f"Use L={length:.2f} m, b={width * 1000:.0f} mm, h={height * 1000:.0f} mm. "
            "Show a Python simulation workflow and make a recommendation."
        )
        response = f"""I will use NumPy for a first-pass Euler-Bernoulli beam model.

**Governing equations.** For a rectangular section, $I=bh^3/12$. The fixed-end bending stress is $\\sigma_{{max}}=PL(h/2)/I$ and tip deflection is $\\delta=PL^3/(3EI)$.

**Assumptions.** Static linear-elastic response, small deflection, a prismatic beam, and a point load at the free end. This is a screening model; it excludes stress concentrations, fatigue, buckling, and fixture compliance.

```python
import numpy as np
P, L = {force:.1f}, {length:.3f}
b, h, E = {width:.4f}, {height:.4f}, {elastic_modulus:.3g}
I = b * h**3 / 12
x = np.linspace(0, L, 201)
M = P * (L - x)
sigma = M * (h / 2) / I
delta_tip = P * L**3 / (3 * E * I)
print(I, sigma[0], delta_tip)
```

**Simulation result.** $I={inertia:.3e}$ m$^4$, maximum bending stress is {stress / 1e6:.1f} MPa, and predicted tip deflection is {deflection * 1000:.2f} mm. With a {yield_strength / 1e6:.0f} MPa yield strength, the static yield safety factor is {fos:.2f}.

**Validation and recommendation.** Check units, confirm the moment diagram peaks at the fixed end, then compare this hand calculation with a beam/FEM mesh refinement. {"The static margin is acceptable for a preliminary design." if fos >= 2 else "Increase section height or reduce the load before releasing this design."} Validate against the actual load spectrum and local mounting geometry before manufacture."""
        yield {"text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"}


def diverse_engineering_workflows(seed: int) -> Iterator[dict[str, str]]:
    """Numerically-grounded thermal, fluid, circuit, and controls examples."""
    rng = random.Random(seed)
    beam_rows = engineering_workflows(seed + 1)
    while True:
        domain = rng.choice(["beam", "thermal", "pipe", "circuit", "control"])
        if domain == "beam":
            yield next(beam_rows)
            continue
        if domain == "thermal":
            power, resistance, ambient = rng.choice([20.0, 40.0, 60.0]), rng.choice([0.6, 1.0, 1.5]), 25.0
            result = ambient + power * resistance
            prompt = f"Model a {power:.0f} W enclosure with thermal resistance {resistance:.1f} K/W at {ambient:.0f} C ambient. Show a Python simulation and validation plan."
            answer = f"""I will start with the lumped thermal model $T=T_a+P R_\\theta$.
```python
P, R_theta, T_ambient = {power}, {resistance}, {ambient}
temperature = T_ambient + P * R_theta
print(temperature)
```
The estimated steady temperature is **{result:.1f} C**. Validate with thermocouples after steady state, then use thermal FEA/CFD if airflow or hot spots matter. This assumes one heat path and constant thermal resistance."""
        elif domain == "pipe":
            flow, diameter, length = 0.001, rng.choice([0.015, 0.020, 0.025]), rng.choice([5.0, 10.0, 20.0])
            rho, mu = 998.0, 1e-3
            velocity = flow / (math.pi * diameter**2 / 4)
            reynolds = rho * velocity * diameter / mu
            friction = 64 / reynolds if reynolds < 2300 else 0.3164 / reynolds**0.25
            drop = friction * (length / diameter) * rho * velocity**2 / 2
            prompt = f"Estimate water pressure drop through {length:.0f} m of {diameter*1000:.0f} mm ID pipe at 60 L/min. Use Python and state assumptions."
            answer = f"""I will apply continuity and Darcy-Weisbach.
```python
import math
Q, D, L, rho, mu = {flow}, {diameter}, {length}, {rho}, {mu}
v = Q / (math.pi * D**2 / 4); Re = rho*v*D/mu
f = 64/Re if Re < 2300 else 0.3164/Re**0.25
delta_p = f*(L/D)*rho*v**2/2
print(v, Re, delta_p)
```
Velocity is {velocity:.2f} m/s, Reynolds number is {reynolds:.0f}, and straight-pipe loss is **{drop/1000:.2f} kPa**. Add fitting $K$ losses and validate with gauges; smooth pipe and isothermal water are assumed."""
        elif domain == "circuit":
            resistance, capacitance, voltage = rng.choice([1000.0, 2200.0, 4700.0]), rng.choice([47e-6, 100e-6, 220e-6]), 5.0
            tau = resistance * capacitance
            prompt = f"Simulate a {voltage:.0f} V RC charge with R={resistance:.0f} ohm and C={capacitance*1e6:.0f} uF using NumPy. Explain validation."
            answer = f"""The governing equation is $V_C(t)=V_s(1-e^{{-t/RC}})$.
```python
import numpy as np
R, C, Vs = {resistance}, {capacitance}, {voltage}
tau = R*C; t = np.linspace(0, 5*tau, 300)
vc = Vs * (1 - np.exp(-t/tau))
print(tau, vc[-1])
```
The time constant is **{tau:.4f} s**. Validate it on an oscilloscope and extend the model with source resistance, ESR, leakage, and measurement loading."""
        else:
            kp, plant_tau = rng.choice([1.0, 2.0, 4.0]), rng.choice([0.2, 0.5, 1.0])
            prompt = f"Simulate proportional control with Kp={kp:.1f} on a first-order plant with time constant {plant_tau:.1f} s. Show Python and discuss limits."
            answer = f"""I will simulate $\\dot y=(-y+u)/\\tau$ with $u=K_p(r-y)$.
```python
import numpy as np
dt, tau, Kp = .002, {plant_tau}, {kp}; t = np.arange(0, 5, dt)
y = np.zeros_like(t)
for k in range(len(t)-1):
    y[k+1] = y[k] + dt*(-y[k] + Kp*(1-y[k]))/tau
print(y[-1])
```
Predicted proportional-only steady output is **{kp/(1+kp):.3f}**, so it retains offset. Validate actuator saturation, delay, sensor noise, and stability margin before adding integral action."""
        yield {"text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"}


def _load_existing(rows_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not rows_path.exists():
        return counts
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                counts[json.loads(line)["source"]] += 1
            except (json.JSONDecodeError, KeyError):
                logger.warning("Ignoring malformed checkpoint line in %s", rows_path)
    return counts


def _remote_rows(source: str, skip: int, retries: int, seed: int) -> Iterator[dict[str, str]]:
    """Restart a single stream after transient Hub/network errors."""
    from vasudha.datasets import VasudhaDatasetLoader

    delivered = 0
    attempt = 0
    while True:
        try:
            iterator = iter(VasudhaDatasetLoader(source, buffer_size=0, seed=seed).load_normalized())
            for row in iterator:
                if delivered < skip:
                    delivered += 1
                    continue
                delivered += 1
                yield row
            return
        except Exception as exc:  # HTTP 429, Xet/S3 timeout, transient connection reset
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"{source} failed after {retries} retries") from exc
            delay = min(120, 2 ** attempt) + random.random()
            logger.warning("%s stream failed (%s); retry %d/%d in %.1fs", source, type(exc).__name__, attempt, retries, delay)
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output directory for Arrow dataset and durable JSONL checkpoint")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--profile", choices=sorted(SOURCE_PROFILES), default="full")
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--retries", type=int, default=6, help="Retries per remote source after 429/network failures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(level="INFO")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    sources = SOURCE_PROFILES[args.profile]
    quotas = _quotas(sources, args.max_samples)
    counts = _load_existing(rows_path)
    log_banner("Resumable Dataset Preparation", f"{args.profile}: {args.max_samples} samples -> {out}")
    logger.info("Existing durable rows: %d", sum(counts.values()))

    with rows_path.open("a", encoding="utf-8", buffering=1) as handle:
        for source, target in quotas.items():
            remaining = target - counts[source]
            if remaining <= 0:
                logger.info("%s already complete (%d/%d)", source, counts[source], target)
                continue
            logger.info("Collecting %s: %d remaining (target %d)", source, remaining, target)
            if source == "synthetic_engineering_workflows":
                rows = engineering_workflows(args.seed)
            elif source == "synthetic_diverse_engineering_workflows":
                rows = diverse_engineering_workflows(args.seed)
            else:
                rows = _remote_rows(source, counts[source], args.retries, args.seed)
            accepted = 0
            for row in rows:
                text = row.get("text", "")
                if not text or not text.strip() or len(text) > args.max_chars:
                    continue
                handle.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
                handle.flush()
                counts[source] += 1
                accepted += 1
                if accepted >= remaining:
                    break
                if accepted % 250 == 0:
                    logger.info("  %s: %d/%d", source, counts[source], target)
            if accepted < remaining:
                raise RuntimeError(f"{source} ended early: got {accepted}/{remaining} required rows")

    from datasets import Dataset

    # Re-read the durable checkpoint so a process interrupt during conversion is
    # safe: the next invocation simply resumes then materializes Arrow again.
    rows = []
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows.append({"text": item["text"]})
    dataset = Dataset.from_list(rows).shuffle(seed=args.seed)
    dataset.save_to_disk(str(out / "arrow"))
    with (out / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({"profile": args.profile, "samples": len(rows), "counts": dict(counts), "quotas": quotas, "seed": args.seed}, handle, indent=2)
    log_banner("Preparation Complete", str(out / "arrow"))
    logger.info("Train with: +data.prepared_path=%s", out / "arrow")


if __name__ == "__main__":
    main()
