"""Procedural engineering problems with exact answers.

These are the tasks the RL stage is rewarded on, and the held-out half of the
engineering evaluation. Every answer is computed here, in Python, from the
same numbers that appear in the question — no model, no API, no textbook
lookups — so a reward can never be wrong about the ground truth.

The families deliberately cover the failure modes this project has measured:

* boundary conditions — beams come as cantilever / simply supported /
  fixed-fixed with point and distributed loads, because v3 recalled
  PL³/48EI (simply supported) for a cantilever (PL³/3EI);
* magnitudes — inputs arrive in mixed prefixes (mm, cm, m; N, kN; MPa, GPa;
  µF, nF) and answers are asked for in a stated unit, because the observed
  errors were exactly 10x and 1000x;
* temperature traps — radiation and Carnot problems give °C and need kelvin.

Held-out families (HELDOUT_FAMILIES) are never generated for training; they
measure whether the skill transfers to problem types the policy never saw.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator, Optional

G = 9.81
SIGMA_SB = 5.670374419e-8
R_GAS = 8.314


@dataclass
class Problem:
    family: str
    question: str
    answer: float
    unit: str                     # canonical target unit ("" dimensionless, "money")
    tol: float = 0.015            # relative tolerance
    givens: list = field(default_factory=list)   # [(value, unit), ...] as stated
    difficulty: int = 1
    requires_tool: bool = True
    signature: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def fmt(x: float, sig: int = 4) -> str:
    """Human formatting: 2.5, 50, 0.025, 1.2e-05 -> '1.2e-5'."""
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e6:
        s = f"{x:.{sig}g}"
        if "e" in s:
            s = f"{x:.{max(0, sig - 1 - int(math.floor(math.log10(ax))))}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    mant, exp = f"{x:.{sig - 1}e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


def pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def a_an(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def cap(text: str) -> str:
    """Capitalise the first letter only ('water at 20 °C' -> 'Water at 20 °C')."""
    return text[:1].upper() + text[1:]


ASK = ["Compute", "Calculate", "Determine", "Find", "Work out", "Estimate"]
UNIT_ASK = [
    "Give the answer in {u}.",
    "Report the result in {u}.",
    "Express the answer in {u}.",
    "Answer in {u}.",
    "State the value in {u}.",
]


def ask_unit(rng, unit_text: str) -> str:
    return pick(rng, UNIT_ASK).format(u=unit_text)


MATERIALS_E = [  # name, E [GPa], alpha [1/K], yield [MPa], density [kg/m^3], G [GPa]
    ("structural steel", 200, 12e-6, 250, 7850, 80),
    ("aluminium alloy", 69, 23e-6, 270, 2700, 26),
    ("titanium alloy", 110, 8.6e-6, 880, 4430, 44),
    ("brass", 100, 19e-6, 200, 8500, 37),
    ("timber (Douglas fir)", 11, 4e-6, 40, 530, 0.7),
    ("copper", 117, 17e-6, 70, 8960, 44),
]


def _section(rng, difficulty):
    """Returns (description, I [m^4], c [m], A [m^2], givens)."""
    kind = pick(rng, ["rect", "rect", "circle", "hollow"] if difficulty >= 2 else ["rect", "rect", "circle"])
    if kind == "rect":
        b = pick(rng, [20, 25, 30, 40, 50, 60, 75, 80, 100])
        h = pick(rng, [40, 50, 60, 80, 100, 120, 150, 200])
        if rng.random() < 0.3:
            desc = f"a rectangular cross-section {fmt(b / 10)} cm wide and {fmt(h / 10)} cm deep"
            givens = [(b / 10, "cm"), (h / 10, "cm")]
        else:
            desc = f"a rectangular cross-section {b} mm wide and {h} mm deep"
            givens = [(b, "mm"), (h, "mm")]
        b_m, h_m = b / 1000, h / 1000
        return desc, b_m * h_m ** 3 / 12, h_m / 2, b_m * h_m, givens, {"section": "rect", "b": b_m, "h": h_m}
    if kind == "circle":
        d = pick(rng, [20, 25, 30, 40, 50, 60, 80, 100])
        d_m = d / 1000
        return (f"a solid circular cross-section of diameter {d} mm", math.pi * d_m ** 4 / 64, d_m / 2,
                math.pi * d_m ** 2 / 4, [(d, "mm")], {"section": "circle", "d": d_m})
    D = pick(rng, [50, 60, 80, 100, 120])
    t = pick(rng, [4, 5, 6, 8, 10])
    d = D - 2 * t
    D_m, d_m = D / 1000, d / 1000
    return (f"a hollow circular tube with outside diameter {D} mm and wall thickness {t} mm",
            math.pi * (D_m ** 4 - d_m ** 4) / 64, D_m / 2, math.pi * (D_m ** 2 - d_m ** 2) / 4,
            [(D, "mm"), (t, "mm")], {"section": "hollow", "D": D_m, "t": t / 1000})


BEAM_CASES = [
    # key, support text, load text, deflection(P|w, L, E, I), Mmax(P|w, L), load kind
    ("cant_point", "a cantilever (fixed at one end, free at the other)", "a point load of {load} at the free end",
     lambda q, L, E, I: q * L ** 3 / (3 * E * I), lambda q, L: q * L, "point"),
    ("cant_udl", "a cantilever (fixed at one end, free at the other)", "a uniformly distributed load of {load} along its full length",
     lambda q, L, E, I: q * L ** 4 / (8 * E * I), lambda q, L: q * L ** 2 / 2, "udl"),
    ("ss_point", "simply supported at both ends", "a point load of {load} at midspan",
     lambda q, L, E, I: q * L ** 3 / (48 * E * I), lambda q, L: q * L / 4, "point"),
    ("ss_udl", "simply supported at both ends", "a uniformly distributed load of {load} over the whole span",
     lambda q, L, E, I: 5 * q * L ** 4 / (384 * E * I), lambda q, L: q * L ** 2 / 8, "udl"),
    ("ff_point", "fixed at both ends", "a point load of {load} at midspan",
     lambda q, L, E, I: q * L ** 3 / (192 * E * I), lambda q, L: q * L / 8, "point"),
    ("ff_udl", "fixed at both ends", "a uniformly distributed load of {load} over the whole span",
     lambda q, L, E, I: q * L ** 4 / (384 * E * I), lambda q, L: q * L ** 2 / 12, "udl"),
]


def _beam_setup(rng, difficulty):
    case = pick(rng, BEAM_CASES if difficulty >= 2 else BEAM_CASES[:1] + BEAM_CASES[2:3] + BEAM_CASES[:1])
    key, support, load_text, defl, mmax, kind = case
    mat = pick(rng, MATERIALS_E[:5])
    L = pick(rng, [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0])
    desc, I, c, A, sec_givens, sec_meta = _section(rng, difficulty)
    if kind == "point":
        P = pick(rng, [200, 500, 800, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 8000, 10000])
        if P >= 1000 and rng.random() < 0.7:
            load, load_given = f"{fmt(P / 1000)} kN", (P / 1000, "kN")
        else:
            load, load_given = f"{P} N", (P, "N")
        q = P
    else:
        w = pick(rng, [0.5, 1, 1.5, 2, 3, 4, 5, 8, 10])  # kN/m
        load, load_given = f"{fmt(w)} kN/m", (w, "kN/m")
        q = w * 1000
    E = mat[1] * 1e9
    span_word = "long" if "cantilever" in support else "span"
    L_text = f"{fmt(L)} m" if rng.random() < 0.8 else f"{fmt(L * 1000)} mm"
    L_given = (L, "m") if L_text.endswith(" m") else (L * 1000, "mm")
    head = (f"{cap(a_an(mat[0]))} beam (E = {mat[1]} GPa), {support}, is {L_text} {span_word} with {desc}. "
            f"It carries {load_text.format(load=load)}.")
    givens = [(mat[1], "GPa"), L_given, load_given] + sec_givens
    distractor = ""
    if difficulty >= 3 and rng.random() < 0.5:
        distractor = f" The yield strength is {mat[3]} MPa and the density is {mat[4]} kg/m³."
        givens += [(mat[3], "MPa"), (mat[4], "kg/m^3")]
    return dict(key=key, head=head + distractor, defl=defl(q, L, E, I), m=mmax(q, L), I=I, c=c, A=A,
                givens=givens, meta={"case": key, "L": L, "E": E, "q": q, **sec_meta})


def fam_beam_deflection(rng, difficulty):
    s = _beam_setup(rng, difficulty)
    unit = pick(rng, ["mm", "mm", "mm", "cm"])
    d_mm = s["defl"] * 1000
    value = d_mm if unit == "mm" else d_mm / 10
    where = "tip" if s["key"].startswith("cant") else "maximum"
    q = f"{s['head']} {pick(rng, ASK)} the {where} deflection. {ask_unit(rng, unit)}"
    return Problem("beam_deflection", q, value, unit, 0.015, s["givens"], difficulty,
                   signature=f"beam_deflection|{s['meta']}", meta=s["meta"])


def fam_beam_stress(rng, difficulty):
    s = _beam_setup(rng, difficulty)
    sigma = s["m"] * s["c"] / s["I"] / 1e6
    q = f"{s['head']} {pick(rng, ASK)} the maximum bending stress. {ask_unit(rng, 'MPa')}"
    return Problem("beam_stress", q, sigma, "MPa", 0.015, s["givens"], difficulty,
                   signature=f"beam_stress|{s['meta']}", meta=s["meta"])


def fam_axial(rng, difficulty):
    mat = pick(rng, MATERIALS_E[:4] + MATERIALS_E[5:])
    d = pick(rng, [8, 10, 12, 16, 20, 25, 30])
    F = pick(rng, [5, 10, 15, 20, 25, 30, 40, 50, 75])  # kN
    L = pick(rng, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    A = math.pi * (d / 1000) ** 2 / 4
    sigma = F * 1000 / A
    givens = [(d, "mm"), (F, "kN"), (L, "m"), (mat[1], "GPa")]
    head = (f"{cap(a_an(mat[0]))} rod of diameter {d} mm and length {fmt(L)} m carries an axial tensile load of {F} kN "
            f"(E = {mat[1]} GPa).")
    if rng.random() < 0.5:
        q = f"{head} {pick(rng, ASK)} the normal stress in the rod. {ask_unit(rng, 'MPa')}"
        return Problem("axial", q, sigma / 1e6, "MPa", 0.015, givens, difficulty, signature=f"axial|s|{d}|{F}|{L}|{mat[0]}")
    delta = sigma * L / (mat[1] * 1e9) * 1000
    q = f"{head} {pick(rng, ASK)} how much the rod elongates. {ask_unit(rng, 'mm')}"
    return Problem("axial", q, delta, "mm", 0.015, givens, difficulty, signature=f"axial|e|{d}|{F}|{L}|{mat[0]}")


def fam_thermal(rng, difficulty):
    mat = pick(rng, [m for m in MATERIALS_E if m[0] != "timber (Douglas fir)"])
    L = pick(rng, [2, 5, 10, 12, 20, 25, 30, 50])
    dT = pick(rng, [15, 20, 25, 30, 40, 50, 60, 80])
    if rng.random() < 0.55 or difficulty == 1:
        dL = mat[2] * L * dT * 1000
        q = (f"A {fmt(L)} m long {mat[0]} member (coefficient of thermal expansion {fmt(mat[2] * 1e6)}×10⁻⁶ /K) "
             f"warms by {dT} K and is free to expand. {pick(rng, ASK)} its change in length. {ask_unit(rng, 'mm')}")
        return Problem("thermal", q, dL, "mm", 0.015, [(L, "m"), (dT, "K"), (mat[2] * 1e6, "")], difficulty,
                       signature=f"thermal|free|{mat[0]}|{L}|{dT}")
    sigma = mat[1] * 1e9 * mat[2] * dT / 1e6
    q = (f"{cap(a_an(mat[0]))} bar (E = {mat[1]} GPa, α = {fmt(mat[2] * 1e6)}×10⁻⁶ /K) is held between rigid walls so it "
         f"cannot expand, then heated by {dT} K. {pick(rng, ASK)} the thermal stress induced. {ask_unit(rng, 'MPa')}")
    return Problem("thermal", q, sigma, "MPa", 0.015, [(mat[1], "GPa"), (dT, "K"), (mat[2] * 1e6, "")], max(2, difficulty),
                   signature=f"thermal|fixed|{mat[0]}|{dT}")


def fam_torsion(rng, difficulty):
    d = pick(rng, [20, 25, 30, 40, 50, 60])
    if rng.random() < 0.5:
        T = pick(rng, [100, 200, 250, 400, 500, 800, 1000, 1500])
        T_text, givens = f"{T} N·m", [(T, "N*m"), (d, "mm")]
        intro = f"A solid steel shaft of diameter {d} mm transmits a torque of {T_text}."
        level = difficulty
    else:
        P = pick(rng, [5, 7.5, 10, 15, 20, 30, 45])  # kW
        n = pick(rng, [600, 900, 1200, 1450, 1500, 1800, 3000])
        T = P * 1000 / (2 * math.pi * n / 60)
        givens = [(P, "kW"), (n, "rpm"), (d, "mm")]
        intro = f"A solid steel shaft of diameter {d} mm transmits {fmt(P)} kW at {n} rpm."
        level = max(2, difficulty)
    r = d / 2000
    J = math.pi * (d / 1000) ** 4 / 32
    tau = T * r / J / 1e6
    if rng.random() < 0.6:
        q = f"{intro} {pick(rng, ASK)} the maximum shear stress in the shaft. {ask_unit(rng, 'MPa')}"
        return Problem("torsion", q, tau, "MPa", 0.015, givens, level, signature=f"torsion|tau|{d}|{T:.4f}")
    L = pick(rng, [0.5, 1.0, 1.5, 2.0])
    Gm = 80e9
    theta = math.degrees(T * L / (Gm * J))
    q = (f"{intro} The shaft is {fmt(L)} m long and G = 80 GPa. {pick(rng, ASK)} the angle of twist. "
         f"{ask_unit(rng, 'degrees')}")
    return Problem("torsion", q, theta, "deg", 0.015, givens + [(L, "m"), (80, "GPa")], min(3, level + 1),
                   signature=f"torsion|theta|{d}|{T:.4f}|{L}")


def fam_pressure_vessel(rng, difficulty):
    D = pick(rng, [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    t = pick(rng, [5, 6, 8, 10, 12, 15, 20])
    if rng.random() < 0.5:
        p_bar = pick(rng, [5, 8, 10, 12, 15, 20, 25])
        p, p_text, p_given = p_bar * 1e5, f"{p_bar} bar", (p_bar, "bar")
    else:
        p_mpa = pick(rng, [0.5, 0.8, 1.0, 1.5, 2.0, 2.5])
        p, p_text, p_given = p_mpa * 1e6, f"{fmt(p_mpa)} MPa", (p_mpa, "MPa")
    r = D / 2
    hoop = p * r / (t / 1000) / 1e6
    longi = hoop / 2
    which = pick(rng, ["hoop", "longitudinal"])
    q = (f"A thin-walled cylindrical pressure vessel has an internal diameter of {fmt(D)} m and a wall thickness "
         f"of {t} mm. It holds gas at an internal gauge pressure of {p_text}. {pick(rng, ASK)} the "
         f"{which} stress in the wall. {ask_unit(rng, 'MPa')}")
    return Problem("pressure_vessel", q, hoop if which == "hoop" else longi, "MPa", 0.015,
                   [(D, "m"), (t, "mm"), p_given], max(2, difficulty), signature=f"pv|{which}|{D}|{t}|{p}")


EULER_ENDS = [
    ("pinned at both ends", 1.0), ("fixed at the base and free at the top", 2.0),
    ("fixed at one end and pinned at the other", 0.7), ("fixed at both ends", 0.5),
]


def fam_euler_buckling(rng, difficulty):
    ends, K = pick(rng, EULER_ENDS)
    mat = pick(rng, MATERIALS_E[:3])
    L = pick(rng, [1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    d = pick(rng, [20, 25, 30, 40, 50, 60])
    I = math.pi * (d / 1000) ** 4 / 64
    P = math.pi ** 2 * mat[1] * 1e9 * I / (K * L) ** 2 / 1000
    q = (f"A {fmt(L)} m long {mat[0]} column (E = {mat[1]} GPa) with a solid circular section of diameter {d} mm "
         f"is {ends}. {pick(rng, ASK)} the Euler critical buckling load. {ask_unit(rng, 'kN')}")
    return Problem("euler_buckling", q, P, "kN", 0.02, [(L, "m"), (mat[1], "GPa"), (d, "mm")], 2,
                   signature=f"euler|{ends}|{mat[0]}|{L}|{d}")


def fam_springs(rng, difficulty):
    k1 = pick(rng, [2, 3, 4, 5, 6, 8, 10, 12, 15, 20])   # kN/m
    k2 = pick(rng, [2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    F = pick(rng, [50, 100, 120, 150, 200, 250, 300])
    arr = pick(rng, ["series", "parallel"])
    k = (1 / (1 / k1 + 1 / k2) if arr == "series" else k1 + k2) * 1000
    x = F / k
    if rng.random() < 0.6:
        q = (f"Two springs with stiffnesses {k1} kN/m and {k2} kN/m are connected in {arr} and support a static load "
             f"of {F} N. {pick(rng, ASK)} the total deflection. {ask_unit(rng, 'mm')}")
        return Problem("springs", q, x * 1000, "mm", 0.015, [(k1, "kN/m"), (k2, "kN/m"), (F, "N")], difficulty,
                       signature=f"springs|x|{arr}|{k1}|{k2}|{F}")
    energy = 0.5 * k * x ** 2
    q = (f"Two springs ({k1} kN/m and {k2} kN/m) are connected in {arr} and loaded with {F} N. {pick(rng, ASK)} the "
         f"elastic energy stored in the spring system. {ask_unit(rng, 'J')}")
    return Problem("springs", q, energy, "J", 0.015, [(k1, "kN/m"), (k2, "kN/m"), (F, "N")], min(3, difficulty + 1),
                   signature=f"springs|E|{arr}|{k1}|{k2}|{F}")


def fam_bolt_shear(rng, difficulty):
    n = pick(rng, [1, 2, 3, 4, 6])
    d = pick(rng, [8, 10, 12, 16, 20, 24])
    F = pick(rng, [10, 20, 30, 40, 50, 60, 80])
    shear = pick(rng, ["single", "double"])
    planes = 1 if shear == "single" else 2
    tau = F * 1000 / (n * planes * math.pi * (d / 1000) ** 2 / 4) / 1e6
    q = (f"A joint uses {n} bolt{'s' if n > 1 else ''} of {d} mm diameter in {shear} shear to carry a load of {F} kN, "
         f"shared equally. {pick(rng, ASK)} the average shear stress in each bolt. {ask_unit(rng, 'MPa')}")
    return Problem("bolt_shear", q, tau, "MPa", 0.015, [(n, ""), (d, "mm"), (F, "kN")], difficulty,
                   signature=f"bolt|{n}|{d}|{F}|{shear}")


def fam_factor_of_safety(rng, difficulty):
    mat = pick(rng, MATERIALS_E[:4])
    d = pick(rng, [10, 12, 16, 20, 25])
    F = pick(rng, [5, 8, 10, 15, 20, 30])
    sigma = F * 1000 / (math.pi * (d / 1000) ** 2 / 4) / 1e6
    fos = mat[3] / sigma
    q = (f"{cap(a_an(mat[0]))} tie rod (yield strength {mat[3]} MPa) of {d} mm diameter carries a {F} kN tensile load. "
         f"{pick(rng, ASK)} the factor of safety against yielding.")
    return Problem("factor_of_safety", q, fos, "", 0.015, [(mat[3], "MPa"), (d, "mm"), (F, "kN")], difficulty,
                   signature=f"fos|{mat[0]}|{d}|{F}")


# -- fluids -----------------------------------------------------------------

FLUIDS = [  # name, rho, mu
    ("water at 20 °C", 998.0, 1.002e-3),
    ("water at 40 °C", 992.2, 0.653e-3),
    ("air at 20 °C", 1.204, 1.825e-5),
    ("light oil", 870.0, 0.035),
    ("ethylene glycol", 1110.0, 0.0161),
]


def fam_reynolds(rng, difficulty):
    name, rho, mu = pick(rng, FLUIDS)
    D = pick(rng, [10, 15, 20, 25, 32, 40, 50, 80, 100])
    v = pick(rng, [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    Re = rho * v * D / 1000 / mu
    q = (f"{cap(name)} (density {fmt(rho)} kg/m³, dynamic viscosity {fmt(mu)} Pa·s) flows at {fmt(v)} m/s "
         f"through a pipe of internal diameter {D} mm. {pick(rng, ASK)} the Reynolds number.")
    return Problem("reynolds", q, Re, "", 0.015, [(rho, "kg/m^3"), (mu, "Pa*s"), (v, "m/s"), (D, "mm")], difficulty,
                   signature=f"re|{name}|{D}|{v}")


def fam_pipe_friction(rng, difficulty):
    name, rho, mu = pick(rng, FLUIDS[:2] + FLUIDS[3:])
    for _ in range(50):
        D = pick(rng, [10, 15, 20, 25, 32, 40, 50])
        v = pick(rng, [0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5])
        Re = rho * v * D / 1000 / mu
        if Re < 2000 or 5000 < Re < 9e4:
            break
    L = pick(rng, [5, 10, 20, 30, 50, 100])
    if Re < 2300:
        f, corr = 64 / Re, "The flow is laminar; use f = 64/Re."
    else:
        f, corr = 0.3164 / Re ** 0.25, "Use the Blasius correlation f = 0.3164 Re^-0.25 for the Darcy friction factor."
    hf = f * (L / (D / 1000)) * v ** 2 / (2 * G)
    givens = [(rho, "kg/m^3"), (mu, "Pa*s"), (v, "m/s"), (D, "mm"), (L, "m")]
    head = (f"{cap(name)} (ρ = {fmt(rho)} kg/m³, μ = {fmt(mu)} Pa·s) flows at {fmt(v)} m/s through {L} m of "
            f"smooth pipe with {D} mm internal diameter. {corr} Take g = 9.81 m/s².")
    if rng.random() < 0.5:
        q = f"{head} {pick(rng, ASK)} the head loss due to friction. {ask_unit(rng, 'm')}"
        return Problem("pipe_friction", q, hf, "m", 0.02, givens, max(2, difficulty), signature=f"pipe|hf|{name}|{D}|{v}|{L}")
    dp = rho * G * hf / 1000
    q = f"{head} {pick(rng, ASK)} the pressure drop due to friction. {ask_unit(rng, 'kPa')}"
    return Problem("pipe_friction", q, dp, "kPa", 0.02, givens, 3, signature=f"pipe|dp|{name}|{D}|{v}|{L}")


def fam_continuity(rng, difficulty):
    D1 = pick(rng, [50, 75, 100, 150, 200])
    D2 = pick(rng, [d for d in [20, 25, 40, 50, 75, 100] if d < D1])
    v1 = pick(rng, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    v2 = v1 * (D1 / D2) ** 2
    if rng.random() < 0.5:
        q = (f"Water flows at {fmt(v1)} m/s in a {D1} mm diameter pipe that reduces to {D2} mm diameter. "
             f"{pick(rng, ASK)} the velocity in the smaller pipe. {ask_unit(rng, 'm/s')}")
        return Problem("continuity", q, v2, "m/s", 0.015, [(D1, "mm"), (D2, "mm"), (v1, "m/s")], difficulty,
                       signature=f"cont|v|{D1}|{D2}|{v1}")
    Q = v1 * math.pi * (D1 / 1000) ** 2 / 4 * 1000
    q = (f"Water flows at {fmt(v1)} m/s through a pipe of {D1} mm internal diameter. {pick(rng, ASK)} the volume "
         f"flow rate. {ask_unit(rng, 'L/s')}")
    return Problem("continuity", q, Q, "L/s", 0.015, [(D1, "mm"), (v1, "m/s")], difficulty, signature=f"cont|Q|{D1}|{v1}")


def fam_torricelli(rng, difficulty):
    h = pick(rng, [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 8.0])
    v = math.sqrt(2 * G * h)
    if rng.random() < 0.5:
        q = (f"Water drains from a large open tank through a small orifice {fmt(h)} m below the free surface. "
             f"Neglecting losses and with g = 9.81 m/s², {pick(rng, ASK).lower()} the efflux velocity. {ask_unit(rng, 'm/s')}")
        return Problem("torricelli", q, v, "m/s", 0.015, [(h, "m")], difficulty, signature=f"torr|v|{h}")
    d = pick(rng, [10, 20, 25, 30, 40, 50])
    cd = pick(rng, [0.6, 0.61, 0.62, 0.64])
    Q = cd * math.pi * (d / 1000) ** 2 / 4 * v * 1000
    q = (f"An open tank discharges through a sharp-edged orifice of {d} mm diameter located {fmt(h)} m below the water "
         f"surface. The discharge coefficient is {cd} and g = 9.81 m/s². {pick(rng, ASK)} the discharge. {ask_unit(rng, 'L/s')}")
    return Problem("torricelli", q, Q, "L/s", 0.015, [(d, "mm"), (h, "m"), (cd, "")], 2, signature=f"torr|Q|{d}|{h}|{cd}")


def fam_hydrostatic(rng, difficulty):
    fluid, rho = pick(rng, [("fresh water", 1000.0), ("seawater", 1025.0), ("oil", 850.0), ("mercury", 13600.0)])
    h = pick(rng, [0.5, 1, 2, 3, 5, 8, 10, 20, 50])
    p = rho * G * h
    if rng.random() < 0.6:
        q = (f"{pick(rng, ASK)} the gauge pressure at a depth of {fmt(h)} m in {fluid} (density {fmt(rho)} kg/m³, "
             f"g = 9.81 m/s²). {ask_unit(rng, 'kPa')}")
        return Problem("hydrostatic", q, p / 1000, "kPa", 0.015, [(h, "m"), (rho, "kg/m^3")], difficulty, signature=f"hyd|p|{fluid}|{h}")
    a = pick(rng, [0.5, 1.0, 1.5, 2.0])
    b = pick(rng, [0.5, 1.0, 2.0, 3.0])
    F = p * a * b / 1000
    q = (f"A horizontal rectangular hatch {fmt(a)} m × {fmt(b)} m lies on the bottom of a tank, {fmt(h)} m below the "
         f"surface of the {fluid} (density {fmt(rho)} kg/m³, g = 9.81 m/s²). {pick(rng, ASK)} the hydrostatic force on "
         f"the hatch, ignoring atmospheric pressure. {ask_unit(rng, 'kN')}")
    return Problem("hydrostatic", q, F, "kN", 0.015, [(a, "m"), (b, "m"), (h, "m"), (rho, "kg/m^3")], 2,
                   signature=f"hyd|F|{fluid}|{h}|{a}|{b}")


def fam_buoyancy(rng, difficulty):
    V = pick(rng, [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5])
    fluid, rho = pick(rng, [("fresh water", 1000.0), ("seawater", 1025.0)])
    Fb = rho * G * V
    q = (f"A sealed {fmt(V * 1000)} litre container is held fully submerged in {fluid} (density {fmt(rho)} kg/m³). "
         f"With g = 9.81 m/s², {pick(rng, ASK).lower()} the buoyant force on it. {ask_unit(rng, 'N')}")
    return Problem("buoyancy", q, Fb, "N", 0.015, [(V * 1000, "L"), (rho, "kg/m^3")], difficulty, signature=f"buoy|{V}|{fluid}")


def fam_pump_power(rng, difficulty):
    Q = pick(rng, [5, 10, 15, 20, 30, 50, 80])   # L/s
    H = pick(rng, [5, 10, 15, 20, 30, 40, 60])
    eta = pick(rng, [0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
    P = 1000 * G * Q / 1000 * H / eta / 1000
    q = (f"A pump delivers {Q} L/s of water (density 1000 kg/m³) against a total head of {H} m with an overall "
         f"efficiency of {int(eta * 100)}%. Using g = 9.81 m/s², {pick(rng, ASK).lower()} the shaft power required? "
         f"{ask_unit(rng, 'kW')}")
    return Problem("pump_power", q, P, "kW", 0.015, [(Q, "L/s"), (H, "m"), (int(eta * 100), "%")], 2, signature=f"pump|{Q}|{H}|{eta}")


def fam_drag(rng, difficulty):
    v_kmh = pick(rng, [36, 54, 72, 90, 108, 120])
    cd = pick(rng, [0.25, 0.28, 0.3, 0.32, 0.35, 0.4])
    A = pick(rng, [1.8, 2.0, 2.2, 2.4, 2.6])
    rho = 1.2
    F = 0.5 * rho * (v_kmh / 3.6) ** 2 * cd * A
    q = (f"A car with frontal area {fmt(A)} m² and drag coefficient {cd} travels at {v_kmh} km/h in air of density "
         f"1.2 kg/m³. {pick(rng, ASK)} the aerodynamic drag force. {ask_unit(rng, 'N')}")
    return Problem("drag", q, F, "N", 0.015, [(A, "m^2"), (cd, ""), (v_kmh, "km/h"), (1.2, "kg/m^3")], 2,
                   signature=f"drag|{v_kmh}|{cd}|{A}")


# -- heat & thermo ---------------------------------------------------------

WALLS = [("brick", 0.72), ("concrete", 1.4), ("glass", 0.8), ("insulation board", 0.04), ("plaster", 0.5), ("wood", 0.12)]


def fam_wall_conduction(rng, difficulty):
    A = pick(rng, [4, 6, 8, 10, 12, 15, 20])
    Ti = pick(rng, [18, 20, 22, 25, 30])
    To = pick(rng, [-10, -5, 0, 5, 10])
    if difficulty == 1 or rng.random() < 0.4:
        name, k = pick(rng, WALLS)
        L = pick(rng, [0.1, 0.15, 0.2, 0.25, 0.3])
        Q = k * A * (Ti - To) / L
        q = (f"A plane {name} wall is {fmt(L)} m thick with thermal conductivity {fmt(k)} W/(m·K) and area {A} m². "
             f"The inner surface is at {Ti} °C and the outer surface at {To} °C (steady state). {pick(rng, ASK)} the "
             f"rate of heat transfer through the wall. {ask_unit(rng, 'W')}")
        return Problem("wall_conduction", q, Q, "W", 0.015, [(L, "m"), (k, "W/m*K"), (A, "m^2"), (Ti, "degC"), (To, "degC")],
                       1, signature=f"wall|1|{name}|{L}|{A}|{Ti}|{To}")
    (n1, k1), (n2, k2) = rng.sample(WALLS, 2)
    L1 = pick(rng, [0.05, 0.1, 0.15, 0.2])
    L2 = pick(rng, [0.02, 0.05, 0.08, 0.1])
    R = L1 / (k1 * A) + L2 / (k2 * A)
    text_conv = ""
    givens = [(L1, "m"), (k1, "W/m*K"), (L2, "m"), (k2, "W/m*K"), (A, "m^2"), (Ti, "degC"), (To, "degC")]
    if difficulty >= 3:
        hi, ho = pick(rng, [8, 10]), pick(rng, [20, 25, 30])
        R += 1 / (hi * A) + 1 / (ho * A)
        text_conv = (f" Convection coefficients are {hi} W/(m²·K) inside and {ho} W/(m²·K) outside, and the "
                     f"temperatures given are the room and outdoor air temperatures.")
        givens += [(hi, "W/m^2*K"), (ho, "W/m^2*K")]
    Q = (Ti - To) / R
    surf = "air" if text_conv else "surface"
    q = (f"A composite wall of area {A} m² has a {fmt(L1 * 1000)} mm layer of {n1} (k = {fmt(k1)} W/(m·K)) and a "
         f"{fmt(L2 * 1000)} mm layer of {n2} (k = {fmt(k2)} W/(m·K)) in series. The inside {surf} is at {Ti} °C and the "
         f"outside {surf} at {To} °C.{text_conv} {pick(rng, ASK)} the steady heat loss through the wall. {ask_unit(rng, 'W')}")
    return Problem("wall_conduction", q, Q, "W", 0.015, givens, difficulty, signature=f"wall|c|{n1}|{n2}|{L1}|{L2}|{A}|{Ti}|{To}|{text_conv!=''}")


def fam_cylinder_conduction(rng, difficulty):
    r1 = pick(rng, [10, 15, 20, 25, 30])
    t = pick(rng, [10, 20, 25, 30, 40, 50])
    r2 = r1 + t
    k = pick(rng, [0.035, 0.04, 0.05, 0.07])
    L = pick(rng, [1, 2, 5, 10])
    Ti, To = pick(rng, [100, 120, 150, 200]), pick(rng, [20, 25, 30])
    Q = 2 * math.pi * k * L * (Ti - To) / math.log(r2 / r1)
    q = (f"A {L} m length of pipe with outer radius {r1} mm is covered by {t} mm of insulation (k = {fmt(k)} W/(m·K)). "
         f"The inner surface of the insulation is at {Ti} °C and its outer surface at {To} °C. {pick(rng, ASK)} the "
         f"radial heat loss through the insulation. {ask_unit(rng, 'W')}")
    return Problem("cylinder_conduction", q, Q, "W", 0.015, [(L, "m"), (r1, "mm"), (t, "mm"), (k, "W/m*K"), (Ti, "degC"), (To, "degC")],
                   2, signature=f"cyl|{r1}|{t}|{k}|{L}|{Ti}|{To}")


def fam_convection(rng, difficulty):
    h = pick(rng, [5, 10, 15, 25, 50, 100, 250])
    A = pick(rng, [0.1, 0.25, 0.5, 1.0, 2.0])
    Ts, Tinf = pick(rng, [40, 60, 80, 100, 150]), pick(rng, [15, 20, 25])
    Q = h * A * (Ts - Tinf)
    q = (f"A surface of area {fmt(A)} m² at {Ts} °C is cooled by air at {Tinf} °C with a convection coefficient of "
         f"{h} W/(m²·K). {pick(rng, ASK)} the convective heat loss. {ask_unit(rng, 'W')}")
    return Problem("convection", q, Q, "W", 0.015, [(A, "m^2"), (Ts, "degC"), (Tinf, "degC"), (h, "W/m^2*K")], 1,
                   signature=f"conv|{h}|{A}|{Ts}|{Tinf}")


def fam_radiation(rng, difficulty):
    eps = pick(rng, [0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    A = pick(rng, [0.5, 1.0, 1.5, 2.0])
    Ts, Tsur = pick(rng, [100, 150, 200, 300, 400]), pick(rng, [20, 25, 30])
    Q = eps * SIGMA_SB * A * ((Ts + 273.15) ** 4 - (Tsur + 273.15) ** 4)
    q = (f"A plate of area {fmt(A)} m² with emissivity {eps} is at {Ts} °C and exchanges radiation with large "
         f"surroundings at {Tsur} °C. {pick(rng, ASK)} the net radiative heat loss (σ = 5.67×10⁻⁸ W/(m²·K⁴)). "
         f"{ask_unit(rng, 'W')}")
    return Problem("radiation", q, Q, "W", 0.015, [(A, "m^2"), (eps, ""), (Ts, "degC"), (Tsur, "degC")], 3,
                   signature=f"rad|{eps}|{A}|{Ts}|{Tsur}")


def fam_sensible_heat(rng, difficulty):
    sub, c = pick(rng, [("water", 4186), ("water", 4186), ("aluminium", 900), ("copper", 385), ("steel", 490), ("engine oil", 1900)])
    m = pick(rng, [0.5, 1, 2, 5, 10, 20, 50])
    T1, T2 = pick(rng, [10, 15, 20, 25]), pick(rng, [40, 50, 60, 80, 90])
    Q = m * c * (T2 - T1)
    if rng.random() < 0.6:
        q = (f"{pick(rng, ASK)} the energy needed to heat {fmt(m)} kg of {sub} from {T1} °C to {T2} °C "
             f"(specific heat {c} J/(kg·K)). {ask_unit(rng, 'kJ')}")
        return Problem("sensible_heat", q, Q / 1000, "kJ", 0.015, [(m, "kg"), (T1, "degC"), (T2, "degC"), (c, "J/kg*K")],
                       1, signature=f"heat|Q|{sub}|{m}|{T1}|{T2}")
    P = pick(rng, [500, 1000, 1500, 2000, 3000])
    t_min = Q / P / 60
    q = (f"A {P} W heater warms {fmt(m)} kg of {sub} (c = {c} J/(kg·K)) from {T1} °C to {T2} °C. Assuming no losses, "
         f"how long does it take. {ask_unit(rng, 'minutes')}")
    return Problem("sensible_heat", q, t_min, "min", 0.015, [(P, "W"), (m, "kg"), (T1, "degC"), (T2, "degC"), (c, "J/kg*K")],
                   2, signature=f"heat|t|{sub}|{m}|{T1}|{T2}|{P}")


def fam_ideal_gas(rng, difficulty):
    gas, M = pick(rng, [("nitrogen", 28.0), ("oxygen", 32.0), ("air", 28.97), ("carbon dioxide", 44.0), ("helium", 4.0)])
    m = pick(rng, [0.1, 0.25, 0.5, 1.0, 2.0])     # kg
    V = pick(rng, [0.05, 0.1, 0.2, 0.5, 1.0])     # m^3
    T = pick(rng, [0, 20, 25, 50, 100])          # degC
    n = m * 1000 / M
    P = n * R_GAS * (T + 273.15) / V
    q = (f"A rigid tank of {fmt(V)} m³ holds {fmt(m)} kg of {gas} (molar mass {fmt(M)} g/mol) at {T} °C. Treating it as "
         f"an ideal gas with R = 8.314 J/(mol·K), {pick(rng, ASK).lower()} the absolute pressure. {ask_unit(rng, 'kPa')}")
    return Problem("ideal_gas", q, P / 1000, "kPa", 0.015, [(V, "m^3"), (m, "kg"), (M, "g/mol"), (T, "degC")], 2,
                   signature=f"gas|{gas}|{m}|{V}|{T}")


def fam_carnot(rng, difficulty):
    Th, Tc = pick(rng, [200, 300, 400, 500, 600, 800]), pick(rng, [15, 20, 25, 30, 40])
    eta = 1 - (Tc + 273.15) / (Th + 273.15)
    q = (f"A heat engine operates between a hot reservoir at {Th} °C and a cold reservoir at {Tc} °C. {pick(rng, ASK)} "
         f"its maximum possible (Carnot) efficiency. {ask_unit(rng, 'percent')}")
    return Problem("carnot", q, eta * 100, "%", 0.01, [(Th, "degC"), (Tc, "degC")], 2, signature=f"carnot|{Th}|{Tc}")


def fam_lmtd(rng, difficulty):
    Thi, Tho = pick(rng, [120, 110, 100, 90]), pick(rng, [60, 55, 50, 45])
    Tci, Tco = pick(rng, [15, 20, 25]), pick(rng, [35, 40, 45])
    d1, d2 = Thi - Tco, Tho - Tci
    lm = (d1 - d2) / math.log(d1 / d2) if abs(d1 - d2) > 1e-9 else d1
    U, A = pick(rng, [300, 500, 800, 1000, 1500]), pick(rng, [2, 5, 8, 10, 15])
    Q = U * A * lm / 1000
    q = (f"In a counter-flow heat exchanger the hot stream cools from {Thi} °C to {Tho} °C while the cold stream warms "
         f"from {Tci} °C to {Tco} °C. With U = {U} W/(m²·K) and area {A} m², {pick(rng, ASK).lower()} the heat transfer "
         f"rate using the log-mean temperature difference. {ask_unit(rng, 'kW')}")
    return Problem("lmtd", q, Q, "kW", 0.015, [(Thi, "degC"), (Tho, "degC"), (Tci, "degC"), (Tco, "degC"), (U, "W/m^2*K"), (A, "m^2")],
                   3, signature=f"lmtd|{Thi}|{Tho}|{Tci}|{Tco}|{U}|{A}")


# -- electrical ------------------------------------------------------------

def _res_text(R: float) -> tuple[str, tuple]:
    if R >= 1e6:
        return f"{fmt(R / 1e6)} MΩ", (R / 1e6, "Mohm")
    if R >= 1000:
        return f"{fmt(R / 1000)} kΩ", (R / 1000, "kohm")
    return f"{fmt(R)} Ω", (R, "ohm")


E12 = [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82]


def _e12(rng, decades=(1, 10, 100, 1000)):
    return pick(rng, E12) * pick(rng, decades)


def fam_ohm_power(rng, difficulty):
    V = pick(rng, [3.3, 5, 9, 12, 24, 48, 230])
    R = _e12(rng, (1, 10, 100, 1000))
    r_text, r_given = _res_text(R)
    which = pick(rng, ["current", "power"])
    if which == "current":
        I = V / R
        unit = "mA" if I < 1 else "A"
        val = I * 1000 if unit == "mA" else I
        q = f"A {r_text} resistor is connected across a {fmt(V)} V supply. {pick(rng, ASK)} the current. {ask_unit(rng, unit)}"
        return Problem("ohm_power", q, val, unit, 0.015, [(V, "V"), r_given], 1, signature=f"ohm|I|{V}|{R}")
    P = V ** 2 / R
    unit = "mW" if P < 1 else "W"
    val = P * 1000 if unit == "mW" else P
    q = f"{pick(rng, ASK)} the power dissipated in a {r_text} resistor connected across {fmt(V)} V. {ask_unit(rng, unit)}"
    return Problem("ohm_power", q, val, unit, 0.015, [(V, "V"), r_given], 1, signature=f"ohm|P|{V}|{R}")


def fam_resistor_network(rng, difficulty):
    R1, R2, R3 = (_e12(rng, (1, 10, 100)) for _ in range(3))
    par = lambda a, b: a * b / (a + b)
    topo = pick(rng, ["par2", "par3", "s_p", "p_s"])
    (t1, g1), (t2, g2), (t3, g3) = _res_text(R1), _res_text(R2), _res_text(R3)
    if topo == "par2":
        Req, desc = par(R1, R2), f"{t1} and {t2} connected in parallel"
    elif topo == "par3":
        Req, desc = par(par(R1, R2), R3), f"{t1}, {t2} and {t3} all connected in parallel"
    elif topo == "s_p":
        Req, desc = R1 + par(R2, R3), f"{t1} in series with the parallel combination of {t2} and {t3}"
    else:
        Req, desc = par(R1 + R2, R3), f"{t1} and {t2} in series, with that string in parallel with {t3}"
    q = f"{pick(rng, ASK)} the equivalent resistance of {desc}. {ask_unit(rng, 'ohms')}"
    givens = [g1, g2] + ([g3] if topo != "par2" else [])
    return Problem("resistor_network", q, Req, "ohm", 0.015, givens, 1 if topo == "par2" else 2,
                   signature=f"rnet|{topo}|{R1}|{R2}|{R3}")


def fam_voltage_divider(rng, difficulty):
    Vin = pick(rng, [3.3, 5, 9, 12, 15, 24])
    R1, R2 = _e12(rng, (100, 1000, 10000)), _e12(rng, (100, 1000, 10000))
    Vout = Vin * R2 / (R1 + R2)
    t1, g1 = _res_text(R1); t2, g2 = _res_text(R2)
    q = (f"A voltage divider has R1 = {t1} (top) and R2 = {t2} (bottom) across a {fmt(Vin)} V supply. With no load, "
         f"{pick(rng, ASK).lower()} the output voltage across R2. {ask_unit(rng, 'V')}")
    return Problem("voltage_divider", q, Vout, "V", 0.015, [(Vin, "V"), g1, g2], 1, signature=f"vdiv|{Vin}|{R1}|{R2}")


def _cap_text(C: float) -> tuple[str, tuple]:
    if C >= 1e-6:
        return f"{fmt(C * 1e6)} µF", (C * 1e6, "uF")
    if C >= 1e-9:
        return f"{fmt(C * 1e9)} nF", (C * 1e9, "nF")
    return f"{fmt(C * 1e12)} pF", (C * 1e12, "pF")


def fam_rc(rng, difficulty):
    for _ in range(100):
        R = _e12(rng, (100, 1000, 10000))
        C = pick(rng, E12) * pick(rng, (1e-9, 10e-9, 100e-9, 1e-6))
        if 1.0 <= 1 / (2 * math.pi * R * C) <= 2e5:
            break
    rt, rg = _res_text(R); ct, cg = _cap_text(C)
    which = pick(rng, ["fc", "fc", "tau", "charge"])
    if which == "fc":
        fc = 1 / (2 * math.pi * R * C)
        unit = "kHz" if fc >= 10000 else "Hz"
        val = fc / 1000 if unit == "kHz" else fc
        q = (f"An RC low-pass filter uses R = {rt} and C = {ct}. {pick(rng, ASK)} the −3 dB cutoff frequency. "
             f"{ask_unit(rng, unit)}")
        return Problem("rc", q, val, unit, 0.015, [rg, cg], 1, signature=f"rc|fc|{R}|{C}")
    tau = R * C
    if which == "tau":
        q = f"{pick(rng, ASK)} the time constant of a series RC circuit with R = {rt} and C = {ct}. {ask_unit(rng, 'ms')}"
        return Problem("rc", q, tau * 1000, "ms", 0.015, [rg, cg], 1, signature=f"rc|tau|{R}|{C}")
    frac = pick(rng, [0.5, 0.63, 0.9, 0.95, 0.99])
    t = -tau * math.log(1 - frac)
    q = (f"A capacitor of {ct} charges through {rt} from 0 V toward a fixed supply voltage. {pick(rng, ASK)} the time to "
         f"reach {int(frac * 100)}% of the supply voltage. {ask_unit(rng, 'ms')}")
    return Problem("rc", q, t * 1000, "ms", 0.015, [rg, cg, (int(frac * 100), "%")], 2, signature=f"rc|t|{R}|{C}|{frac}")


def fam_lc(rng, difficulty):
    L = pick(rng, [1e-6, 10e-6, 100e-6, 1e-3, 10e-3, 100e-3]) * pick(rng, [1, 2.2, 4.7])
    C = pick(rng, E12) * pick(rng, (1e-12, 10e-12, 1e-9, 10e-9, 100e-9))
    f0 = 1 / (2 * math.pi * math.sqrt(L * C))
    unit = "MHz" if f0 >= 1e6 else ("kHz" if f0 >= 1e3 else "Hz")
    val = f0 / {"MHz": 1e6, "kHz": 1e3, "Hz": 1}[unit]
    l_text = f"{fmt(L * 1e6)} µH" if L < 1e-3 else f"{fmt(L * 1e3)} mH"
    l_given = (L * 1e6, "uH") if L < 1e-3 else (L * 1e3, "mH")
    ct, cg = _cap_text(C)
    q = f"{pick(rng, ASK)} the resonant frequency of an LC circuit with L = {l_text} and C = {ct}. {ask_unit(rng, unit)}"
    return Problem("lc", q, val, unit, 0.015, [l_given, cg], 1, signature=f"lc|{L}|{C}")


def fam_stored_energy(rng, difficulty):
    if rng.random() < 0.6:
        C = pick(rng, [100e-6, 470e-6, 1000e-6, 2200e-6, 4700e-6, 10e-6])
        V = pick(rng, [5, 12, 25, 50, 100, 400])
        E = 0.5 * C * V ** 2
        unit = "mJ" if E < 1 else "J"
        val = E * 1000 if unit == "mJ" else E
        ct, cg = _cap_text(C)
        q = f"{pick(rng, ASK)} the energy stored in a {ct} capacitor charged to {V} V. {ask_unit(rng, unit)}"
        return Problem("stored_energy", q, val, unit, 0.015, [cg, (V, "V")], 1, signature=f"energy|C|{C}|{V}")
    L = pick(rng, [1e-3, 2.2e-3, 10e-3, 47e-3, 100e-3, 0.5])
    I = pick(rng, [0.5, 1, 2, 3, 5, 10])
    E = 0.5 * L * I ** 2
    unit = "mJ" if E < 1 else "J"
    val = E * 1000 if unit == "mJ" else E
    q = (f"{pick(rng, ASK)} the energy stored in a {fmt(L * 1000)} mH inductor carrying {fmt(I)} A. {ask_unit(rng, unit)}")
    return Problem("stored_energy", q, val, unit, 0.015, [(L * 1000, "mH"), (I, "A")], 1, signature=f"energy|L|{L}|{I}")


def fam_impedance(rng, difficulty):
    f = pick(rng, [50, 60, 100, 1000, 10000])
    which = pick(rng, ["xc", "xl", "z"])
    if which == "xc":
        C = pick(rng, E12) * pick(rng, (1e-9, 10e-9, 100e-9, 1e-6))
        X = 1 / (2 * math.pi * f * C)
        ct, cg = _cap_text(C)
        q = f"{pick(rng, ASK)} the reactance of a {ct} capacitor at {f} Hz. {ask_unit(rng, 'ohms')}"
        return Problem("impedance", q, X, "ohm", 0.015, [cg, (f, "Hz")], 1, signature=f"imp|xc|{C}|{f}")
    if which == "xl":
        L = pick(rng, [1e-3, 10e-3, 47e-3, 100e-3, 0.5])
        X = 2 * math.pi * f * L
        q = f"{pick(rng, ASK)} the inductive reactance of a {fmt(L * 1000)} mH inductor at {f} Hz. {ask_unit(rng, 'ohms')}"
        return Problem("impedance", q, X, "ohm", 0.015, [(L * 1000, "mH"), (f, "Hz")], 1, signature=f"imp|xl|{L}|{f}")
    R = pick(rng, [10, 22, 47, 100, 220])
    L = pick(rng, [10e-3, 47e-3, 100e-3])
    C = pick(rng, [10e-6, 47e-6, 100e-6])
    Z = math.sqrt(R ** 2 + (2 * math.pi * f * L - 1 / (2 * math.pi * f * C)) ** 2)
    ct, cg = _cap_text(C)
    q = (f"A series RLC circuit has R = {R} Ω, L = {fmt(L * 1000)} mH and C = {ct}. {pick(rng, ASK)} the magnitude of its "
         f"impedance at {f} Hz. {ask_unit(rng, 'ohms')}")
    return Problem("impedance", q, Z, "ohm", 0.015, [(R, "ohm"), (L * 1000, "mH"), cg, (f, "Hz")], 3, signature=f"imp|z|{R}|{L}|{C}|{f}")


def fam_three_phase(rng, difficulty):
    VL = pick(rng, [400, 415, 440, 690, 11000])
    I = pick(rng, [10, 16, 25, 32, 50, 80, 100])
    pf = pick(rng, [0.8, 0.85, 0.9, 0.95])
    P = math.sqrt(3) * VL * I * pf / 1000
    q = (f"A balanced three-phase load draws {I} A per line from a {VL} V (line-to-line) supply at a power factor of "
         f"{pf} lagging. {pick(rng, ASK)} the real power consumed. {ask_unit(rng, 'kW')}")
    return Problem("three_phase", q, P, "kW", 0.015, [(I, "A"), (VL, "V"), (pf, "")], 2, signature=f"3ph|{VL}|{I}|{pf}")


def fam_battery(rng, difficulty):
    Ah = pick(rng, [2, 5, 7, 10, 20, 50, 100])
    V = pick(rng, [3.7, 12, 24, 48])
    P = pick(rng, [5, 10, 20, 50, 100, 200, 500])
    dod = pick(rng, [0.5, 0.8, 0.9, 1.0])
    hours = Ah * V * dod / P
    dod_text = "" if dod == 1.0 else f" using only {int(dod * 100)}% of its capacity"
    q = (f"A {fmt(V)} V, {Ah} Ah battery powers a constant {P} W load{dod_text}. Ignoring conversion losses, "
         f"{pick(rng, ASK).lower()} the run time. {ask_unit(rng, 'hours')}")
    return Problem("battery", q, hours, "h", 0.015, [(V, "V"), (Ah, "Ah"), (P, "W"), (int(dod * 100), "%")], 1,
                   signature=f"batt|{Ah}|{V}|{P}|{dod}")


def fam_led_resistor(rng, difficulty):
    Vs = pick(rng, [3.3, 5, 9, 12, 24])
    Vf = pick(rng, [1.8, 2.0, 2.1, 3.0, 3.2])
    I = pick(rng, [5, 10, 15, 20, 25])  # mA
    if Vs <= Vf + 0.5:
        Vs = 5.0
    R = (Vs - Vf) / (I / 1000)
    q = (f"An LED with a forward voltage of {fmt(Vf)} V should run at {I} mA from a {fmt(Vs)} V supply. "
         f"{pick(rng, ASK)} the series resistor value. {ask_unit(rng, 'ohms')}")
    return Problem("led_resistor", q, R, "ohm", 0.015, [(Vf, "V"), (I, "mA"), (Vs, "V")], 1, signature=f"led|{Vs}|{Vf}|{I}")


# -- dynamics ----------------------------------------------------------------

def fam_projectile(rng, difficulty):
    v = pick(rng, [10, 15, 20, 25, 30, 35, 40, 50])
    th = pick(rng, [15, 20, 25, 30, 35, 40, 45, 50, 60, 70])
    s, c = math.sin(math.radians(th)), math.cos(math.radians(th))
    which = pick(rng, ["range", "height", "time", "range_h"] if difficulty >= 2 else ["range", "height", "time"])
    base = f"A projectile is launched at {v} m/s at {th}° above the horizontal (g = 9.81 m/s², no air resistance)."
    if which == "range":
        R = v ** 2 * math.sin(math.radians(2 * th)) / G
        return Problem("projectile", f"{base} Starting and landing on level ground, {pick(rng, ASK).lower()} the horizontal range. {ask_unit(rng, 'm')}",
                       R, "m", 0.015, [(v, "m/s"), (th, "deg")], 1, signature=f"proj|R|{v}|{th}")
    if which == "height":
        H = (v * s) ** 2 / (2 * G)
        return Problem("projectile", f"{base} {pick(rng, ASK)} the maximum height above the launch point. {ask_unit(rng, 'm')}",
                       H, "m", 0.015, [(v, "m/s"), (th, "deg")], 1, signature=f"proj|H|{v}|{th}")
    if which == "time":
        T = 2 * v * s / G
        return Problem("projectile", f"{base} Over level ground, {pick(rng, ASK).lower()} the time of flight. {ask_unit(rng, 's')}",
                       T, "s", 0.015, [(v, "m/s"), (th, "deg")], 1, signature=f"proj|T|{v}|{th}")
    h = pick(rng, [5, 10, 15, 20, 30, 50])
    t = (v * s + math.sqrt((v * s) ** 2 + 2 * G * h)) / G
    R = v * c * t
    q = (f"A projectile is launched at {v} m/s at {th}° above the horizontal from the top of a {h} m high cliff "
         f"(g = 9.81 m/s², no air resistance). {pick(rng, ASK)} the horizontal distance from the cliff base to where it "
         f"lands. {ask_unit(rng, 'm')}")
    return Problem("projectile", q, R, "m", 0.015, [(v, "m/s"), (th, "deg"), (h, "m")], 3, signature=f"proj|Rh|{v}|{th}|{h}")


def fam_free_fall(rng, difficulty):
    h = pick(rng, [2, 5, 10, 20, 45, 80, 100])
    if rng.random() < 0.5:
        t = math.sqrt(2 * h / G)
        q = f"An object is dropped from rest from {h} m (g = 9.81 m/s², no drag). {pick(rng, ASK)} the time to hit the ground. {ask_unit(rng, 's')}"
        return Problem("free_fall", q, t, "s", 0.015, [(h, "m")], 1, signature=f"ff|t|{h}")
    v = math.sqrt(2 * G * h)
    q = f"An object is dropped from rest from {h} m (g = 9.81 m/s², no drag). {pick(rng, ASK)} its impact speed. {ask_unit(rng, 'm/s')}"
    return Problem("free_fall", q, v, "m/s", 0.015, [(h, "m")], 1, signature=f"ff|v|{h}")


def fam_circular(rng, difficulty):
    m = pick(rng, [0.5, 1, 2, 5, 1200, 1500])
    v_kmh = pick(rng, [18, 36, 54, 72])
    r = pick(rng, [0.5, 1, 2, 20, 50, 80])
    F = m * (v_kmh / 3.6) ** 2 / r
    q = (f"A {fmt(m)} kg mass moves at a constant {v_kmh} km/h around a circle of radius {fmt(r)} m. {pick(rng, ASK)} the "
         f"centripetal force required. {ask_unit(rng, 'N')}")
    return Problem("circular", q, F, "N", 0.015, [(m, "kg"), (v_kmh, "km/h"), (r, "m")], 2, signature=f"circ|{m}|{v_kmh}|{r}")


def fam_oscillation(rng, difficulty):
    if rng.random() < 0.5:
        k = pick(rng, [100, 200, 500, 1000, 2000, 5000])
        m = pick(rng, [0.1, 0.2, 0.5, 1, 2, 5])
        f = math.sqrt(k / m) / (2 * math.pi)
        q = f"{pick(rng, ASK)} the natural frequency of a {fmt(m)} kg mass on a spring of stiffness {k} N/m. {ask_unit(rng, 'Hz')}"
        return Problem("oscillation", q, f, "Hz", 0.015, [(m, "kg"), (k, "N/m")], 1, signature=f"osc|sm|{k}|{m}")
    L = pick(rng, [0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
    T = 2 * math.pi * math.sqrt(L / G)
    q = f"{pick(rng, ASK)} the period of a simple pendulum {fmt(L)} m long (small swings, g = 9.81 m/s²). {ask_unit(rng, 's')}"
    return Problem("oscillation", q, T, "s", 0.015, [(L, "m")], 1, signature=f"osc|pend|{L}")


def fam_braking(rng, difficulty):
    v_kmh = pick(rng, [30, 40, 50, 60, 80, 100, 120])
    mu = pick(rng, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    d = (v_kmh / 3.6) ** 2 / (2 * mu * G)
    q = (f"A car travelling at {v_kmh} km/h brakes to a stop on a level road with friction coefficient {mu} "
         f"(g = 9.81 m/s²). Ignoring reaction time, {pick(rng, ASK).lower()} the braking distance. {ask_unit(rng, 'm')}")
    return Problem("braking", q, d, "m", 0.015, [(v_kmh, "km/h"), (mu, "")], 2, signature=f"brake|{v_kmh}|{mu}")


def fam_collision(rng, difficulty):
    m1, m2 = pick(rng, [1, 2, 5, 1000, 1500]), pick(rng, [1, 3, 4, 800, 2000])
    v1, v2 = pick(rng, [2, 5, 10, 15, 20]), pick(rng, [0, 0, 1, 3, -5])
    vf = (m1 * v1 + m2 * v2) / (m1 + m2)
    q = (f"A {m1} kg body moving at {v1} m/s collides with a {m2} kg body moving at {v2} m/s along the same line and "
         f"they stick together. {pick(rng, ASK)} their common velocity just after impact. {ask_unit(rng, 'm/s')}")
    return Problem("collision", q, vf, "m/s", 0.015, [(m1, "kg"), (m2, "kg"), (v1, "m/s"), (v2, "m/s")], 2,
                   signature=f"coll|{m1}|{m2}|{v1}|{v2}")


def fam_lifting(rng, difficulty):
    m = pick(rng, [50, 100, 250, 500, 1000, 2000])
    h = pick(rng, [2, 5, 10, 15, 20, 30])
    t = pick(rng, [5, 10, 15, 20, 30, 60])
    eta = pick(rng, [1.0, 0.7, 0.75, 0.8, 0.85])
    P = m * G * h / t / eta
    eff = "" if eta == 1.0 else f" The hoist is {int(eta * 100)}% efficient; give the input power."
    unit = "kW" if P >= 1000 else "W"
    val = P / 1000 if unit == "kW" else P
    q = (f"A hoist lifts a {m} kg load through {h} m at constant speed in {t} s (g = 9.81 m/s²).{eff} {pick(rng, ASK)} the "
         f"power required. {ask_unit(rng, unit)}")
    return Problem("lifting", q, val, unit, 0.015, [(m, "kg"), (h, "m"), (t, "s"), (int(eta * 100), "%")], 1 if eta == 1 else 2,
                   signature=f"lift|{m}|{h}|{t}|{eta}")


def fam_kinetic_energy(rng, difficulty):
    m = pick(rng, [2, 10, 70, 800, 1200, 1500, 20000])
    v_kmh = pick(rng, [18, 36, 50, 72, 90, 108])
    KE = 0.5 * m * (v_kmh / 3.6) ** 2
    unit = "kJ" if KE >= 1000 else "J"
    val = KE / 1000 if unit == "kJ" else KE
    q = f"{pick(rng, ASK)} the kinetic energy of a {m} kg body moving at {v_kmh} km/h. {ask_unit(rng, unit)}"
    return Problem("kinetic_energy", q, val, unit, 0.015, [(m, "kg"), (v_kmh, "km/h")], 1, signature=f"ke|{m}|{v_kmh}")


def fam_gears(rng, difficulty):
    n_in = pick(rng, [900, 1200, 1450, 1500, 1800, 3000])
    z1, z2 = pick(rng, [12, 15, 18, 20, 24]), pick(rng, [36, 40, 45, 48, 60, 72])
    n_out = n_in * z1 / z2
    if rng.random() < 0.5:
        q = (f"A {z1}-tooth pinion driven at {n_in} rpm meshes with a {z2}-tooth gear. {pick(rng, ASK)} the gear speed. "
             f"{ask_unit(rng, 'rpm')}")
        return Problem("gears", q, n_out, "rpm", 0.015, [(z1, ""), (z2, ""), (n_in, "rpm")], 1, signature=f"gear|n|{n_in}|{z1}|{z2}")
    T_in = pick(rng, [10, 20, 30, 50, 80])
    eta = pick(rng, [0.95, 0.97, 0.98])
    T_out = T_in * z2 / z1 * eta
    q = (f"A {z1}-tooth pinion delivers {T_in} N·m to a {z2}-tooth gear through a mesh that is {int(eta * 100)}% efficient. "
         f"{pick(rng, ASK)} the output torque on the gear shaft. {ask_unit(rng, 'N·m')}")
    return Problem("gears", q, T_out, "N*m", 0.015, [(z1, ""), (z2, ""), (T_in, "N*m"), (int(eta * 100), "%")], 2,
                   signature=f"gear|T|{T_in}|{z1}|{z2}|{eta}")


# -- chemistry ---------------------------------------------------------------

COMPOUNDS = [("NaCl", 58.44), ("H2SO4", 98.08), ("NaOH", 40.00), ("CaCO3", 100.09), ("glucose (C6H12O6)", 180.16),
             ("KMnO4", 158.03), ("HCl", 36.46), ("CuSO4", 159.61)]


def fam_moles(rng, difficulty):
    name, M = pick(rng, COMPOUNDS)
    if rng.random() < 0.5:
        m = pick(rng, [5, 10, 12.5, 25, 50, 100])
        q = f"How many moles are in {fmt(m)} g of {name} (molar mass {M} g/mol). {ask_unit(rng, 'mol')}"
        return Problem("moles", q, m / M, "mol", 0.01, [(m, "g"), (M, "g/mol")], 1, signature=f"mol|n|{name}|{m}")
    n = pick(rng, [0.1, 0.25, 0.35, 0.5, 1.2, 2.5])
    q = f"{pick(rng, ASK)} the mass of {fmt(n)} mol of {name} (molar mass {M} g/mol). {ask_unit(rng, 'g')}"
    return Problem("moles", q, n * M, "g", 0.01, [(n, "mol"), (M, "g/mol")], 1, signature=f"mol|m|{name}|{n}")


def fam_solutions(rng, difficulty):
    name, M = pick(rng, COMPOUNDS[:3] + COMPOUNDS[4:])
    if rng.random() < 0.5:
        m = pick(rng, [2, 5, 10, 20, 29.22])
        V = pick(rng, [100, 250, 500, 1000])
        C = m / M / (V / 1000)
        q = (f"{fmt(m)} g of {name} (M = {M} g/mol) is dissolved in water to make {V} mL of solution. {pick(rng, ASK)} the "
             f"molarity. {ask_unit(rng, 'mol/L')}")
        return Problem("solutions", q, C, "mol/L", 0.01, [(m, "g"), (M, "g/mol"), (V, "mL")], 1, signature=f"sol|C|{name}|{m}|{V}")
    C1 = pick(rng, [1, 2, 5, 6, 12])
    C2 = pick(rng, [0.1, 0.2, 0.25, 0.5])
    V2 = pick(rng, [100, 250, 500, 1000])
    V1 = C2 * V2 / C1
    q = (f"What volume of {C1} M stock solution is needed to prepare {V2} mL of {fmt(C2)} M solution by dilution? "
         f"{ask_unit(rng, 'mL')}")
    return Problem("solutions", q, V1, "mL", 0.01, [(C1, "mol/L"), (C2, "mol/L"), (V2, "mL")], 1, signature=f"sol|V|{C1}|{C2}|{V2}")


def fam_ph(rng, difficulty):
    c = pick(rng, [1e-1, 5e-2, 1e-2, 2e-3, 1e-3, 5e-4, 1e-4])
    if rng.random() < 0.6:
        acid = pick(rng, ["HCl", "HNO3"])
        q = f"{pick(rng, ASK)} the pH of a {fmt(c)} M {acid} solution (a strong acid, fully dissociated, 25 °C)."
        return Problem("ph", q, -math.log10(c), "", 0.005, [(c, "mol/L")], 1, signature=f"ph|a|{c}")
    q = f"{pick(rng, ASK)} the pH of a {fmt(c)} M NaOH solution at 25 °C (strong base, pKw = 14)."
    return Problem("ph", q, 14 + math.log10(c), "", 0.005, [(c, "mol/L")], 2, signature=f"ph|b|{c}")


# -- finance / statistics / maths -----------------------------------------

def fam_compound_interest(rng, difficulty):
    P = pick(rng, [10000, 25000, 50000, 100000, 250000])
    r = pick(rng, [5, 6, 6.5, 7, 7.5, 8, 9, 10])
    t = pick(rng, [2, 3, 5, 6, 8, 10])
    comp, n = pick(rng, [("annually", 1), ("quarterly", 4), ("monthly", 12)])
    A = P * (1 + r / 100 / n) ** (n * t)
    cur = pick(rng, ["₹", "$"])
    q = (f"{cur}{P:,} is invested at {fmt(r)}% per year compounded {comp} for {t} years. {pick(rng, ASK)} the final "
         f"amount. Give the amount in {'rupees' if cur == '₹' else 'dollars'}.")
    return Problem("compound_interest", q, A, "money", 0.002, [(P, "money"), (r, "%")], 1 if n == 1 else 2,
                   signature=f"ci|{P}|{r}|{t}|{n}")


def fam_emi(rng, difficulty):
    P = pick(rng, [200000, 500000, 1000000, 2500000])
    r = pick(rng, [7, 8, 8.5, 9, 10.5, 12])
    years = pick(rng, [2, 3, 5, 10, 15, 20])
    i = r / 100 / 12
    n = years * 12
    emi = P * i * (1 + i) ** n / ((1 + i) ** n - 1)
    q = (f"A loan of ₹{P:,} is repaid in equal monthly instalments over {years} years at {fmt(r)}% annual interest, "
         f"compounded monthly. {pick(rng, ASK)} the monthly instalment (EMI). Give the amount in rupees.")
    return Problem("emi", q, emi, "money", 0.002, [(P, "money"), (r, "%")], 2, signature=f"emi|{P}|{r}|{years}")


def fam_statistics(rng, difficulty):
    n = rng.randint(6, 10)
    data = [rng.randint(1, 30) for _ in range(n)]
    which = pick(rng, ["pop_std", "sample_std", "mean", "median"])
    mean = sum(data) / n
    if which == "mean":
        val, text = mean, "the mean"
    elif which == "median":
        s = sorted(data)
        val = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        text = "the median"
    else:
        ddof = 0 if which == "pop_std" else 1
        val = math.sqrt(sum((x - mean) ** 2 for x in data) / (n - ddof))
        text = "the population standard deviation" if ddof == 0 else "the sample standard deviation"
    q = f"{pick(rng, ASK)} {text} of: {', '.join(map(str, data))}."
    return Problem("statistics", q, val, "", 0.005, [(x, "") for x in sorted(set(data))], 1 if which in ("mean", "median") else 2,
                   signature=f"stats|{which}|{data}")


def fam_combinatorics(rng, difficulty):
    which = pick(rng, ["fact", "ncr", "npr"])
    if which == "fact":
        n = rng.randint(10, 20)
        return Problem("combinatorics", f"{pick(rng, ASK)} {n} factorial ({n}!). Give the exact integer.",
                       float(math.factorial(n)), "", 1e-9, [(n, "")], 1, signature=f"comb|f|{n}")
    n = rng.randint(10, 40)
    r = rng.randint(3, min(8, n - 1))
    if which == "ncr":
        return Problem("combinatorics", f"In how many ways can a committee of {r} be chosen from {n} people? Give the exact number.",
                       float(math.comb(n, r)), "", 1e-9, [(n, ""), (r, "")], 1, signature=f"comb|c|{n}|{r}")
    return Problem("combinatorics", f"How many ordered arrangements of {r} items can be taken from {n} distinct items? Give the exact number.",
                   float(math.perm(n, r)), "", 1e-9, [(n, ""), (r, "")], 1, signature=f"comb|p|{n}|{r}")


# -- unit conversion traps --------------------------------------------------

CONVERSIONS = [
    # value choices, from unit text, from canonical, to unit text, to canonical
    ([1.5, 2.5, 69, 110, 200], "GPa", "GPa", "N/mm²", "MPa"),
    ([470, 1000, 2200, 4700, 10000], "µF", "uF", "farads", "F"),
    ([100, 220, 470], "nF", "nF", "µF", "uF"),
    ([1, 2, 3.5], "atm", "atm", "kPa", "kPa"),
    ([1, 2, 5], "atm", "atm", "psi", "psi"),
    ([36, 72, 90, 120], "km/h", "km/h", "m/s", "m/s"),
    ([600, 1450, 1500, 3000], "rpm", "rpm", "rad/s", "rad/s"),
    ([5, 10, 25, 100], "hp", "hp", "kW", "kW"),
    ([1.5, 3.2, 12], "kWh", "kWh", "MJ", "MJ"),
    ([150, 250, 350], "°F", "degF", "°C", "degC"),
    ([2.5, 8, 15], "kN·m", "kN*m", "N·mm", "N*mm"),
    ([30, 45, 90], "psi", "psi", "bar", "bar"),
    ([2, 5, 12], "inches", "inch", "mm", "mm"),
    ([0.5, 2.5, 7.85], "g/cm³", "g/cm^3", "kg/m³", "kg/m^3"),
    ([120, 450, 900], "L/min", "L/min", "m³/h", "m^3/h"),
]


def fam_unit_conversion(rng, difficulty):
    values, ftext, fcanon, ttext, tcanon = pick(rng, CONVERSIONS)
    from neo.units import convert
    v = pick(rng, values)
    ans = convert(v, fcanon, tcanon)
    sci = " in scientific notation" if abs(ans) < 1e-2 or abs(ans) >= 1e6 else ""
    q = f"Convert {fmt(v)} {ftext} to {ttext}{sci}."
    return Problem("unit_conversion", q, ans, tcanon, 0.005, [], 1, signature=f"conv|{fcanon}|{tcanon}|{v}")


def fam_density(rng, difficulty):
    mat, rho = pick(rng, [("steel", 7.85), ("aluminium", 2.70), ("copper", 8.96), ("brass", 8.50), ("titanium", 4.43)])
    a, b, c = pick(rng, [10, 20, 25]), pick(rng, [20, 30, 40]), pick(rng, [5, 10, 15])
    vol_cm3 = a * b * c / 1000
    m = round(rho * vol_cm3, 1)
    dens = m / vol_cm3
    q = (f"A rectangular block measures {a} × {b} × {c} mm and has a mass of {fmt(m)} g. {pick(rng, ASK)} its density. "
         f"{ask_unit(rng, 'g/cm³')}")
    return Problem("density", q, dens, "g/cm^3", 0.01, [(a, "mm"), (b, "mm"), (c, "mm"), (m, "g")], 1,
                   signature=f"dens|{mat}|{a}|{b}|{c}", meta={"material": mat})


FAMILIES: dict[str, Callable] = {
    "beam_deflection": fam_beam_deflection, "beam_stress": fam_beam_stress, "axial": fam_axial,
    "thermal": fam_thermal, "torsion": fam_torsion, "pressure_vessel": fam_pressure_vessel,
    "euler_buckling": fam_euler_buckling, "springs": fam_springs, "bolt_shear": fam_bolt_shear,
    "factor_of_safety": fam_factor_of_safety, "reynolds": fam_reynolds, "pipe_friction": fam_pipe_friction,
    "continuity": fam_continuity, "torricelli": fam_torricelli, "hydrostatic": fam_hydrostatic,
    "buoyancy": fam_buoyancy, "pump_power": fam_pump_power, "drag": fam_drag,
    "wall_conduction": fam_wall_conduction, "cylinder_conduction": fam_cylinder_conduction,
    "convection": fam_convection, "radiation": fam_radiation, "sensible_heat": fam_sensible_heat,
    "ideal_gas": fam_ideal_gas, "carnot": fam_carnot, "lmtd": fam_lmtd, "ohm_power": fam_ohm_power,
    "resistor_network": fam_resistor_network, "voltage_divider": fam_voltage_divider, "rc": fam_rc,
    "lc": fam_lc, "stored_energy": fam_stored_energy, "impedance": fam_impedance, "three_phase": fam_three_phase,
    "battery": fam_battery, "led_resistor": fam_led_resistor, "projectile": fam_projectile,
    "free_fall": fam_free_fall, "circular": fam_circular, "oscillation": fam_oscillation,
    "braking": fam_braking, "collision": fam_collision, "lifting": fam_lifting,
    "kinetic_energy": fam_kinetic_energy, "gears": fam_gears, "moles": fam_moles, "solutions": fam_solutions,
    "ph": fam_ph, "compound_interest": fam_compound_interest, "emi": fam_emi, "statistics": fam_statistics,
    "combinatorics": fam_combinatorics, "unit_conversion": fam_unit_conversion, "density": fam_density,
}

# Never generated for training. These measure transfer, not recall: if the
# policy only memorised templates, it will fall over on these.
HELDOUT_FAMILIES = frozenset({"euler_buckling", "lmtd", "three_phase", "radiation", "emi", "circular"})

# Families whose main failure is a recalled formula with the wrong case get
# extra weight in training — the cantilever/simply-supported mix-up is the
# single best-documented bug in this model.
FAMILY_WEIGHTS = {"beam_deflection": 4.0, "beam_stress": 3.0, "pipe_friction": 1.5, "torsion": 1.5,
                  "wall_conduction": 1.5, "unit_conversion": 2.0, "projectile": 1.5}


def _degenerate(p: Problem) -> bool:
    """True when the answer coincides (within 2%) with a stated input of the
    same kind — e.g. a 1000 kg body hitting a 1 kg one. Grading cannot tell an
    answer from a restated given there, so such problems are not used."""
    from neo.units import dimension, lookup, to_si
    unit = lookup(p.unit)
    if unit is None:
        return True
    target_si, dim = to_si(p.answer, unit), dimension(unit)
    for value, given_unit in p.givens:
        g = lookup(given_unit or "")
        if g is None:
            continue
        if dimension(g) == dim and math.isclose(to_si(float(value), g), target_si, rel_tol=0.02):
            return True
        if dim == "dimensionless" and math.isclose(float(value), p.answer, rel_tol=0.02):
            return True
    return False


def generate(n: int, seed: int, split: str = "train", difficulty_weights=(0.25, 0.45, 0.30),
             exclude_signatures: Optional[set] = None, families: Optional[list] = None) -> list[Problem]:
    """`n` problems. split="train" never emits HELDOUT_FAMILIES; "heldout" emits only those;
    "eval" emits every family (in-distribution and held-out)."""
    rng = random.Random(seed)
    if families is None:
        if split == "train":
            families = [f for f in FAMILIES if f not in HELDOUT_FAMILIES]
        elif split == "heldout":
            families = sorted(HELDOUT_FAMILIES)
        else:
            families = list(FAMILIES)
    weights = [FAMILY_WEIGHTS.get(f, 1.0) if split == "train" else 1.0 for f in families]
    exclude = set(exclude_signatures or ())
    seen: set = set()
    out: list[Problem] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        fam = rng.choices(families, weights=weights)[0]
        difficulty = rng.choices([1, 2, 3], weights=difficulty_weights)[0]
        p = FAMILIES[fam](rng, difficulty)
        if not p.signature:
            p.signature = f"{fam}|{p.question}"
        if p.signature in seen or p.signature in exclude:
            continue
        if not (math.isfinite(p.answer)) or p.answer == 0 or _degenerate(p):
            continue
        seen.add(p.signature)
        out.append(p)
    return out


def iter_problems(seed: int, split: str = "train") -> Iterator[Problem]:
    i = 0
    while True:
        for p in generate(256, seed + i, split):
            yield p
        i += 1
