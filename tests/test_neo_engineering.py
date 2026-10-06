"""The procedural engineering generator: coverage, ground truth, splits."""
from __future__ import annotations

import math

import pytest

from neo.engineering import FAMILIES, HELDOUT_FAMILIES, fam_beam_deflection, generate
from neo.grading import grade_numeric
from neo.units import lookup


def test_every_family_generates_and_self_grades():
    problems = generate(4000, seed=3, split="eval")
    assert {p.family for p in problems} == set(FAMILIES)
    for p in problems:
        assert lookup(p.unit) is not None
        assert math.isfinite(p.answer) and p.answer != 0
        unit_text = {"": "", "money": "", "deg": "degrees", "degC": "°C", "ohm": "Ω"}.get(p.unit, p.unit)
        value = repr(p.answer) if p.tol < 1e-6 else f"{p.answer:.6g}"
        reply = f"The answer is **{'₹' if p.unit == 'money' else ''}{value} {unit_text}**."
        assert grade_numeric(reply, p.answer, p.unit, p.tol, givens=p.givens).score == 1.0, (p.family, reply)


def test_train_split_never_contains_heldout_families():
    assert not {p.family for p in generate(3000, seed=5, split="train")} & HELDOUT_FAMILIES
    assert {p.family for p in generate(300, seed=5, split="heldout")} <= HELDOUT_FAMILIES


def test_exclusion_by_signature():
    held = generate(200, seed=11, split="eval")
    train = generate(2000, seed=11, split="eval", exclude_signatures={p.signature for p in held})
    assert not {p.signature for p in held} & {p.signature for p in train}


def test_cantilever_matches_hand_calculation():
    # the bug that motivated the beam families: PL^3/(3EI), not PL^3/(48EI)
    import random
    for seed in range(200):
        p = fam_beam_deflection(random.Random(seed), 2)
        if p.meta["case"] == "cant_point" and p.meta["section"] == "rect":
            m = p.meta
            I = m["b"] * m["h"] ** 3 / 12
            expected_mm = m["q"] * m["L"] ** 3 / (3 * m["E"] * I) * 1000
            got_mm = p.answer if p.unit == "mm" else p.answer * 10
            assert got_mm == pytest.approx(expected_mm)
            return
    pytest.fail("no cantilever point-load case generated")


def test_deterministic():
    a = [p.question for p in generate(50, seed=9)]
    b = [p.question for p in generate(50, seed=9)]
    assert a == b
