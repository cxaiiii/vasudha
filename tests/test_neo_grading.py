"""Units and numeric grading — the RL reward for every engineering task."""
from __future__ import annotations

import pytest

from neo.grading import grade_lenient, grade_numeric, givens_from_text, value_appears_in
from neo.units import convert, extract_quantities, lookup


@pytest.mark.parametrize("text, value, unit", [
    ("The maximum tip deflection is **16.0 mm**.", 16.0, "mm"),
    ("₹77,165 after six years", 77165.0, "money"),
    ("Rs. 500", 500.0, "money"),
    ("4.7 × 10⁻³ F", 4.7e-3, "F"),
    ("1.2 × 10^(-3) m", 1.2e-3, "m"),
    ("2500 N/mm²", 2500.0, "MPa"),
    ("1101208 J/(kg · K)", 1101208.0, "J/kg*K"),
    ("k = 0.8 W/(m·K)", 0.8, "W/m*K"),
    ("25 deg C", 25.0, "degC"),
    ("150 N·m", 150.0, "N*m"),
    ("$\\delta = 16\\,\\text{mm}$", 16.0, "mm"),
    ("5 kN(approx.)", 5.0, "kN"),
])
def test_extract(text, value, unit):
    q = extract_quantities(text)[-1]
    assert q.value == pytest.approx(value)
    assert q.unit == unit


def test_words_are_not_units():
    # "steps" must not become seconds, "in this case" must not become inches
    qs = extract_quantities("5 steps; the deflection is 16 in this case")
    assert [q.unit for q in qs] == ["", ""]


def test_currency_prefix_not_swallowed():
    qs = extract_quantities("also $12,672.96 and Rs. 500")
    assert [(q.value, q.unit) for q in qs] == [(12672.96, "money"), (500.0, "money")]


def test_convert():
    assert convert(0.016, "m", "mm") == pytest.approx(16)
    assert convert(25, "degC", "K") == pytest.approx(298.15)
    assert convert(1, "atm", "psi") == pytest.approx(14.696, rel=1e-4)
    assert lookup("mpa") == "MPa" and lookup("mPa") is None


GIVENS = [(2, "m"), (50, "mm"), (100, "mm"), (5, "kN"), (200, "GPa")]


@pytest.mark.parametrize("reply, score", [
    ("The maximum tip deflection is **16.0 mm**.", 1.0),
    ("For a 50 mm × 100 mm section the tip deflection is 16 mm.", 1.0),
    ("Deflection = 0.016 m (16 mm).", 1.0),
    ("The tool printed 0.016, so the deflection is 0.016 mm.", 0.0),      # mislabelled unit: wrong
    ("It is either 1.6 mm or 16 mm.", 0.5),                                # hedged
    ("It is either 16 mm or 1.6 mm.", 0.0),                                # committed to the wrong one
    ("**1.6 mm** ... actually **16 mm**", 0.5),
    ("The deflection is 16.", 0.5),                                        # no unit
    ("**16 mm** - that's a big deflection for a 2 m span.", 1.0),
])
def test_grade_deflection(reply, score):
    assert grade_numeric(reply, 16.0, "mm", 0.015, givens=GIVENS).score == score


def test_v3_unit_failure_scores_zero():
    # observed on v3: correct 1.2e8 Pa, reported as 120,000 MPa
    assert grade_numeric("σ = 1.2e8 Pa ≈ 120,000 MPa", 120.0, "MPa").score == 0.0


def test_dimensionless_with_thresholds_and_operands():
    reply = "Re = 998 × 2 × 0.025 / 1.002e-3 ≈ 49,800, so the flow is turbulent (Re > 4000)."
    givens = [(998, "kg/m^3"), (2, "m/s"), (25, "mm"), (1.002e-3, "Pa*s")]
    assert grade_numeric(reply, 49800, "", givens=givens).score == 1.0
    carnot = "η = 1 − Tc/Th = 1 − 288.15/773.15 = 0.627, i.e. 62.7 %."
    assert grade_numeric(carnot, 62.73, "%", 0.01, givens=[(500, "degC"), (15, "degC")]).score == 1.0


def test_unit_conversion_given_equal_to_answer_is_ignored():
    assert grade_numeric("2.5 GPa = **2500 N/mm²**.", 2500, "MPa", givens=givens_from_text("Convert 2.5 GPa to N/mm².")).score == 1.0


def test_any_unit_mode():
    assert grade_numeric("The share price is **$65.38**.", 65.38, None).score == 1.0
    assert grade_numeric("about 21,500", 21500, None).score == 1.0
    assert grade_numeric("about 21.5", 21500, None).score == 0.0


def test_lenient_and_provenance():
    assert grade_lenient("so 0.016 mm", 0.016)
    assert value_appears_in("16.0 mm\n", 16.0, "mm")
    assert value_appears_in("0.016\n", 16.0, "mm")       # printed in SI
    assert not value_appears_in("1.6\n", 16.0, "mm")
