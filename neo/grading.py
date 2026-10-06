"""Grade a free-text reply against a numeric ground truth.

The reward for the engineering tasks is this file, so it is written to be hard
to game rather than easy to pass:

* A value only counts with a unit of the right dimension, converted to SI
  before comparison. "0.016 mm" for a 16 mm answer is wrong; "1.6 cm" is right.
* What counts as "the answer" is what the reply commits to: the bolded values
  if anything is bolded, otherwise the first and the last candidate value.
  Quantities that merely repeat the problem's givens ("for the 2 m span"),
  thresholds ("turbulent since Re > 4000"), list markers and formula
  coefficients (the 3 in PL³/3EI) are not candidates.
* The last committed value is the answer. If an earlier committed value
  disagrees with it, the score is halved even when the last one is right —
  otherwise RL quickly learns that listing a few guesses beats committing to
  one. "1.2e8 Pa ≈ 120,000 MPa" (a real failure of v3) scores zero.
* A right value with no unit at all is half credit: that is how 1000x errors
  hide in engineering answers.

`grade_lenient` reproduces scripts/bench_numeric.py's older rule (any of the
last eight numbers within tolerance) so reports can show both side by side.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from neo.units import Quantity, dimension, extract_quantities, lookup, to_si


@dataclass
class NumericGrade:
    correct: bool                # the committed answer is right (possibly hedged)
    score: float                 # 1.0 clean, 0.5 hedged or unitless, 0.25 both, 0.0 wrong
    primary: Optional[Quantity]
    hedged: bool = False
    missing_unit: bool = False
    reason: str = ""
    candidates: list[Quantity] = field(default_factory=list)


def _close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


@dataclass(frozen=True)
class _Given:
    written: float      # value as it appears in the problem
    si: float           # value in SI
    dim: str


def parse_givens(givens: Iterable) -> list[_Given]:
    """Givens as (value, unit) pairs, as stated in the problem."""
    out = []
    for item in givens or []:
        try:
            value, unit = item
            canonical = lookup(unit or "")
            if canonical is None:
                continue
            value = float(value)
            out.append(_Given(value, to_si(value, canonical), dimension(canonical)))
        except (TypeError, ValueError, KeyError):
            continue
    return out


def givens_from_text(question: str) -> list[tuple[float, str]]:
    """Every quantity stated in a question, as (value, unit) givens."""
    return [(q.value, q.unit) for q in extract_quantities(question or "")]


def _drop_target(givens: list[_Given], target: float, target_si: float, dim: str) -> list[_Given]:
    """A given equal to the answer cannot be told apart from the answer (unit
    conversions: "convert 2.5 GPa to N/mm²"), so it is not used to exclude."""
    out = []
    for g in givens:
        if g.dim == dim and _close(g.si, target_si, 0.02, 1e-15):
            continue
        if dim == "dimensionless" and _close(g.written, target, 0.02, 1e-15):
            continue
        out.append(g)
    return out


def _is_given(q: Quantity, givens: Sequence[_Given]) -> bool:
    for g in givens:
        if q.unit:
            if q.dim == g.dim and _close(q.si, g.si, 1e-3, 1e-15):
                return True
        else:
            # A bare number equal to a given as written ("998") or in SI
            # ("0.025" for 25 mm) is the given being substituted, not an answer.
            if _close(q.value, g.written, 1e-3, 1e-15) or _close(q.value, g.si, 1e-3, 1e-15):
                return True
    return False


_THRESHOLD_BEFORE = re.compile(
    r"(?:[<>≤≥]=?|\b(?:above|below|than|over|under|exceeds?|beyond|at least|at most|up to|"
    r"limit(?: of)?|threshold(?: of)?|critical(?: value)?(?: of)?))\s*$",
    re.I,
)
_LIST_MARKER_BEFORE = re.compile(r"(?:^|\n)\s*(?:[-*+]\s*)?(?:step\s*)?$", re.I)


_OPERATORS = "+-*/×·^÷"


def _is_noise(q: Quantity) -> bool:
    """Numbers that are clearly not an answer: thresholds, list markers,
    formula coefficients and operands inside an expression."""
    if _THRESHOLD_BEFORE.search(q.before):
        return True        # "turbulent since Re > 4000", "limit of 20 mm"
    before, after = q.before.rstrip(), q.after.lstrip()
    prev, nxt = before[-1:], after[:1]
    # "**" is markdown bold, not multiplication
    prev_op = bool(prev) and prev in _OPERATORS and not before.endswith("**")
    next_op = bool(nxt) and nxt in _OPERATORS and not after.startswith("**")
    if prev_op or next_op:
        return True        # "998 × 2 × 0.025 / 1.002e-3", "1 − Tc/Th": operands, not results
    if q.unit:
        return False
    first, second = q.after[:1], q.after[1:2]
    if first in (".", ")") and second in (" ", "\t") and _LIST_MARKER_BEFORE.search(q.before):
        return True        # "1. " / "2) " at the start of a line
    if first and (first.isalpha() or first == "π"):
        return True        # "3EI", "2πr": a coefficient glued to symbols
    return False


def candidate_quantities(reply: str, target_unit: str, givens: Iterable = (),
                        target: Optional[float] = None) -> tuple[list[Quantity], bool]:
    """Values in `reply` that could be the answer, and whether they are bare
    numbers standing in for a missing unit."""
    canonical = lookup(target_unit or "")
    if canonical is None:
        raise ValueError(f"unknown target unit {target_unit!r}")
    target_dim = dimension(canonical)
    parsed_givens = parse_givens(givens)
    if target is not None:
        parsed_givens = _drop_target(parsed_givens, target, to_si(target, canonical), target_dim)
    quantities = [q for q in extract_quantities(reply or "") if not _is_given(q, parsed_givens)]

    candidates = [q for q in quantities if q.dim == target_dim and not _is_noise(q)]
    if candidates or target_dim == "dimensionless":
        return candidates, False
    bare = [q for q in quantities if q.unit == "" and not _is_noise(q)]
    as_target = [Quantity(q.value, canonical, q.raw, q.start, q.end, q.bold, q.before, q.after) for q in bare]
    return as_target, bool(as_target)


def grade_numeric(
    reply: str,
    target: float,
    unit: Optional[str] = "",
    rel_tol: float = 0.015,
    abs_tol: float = 1e-12,
    givens: Iterable = (),
    allow_unitless: bool = True,
    absolute: bool = False,
) -> NumericGrade:
    """Is the answer `reply` commits to equal to `target` (expressed in `unit`)?

    `absolute=True` compares magnitudes only — for scraped references whose
    sign follows a convention the question does not state ("-21.4 mm" for a
    downward deflection).

    `unit=None` means the reference carries no unit and the question does not
    pin one down (common in scraped datasets): every value counts as a
    candidate and raw values are compared.
    """
    if unit is None:
        return _grade_any_unit(reply, target, rel_tol, abs_tol, givens, absolute)
    canonical = lookup(unit or "")
    if canonical is None:
        raise ValueError(f"unknown target unit {unit!r}")
    target_si = to_si(target, canonical)
    candidates, missing_unit = candidate_quantities(reply, unit, givens, target)
    if missing_unit and not allow_unitless:
        candidates, missing_unit = [], False
    if not candidates:
        return NumericGrade(False, 0.0, None, reason="no candidate value")

    bold = [q for q in candidates if q.bold]
    committed = bold if bold else [candidates[0], candidates[-1]]
    primary = committed[-1]
    consistent = all(_close(q.si, primary.si, rel_tol, abs_tol) for q in committed)
    # The last committed value is the answer; an earlier, different value only
    # costs half the credit (it is a hedge or a stray intermediate).
    value_si = abs(primary.si) if absolute else primary.si
    if not _close(value_si, abs(target_si) if absolute else target_si, rel_tol, abs_tol):
        return NumericGrade(False, 0.0, primary, hedged=not consistent, missing_unit=missing_unit,
                            reason=f"committed {primary.raw!r}", candidates=candidates)
    score, reasons = 1.0, []
    if not consistent:
        score *= 0.5
        reasons.append("hedged between values")
    if missing_unit:
        score *= 0.5
        reasons.append("missing unit")
    return NumericGrade(True, score, primary, hedged=not consistent, missing_unit=missing_unit,
                        reason="; ".join(reasons) or "ok", candidates=candidates)


def _grade_any_unit(reply: str, target: float, rel_tol: float, abs_tol: float, givens: Iterable,
                    absolute: bool = False) -> NumericGrade:
    parsed_givens = [g for g in parse_givens(givens) if not _close(g.written, target, 0.02, 1e-15)]
    candidates = [q for q in extract_quantities(reply or "") if not _is_given(q, parsed_givens) and not _is_noise(q)]
    if not candidates:
        return NumericGrade(False, 0.0, None, reason="no candidate value")
    bold = [q for q in candidates if q.bold]
    committed = bold if bold else [candidates[0], candidates[-1]]
    primary = committed[-1]
    consistent = all(_close(q.value, primary.value, rel_tol, abs_tol) for q in committed)
    value = abs(primary.value) if absolute else primary.value
    if not _close(value, abs(target) if absolute else target, rel_tol, abs_tol):
        return NumericGrade(False, 0.0, primary, hedged=not consistent, reason=f"committed {primary.raw!r}",
                            candidates=candidates)
    score = 1.0 if consistent else 0.5
    return NumericGrade(True, score, primary, hedged=not consistent,
                        reason="ok" if consistent else "hedged between values", candidates=candidates)


_ANY_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def grade_lenient(reply: str, target: float, rel_tol: float = 0.02) -> bool:
    """bench_numeric.py's rule: any of the last 8 bare numbers within tolerance."""
    values = []
    for match in _ANY_NUMBER.finditer(reply or ""):
        try:
            values.append(float(match.group().replace(",", "")))
        except ValueError:
            pass
    return any(target and abs(v - target) / abs(target) <= rel_tol for v in values[-8:])


def value_appears_in(text: str, target: float, unit: str, rel_tol: float = 0.015) -> bool:
    """Does `text` (typically sandbox stdout) contain the target value?

    Accepts the value printed with any unit of the right dimension, or printed
    bare either in the requested unit or in SI. This is the provenance check:
    the number in the answer should have come out of the sandbox, not out of
    the model's head.
    """
    canonical = lookup(unit or "")
    if canonical is None:
        return False
    target_si = to_si(target, canonical)
    dim = dimension(canonical)
    for q in extract_quantities(text or ""):
        if q.unit:
            if q.dim == dim and _close(q.si, target_si, rel_tol, 1e-15):
                return True
        else:
            if _close(q.value, target, rel_tol, 1e-15):
                return True
            if dim != "temperature" and _close(q.value, target_si, rel_tol, 1e-15):
                return True
    return False
