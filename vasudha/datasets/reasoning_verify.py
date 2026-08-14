"""
Mechanical verification for generated reasoning records — real execution,
not another LLM's subjective judgment, wherever a claim is checkable.

Reuses web/tools.py's SandboxedCodeExecutor as-is (temp-dir isolation,
stripped env vars, timeout, cleanup) rather than reimplementing sandboxing.
That module lives under web/, which has no __init__.py (confirmed), so it's
imported the same way scripts/prepare_dataset.py already reaches into
vasudha/ from scripts/ — via an explicit sys.path insertion, not a package
import.

Scope note: `reference_calculation` is generated per-candidate (each
candidate proposes its own derivation alongside its answer), so this checks
per-candidate self-consistency — does a candidate's own supporting
calculation actually produce the value it claims? — rather than an
independent third-party ground-truth oracle. That's a real, useful signal on
its own: it's exactly the failure mode caught live during this project's own
testing, where a model's printed code output (19) didn't match the number it
asserted in prose (28). A fully independent reference-oracle call is a
natural future upgrade, not required for the pilot this module supports.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from web.tools import SandboxedCodeExecutor  # noqa: E402
from vasudha.datasets.reasoning_schema import Candidate, ReasoningRecord, VerificationResult  # noqa: E402

_executor = SandboxedCodeExecutor(timeout=10)

_NUMBER_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")


def _extract_stdout(raw_result: str) -> Optional[str]:
    """Pull the raw stdout text back out of SandboxedCodeExecutor's
    formatted return string ("[Stdout]:\\n...\\n\\n[Finished in Xs]")."""
    if not raw_result.startswith("[Stdout]:"):
        return None
    body = raw_result[len("[Stdout]:"):]
    # Stop at the next bracketed section header ([Stderr / Errors] or [Finished ...])
    end_match = re.search(r"\n\n\[(?:Stderr / Errors|Finished)", body)
    return (body[:end_match.start()] if end_match else body).strip()


def _extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    match = _NUMBER_RE.search(text)
    return float(match.group()) if match else None


def _head_tail(text: str, head: int = 150, tail: int = 350) -> str:
    """Python tracebacks put the actual exception type/message at the end —
    a plain text[:N] cap can bury it behind a long echoed source line (e.g.
    a candidate's semicolon-chained one-liner), showing everything except
    the one thing needed to diagnose the failure."""
    if len(text) <= head + tail:
        return text
    return f"{text[:head]} ... {text[-tail:]}"


def verify_candidate_numeric(candidate: Candidate, tolerance: float = 0.05) -> VerificationResult:
    """Runs a candidate's own reference_calculation and checks it actually
    produces the value the candidate claims in final_answer_value. tolerance
    is a relative fraction (0.05 = 5%)."""
    if not candidate.reference_calculation:
        return VerificationResult(method="numeric", ran=False, passed=None,
                                   detail="No reference_calculation provided.")

    claimed = _extract_number(candidate.final_answer_value)
    if claimed is None:
        return VerificationResult(method="numeric", ran=False, passed=None,
                                   detail="final_answer_value has no parseable number.")

    raw = _executor.execute_python(candidate.reference_calculation)
    stdout = _extract_stdout(raw)
    computed = _extract_number(stdout)

    if computed is None:
        return VerificationResult(method="numeric", ran=True, passed=None,
                                   detail=f"reference_calculation produced no parseable number. Raw: {_head_tail(raw)}")

    passed = abs(computed - claimed) <= tolerance * max(abs(claimed), 1e-9)
    return VerificationResult(
        method="numeric", ran=True, passed=passed,
        detail=f"claimed={claimed}, computed={computed}, tolerance={tolerance:.0%}",
    )


def verify_candidate_code(candidate_code: str, test_cases: list[str]) -> VerificationResult:
    """Runs candidate_code followed by a list of assert-statement test cases
    in one script. Passes only if every assertion holds and nothing else
    raises."""
    if not test_cases:
        return VerificationResult(method="code_tests", ran=False, passed=None,
                                   detail="No test cases provided.")

    script = candidate_code.strip() + "\n\n" + "\n".join(test_cases) + "\nprint('__ALL_TESTS_PASSED__')"
    raw = _executor.execute_python(script)
    stdout = _extract_stdout(raw) or ""
    passed = "__ALL_TESTS_PASSED__" in stdout and "[Stderr" not in raw and "[Error" not in raw
    return VerificationResult(method="code_tests", ran=True, passed=passed, detail=_head_tail(raw, head=150, tail=350))


def verify_record(record: ReasoningRecord, test_cases: Optional[list[str]] = None) -> VerificationResult:
    """Top-level entry point: verifies the judge's winning candidate
    (that's the one that actually becomes the training example), dispatching
    on record.reference_check_method. Returns a VerificationResult ready to
    attach to record.verification — never raises; a failure to verify is
    itself a valid (method, ran=False) result, not an exception."""
    method = record.reference_check_method
    try:
        winner = record.winning_candidate()
    except ValueError as e:
        return VerificationResult(method=method, ran=False, passed=None, detail=str(e))

    if method == "numeric":
        return verify_candidate_numeric(winner)
    if method == "code_tests":
        return verify_candidate_code(winner.code or "", test_cases or [])
    return VerificationResult(method="none", ran=False, passed=None, detail="No mechanical check for this domain.")
