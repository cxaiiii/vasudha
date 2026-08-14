"""
Dry-run tests for the reasoning-trace distillation pipeline's schema,
verification, and formatting modules — zero API cost, zero network access.
Three hand-written fixtures cover the three reference_check_method shapes
the pipeline actually produces: numeric, code_tests, and none (judge-only).

See C:\\Users\\saxen\\.claude\\plans\\hidden-sleeping-petal.md for the design
this exercises.
"""

from __future__ import annotations

import random

import pytest

from vasudha.datasets.reasoning_schema import (
    Candidate,
    CandidateCritique,
    JudgeVerdict,
    ReasoningRecord,
    DOMAIN_PROFILES,
    domain_description,
)
from vasudha.datasets.reasoning_format import (
    format_main_example,
    format_self_critique_example,
    flatten_record,
    estimate_tokens,
)
from vasudha.datasets.reasoning_verify import (
    verify_candidate_numeric,
    verify_candidate_code,
    verify_record,
)


def _numeric_fixture() -> ReasoningRecord:
    """Physics domain, reference_check_method="numeric". The losing
    candidate mirrors the exact bug caught live during this project's own
    testing: it states an answer that its own supporting arithmetic doesn't
    produce."""
    winner = Candidate(
        source="pro_a",
        concepts=["Doppler effect", "perpendicular-bisector symmetry"],
        constraints=["car speed constant", "far-field approximation"],
        governing_relations=["beat = |f1-f2| * (1 + v_r/c)"],
        tradeoffs=["small-angle approximation near the midpoint"],
        final_answer_text="Statement A is true: Vp+VR = 2*VQ by symmetry.",
        final_answer_value="6",
        reference_calculation="print(3.1515 + 2.8485)",
    )
    loser = Candidate(
        source="flash_c",
        final_answer_text="The beat frequency stays constant at 3 Hz throughout.",
        final_answer_value="3",
        reference_calculation="print(3)",
    )
    judge = JudgeVerdict(
        winner_position=1,
        order_map=["pro_a", "flash_c"],
        justification="pro_a correctly derives the Doppler-scaled beat frequency.",
        critiques=[
            CandidateCritique(position=1, summary="rigorous", strengths=["uses symmetry"], flaws=[]),
            CandidateCritique(
                position=2, summary="wrong",
                flaws=["incorrectly assumes beat frequency is constant; ignores the Doppler shift entirely"],
            ),
        ],
    )
    return ReasoningRecord(
        domain="physics_derivation",
        instruction="Two loudspeakers 20m apart emit 118Hz and 121Hz; a car crosses "
                    "the perpendicular bisector. Which statements about the beat "
                    "frequency are true?",
        candidates=[winner, loser],
        judge=judge,
        reference_check_method="numeric",
    )


def _code_fixture() -> ReasoningRecord:
    """Debugging domain, reference_check_method="code_tests"."""
    winner = Candidate(
        source="pro_a",
        concepts=["off-by-one error"],
        final_answer_text="The loop excluded the last element; fixed the range bound.",
        code="def sum_to_n(n):\n    return sum(range(1, n + 1))",
    )
    loser = Candidate(
        source="flash_c",
        final_answer_text="Left the range as-is, assuming Python's range is inclusive.",
        code="def sum_to_n(n):\n    return sum(range(1, n))",
    )
    judge = JudgeVerdict(
        winner_position=1,
        order_map=["pro_a", "flash_c"],
        justification="pro_a's fix passes the stated test cases.",
        critiques=[
            CandidateCritique(position=1, summary="correct", flaws=[]),
            CandidateCritique(position=2, summary="still off by one", flaws=["range(1, n) excludes n, same bug"]),
        ],
    )
    return ReasoningRecord(
        domain="debugging",
        instruction="Fix sum_to_n(5) which should return 15 but returns 10.",
        candidates=[winner, loser],
        judge=judge,
        reference_check_method="code_tests",
    )


def _judge_only_fixture() -> ReasoningRecord:
    """Scientific-critique domain, reference_check_method="none" — no
    mechanical check exists, judge verdict is the only signal."""
    winner = Candidate(
        source="pro_a",
        concepts=["precipitation hardening", "operating temperature limits"],
        final_answer_text="Al-Mg-Si is unsuitable above ~200C; gamma-TiAl is the better choice near 600C.",
    )
    loser = Candidate(
        source="flash_c",
        final_answer_text="Al-Mg-Si works fine since aluminum alloys are generally lightweight and strong.",
    )
    judge = JudgeVerdict(
        winner_position=1,
        order_map=["pro_a", "flash_c"],
        justification="pro_a correctly identifies the temperature ceiling for precipitation-hardened aluminum.",
        critiques=[
            CandidateCritique(position=1, summary="correct", flaws=[]),
            CandidateCritique(position=2, summary="ignores temperature constraint",
                               flaws=["precipitation strengthening degrades well below 600C; never addresses this"]),
        ],
    )
    return ReasoningRecord(
        domain="scientific_critique",
        instruction="Critique this alloy choice: Al-Mg-Si for a 600C aerospace application.",
        candidates=[winner, loser],
        judge=judge,
        reference_check_method="none",
    )


class TestSchema:
    def test_domain_weights_sum_to_one(self):
        assert sum(DOMAIN_PROFILES.values()) == pytest.approx(1.0)

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError):
            domain_description("not_a_real_domain")

    def test_record_round_trips_through_dict(self):
        record = _numeric_fixture()
        restored = ReasoningRecord.from_dict(record.to_dict())
        assert restored.to_dict() == record.to_dict()

    def test_winning_candidate_resolves_via_judge(self):
        record = _numeric_fixture()
        assert record.winning_candidate().source == "pro_a"


class TestVerification:
    def test_numeric_candidate_self_consistent_passes(self):
        record = _numeric_fixture()
        result = verify_candidate_numeric(record.winning_candidate())
        assert result.ran is True
        assert result.passed is True

    def test_numeric_candidate_catches_mismatch(self):
        """The exact bug class caught live in this project: a candidate's
        own reference calculation disagrees with its stated answer."""
        mismatched = Candidate(source="x", final_answer_value="28", reference_calculation="print(19)")
        result = verify_candidate_numeric(mismatched)
        assert result.ran is True
        assert result.passed is False

    def test_code_candidate_passes_real_tests(self):
        record = _code_fixture()
        result = verify_candidate_code(record.winning_candidate().code, ["assert sum_to_n(5) == 15"])
        assert result.passed is True

    def test_code_candidate_fails_real_tests(self):
        record = _code_fixture()
        loser = record.candidates[1]
        result = verify_candidate_code(loser.code, ["assert sum_to_n(5) == 15"])
        assert result.passed is False

    def test_verify_record_dispatches_on_method(self):
        numeric_result = verify_record(_numeric_fixture())
        assert numeric_result.method == "numeric"

        code_result = verify_record(_code_fixture(), test_cases=["assert sum_to_n(5) == 15"])
        assert code_result.method == "code_tests"
        assert code_result.passed is True

        none_result = verify_record(_judge_only_fixture())
        assert none_result.method == "none"
        assert none_result.ran is False


class TestFormat:
    @pytest.mark.parametrize("fixture_fn", [_numeric_fixture, _code_fixture, _judge_only_fixture])
    def test_main_example_wraps_thinking_in_think_tags(self, fixture_fn):
        record = fixture_fn()
        flat = flatten_record(record, self_critique_ratio=0.0, rng=random.Random(0))
        assert flat is not None
        assert "<think>" in flat["text"]
        assert "</think>" in flat["text"]
        # The visible answer must appear after the closing tag, not buried
        # inside the reasoning block — this is what keeps replies concise.
        think_close = flat["text"].index("</think>")
        answer_start = flat["text"].index(record.winning_candidate().final_answer_text)
        assert answer_start > think_close

    def test_self_critique_example_uses_real_losing_candidate(self):
        record = _numeric_fixture()
        messages = format_self_critique_example(record)
        assert messages is not None
        thinking = messages[1]["thinking"]
        # Must reference the loser's actual (real, not fabricated) answer.
        assert "constant at 3 Hz" in thinking
        assert "Doppler shift entirely" in thinking

    def test_self_critique_falls_back_when_no_flaws_recorded(self):
        record = _numeric_fixture()
        record.judge.critiques[1].flaws = []  # no flaws recorded for the loser
        assert format_self_critique_example(record) is None

    def test_length_cap_rejects_oversized_record(self):
        record = _numeric_fixture()
        assert flatten_record(record, max_tokens=5) is None

    def test_reasonable_record_stays_within_default_cap(self):
        for fixture_fn in (_numeric_fixture, _code_fixture, _judge_only_fixture):
            flat = flatten_record(fixture_fn(), self_critique_ratio=0.0)
            assert flat is not None
            assert estimate_tokens(flat["text"]) < 3800

    def test_main_example_has_no_raw_json_leaking_into_thinking(self):
        """A common failure mode worth guarding against directly: the
        flattener should render prose/labeled lines, not dump the raw
        candidate JSON into the think block."""
        record = _numeric_fixture()
        messages = format_main_example(record)
        thinking = messages[1]["thinking"]
        assert "{" not in thinking and "}" not in thinking
