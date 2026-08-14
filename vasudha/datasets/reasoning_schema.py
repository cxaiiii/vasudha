"""
Schema + prompt templates for the expert-reasoning-trace distillation pipeline.

See C:\\Users\\saxen\\.claude\\plans\\hidden-sleeping-petal.md for the full design
rationale. Short version: each training example is built from three
independently-generated candidate solutions (two from a reasoning-tier
teacher, one deliberately from a cheaper/weaker model so its flaws are real
rather than fabricated) plus a blind judge pass that critiques all three, not
just the winner. Kept as plain dataclasses + to_dict/from_dict so records
round-trip cleanly through the JSONL checkpoint in
scripts/generate_reasoning_dataset.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# Weights are relative, not percentages of a fixed total — consumed by the
# same largest-remainder _quotas() logic scripts/prepare_dataset.py already
# uses, so exact values don't need to sum to 100.
DOMAIN_PROFILES: dict[str, float] = {
    "physics_derivation": 0.16,
    "engineering_design": 0.12,
    "programming_reasoning": 0.12,
    "mathematics": 0.08,
    "debugging": 0.08,
    "scientific_critique": 0.08,
    "simulation": 0.08,
    "electronics": 0.04,
    "chemistry": 0.04,
    # Added for the bio/metallurgy/CAD-adjacent push: no ready-made dataset
    # exists for any of these (checked live — real search turned up general
    # biomedical QA, not bioengineering; metallurgy has real external data
    # like Materials Project, but nothing in a format this pipeline can
    # ingest directly), so they're generated the same way as every other
    # domain here — teacher-generated problems, not pulled from an external
    # source. cad_design is the parametric/geometric-reasoning half of "CAD"
    # (dimensions, tolerances, constraints, manufacturability) — this text
    # pipeline can't produce real CAD geometry files (that's Text2CAD/
    # DeepCAD territory, a wholly different data shape), so it's scoped to
    # what a reasoning-trace record actually can teach.
    "bioengineering": 0.07,
    "metallurgy": 0.07,
    "cad_design": 0.06,
}

# Short natural-language description used to seed problem generation per
# domain — deliberately concrete ("aerospace structural design under a
# temperature constraint") rather than generic ("engineering") so the
# teacher doesn't default to trivial/textbook problems.
DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "physics_derivation": "a physics problem requiring a multi-step derivation from first principles (mechanics, thermodynamics, waves, electromagnetism, or orbital mechanics)",
    "engineering_design": "a real engineering design problem with explicit constraints (material selection, structural, thermal, or mechanical design under stated operating conditions)",
    "programming_reasoning": "a non-trivial algorithm design or optimization problem requiring reasoning about complexity and correctness, not just syntax",
    "mathematics": "a multi-step applied mathematics problem (calculus, linear algebra, differential equations, or numerical methods) grounded in a physical or engineering context",
    "debugging": "a realistic buggy code snippet (with the bug and symptom described) that requires root-cause reasoning to fix, not just a syntax error",
    "scientific_critique": "a plausible-sounding but flawed scientific or engineering claim that requires critical evaluation to identify the flaw",
    "simulation": "a numerical simulation setup problem (specifying governing equations, boundary conditions, and assumptions) for a physical system",
    "electronics": "a circuit analysis or design problem with explicit component values and operating requirements",
    "chemistry": "a quantitative chemistry problem (stoichiometry, thermochemistry, reaction kinetics, or materials chemistry) grounded in a practical context",
    "bioengineering": "a quantitative bioengineering problem with concrete numbers (drug dosage/infusion-rate calculation, bioreactor scale-up, scaffold porosity and mechanical properties for tissue engineering, biomechanical load analysis, or a biomedical device design constraint) — not general biology trivia",
    "metallurgy": "a quantitative metallurgy/materials-science problem with concrete numbers (alloy composition and phase-diagram reasoning, heat-treatment schedule and resulting hardness/strength, failure analysis via a real material-property calculation, or corrosion-rate estimation)",
    "cad_design": "a parametric mechanical CAD design problem with concrete dimensions, tolerances, and constraints (fit/clearance calculation between mating parts, stack-up tolerance analysis, a part dimensioned to satisfy a stated load or manufacturability constraint) — describe geometry precisely in words/numbers, not as an image",
}

SCHEMA_VERSION = "1.0"


@dataclass
class Candidate:
    """One independently-generated attempt at solving the instruction."""
    source: str  # "pro_a" | "pro_b" | "flash_c"
    # Which actual model produced this candidate (e.g. "deepseek-v4-pro" or
    # "kimi-k3") — tracked per-candidate, not assumed uniform per role,
    # since candidate B can come from a different provider entirely when
    # diversifying reasoning lineage rather than just sampling the same
    # model at a different temperature.
    model: str = ""
    concepts: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    governing_relations: list[str] = field(default_factory=list)
    tradeoffs: list[str] = field(default_factory=list)
    final_answer_text: str = ""
    # Machine-parseable value (e.g. "42.3 MPa") kept separate from the prose
    # answer so verification doesn't have to regex it out — None when the
    # problem has no single checkable value.
    final_answer_value: Optional[str] = None
    # Python snippet deriving final_answer_value from the problem's given
    # quantities (not from the candidate's own reasoning) — the ground-truth
    # oracle for reasoning_verify.py, not a restatement of the candidate's
    # work. None when not numerically computable.
    reference_calculation: Optional[str] = None
    # Complete, executable solution code — populated only for
    # programming_reasoning/debugging domains. final_answer_text stays prose
    # (a summary of the fix/approach) for every domain including these two;
    # this is the separate field reasoning_verify.verify_candidate_code
    # actually runs, so a prose answer never gets mistaken for a script.
    code: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CandidateCritique:
    position: int  # 1-3, the randomized slot this candidate was shown in
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    flaws: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateCritique":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class JudgeVerdict:
    winner_position: int  # 1-3, position in the randomized order shown to the judge
    # order_map[i] is the real candidate.source that sat at position i+1 —
    # what lets a later audit check for position bias (does the judge favor
    # position 1 regardless of content?).
    order_map: list[str] = field(default_factory=list)
    justification: str = ""
    critiques: list[CandidateCritique] = field(default_factory=list)

    def winner_source(self) -> str:
        return self.order_map[self.winner_position - 1]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "JudgeVerdict":
        critiques = [CandidateCritique.from_dict(c) for c in d.get("critiques", [])]
        return cls(
            winner_position=d["winner_position"],
            order_map=d.get("order_map", []),
            justification=d.get("justification", ""),
            critiques=critiques,
        )


@dataclass
class VerificationResult:
    method: str = "none"  # "numeric" | "code_tests" | "none"
    ran: bool = False
    passed: Optional[bool] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VerificationResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReasoningRecord:
    domain: str
    instruction: str
    candidates: list[Candidate]  # exactly 3, generation order [pro_a, pro_b, flash_c]
    judge: JudgeVerdict
    reference_check_method: str = "none"  # "numeric" | "code_tests" | "none"
    verification: Optional[VerificationResult] = None
    # Per-candidate provenance lives on Candidate.model now (candidate B can
    # be a different provider than A/C when diversifying reasoning
    # lineage) — the judge isn't a Candidate, so it gets its own field here.
    judge_model: str = ""
    schema_version: str = SCHEMA_VERSION
    generated_at: str = ""  # ISO 8601 timestamp
    token_estimate: Optional[int] = None

    def winning_candidate(self) -> Candidate:
        winner_source = self.judge.winner_source()
        for c in self.candidates:
            if c.source == winner_source:
                return c
        raise ValueError(f"Judge winner_source {winner_source!r} not found among candidates")

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "instruction": self.instruction,
            "candidates": [c.to_dict() for c in self.candidates],
            "judge": self.judge.to_dict(),
            "reference_check_method": self.reference_check_method,
            "verification": self.verification.to_dict() if self.verification else None,
            "judge_model": self.judge_model,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "token_estimate": self.token_estimate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReasoningRecord":
        verification = d.get("verification")
        return cls(
            domain=d["domain"],
            instruction=d["instruction"],
            candidates=[Candidate.from_dict(c) for c in d["candidates"]],
            judge=JudgeVerdict.from_dict(d["judge"]),
            reference_check_method=d.get("reference_check_method", "none"),
            verification=VerificationResult.from_dict(verification) if verification else None,
            judge_model=d.get("judge_model", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            generated_at=d.get("generated_at", ""),
            token_estimate=d.get("token_estimate"),
        )


# ── Prompt templates ─────────────────────────────────────────────────────

PROBLEM_PROMPT_TEMPLATE = """Generate ONE realistic, specific, non-trivial problem: {domain_description}.

Requirements:
- Give concrete numbers/quantities where relevant (not "a beam of some length" but "a 2m steel beam under a 500N load").
- A practicing engineer or scientist should find this plausible, not a generic textbook flashcard.
- Return ONLY the problem statement as plain text. No preamble, no solution, no markdown formatting."""

SOLUTION_PROMPT_TEMPLATE = """Problem: {instruction}

Solve this rigorously and respond with ONLY valid JSON (no markdown fences, no commentary) matching exactly this schema:
{{
  "concepts": ["relevant concept 1", "relevant concept 2", ...],
  "constraints": ["constraint 1", "constraint 2", ...],
  "governing_relations": ["equation or principle 1", ...],
  "tradeoffs": ["tradeoff or limitation 1", ...],
  "final_answer_text": "concise prose answer — a summary/explanation, never raw code even for programming problems",
  "final_answer_value": "machine-parseable value with units, e.g. '42.3 MPa', or null if not applicable",
  "reference_calculation": "a standalone Python snippet that computes final_answer_value purely from the problem's given quantities (not from your reasoning above) — include print(result) as the last line. Use real newline characters (\\n) between statements for any loop, def, or multi-step logic — do NOT chain control-flow statements (while/for/def) together with semicolons on one line, that is invalid Python. Null if the problem has no single numeric answer.",
  "code": "complete, executable solution code, ONLY if the problem is a programming/debugging problem — otherwise null"
}}
{strategy_hint}"""

STRATEGY_HINTS: dict[str, str] = {
    "pro_a": "Use a careful, first-principles approach — derive from fundamentals rather than pattern-matching to a remembered formula.",
    "pro_b": "",
    "flash_c": "",
}

JUDGE_PROMPT_TEMPLATE = """Problem: {instruction}

Three independent candidate solutions follow, labeled by position only — you do not know which model produced which, and should not assume any of them is more likely correct than another.

Position 1:
{candidate_1_json}

Position 2:
{candidate_2_json}

Position 3:
{candidate_3_json}

Evaluate all three on correctness, rigor, and completeness. Respond with ONLY valid JSON (no markdown fences, no commentary):
{{
  "winner": 1,
  "justification": "why the winner is best, referencing specific reasoning steps",
  "critiques": [
    {{"position": 1, "summary": "one-line assessment", "strengths": ["..."], "flaws": ["..."]}},
    {{"position": 2, "summary": "...", "strengths": ["..."], "flaws": ["..."]}},
    {{"position": 3, "summary": "...", "strengths": ["..."], "flaws": ["..."]}}
  ]
}}"""


def domain_description(domain: str) -> str:
    if domain not in DOMAIN_DESCRIPTIONS:
        raise ValueError(f"Unknown domain {domain!r}. Known domains: {sorted(DOMAIN_DESCRIPTIONS)}")
    return DOMAIN_DESCRIPTIONS[domain]
