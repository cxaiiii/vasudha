"""Synthesise instruction-following tasks with verifiable constraints.

Base prompts come from an open prompt set; one to three constraints are drawn
from the IFEval instruction registry (lm-eval's copy of Google's reference
checkers) and appended. The same registry then verifies the reply, so the
reward is exact rather than judged.

This is the "IF-RLVR" recipe used for Tulu 3: same constraint *types* as the
IFEval benchmark, entirely different base prompts. Reports should say so —
IFEval gains from this are in-distribution in constraint space.
"""
from __future__ import annotations

import random
from typing import Iterable

# Constraints that are awkward to satisfy jointly with arbitrary prompts or
# that make grading depend on things other than the response.
_SKIP = {"combination:repeat_prompt"}


def _registry():
    from lm_eval.tasks.ifeval import instructions_registry as registry
    return registry


def _conflicts(registry) -> dict:
    conflicts = getattr(registry, "INSTRUCTION_CONFLICTS", {}) or {}
    try:
        return registry.conflict_make(dict(conflicts))
    except Exception:  # noqa: BLE001 - older lm-eval: symmetric closure by hand
        out = {k: set(v) for k, v in conflicts.items()}
        for k, vs in list(out.items()):
            for v in vs:
                out.setdefault(v, set()).add(k)
        return out


def synthesize(base_prompts: Iterable[str], n: int, seed: int, max_constraints: int = 3) -> list[dict]:
    """Returns task dicts: {type: ifeval, prompt_text, instruction_id_list, kwargs}."""
    registry = _registry()
    conflicts = _conflicts(registry)
    ids = [i for i in registry.INSTRUCTION_DICT if i not in _SKIP]
    rng = random.Random(seed)
    state = random.getstate()
    random.seed(seed)           # the registry's builders draw from the global RNG
    out = []
    try:
        prompts = [p for p in base_prompts if p and 20 <= len(p) <= 1500]
        rng.shuffle(prompts)
        for base in prompts:
            if len(out) >= n:
                break
            k = rng.choices([1, 2, 3], weights=[0.45, 0.4, 0.15])[0]
            k = min(k, max_constraints)
            chosen: list[str] = []
            for _ in range(20):
                if len(chosen) == k:
                    break
                cand = rng.choice(ids)
                if cand in chosen or any(cand in conflicts.get(c, set()) or c in conflicts.get(cand, set())
                                         for c in chosen):
                    continue
                if cand.split(":")[0] in {c.split(":")[0] for c in chosen} and cand.startswith(("length_constraints",
                                                                                              "change_case")):
                    continue
                chosen.append(cand)
            descs, kwargs = [], []
            for iid in chosen:
                inst = registry.INSTRUCTION_DICT[iid](iid)
                desc = inst.build_description()
                args = inst.get_instruction_args() or {}
                descs.append(desc)
                kwargs.append({k2: v for k2, v in args.items()})
            prompt_text = base.rstrip() + "\n\n" + " ".join(d.strip() for d in descs)
            out.append({"type": "ifeval", "prompt_text": prompt_text, "instruction_id_list": chosen,
                        "kwargs": kwargs})
    finally:
        random.setstate(state)
    return out
