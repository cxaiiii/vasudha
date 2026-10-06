"""Evaluation suites. Every suite is harness-free: standard prompts, no system
prompt, tools only where the suite is explicitly about tools.

Suites (name -> what it measures):
  gsm8k, math500, aime25         math (\\boxed{} answers, math-verify)
  humaneval, mbpp                code (sandboxed unit tests)
  ifeval                         instruction following (reference checkers)
  mmlu_pro                       knowledge + reasoning (10-way multiple choice)
  bfcl                           function calling, BFCL v3 AST match
  eng_text / eng_tool            engineering numerics without / with a python tool
                                 (includes HELDOUT families never trained on)
  classic_text / classic_tool    the original Vasudha sheet (docs/EVALUATION.md)
  webinstruct                    held-out real science/finance numeric problems
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

from neo.data import EVAL_SEED, MATH_SUFFIX, _listish, _load, eng_eval_problems, make_prompt, numeric_task, \
    webinstruct_rows
from neo.engineering import HELDOUT_FAMILIES
from neo.grading import givens_from_text
from neo.toolcalls import render_tools_system, to_openai_tool


def _row(id_: str, source: str, user: str, task: dict, environment: str = "text",
         system: Optional[str] = None, prompt: Optional[list] = None) -> dict:
    return {"id": id_, "source": source, "prompt": prompt or make_prompt(user, system), "environment": environment,
            "task": json.dumps(task, ensure_ascii=False)}


# -- the classic sheet: docs/EVALUATION.md sections A and B --------------------

CLASSIC = [
    ("A1", "A steel cantilever beam is 2 m long, 50 mm wide and 100 mm deep, with a 5 kN point load at the free end. E = 200 GPa. Maximum tip deflection in mm?", 16.0, "mm"),
    ("A2", "A steel cantilever beam is 2 m long with a rectangular section 50 mm wide and 100 mm deep, and a 5 kN point load at the free end. Maximum bending stress in MPa?", 120.0, "MPa"),
    ("A3", "A simply supported steel beam has a 2 m span and a rectangular section 50 mm wide and 100 mm deep, with 5 kN at midspan. E = 200 GPa. Deflection at the centre in mm?", 1.0, "mm"),
    ("A4", "A 12 mm diameter steel rod carries 25 kN in tension. Stress in MPa?", 221.0, "MPa"),
    ("A5", "A 12 mm diameter steel rod, 1.5 m long, carries 25 kN in tension. E = 200 GPa. How much does it stretch, in mm?", 1.658, "mm"),
    ("A6", "Water (998 kg/m³, 1.002e-3 Pa·s) flows at 2 m/s through a 25 mm pipe. Reynolds number?", 49800.0, ""),
    ("A7", "Water (998 kg/m³, 1.002e-3 Pa·s) flows at 2 m/s through a 25 mm pipe. Estimate the Darcy friction factor using the Blasius correlation.", 0.02119, ""),
    ("A8", "Water (998 kg/m³, 1.002e-3 Pa·s) flows at 2 m/s through a 25 mm smooth pipe. Using the Blasius correlation, what is the head loss over 10 m of that pipe, in metres?", 1.728, "m"),
    ("A9", "Water discharges from a tank through a small orifice 3 m below the surface. Efflux velocity in m/s?", 7.672, "m/s"),
    ("A10", "A plane wall is 0.2 m thick, k = 0.8 W/(m·K), area 10 m², inner face 30 °C, outer 5 °C, steady state. Heat transfer rate in W?", 1000.0, "W"),
    ("A11", "Energy to heat 2 kg of water from 20 °C to 80 °C, c = 4186 J/(kg·K), in kJ?", 502.3, "kJ"),
    ("A12", "A 25 m steel rail warms by 40 K, α = 12e-6 /K. Expansion in mm?", 12.0, "mm"),
    ("A13", "RC low-pass filter, R = 4.7 kΩ, C = 100 nF. −3 dB cutoff frequency in Hz?", 338.6, "Hz"),
    ("A14", "Resonant frequency of an LC circuit with L = 10 mH and C = 100 nF, in Hz?", 5033.0, "Hz"),
    ("A15", "330 Ω and 470 Ω in parallel — equivalent resistance in ohms?", 193.9, "ohm"),
    ("A16", "Power dissipated in a 470 Ω resistor across 12 V, in watts?", 0.3064, "W"),
    ("A17", "How many moles in 25 g of NaCl (M = 58.44 g/mol)?", 0.4278, "mol"),
    ("A18", "Mass of 0.35 mol of H₂SO₄ (M = 98.08 g/mol), in grams?", 34.33, "g"),
    ("A19", "pH of a 1e-3 M HCl solution?", 3.0, ""),
    ("A20", "A block 20 × 30 × 5 mm has mass 23.4 g. Density in g/cm³?", 7.8, "g/cm^3"),
    ("A21", "₹50,000 invested at 7.5% compounded annually for 6 years — final amount in rupees?", 77165.0, "money"),
    ("A22", "Population standard deviation of 2, 4, 4, 4, 5, 5, 7, 9?", 2.0, ""),
    ("A23", "A projectile is launched at 30 m/s, 40° above horizontal, g = 9.81, no drag. Horizontal range in m?", 90.35, "m"),
    ("A24", "A projectile is launched at 30 m/s, 40° above horizontal, g = 9.81, no drag. Maximum height in m?", 18.953, "m"),
    ("B1", "Convert 2.5 GPa to N/mm².", 2500.0, "MPa"),
    ("B2", "A capacitor is 4700 µF. Express it in farads in scientific notation.", 4.7e-3, "F"),
    ("B3a", "What is a pressure of 1 atm in kPa?", 101.325, "kPa"),
    ("B3b", "What is a pressure of 1 atm in psi?", 14.696, "psi"),
    ("B4", "What is 15 factorial?", 1307674368000.0, ""),
]


def classic(env: str) -> list[dict]:
    rows = []
    for key, q, ans, unit in CLASSIC:
        tol = 1e-9 if key == "B4" else 0.02
        task = {"type": "numeric", "answer": ans, "unit": unit, "tol": tol, "givens": givens_from_text(q)}
        rows.append(_row(f"classic-{key}", f"classic/{key}", q, task, env))
    return rows


def eng(env: str, n: int = 300) -> list[dict]:
    rows = []
    for i, p in enumerate(eng_eval_problems(n)):
        tag = "eng-heldout" if p.family in HELDOUT_FAMILIES else "eng"
        rows.append(_row(f"eng-eval-{i}", f"{tag}/{p.family}", p.question, numeric_task(p), env))
    return rows


def gsm8k(limit: Optional[int] = None) -> list[dict]:
    ds = _load("openai/gsm8k", "main", "test")
    rows = []
    for i, r in enumerate(ds):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        rows.append(_row(f"gsm8k-{i}", "gsm8k", r["question"] + MATH_SUFFIX, {"type": "math", "answer": ans}))
    return rows[:limit] if limit else rows


def math500(limit: Optional[int] = None) -> list[dict]:
    ds = _load("HuggingFaceH4/MATH-500", split="test")
    rows = [_row(f"math500-{i}", f"math500/{r['subject']}", r["problem"] + MATH_SUFFIX,
                 {"type": "math", "answer": r["answer"]}) for i, r in enumerate(ds)]
    return rows[:limit] if limit else rows


def aime25(limit: Optional[int] = None) -> list[dict]:
    ds = _load("math-ai/aime25", split="test")
    rows = [_row(f"aime25-{i}", "aime25", r["problem"] + MATH_SUFFIX, {"type": "math", "answer": str(r["answer"])})
            for i, r in enumerate(ds)]
    return rows[:limit] if limit else rows


def humaneval(limit: Optional[int] = None) -> list[dict]:
    ds = _load("openai/openai_humaneval", split="test")
    rows = []
    for r in ds:
        user = ("Complete the following Python function. Return the complete function in a single ```python code "
                f"block.\n\n```python\n{r['prompt']}```")
        task = {"type": "code", "test_style": "humaneval", "tests": r["test"], "entry_point": r["entry_point"],
                "prompt_code": r["prompt"], "timeout": 10}
        rows.append(_row(r["task_id"].replace("/", "-"), "humaneval", user, task))
    return rows[:limit] if limit else rows


def mbpp(limit: Optional[int] = None) -> list[dict]:
    ds = _load("google-research-datasets/mbpp", "sanitized", "test")
    rows = []
    for r in ds:
        tests = _listish(r["test_list"])
        imports = _listish(r["test_imports"]) or []
        user = (f"{r['prompt']}\nYour code should pass these tests:\n" + "\n".join(tests)
                + "\n\nGive the complete solution in a single ```python code block.")
        task = {"type": "code", "test_style": "asserts", "tests": tests, "setup": "\n".join(imports), "timeout": 10}
        rows.append(_row(f"mbpp-{r['task_id']}", "mbpp", user, task))
    return rows[:limit] if limit else rows


def ifeval(limit: Optional[int] = None) -> list[dict]:
    ds = _load("google/IFEval")
    rows = []
    for r in ds:
        kwargs = _listish(r["kwargs"])
        task = {"type": "ifeval", "prompt_text": r["prompt"], "instruction_id_list": _listish(r["instruction_id_list"]),
                "kwargs": kwargs}
        rows.append(_row(f"ifeval-{r['key']}", "ifeval", r["prompt"], task))
    return rows[:limit] if limit else rows


LETTERS = "ABCDEFGHIJ"


def mmlu_pro(per_category: int = 100, seed: int = EVAL_SEED) -> list[dict]:
    ds = _load("TIGER-Lab/MMLU-Pro", split="test")
    by_cat: dict[str, list[int]] = {}
    for i, cat in enumerate(ds["category"]):
        by_cat.setdefault(cat, []).append(i)
    rng = random.Random(seed)
    rows = []
    for cat, idx in sorted(by_cat.items()):
        rng.shuffle(idx)
        for i in idx[:per_category]:
            r = ds[i]
            options = _listish(r["options"])
            opts = "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(options))
            user = (f"The following is a multiple choice question about {cat}. Think step by step and then finish "
                    f'your answer with "the answer is (X)" where X is the correct letter choice.\n\n'
                    f"Question: {r['question']}\nOptions:\n{opts}")
            rows.append(_row(f"mmlupro-{r['question_id']}", f"mmlu_pro/{cat}", user,
                             {"type": "choice", "answer": r["answer"]}))
    return rows


BFCL_CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance", "live_simple",
                   "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance"]
BFCL_LIMITS = {"live_multiple": 300, "live_irrelevance": 240}


def _bfcl_file(path: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", path, repo_type="dataset")
    with open(local, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def bfcl(categories: list[str] = BFCL_CATEGORIES, seed: int = EVAL_SEED) -> list[dict]:
    rows = []
    rng = random.Random(seed)
    for cat in categories:
        items = _bfcl_file(f"BFCL_v3_{cat}.json")
        answers = {}
        if not cat.endswith("relevance"):
            answers = {a["id"]: a["ground_truth"] for a in _bfcl_file(f"possible_answer/BFCL_v3_{cat}.json")}
        if cat in BFCL_LIMITS and len(items) > BFCL_LIMITS[cat]:
            items = rng.sample(items, BFCL_LIMITS[cat])
        for item in items:
            tools = [to_openai_tool(f) for f in item["function"]]
            turn = item["question"][0]
            system = "\n".join(m["content"] for m in turn if m["role"] == "system") or None
            convo = [m for m in turn if m["role"] != "system"]
            prompt = [{"role": "system", "content": render_tools_system(tools, system)}] + convo
            task = {"type": "fc", "tools": tools, "bfcl_category": cat,
                    "bfcl_ground_truth": answers.get(item["id"])}
            rows.append(_row(item["id"], f"bfcl/{cat}", "", task, prompt=prompt))
    return rows


def webinstruct(n: int = 300) -> list[dict]:
    rows = webinstruct_rows(n, EVAL_SEED, split="test", p_python=0.0)
    for r in rows:
        r["prompt"] = [m for m in r["prompt"] if m["role"] != "system"]
    return rows


SUITES = {
    "gsm8k": lambda limit: gsm8k(limit),
    "math500": lambda limit: math500(limit),
    "aime25": lambda limit: aime25(limit),
    "humaneval": lambda limit: humaneval(limit),
    "mbpp": lambda limit: mbpp(limit),
    "ifeval": lambda limit: ifeval(limit),
    "mmlu_pro": lambda limit: mmlu_pro(per_category=max(1, (limit or 1400) // 14)),
    "bfcl": lambda limit: bfcl()[:limit] if limit else bfcl(),
    "eng_text": lambda limit: eng("text", limit or 300),
    "eng_tool": lambda limit: eng("python", limit or 300),
    "classic_text": lambda limit: classic("text")[:limit] if limit else classic("text"),
    "classic_tool": lambda limit: classic("python")[:limit] if limit else classic("python"),
    "webinstruct": lambda limit: webinstruct(limit or 300),
}

FAST_SUITES = ["gsm8k", "math500", "humaneval", "mbpp", "ifeval", "mmlu_pro", "bfcl", "eng_text", "eng_tool",
               "classic_text", "classic_tool", "webinstruct"]
THINKING_SUITES = ["math500", "aime25", "mmlu_pro", "eng_text", "classic_text", "ifeval"]
THINKING_LIMITS = {"math500": 250, "mmlu_pro": 560}


def load_suite(name: str, limit: Optional[int] = None, cache_dir: Optional[str] = None) -> list[dict]:
    """Rows for a suite, cached as JSONL so every model sees identical items."""
    if cache_dir:
        path = os.path.join(cache_dir, f"{name}{'' if limit is None else f'-{limit}'}.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
    rows = SUITES[name](limit)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows
