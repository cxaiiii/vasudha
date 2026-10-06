"""Build the two training sets: distillation prompts and verifiable RL tasks.

Nothing here contains a model-written answer to imitate. Stage 1 needs only
prompts (the student writes its own completions and the teacher scores them);
stage 2 needs prompts plus a way to check the answer.

Row formats (JSONL / Arrow):

  OPD row: {id, source, prompt}
  RL row:  {id, source, prompt, environment, task}
           prompt      — chat messages (optional system + user)
           environment — "text" (no tools) or a python-tool variant (neo.envs)
           task        — JSON string: {"type": ..., verification fields}

Tools for function-calling tasks are rendered into the system message with
the model's own template text (neo.toolcalls.render_tools_system), so each
example can carry its own tool set while the trainer uses one template.

Contamination guards: engineering eval problems are generated first and their
signatures excluded from training; the classic Vasudha sheet is excluded by
its key numbers; MATH-500 problems are drawn from MATH's test split, and only
the train split is used here; IFEval-style constraints are attached to
UltraFeedback prompts, never IFEval's own.
"""
from __future__ import annotations

import ast
import json
import random
import re
from typing import Any, Callable, Iterable, Optional

from neo.engineering import Problem, generate
from neo.grading import givens_from_text
from neo.envs import PYTHON_ENVS, python_tool_schema
from neo.toolcalls import render_tools_system, to_openai_tool
from neo.units import extract_quantities, lookup

TRAIN_SEED = 20261006
EVAL_SEED = 777

GENERIC_SYSTEMS = [
    "You are a helpful assistant.",
    "You are Vasudha, a helpful AI assistant.",
    "You are a precise engineering assistant. Show the key steps and give the final answer with its unit.",
    "You are a knowledgeable assistant. Be accurate and concise.",
    "You are an expert problem solver. Think carefully and answer correctly.",
]

MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def pick_system(rng: random.Random, p_none: float = 0.7) -> Optional[str]:
    return None if rng.random() < p_none else rng.choice(GENERIC_SYSTEMS)


def make_prompt(user: str, system: Optional[str] = None) -> list[dict]:
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": user}]


def rl_row(id_: str, source: str, user: str, task: dict, environment: str = "text",
           system: Optional[str] = None) -> dict:
    return {"id": id_, "source": source, "prompt": make_prompt(user, system), "environment": environment,
            "task": json.dumps(task, ensure_ascii=False)}


def _load(name: str, config: Optional[str] = None, split: str = "train"):
    from datasets import load_dataset
    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def _listish(value: Any) -> Any:
    """HF viewer-style stringified lists/dicts -> Python objects."""
    if isinstance(value, str) and value[:1] in "[{":
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(value)
            except (ValueError, SyntaxError):
                continue
    return value


# ---------------------------------------------------------------------------
# engineering
# ---------------------------------------------------------------------------

# The classic Vasudha sheet (docs/EVALUATION.md). Training problems that
# contain all of an item's key numbers are dropped, so the sheet stays held out.
CLASSIC_KEYS = [
    ["2 m", "50 mm", "100 mm", "5 kN"], ["12 mm", "25 kN"], ["998", "1.002", "2 m/s", "25 mm"],
    ["0.2 m", "0.8 W", "10 m²", "30 °C", "5 °C"], ["2 kg", "20 °C", "80 °C"], ["25 m", "40 K"],
    ["4.7 kΩ", "100 nF"], ["10 mH", "100 nF"], ["330 Ω", "470 Ω"], ["470 Ω", "12 V"], ["25 g", "NaCl"],
    ["0.35 mol", "H2SO4"], ["30 m/s", "40°"], ["50,000", "7.5%", "6 years"], ["2, 4, 4, 4, 5, 5, 7, 9"],
    ["3 m below"], ["2.5 GPa"], ["4700 µF"], ["15 factorial"], ["20 × 30 × 5"],
]


def overlaps_classic(question: str) -> bool:
    return any(all(k in question for k in keys) for keys in CLASSIC_KEYS)


def drop_classic(rows: list[dict]) -> list[dict]:
    """Rows whose user prompt does not overlap the classic sheet (checked on
    every source, not just generated problems: the population-SD list, for
    one, is a textbook example that turns up in scraped data)."""
    return [r for r in rows if not overlaps_classic(str(r["prompt"][-1].get("content") or ""))]


def numeric_task(p: Problem) -> dict:
    return {"type": "numeric", "answer": p.answer, "unit": p.unit, "tol": p.tol, "givens": p.givens,
            "family": p.family, "difficulty": p.difficulty}


def eng_eval_problems(n: int = 400) -> list[Problem]:
    return generate(n, seed=EVAL_SEED, split="eval")


def eng_rows(n: int, seed: int, p_python: float = 0.5, exclude: Optional[set] = None) -> list[dict]:
    rng = random.Random(seed)
    problems = [p for p in generate(int(n * 1.2) + 10, seed=seed, split="train", exclude_signatures=exclude)
                if not overlaps_classic(p.question)][:n]
    rows = []
    for i, p in enumerate(problems):
        env = rng.choice(PYTHON_ENVS) if rng.random() < p_python else "text"
        rows.append(rl_row(f"eng-{seed}-{i}", f"eng/{p.family}", p.question, numeric_task(p), env, pick_system(rng)))
    return rows


# ---------------------------------------------------------------------------
# WebInstruct-verified: real physics / chemistry / finance problems
# ---------------------------------------------------------------------------

WEBINSTRUCT_CATEGORIES = {"Physics", "Chemistry", "Engineering", "Finance", "Economics", "Business",
                          "Other STEM", "Mathematics", "Biology"}
_AMBIGUOUS_UNIT_TEXT = re.compile(r"^\s*[-+$₹€]?[\d.,eE+\-×^ ]+\s*(C|F)\s*\.?\s*$")


def parse_reference(answer: str) -> Optional[tuple[float, str]]:
    """'$12,672.96' -> (12672.96, 'money'); '30.5 kPa' -> (30.5, 'kPa');
    '3630' -> (3630, '*'); anything with extra words -> None."""
    text = (answer or "").strip().rstrip(".")
    if not text or len(text) > 40 or _AMBIGUOUS_UNIT_TEXT.match(text):
        return None
    qs = extract_quantities(text)
    if len(qs) != 1:
        return None
    q = qs[0]
    leftover = (text[:q.start] + text[q.end:]).strip(" .,")
    if leftover:
        return None
    unit = q.unit if q.unit else "*"
    return q.value, unit


def webinstruct_rows(n: int, seed: int, split: str = "train", p_python: float = 0.4) -> list[dict]:
    ds = _load("TIGER-Lab/WebInstruct-verified", split=split)
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    rows = []
    for i in idx:
        if len(rows) >= n:
            break
        r = ds[i]
        if r["answer_type"] not in ("Float", "Integer", "Percentage") or r["category"] not in WEBINSTRUCT_CATEGORIES:
            continue
        if len(r["question"]) > 2500 or "{eq}" in r["question"] and r["question"].count("{eq}") > 12:
            continue
        ref = parse_reference(r["answer"])
        if ref is None or ref[0] == 0:
            continue
        value, unit = ref
        question = r["question"].replace("{eq}", "$").replace("{/eq}", "$")
        task = {"type": "numeric", "answer": value, "unit": unit, "tol": 0.02, "givens": givens_from_text(question),
                "absolute": True, "category": r["category"]}
        env = rng.choice(PYTHON_ENVS) if rng.random() < p_python else "text"
        rows.append(rl_row(f"wi-{split}-{r['id']}", f"webinstruct/{r['category']}", question, task, env,
                           pick_system(rng)))
    return rows


def webinstruct_questions(n: int, seed: int) -> list[list[dict]]:
    """Any answer type — for distillation, where only the question matters."""
    ds = _load("TIGER-Lab/WebInstruct-verified", split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        if len(out) >= n:
            break
        q = ds[i]["question"]
        if 30 <= len(q) <= 2500:
            out.append(make_prompt(q.replace("{eq}", "$").replace("{/eq}", "$"), pick_system(rng)))
    return out


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------

def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} (brace-matched)."""
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def math_rows(n: int, seed: int, p_python: float = 0.3, dapo_frac: float = 0.6) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    dapo = _load("open-r1/DAPO-Math-17k-Processed", "en")
    idx = list(range(len(dapo)))
    rng.shuffle(idx)
    for i in idx[: int(n * dapo_frac)]:
        r = dapo[i]
        env = rng.choice(PYTHON_ENVS) if rng.random() < p_python else "text"
        rows.append(rl_row(f"dapo-{i}", "math/dapo", r["prompt"] + MATH_SUFFIX,
                           {"type": "math", "answer": str(r["solution"])}, env, pick_system(rng)))
    mth = _load("DigitalLearningGmbH/MATH-lighteval", "default", "train")
    idx = [i for i in range(len(mth)) if mth[i]["level"] in ("Level 3", "Level 4", "Level 5")]
    rng.shuffle(idx)
    for i in idx:
        if len(rows) >= n:
            break
        r = mth[i]
        ans = last_boxed(r["solution"])
        if not ans:
            continue
        env = rng.choice(PYTHON_ENVS) if rng.random() < p_python else "text"
        rows.append(rl_row(f"math-{i}", f"math/{r['type']}", r["problem"] + MATH_SUFFIX,
                           {"type": "math", "answer": ans}, env, pick_system(rng)))
    return rows


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------

def _kodcode_prompt(r: dict) -> Optional[str]:
    info = _listish(r.get("test_info"))
    decl = ""
    if isinstance(info, list) and info and isinstance(info[0], dict):
        decl = info[0].get("function_declaration", "")
    if not decl:
        return None
    return (f"{r['question'].strip()}\n\nImplement it in Python as `{decl.strip().rstrip(':')}` and give the complete "
            "solution in a single ```python code block.")


def code_rows(n: int, seed: int, validate: bool = True, workers: int = 8) -> list[dict]:
    ds = _load("KodCode/KodCode-Light-RL-10K")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    cands = []
    for i in idx:
        r = ds[i]
        prompt = _kodcode_prompt(r)
        if prompt is None or len(r["test"]) > 6000:
            continue
        cands.append((i, prompt, {"type": "code", "test_style": "kodcode", "tests": r["test"], "timeout": 10},
                      r["solution"]))
        if len(cands) >= int(n * 1.4):
            break
    if validate:
        # Keep only tasks whose reference solution passes our own runner: a
        # task the reference fails would hand out a reward nobody can earn.
        from concurrent.futures import ThreadPoolExecutor

        from neo.sandbox import SandboxPool
        from neo.verify import run_tests

        pool = SandboxPool(workers=workers, timeout=10)
        try:
            with ThreadPoolExecutor(workers) as ex:
                ok = list(ex.map(lambda c: run_tests(c[2], c[3], pool)[0], cands))
        finally:
            pool.close()
        cands = [c for c, good in zip(cands, ok) if good]
    rows = []
    for i, prompt, task, _ in cands[:n]:
        rows.append(rl_row(f"kod-{i}", "code/kodcode", prompt, task, "text", pick_system(rng, 0.8)))
    return rows


# ---------------------------------------------------------------------------
# function calling (xLAM / APIGen) with synthetic irrelevance
# ---------------------------------------------------------------------------

def fc_rows(n: int, seed: int, p_irrelevant: float = 0.15) -> list[dict]:
    ds = _load("argilla/apigen-function-calling")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    pool_tools: list[dict] = []
    rows = []
    for i in idx:
        if len(rows) >= n:
            break
        r = ds[i]
        try:
            tools = [to_openai_tool(t) for t in _listish(r["tools"])]
            gold = _listish(r["answers"])
        except Exception:  # noqa: BLE001
            continue
        if not tools or not isinstance(gold, list) or len(tools) > 6 or len(gold) > 4:
            continue
        if any(not isinstance(g, dict) or "name" not in g for g in gold):
            continue
        pool_tools.extend(tools)
        query = r["query"]
        if rng.random() < p_irrelevant and len(pool_tools) > 50:
            gold_names = {g["name"] for g in gold}
            names_in = {t["function"]["name"] for t in tools}
            others = [t for t in rng.sample(pool_tools, 30)
                      if t["function"]["name"] not in gold_names and t["function"]["name"] not in names_in]
            if not others:
                continue
            tools = others[: rng.randint(1, 3)]
            gold, source = [], "fc/irrelevance"
        else:
            source = "fc/xlam"
        gold = [{"name": g["name"], "arguments": g.get("arguments") or {}} for g in gold]
        system = render_tools_system(tools, pick_system(rng, 0.75))
        rows.append(rl_row(f"fc-{i}", source, query, {"type": "fc", "tools": tools, "gold_calls": gold},
                           "text", system))
    return rows


# ---------------------------------------------------------------------------
# instruction following
# ---------------------------------------------------------------------------

def _ultrafeedback_texts(limit: int, seed: int) -> list[str]:
    ds = _load("trl-lib/ultrafeedback-prompt")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out = []
    for i in idx[: limit * 2]:
        prompt = _listish(ds[i]["prompt"])
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = prompt[0].get("content", "")
        else:
            text = str(prompt)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def ifeval_rows(n: int, seed: int) -> list[dict]:
    from neo.ifsynth import synthesize

    rng = random.Random(seed)
    tasks = synthesize(_ultrafeedback_texts(n * 2, seed), n, seed)
    return [rl_row(f"if-{seed}-{i}", "ifeval/synth", t["prompt_text"], t, "text", pick_system(rng, 0.85))
            for i, t in enumerate(tasks)]


# ---------------------------------------------------------------------------
# assembling the two sets
# ---------------------------------------------------------------------------

RL_MIX = {"eng": 0.28, "webinstruct": 0.12, "math": 0.20, "code": 0.12, "fc": 0.16, "ifeval": 0.12}

OPD_MIX = {"math": 0.18, "code": 0.12, "science": 0.13, "eng": 0.12, "fc": 0.12, "ifeval": 0.10,
           "general": 0.13, "gsm8k": 0.04, "python_available": 0.06}


def _quotas(mix: dict, total: int) -> dict:
    raw = {k: total * v / sum(mix.values()) for k, v in mix.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def build_rl_pool(total: int, seed: int = TRAIN_SEED, mix: dict = RL_MIX,
                  exclude_eng: Optional[set] = None, log: Callable = print) -> list[dict]:
    q = _quotas(mix, total)
    builders = {
        "eng": lambda k: eng_rows(k, seed, exclude=exclude_eng),
        "webinstruct": lambda k: webinstruct_rows(k, seed + 1),
        "math": lambda k: math_rows(k, seed + 2),
        "code": lambda k: code_rows(k, seed + 3),
        "fc": lambda k: fc_rows(k, seed + 4),
        "ifeval": lambda k: ifeval_rows(k, seed + 5),
    }
    rows = []
    for name, k in q.items():
        if k <= 0:
            continue
        got = drop_classic(builders[name](k))
        log(f"  rl/{name}: {len(got)} / {k}")
        rows.extend(got)
    random.Random(seed).shuffle(rows)
    return rows


def _opd_row(id_: str, source: str, prompt: list[dict]) -> dict:
    return {"id": id_, "source": source, "prompt": prompt}


def build_opd_prompts(total: int, seed: int = TRAIN_SEED + 100, mix: dict = OPD_MIX,
                      exclude_eng: Optional[set] = None, log: Callable = print) -> list[dict]:
    """Prompts only. Tasks are borrowed from the RL builders (different seeds,
    so the two stages do not see the same items) and stripped of answers."""
    q = _quotas(mix, total)
    rng = random.Random(seed)
    rows: list[dict] = []

    def from_rl(rl: Iterable[dict], source_prefix: str) -> list[dict]:
        out = []
        for r in rl:
            prompt = r["prompt"]
            if r["environment"] != "text":
                # No tool execution during distillation: the python tool is
                # declared in the system prompt instead, and the episode ends at
                # the student's first turn (a direct answer or a tool call).
                sys_text = prompt[0]["content"] if prompt[0]["role"] == "system" else None
                user = prompt[-1]
                prompt = [{"role": "system", "content": render_tools_system(
                    [python_tool_schema(r["environment"])], sys_text)}, user]
            out.append(_opd_row(r["id"], f"{source_prefix}/{r['source']}", prompt))
        return out

    builders = {
        "math": lambda k: from_rl(math_rows(k, seed + 2, p_python=0.0), "opd"),
        "code": lambda k: from_rl(code_rows(k, seed + 3, validate=False), "opd"),
        "science": lambda k: [_opd_row(f"wiq-{i}", "opd/science", p)
                              for i, p in enumerate(webinstruct_questions(k, seed + 1))],
        "eng": lambda k: from_rl(eng_rows(k, seed, p_python=0.0, exclude=exclude_eng), "opd"),
        "fc": lambda k: from_rl(fc_rows(k, seed + 4), "opd"),
        "ifeval": lambda k: from_rl(ifeval_rows(k, seed + 5), "opd"),
        "general": lambda k: [_opd_row(f"uf-{i}", "opd/general", make_prompt(t, pick_system(rng)))
                              for i, t in enumerate(_ultrafeedback_texts(k, seed + 6))],
        "gsm8k": lambda k: [_opd_row(f"gsm-{i}", "opd/gsm8k", make_prompt(r["question"] + MATH_SUFFIX, pick_system(rng)))
                            for i, r in enumerate(_load("openai/gsm8k", "main", "train").shuffle(seed=seed).select(range(k)))],
        "python_available": lambda k: from_rl(eng_rows(k // 2, seed + 7, p_python=1.0, exclude=exclude_eng)
                                              + math_rows(k - k // 2, seed + 8, p_python=1.0), "opd"),
    }
    for name, k in q.items():
        if k <= 0:
            continue
        got = drop_classic(builders[name](k))
        log(f"  opd/{name}: {len(got)} / {k}")
        rows.extend(got)
    rng.shuffle(rows)
    return rows


def prompt_tokens(tokenizer, prompt: list[dict], tools: Optional[list] = None, thinking: bool = False) -> int:
    """Length of a prompt as the trainer will render it."""
    ids = tokenizer.apply_chat_template(prompt, tools=tools, add_generation_prompt=True, tokenize=True,
                                        enable_thinking=thinking)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return len(ids[0] if ids and isinstance(ids[0], list) else ids)


def write_jsonl(rows: list[dict], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
