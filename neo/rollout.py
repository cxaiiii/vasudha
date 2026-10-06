"""Multi-turn rollouts for evaluation and difficulty calibration.

A standalone agent loop, independent of any application: render the
conversation with the model's own chat template (tools via `tools=`), sample,
parse Qwen3.5 tool calls from the raw text, run python calls in the sandbox,
append the results, repeat. Function-calling tasks stop after the first turn:
their calls are graded, not executed.

Two generation backends share one interface:
  * VLLMBackend — the real thing, batched across every active episode;
  * HFBackend   — plain transformers, for CPU tests of the loop itself.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from neo.envs import PYTHON_ENVS, python_tool_schema
from neo.toolcalls import parse_reply


@dataclass
class GenOut:
    text: str
    finish_reason: str          # "stop" | "length"
    tokens: int


class VLLMBackend:
    def __init__(self, llm: Any):
        self.llm = llm

    def generate(self, prompts: list[list[int]], max_tokens: list[int], temperature: float, top_p: float,
                 top_k: int, seed: Optional[int] = None) -> list[GenOut]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = [SamplingParams(max_tokens=m, temperature=temperature, top_p=top_p, top_k=top_k,
                                 seed=None if seed is None else seed + i, skip_special_tokens=True)
                  for i, m in enumerate(max_tokens)]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
        return [GenOut(o.outputs[0].text, o.outputs[0].finish_reason or "stop", len(o.outputs[0].token_ids))
                for o in outs]


class HFBackend:
    """Sequential transformers generation. Slow; for tests only."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model, self.tokenizer = model, tokenizer

    def generate(self, prompts, max_tokens, temperature, top_p, top_k, seed=None) -> list[GenOut]:
        import torch

        outs = []
        for ids, m in zip(prompts, max_tokens):
            x = torch.tensor([ids])
            kwargs = dict(max_new_tokens=m, do_sample=temperature > 0, pad_token_id=self.tokenizer.pad_token_id
                          or self.tokenizer.eos_token_id)
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k if top_k > 0 else None)
            with torch.no_grad():
                y = self.model.generate(x, **kwargs)[0, len(ids):]
            eos = {self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")}
            stopped = len(y) > 0 and int(y[-1]) in eos
            outs.append(GenOut(self.tokenizer.decode(y, skip_special_tokens=True), "stop" if stopped or len(y) < m
                               else "length", len(y)))
        return outs


@dataclass
class Episode:
    row: dict
    task: dict
    messages: list[dict]                         # full conversation so far
    completion: list[dict] = field(default_factory=list)   # generated turns only
    tools: Optional[list[dict]] = None           # schemas passed to the template
    tool_name: Optional[str] = None              # the executable python tool, if any
    turns: int = 0
    done: bool = False
    finish: str = ""
    tokens: int = 0
    thinking_tokens: int = 0


def make_episode(row: dict) -> Episode:
    task = json.loads(row["task"]) if isinstance(row.get("task"), str) else (row.get("task") or {})
    env = row.get("environment", "text")
    tools, tool_name = None, None
    if env in PYTHON_ENVS:
        tools, tool_name = [python_tool_schema(env)], env
    return Episode(row=row, task=task, messages=[dict(m) for m in row["prompt"]], tools=tools, tool_name=tool_name)


class RolloutEngine:
    def __init__(self, backend: Any, tokenizer: Any, pool: Any, thinking: bool = False, max_turns: int = 4,
                 max_tokens: int = 2048, max_model_len: int = 8192, temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = -1, seed: Optional[int] = 0):
        self.backend, self.tokenizer, self.pool = backend, tokenizer, pool
        self.thinking, self.max_turns, self.max_tokens = thinking, max_turns, max_tokens
        self.max_model_len = max_model_len
        self.temperature, self.top_p, self.top_k, self.seed = temperature, top_p, top_k, seed

    def _render(self, ep: Episode) -> list[int]:
        ids = self.tokenizer.apply_chat_template(ep.messages, tools=ep.tools, add_generation_prompt=True,
                                                 tokenize=True, enable_thinking=self.thinking)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)

    def run(self, episodes: list[Episode]) -> list[Episode]:
        active = [e for e in episodes if not e.done]
        while active:
            prompts, budgets, live = [], [], []
            for ep in active:
                ids = self._render(ep)
                room = self.max_model_len - len(ids)
                if room < 32:
                    ep.done, ep.finish = True, "context"
                    continue
                prompts.append(ids)
                budgets.append(min(self.max_tokens, room))
                live.append(ep)
            if not live:
                break
            outs = self.backend.generate(prompts, budgets, self.temperature, self.top_p, self.top_k, self.seed)
            to_exec: list[tuple[Episode, str, str]] = []
            for ep, out in zip(live, outs):
                ep.tokens += out.tokens
                ep.turns += 1
                parsed = parse_reply(out.text, ep.tools or ep.task.get("tools"))
                content = parsed.content
                if self.thinking and "</think>" not in out.text:
                    content, parsed.calls = "", []       # never finished thinking: no answer
                    ep.thinking_tokens += out.tokens
                elif parsed.reasoning:
                    ep.thinking_tokens += len(self.tokenizer.encode(parsed.reasoning, add_special_tokens=False))
                msg: dict = {"role": "assistant", "content": content}
                if parsed.reasoning:
                    msg["reasoning_content"] = parsed.reasoning
                if parsed.calls:
                    msg["tool_calls"] = [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                         for c in parsed.calls]
                ep.messages.append(msg)
                ep.completion.append(msg)
                executable = ep.tool_name is not None and parsed.calls and out.finish_reason != "length"
                if not executable or ep.turns >= self.max_turns:
                    ep.done = True
                    ep.finish = out.finish_reason if not parsed.calls else ("tool_call" if not executable else "max_turns")
                    continue
                for call in parsed.calls:
                    to_exec.append((ep, call.name, call.arguments.get("code", "") if isinstance(call.arguments, dict) else ""))
            if to_exec:
                def run_one(item):
                    ep, name, code = item
                    if name != ep.tool_name:
                        return f"Error: unknown tool '{name}'"
                    return self.pool.run(str(code))
                with ThreadPoolExecutor(max_workers=16) as ex:
                    results = list(ex.map(run_one, to_exec))
                for (ep, name, _), result in zip(to_exec, results):
                    tool_msg = {"role": "tool", "name": name, "content": result}
                    ep.messages.append(tool_msg)
                    ep.completion.append(tool_msg)
            active = [e for e in active if not e.done]
        return episodes


def expand(rows: Sequence[dict], k: int) -> list[Episode]:
    return [make_episode(r) for r in rows for _ in range(k)]
