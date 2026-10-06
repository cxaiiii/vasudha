"""The small amount of glue Neo needs on top of TRL 1.14.

* `NeoGRPOTrainer` — tasks routed to the tool-less "text" environment may
  still emit tool calls (function-calling tasks declare their tools in the
  system prompt). Those calls are the answer to be graded, not something to
  execute; stock TRL would try to run them and continue the episode with a
  "tool not found" error. The override hides them from the tool loop and puts
  them back afterwards. Nothing else about GRPO changes.
* `neo_reward` — neo.verify.score_episode as a TRL reward function, logging
  per-task-type accuracy so the training curves say what is improving.
* `lora_config` — LoRA on every linear layer of the language model (attention,
  Gated DeltaNet projections, MLP), as "LoRA Without Regret" recommends.
"""
from __future__ import annotations

import json
from collections import defaultdict

from trl import GRPOTrainer

from neo.verify import ToolRecord, score_episode

TEXT_ENVS = frozenset({"text"})


class NeoGRPOTrainer(GRPOTrainer):
    def _tool_call_loop(self, prompts, prompt_ids, completion_ids, completions, *args, **kwargs):
        envs = getattr(self, "_batch_environments", None) or [None] * len(completions)
        hidden = {}
        for i, name in enumerate(envs):
            if name in TEXT_ENVS and completions[i] and completions[i][0].get("tool_calls"):
                hidden[i] = completions[i][0].pop("tool_calls")
        result = super()._tool_call_loop(prompts, prompt_ids, completion_ids, completions, *args, **kwargs)
        out_completions = result[1]
        for i, calls in hidden.items():
            out_completions[i][0]["tool_calls"] = calls
        return result


def neo_reward(prompts, completions, task, environments=None, log_metric=None, **kwargs):
    rewards = []
    per_type = defaultdict(list)
    for i, (completion, raw) in enumerate(zip(completions, task)):
        t = json.loads(raw) if isinstance(raw, str) else raw
        tools = None
        env = environments[i] if environments else None
        if env is not None and getattr(env, "calls", None) is not None and t["type"] != "fc":
            tools = [ToolRecord(c.name, c.result, c.ok) for c in env.calls]
        messages = completion if isinstance(completion, list) else [{"role": "assistant", "content": completion}]
        s = score_episode(t, messages, tools)
        rewards.append(float(s.reward))
        per_type[t["type"]].append(s)
    if log_metric is not None:
        for kind, scores in per_type.items():
            log_metric(f"neo/{kind}/correct", sum(x.correct for x in scores) / len(scores))
            log_metric(f"neo/{kind}/reward", sum(x.reward for x in scores) / len(scores))
        empties = [x.metrics.get("empty_reply", 0.0) for xs in per_type.values() for x in xs]
        if empties:
            log_metric("neo/empty_reply", sum(empties) / len(empties))
    return rewards


def lora_config(r: int, alpha: int, dropout: float = 0.0):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                      target_modules="all-linear")
