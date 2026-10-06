"""Hard money limits.

Two mechanisms, because a training run can fail in two ways:

* Ledger — every Modal stage appends its measured wall-clock and cost to a
  JSONL file on the Volume. The pipeline reads it before each stage and
  refuses to start anything that could push the total past the cap.
* TimeBudgetCallback — a training stage gets a time allowance derived from the
  remaining money. It stops cleanly (saving the adapter) before the allowance
  runs out, instead of being killed by a timeout with nothing saved.

Rates are Modal's published prices (Oct 2026): GPU per hour plus CPU per
physical core-hour and memory per GiB-hour, which Modal bills on top.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

GPU_PER_HOUR = {"H100": 3.95, "H200": 4.54, "A100-80GB": 2.50, "A100": 2.10, "L40S": 1.95, "B200": 6.25,
                "RTX-PRO-6000": 3.03, "A10G": 1.10, "L4": 0.80, "T4": 0.59, None: 0.0}
CPU_CORE_PER_HOUR = 0.0472
MEM_GIB_PER_HOUR = 0.008


def hourly_rate(gpu: Optional[str], cpu: float, memory_gib: float) -> float:
    return GPU_PER_HOUR.get(gpu, 3.95 if gpu else 0.0) + cpu * CPU_CORE_PER_HOUR + memory_gib * MEM_GIB_PER_HOUR


@dataclass
class Entry:
    stage: str
    gpu: Optional[str]
    seconds: float
    cost: float
    ok: bool
    note: str = ""
    at: float = 0.0


class Ledger:
    def __init__(self, path: str, cap: float):
        self.path, self.cap = path, cap

    def entries(self) -> list[Entry]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            return [Entry(**json.loads(line)) for line in fh if line.strip()]

    def spent(self) -> float:
        return sum(e.cost for e in self.entries())

    def remaining(self) -> float:
        return self.cap - self.spent()

    def record(self, stage: str, gpu: Optional[str], seconds: float, rate: float, ok: bool, note: str = "") -> Entry:
        entry = Entry(stage, gpu, seconds, seconds / 3600 * rate, ok, note, time.time())
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def ensure(self, stage: str, estimate: float) -> None:
        if self.spent() + estimate > self.cap + 1e-9:
            raise BudgetExceeded(f"{stage} needs ~${estimate:.2f} but only ${self.remaining():.2f} of "
                                 f"${self.cap:.2f} remains (spent ${self.spent():.2f}).")

    def table(self) -> str:
        rows = ["| stage | gpu | minutes | cost | ok |", "|---|---|---|---|---|"]
        for e in self.entries():
            rows.append(f"| {e.stage} | {e.gpu or 'cpu'} | {e.seconds / 60:.1f} | ${e.cost:.2f} | {'yes' if e.ok else 'NO'} |")
        rows.append(f"| **total** | | | **${self.spent():.2f}** of ${self.cap:.2f} | |")
        return "\n".join(rows)


class BudgetExceeded(RuntimeError):
    pass


def make_time_budget_callback(max_seconds: float, safety_seconds: float = 120.0):
    """Stops training before `max_seconds` of wall clock, leaving time to save."""
    from transformers import TrainerCallback

    class TimeBudgetCallback(TrainerCallback):
        def __init__(self) -> None:
            self.start = time.time()
            self.first_step_at: Optional[float] = None
            self.first_step = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            now = time.time()
            if self.first_step_at is None:
                self.first_step_at, self.first_step = now, state.global_step
                return control
            steps = max(1, state.global_step - self.first_step)
            per_step = (now - self.first_step_at) / steps
            if now - self.start + 1.2 * per_step + safety_seconds > max_seconds:
                print(f"[budget] stopping at step {state.global_step}: {(now - self.start) / 60:.1f} min used, "
                      f"{per_step:.0f}s/step, allowance {max_seconds / 60:.1f} min", flush=True)
                control.should_training_stop = True
                control.should_save = True
            return control

    return TimeBudgetCallback()
