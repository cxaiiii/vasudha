import os
import traceback
from typing import Optional, List, Dict
from .metrics import BenchmarkResult
from .benchmarks.gsm8k import GSM8KBenchmark

class EvaluationRunner:
    def __init__(
        self,
        model,
        tokenizer,
        benchmarks: Optional[List[str]] = None,
        batch_size: int = 4,
        device: str = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.benchmarks = benchmarks or ["gsm8k", "math500", "humaneval", "mbpp", "longbench", "needle"]
        self.batch_size = batch_size
        self.device = device
        
    def run(
        self,
        benchmarks: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, BenchmarkResult]:
        """Run all specified benchmarks. Saves results to output_dir if given."""
        targets = benchmarks or self.benchmarks
        results = {}
        
        for b in targets:
            try:
                print(f"Running benchmark: {b}")
                res = self.run_single(b, max_samples=max_samples)
                results[b] = res
            except Exception as e:
                # Bare str(e) hid exactly which line failed on the last crash
                # ('NoneType' object has no attribute 'dim' — no file, no
                # number). Print the full traceback so the next failure is
                # diagnosable from one log instead of a guessing game.
                print(f"Error running benchmark {b}: {e}")
                traceback.print_exc()
                
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Need to import EvalReport here to avoid circular imports if it depends on runner
            from .report import EvalReport
            report = EvalReport(results)
            report.save_json(os.path.join(output_dir, "results.json"))
            report.save_markdown(os.path.join(output_dir, "results.md"))
            
        return results
        
    def run_single(self, benchmark: str, max_samples: Optional[int] = None, **kwargs) -> BenchmarkResult:
        """Run a single benchmark by name."""
        if benchmark == "gsm8k":
            bench = GSM8KBenchmark(**kwargs)
            return bench.evaluate(self.model, self.tokenizer, batch_size=self.batch_size, max_samples=max_samples)
        elif benchmark == "math500":
            from .benchmarks.math500 import Math500Benchmark
            bench = Math500Benchmark(**kwargs)
            return bench.evaluate(self.model, self.tokenizer, batch_size=self.batch_size, max_samples=max_samples)
        else:
            raise NotImplementedError(f"Benchmark {benchmark} runner integration is pending.")
            
    def __repr__(self) -> str:
        return f"EvaluationRunner(benchmarks={self.benchmarks}, batch_size={self.batch_size})"
