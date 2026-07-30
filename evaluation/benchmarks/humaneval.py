import tempfile
import subprocess
import os
import time
from typing import Optional
from datasets import load_dataset
from .metrics import BenchmarkResult, MetricsCollector
import torch

class HumanEvalBenchmark:
    """HumanEval code evaluation using functional correctness."""
    
    def __init__(self, pass_k: int = 1, timeout: float = 3.0, max_new_tokens: int = 512):
        self.pass_k = pass_k
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        
    def run_code_safe(self, code: str) -> bool:
        """Executes generated code in a subprocess with timeout."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
            
        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                timeout=self.timeout,
                text=True
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def evaluate(
        self,
        model,
        tokenizer,
        max_samples: Optional[int] = None,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        # Load dataset
        try:
            ds = load_dataset("openai_humaneval", split="test")
        except:
            # Fallback for prototype
            ds = [{"prompt": "def add(a, b):", "test": "assert add(1, 2) == 3\n", "task_id": "test/0"}]
            
        if max_samples:
            ds = ds.select(range(max_samples)) if hasattr(ds, 'select') else ds[:max_samples]
            
        collector = MetricsCollector()
        
        for item in ds:
            prompt = item["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            latency_ms = (time.time() - t0) * 1000
            
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Combine generated function with test cases
            full_code = prompt + generated + "\n" + item["test"] + f"\ncheck({item['task_id'].split('/')[-1]})"
            is_correct = self.run_code_safe(full_code)
            
            # Record result (using "1" as correct reference, "1" or "0" as prediction)
            collector.record_sample(str(int(is_correct)), "1", latency_ms)
            
        res = collector.summarize()
        res.benchmark = "HumanEval"
        res.model_name = getattr(model, "name_or_path", "unknown")
        res.extra["pass@k"] = self.pass_k
        return res
        
    def __repr__(self) -> str:
        return f"HumanEvalBenchmark(pass_k={self.pass_k}, timeout={self.timeout}s)"
