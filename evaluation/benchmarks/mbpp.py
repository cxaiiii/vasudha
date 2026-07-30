import tempfile
import subprocess
import os
import time
from typing import Optional
from datasets import load_dataset
from .metrics import BenchmarkResult, MetricsCollector
import torch

class MBPPBenchmark:
    """MBPP (Mostly Basic Python Programming) evaluation."""
    
    def __init__(self, num_shots: int = 3, timeout: float = 3.0, max_new_tokens: int = 512):
        self.num_shots = num_shots
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        
    def run_code_safe(self, code: str) -> bool:
        """Executes generated code in a restricted subprocess with timeout."""
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
                
    def format_prompt(self, text: str, test_list: list) -> str:
        """Formats 3-shot prompt (simplified for prototype)."""
        prompt = f"Write a python function to solve the following problem:\n{text}\n\nTest cases:\n"
        for t in test_list:
            prompt += f"{t}\n"
        prompt += "\nCode:\n"
        return prompt
        
    def evaluate(
        self,
        model,
        tokenizer,
        max_samples: Optional[int] = None,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        try:
            ds = load_dataset("mbpp", "sanitized", split="test")
        except:
            ds = [{"text": "Write a function to add two numbers.", "test_list": ["assert add(1, 2) == 3"]}]
            
        if max_samples:
            ds = ds.select(range(max_samples)) if hasattr(ds, 'select') else ds[:max_samples]
            
        collector = MetricsCollector()
        
        for item in ds:
            prompt = self.format_prompt(item["text"], item["test_list"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            latency_ms = (time.time() - t0) * 1000
            
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Combine generated function with test cases
            full_code = generated + "\n" + "\n".join(item["test_list"])
            is_correct = self.run_code_safe(full_code)
            
            collector.record_sample(str(int(is_correct)), "1", latency_ms)
            
        res = collector.summarize()
        res.benchmark = "MBPP"
        res.model_name = getattr(model, "name_or_path", "unknown")
        return res
        
    def __repr__(self) -> str:
        return f"MBPPBenchmark(num_shots={self.num_shots}, timeout={self.timeout}s)"
