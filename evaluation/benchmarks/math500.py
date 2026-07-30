import json
import re
from typing import Optional
from ..metrics import BenchmarkResult, MetricsCollector
import time
import torch

class Math500Benchmark:
    """4-shot prompting evaluation on MATH-500."""
    
    EXEMPLARS = [
        {"problem": "Find the sum of the first 10 positive integers.", "solution": "The first 10 positive integers are 1, 2, ..., 10. The sum is 10 * 11 / 2 = 55. \\boxed{55}"},
        {"problem": "What is 2 + 2?", "solution": "2 + 2 = 4. \\boxed{4}"},
        {"problem": "Solve for x: 2x + 3 = 7", "solution": "2x = 4 -> x = 2. \\boxed{2}"},
        {"problem": "What is the square root of 144?", "solution": "12 * 12 = 144. \\boxed{12}"},
    ]
    
    def __init__(self, num_shots: int = 4, max_new_tokens: int = 1024):
        self.num_shots = num_shots
        self.max_new_tokens = max_new_tokens
        
    def format_prompt(self, problem: str, num_shots: int = 4) -> str:
        prompt = ""
        for ex in self.EXEMPLARS[:num_shots]:
            prompt += f"Problem: {ex['problem']}\nSolution: {ex['solution']}\n\n"
        prompt += f"Problem: {problem}\nSolution:"
        return prompt
        
    def extract_answer(self, generated_text: str) -> Optional[str]:
        """Extract final answer from \\boxed{}."""
        matches = re.findall(r"\\boxed\{(.*?)\}", generated_text)
        if matches:
            return matches[-1]
        return None
        
    def evaluate(
        self,
        model,
        tokenizer,
        max_samples: Optional[int] = None,
        batch_size: int = 4,
    ) -> BenchmarkResult:
        # Dummy evaluation for prototype
        collector = MetricsCollector()
        
        # In a real scenario, this would load the math500 dataset
        ds = [{"problem": "What is 5 * 5?", "solution": "\\boxed{25}"}]
        
        for item in ds:
            prompt = self.format_prompt(item["problem"], self.num_shots)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            latency_ms = (time.time() - t0) * 1000
            
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred_ans = self.extract_answer(generated)
            ref_ans = self.extract_answer(item["solution"])
            
            collector.record_sample(str(pred_ans), str(ref_ans), latency_ms)
            
        res = collector.summarize()
        res.benchmark = "MATH-500"
        res.model_name = getattr(model, "name_or_path", "unknown")
        return res
        
    def __repr__(self) -> str:
        return f"Math500Benchmark(num_shots={self.num_shots})"
