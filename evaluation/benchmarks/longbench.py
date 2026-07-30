import time
from typing import Optional
import torch
from .metrics import BenchmarkResult, MetricsCollector

class LongBenchBenchmark:
    """LongBench evaluation for long-context capabilities."""
    
    def __init__(self, max_context_length: int = 32000, max_new_tokens: int = 128):
        self.max_context_length = max_context_length
        self.max_new_tokens = max_new_tokens
        
    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score for QA."""
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(pred_tokens == truth_tokens)
            
        common_tokens = set(pred_tokens).intersection(set(truth_tokens))
        if len(common_tokens) == 0:
            return 0.0
            
        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)
        f1 = 2 * (prec * rec) / (prec + rec)
        return f1
        
    def evaluate(
        self,
        model,
        tokenizer,
        max_samples: Optional[int] = None,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        collector = MetricsCollector()
        
        # Prototype synthetic long data
        ds = [{"context": "A " * 1000 + "The secret code is 42. " + "A " * 1000, "question": "What is the secret code?", "answer": "42"}]
        
        for item in ds:
            prompt = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_context_length).to(model.device)
            
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            latency_ms = (time.time() - t0) * 1000
            
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            f1 = self.compute_f1(generated, item["answer"])
            
            # Record thresholded F1 as "correct" if > 0.5 for simple reporting
            collector.record_sample(str(int(f1 > 0.5)), "1", latency_ms)
            
        res = collector.summarize()
        res.benchmark = "LongBench"
        res.model_name = getattr(model, "name_or_path", "unknown")
        res.extra["max_context_length"] = self.max_context_length
        return res
        
    def __repr__(self) -> str:
        return f"LongBenchBenchmark(max_context_length={self.max_context_length})"
