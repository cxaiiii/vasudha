import time
import random
import torch
import numpy as np
from typing import Optional
from .metrics import BenchmarkResult, MetricsCollector

class NeedleBenchmark:
    """Needle-in-Haystack test for long context retrieval."""
    
    def __init__(self, min_len: int = 1000, max_len: int = 32000, num_lengths: int = 5, num_depths: int = 5):
        self.min_len = min_len
        self.max_len = max_len
        self.num_lengths = num_lengths
        self.num_depths = num_depths
        
        self.needle = "The secret passkey for the server is 'Omega-42'."
        self.question = "What is the secret passkey for the server?"
        self.answer = "Omega-42"
        
    def generate_haystack(self, length: int) -> list:
        """Generate dummy text of approximately `length` words/tokens."""
        dummy = "The quick brown fox jumps over the lazy dog. "
        repeats = (length // len(dummy.split())) + 1
        return (dummy * repeats).split()[:length]
        
    def evaluate(
        self,
        model,
        tokenizer,
        max_samples: Optional[int] = None,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        collector = MetricsCollector()
        
        lengths = np.linspace(self.min_len, self.max_len, self.num_lengths, dtype=int)
        depths = np.linspace(0, 1.0, self.num_depths)
        
        heatmap_data = np.zeros((self.num_lengths, self.num_depths))
        
        for i, length in enumerate(lengths):
            for j, depth in enumerate(depths):
                haystack = self.generate_haystack(length)
                insert_idx = int(depth * (len(haystack) - 1))
                
                # Insert needle
                haystack.insert(insert_idx, self.needle)
                context = " ".join(haystack)
                
                prompt = f"{context}\n\nQuestion: {self.question}\nAnswer:"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                t0 = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=32)
                latency_ms = (time.time() - t0) * 1000
                
                generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                is_correct = self.answer.lower() in generated.lower()
                heatmap_data[i, j] = int(is_correct)
                
                collector.record_sample(str(int(is_correct)), "1", latency_ms)
                
        res = collector.summarize()
        res.benchmark = "NeedleInHaystack"
        res.model_name = getattr(model, "name_or_path", "unknown")
        res.extra["heatmap_data"] = heatmap_data.tolist()
        return res
        
    def __repr__(self) -> str:
        return f"NeedleBenchmark(min_len={self.min_len}, max_len={self.max_len})"
