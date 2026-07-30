from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class BenchmarkResult:
    benchmark: str
    score: float
    num_samples: int
    num_correct: int
    tokens_per_second: float
    peak_vram_gb: float
    latency_p50_ms: float
    latency_p99_ms: float
    total_time_seconds: float
    timestamp: str
    model_name: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"BenchmarkResult(benchmark='{self.benchmark}', score={self.score:.4f}, num_samples={self.num_samples})"

class MetricsCollector:
    def __init__(self):
        self.latencies = []
        self.correct_count = 0
        self.total_count = 0
        
    def record_sample(self, prediction: str, reference: str, latency_ms: float) -> None:
        self.latencies.append(latency_ms)
        self.total_count += 1
        if str(prediction).strip() == str(reference).strip():
            self.correct_count += 1
            
    def compute_throughput(self, total_tokens: int, total_seconds: float) -> float:
        return total_tokens / total_seconds if total_seconds > 0 else 0.0
        
    def summarize(self) -> BenchmarkResult:
        import numpy as np
        from datetime import datetime
        import torch
        
        score = self.correct_count / self.total_count if self.total_count > 0 else 0.0
        p50 = float(np.percentile(self.latencies, 50)) if self.latencies else 0.0
        p99 = float(np.percentile(self.latencies, 99)) if self.latencies else 0.0
        
        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
        return BenchmarkResult(
            benchmark="Unknown",
            score=score,
            num_samples=self.total_count,
            num_correct=self.correct_count,
            tokens_per_second=0.0,
            peak_vram_gb=peak_vram,
            latency_p50_ms=p50,
            latency_p99_ms=p99,
            total_time_seconds=sum(self.latencies) / 1000.0,
            timestamp=datetime.now().isoformat(),
            model_name="Unknown"
        )
