import json
from typing import Dict
from .metrics import BenchmarkResult

class EvalReport:
    def __init__(self, results: Dict[str, BenchmarkResult]):
        self.results = results
        
    def print_table(self) -> None:
        """Print a Rich table with all benchmark results."""
        try:
            from rich.console import Console
            from rich.table import Table
            
            console = Console()
            table = Table(title="Vasudha Evaluation Results")
            
            table.add_column("Benchmark", style="cyan", no_wrap=True)
            table.add_column("Score", style="magenta")
            table.add_column("Samples", style="green")
            table.add_column("Latency (p50)", style="yellow")
            
            for name, res in self.results.items():
                table.add_row(
                    name,
                    f"{res.score:.4f}",
                    str(res.num_samples),
                    f"{res.latency_p50_ms:.2f}ms"
                )
                
            console.print(table)
        except ImportError:
            print("Rich library not installed. Printing text table:")
            print(f"{'Benchmark':<15} | {'Score':<10} | {'Samples':<10} | {'Latency(p50)':<15}")
            print("-" * 55)
            for name, res in self.results.items():
                print(f"{name:<15} | {res.score:<10.4f} | {res.num_samples:<10} | {res.latency_p50_ms:<15.2f}")
                
    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            data = {k: v.__dict__ for k, v in self.results.items()}
            json.dump(data, f, indent=2)
            
    def save_markdown(self, path: str) -> None:
        """Save results as a markdown table suitable for README embedding."""
        with open(path, "w") as f:
            f.write("# Evaluation Results\n\n")
            f.write("| Benchmark | Score | Samples | Latency (p50) |\n")
            f.write("|-----------|-------|---------|---------------|\n")
            for name, res in self.results.items():
                f.write(f"| {name} | {res.score:.4f} | {res.num_samples} | {res.latency_p50_ms:.2f}ms |\n")
                
    def plot_needle_heatmap(self, needle_result: BenchmarkResult, output_path: str) -> None:
        """Plot needle-in-haystack retrieval heatmap using matplotlib."""
        if 'heatmap_data' not in needle_result.extra:
            print("No heatmap data found in needle result.")
            return
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            data = needle_result.extra['heatmap_data']
            # Assume data is a 2D array: context_length x depth
            plt.figure(figsize=(10, 8))
            sns.heatmap(data, annot=True, cmap="YlGnBu", cbar=True)
            plt.title("Needle in a Haystack Accuracy")
            plt.xlabel("Document Depth (%)")
            plt.ylabel("Context Length")
            plt.savefig(output_path)
            plt.close()
        except ImportError:
            print("Matplotlib/Seaborn not installed. Cannot plot heatmap.")
            
    def __repr__(self) -> str:
        return f"EvalReport(num_benchmarks={len(self.results)})"
