import json
import os
import re
from typing import Optional
from ..metrics import BenchmarkResult, MetricsCollector
import time
from datetime import datetime
import torch

# The 1319-example test split, snapshotted to evaluation/data/gsm8k_test.jsonl.
# load_dataset() pulled this from the Hub before, and huggingface_hub's HTTP
# calls don't take a timeout kwarg from us, so a stalled connection in the
# Modal container hung silently for 30+ minutes with no error. Shipping the
# data with the repo removes the network dependency entirely.
_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "gsm8k_test.jsonl")

class GSM8KBenchmark:
    """8-shot chain-of-thought evaluation on GSM8K."""
    
    EXEMPLARS = [
        {"question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?", "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."},
        {"question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."},
        {"question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?", "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."},
        {"question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?", "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."},
        {"question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?", "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 + 2 = 4 more toys. Now he has 5 + 4 = 9 toys. The answer is 9."},
        {"question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?", "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29 computers are now in the server room. The answer is 29."},
        {"question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?", "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."},
        {"question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?", "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8."}
    ]
    
    def __init__(self, num_shots: int = 8, max_new_tokens: int = 512):
        self.num_shots = num_shots
        self.max_new_tokens = max_new_tokens
        
    def load_dataset(self):
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
        
    def format_prompt(self, question: str, num_shots: int = 8) -> str:
        prompt = ""
        for ex in self.EXEMPLARS[:num_shots]:
            prompt += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        prompt += f"Question: {question}\nAnswer:"
        return prompt
        
    def extract_answer(self, generated_text: str) -> Optional[float]:
        """Extract final numeric answer. Handles #### marker, \\boxed{}, last number."""
        if "####" in generated_text:
            ans_str = generated_text.split("####")[-1].strip()
            ans_str = ans_str.replace(",", "")
            try:
                return float(ans_str)
            except:
                pass
        
        matches = re.findall(r"[-+]?\d*\.\d+|\d+", generated_text.replace(",", ""))
        if matches:
            try:
                return float(matches[-1])
            except:
                pass
        return None
        
    def evaluate(
        self,
        model,
        tokenizer,
        max_samples: Optional[int] = None,
        batch_size: int = 4,
    ) -> BenchmarkResult:
        ds = self.load_dataset()
        if max_samples:
            ds = ds[:max_samples]
            
        collector = MetricsCollector()
        start_time = time.time()
        n = len(ds)

        # Simplified batch evaluation logic
        for i, item in enumerate(ds):
            prompt = self.format_prompt(item["question"], self.num_shots)
            # Training data was entirely <|im_start|>user...<|im_end|> chat
            # turns (see ChatFormatter._format_qwen3) — a bare completion
            # prompt is out-of-distribution for this model regardless of how
            # well training went, and alone accounts for near-random scores.
            chat_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).to(model.device)

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            latency_ms = (time.time() - t0) * 1000

            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            pred_ans = self.extract_answer(generated)
            ref_ans = self.extract_answer(item["answer"])

            collector.record_sample(str(pred_ans), str(ref_ans), latency_ms)

            if i < 3:
                print(
                    f"  [gsm8k {i + 1} sample] pred={pred_ans} ref={ref_ans} "
                    f"raw={generated[:200]!r}",
                    flush=True,
                )

            elapsed = time.time() - start_time
            eta_min = (elapsed / (i + 1)) * (n - i - 1) / 60
            print(
                f"  [gsm8k {i + 1}/{n}] {latency_ms / 1000:.1f}s/sample, "
                f"eta {eta_min:.1f} min",
                flush=True,
            )

        res = collector.summarize()
        res.benchmark = "GSM8K"
        res.model_name = getattr(model, "name_or_path", "unknown")
        return res
        
    def __repr__(self) -> str:
        return f"GSM8KBenchmark(num_shots={self.num_shots})"
