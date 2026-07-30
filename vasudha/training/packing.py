from __future__ import annotations

import random
from typing import Any, Dict, Iterator, List
from torch.utils.data import IterableDataset
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

class VasudhaConstantLengthDataset(IterableDataset):
    """Packs tokenized examples into fixed-length chunks."""
    def __init__(
        self,
        tokenizer: Any,
        dataset: IterableDataset,
        max_seq_length: int = 2048,
        infinite: bool = True,
        num_of_sequences: int = 1024,
        chars_per_token: float = 3.6,
        eos_token_id: int = 0,
        shuffle: bool = True,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        self.infinite = infinite
        self.num_of_sequences = num_of_sequences
        self.chars_per_token = chars_per_token
        self.eos_token_id = eos_token_id
        self.shuffle = shuffle
        
    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        iterator = iter(self.dataset)
        more_examples = True
        
        while more_examples:
            buffer: List[int] = []
            try:
                for _ in range(self.num_of_sequences):
                    item = next(iterator)
                    
                    if isinstance(item, dict) and "input_ids" in item:
                        tokens = item["input_ids"]
                    elif isinstance(item, dict) and "text" in item:
                        tokens = self.tokenizer(item["text"])["input_ids"]
                    else:
                        tokens = self.tokenizer(str(item))["input_ids"]
                        
                    tokens.append(self.eos_token_id)
                    buffer.extend(tokens)
            except StopIteration:
                more_examples = False
                
            if not buffer:
                break
                
            # Yield chunks of exactly max_seq_length
            i = 0
            while i + self.max_seq_length <= len(buffer):
                chunk = buffer[i : i + self.max_seq_length]
                yield {
                    "input_ids": chunk,
                    "attention_mask": [1] * self.max_seq_length,
                }
                i += self.max_seq_length
                
            if self.infinite and not more_examples:
                iterator = iter(self.dataset)
                more_examples = True
