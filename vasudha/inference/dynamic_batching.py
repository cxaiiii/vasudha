from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional
import logging
import asyncio
from concurrent.futures import Future

import torch
import torch.nn as nn
from .generator import GenerationConfig

logger = logging.getLogger(__name__)

@dataclass
class _BatchRequest:
    prompt: str
    gen_config: GenerationConfig
    future: Future

class DynamicBatcher:
    """Groups incoming requests into batches for efficient GPU utilization."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_batch_size: int = 8,
        max_batch_tokens: int = 8192,
        max_wait_ms: float = 50.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_wait_ms = max_wait_ms
        self.queue: List[_BatchRequest] = []
        self._lock = asyncio.Lock()
    
    def add_request(self, prompt: str, gen_config: GenerationConfig) -> Future:
        """Add a generation request. Returns a Future for the result."""
        future = Future()
        req = _BatchRequest(prompt, gen_config, future)
        self.queue.append(req)
        logger.debug(f"Added request to dynamic batcher. Queue size: {len(self.queue)}")
        return future
    
    def _process_batch(self, requests: List[_BatchRequest]) -> None:
        """Run batched generation for a group of requests."""
        if not requests:
            return
            
        logger.info(f"Processing batch of {len(requests)} requests.")
        prompts = [req.prompt for req in requests]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        config = requests[0].gen_config
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=config.do_sample,
            )
            
        input_lens = [input_ids[i].shape[0] for i in range(len(requests))]
        
        for i, req in enumerate(requests):
            new_tokens = outputs[i][input_lens[i]:]
            result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            req.future.set_result(result)

    def __repr__(self) -> str:
        return f"DynamicBatcher(max_batch_size={self.max_batch_size}, max_wait_ms={self.max_wait_ms})"
