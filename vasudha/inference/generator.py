from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Generator, Tuple, Any
import logging
import threading

try:
    from transformers import TextIteratorStreamer
except ImportError:
    TextIteratorStreamer = None

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    use_cache: bool = True
    thinking_mode: bool = False  # Inject <think> token

class StreamingGenerator:
    """Streaming text generation for Vasudha models."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        gen_config: Optional[GenerationConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.default_config = gen_config or GenerationConfig()
        
    def _prepare_kwargs(self, prompt: str, gen_config: Optional[GenerationConfig] = None) -> dict:
        config = gen_config or self.default_config
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(self.model.parameters()).device)
            
        kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "do_sample": config.do_sample,
            "use_cache": config.use_cache,
        }
        if config.eos_token_id is not None:
            kwargs["eos_token_id"] = config.eos_token_id
        elif hasattr(self.tokenizer, "eos_token_id"):
            kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            
        if config.pad_token_id is not None:
            kwargs["pad_token_id"] = config.pad_token_id
        elif hasattr(self.tokenizer, "pad_token_id"):
            kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            
        return kwargs

    def generate(
        self,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> str:
        """Generate text (blocking, returns full string)."""
        logger.info("Starting blocking generation.")
        kwargs = self._prepare_kwargs(prompt, gen_config)
        with torch.no_grad():
            outputs = self.model.generate(**kwargs)
        
        input_len = kwargs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def stream(
        self,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, None]:
        """Generate text as a streaming token-by-token generator."""
        if TextIteratorStreamer is None:
            raise ImportError("transformers is required for streaming generation.")
            
        logger.info("Starting streaming generation.")
        kwargs = self._prepare_kwargs(prompt, gen_config)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs["streamer"] = streamer
        
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
            
    def generate_with_thinking(
        self,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
    ) -> Tuple[str, str]:
        """Generate with Qwen3 thinking mode. Returns (thinking, response)."""
        cfg = gen_config or self.default_config
        cfg.thinking_mode = True
        
        full_text = self.generate(prompt, cfg)
        
        if "<think>" in full_text and "</think>" in full_text:
            parts = full_text.split("</think>")
            thinking = parts[0].replace("<think>", "").strip()
            response = parts[1].strip() if len(parts) > 1 else ""
            return thinking, response
        else:
            return "", full_text

    def __repr__(self) -> str:
        return f"StreamingGenerator(model={self.model.__class__.__name__})"
