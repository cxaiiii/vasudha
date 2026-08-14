from __future__ import annotations
from typing import List, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)

class ChatFormatter:
    """Formats conversation data into model-ready strings."""
    
    SUPPORTED_FORMATS = ["qwen3", "chatml", "llama3", "gemma"]
    
    def __init__(self, tokenizer: Any, format: str = "qwen3"):
        self.tokenizer = tokenizer
        self.format = format
        if self.format not in self.SUPPORTED_FORMATS:
            logger.warning(f"Format {self.format} not in officially supported formats: {self.SUPPORTED_FORMATS}")

    def format_conversation(self, conversation: List[Dict[str, Any]]) -> str:
        """Format a conversation (list of role/content dicts) to a string."""
        if self.format == "qwen3":
            return self._format_qwen3(conversation)
        elif hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(conversation, tokenize=False)
        else:
            raise ValueError(f"Fallback formatting failed for {self.format} and no apply_chat_template available.")

    def _format_qwen3(self, conversation: List[Dict[str, Any]]) -> str:
        """Qwen3 specific formatting including <think> tags."""
        result = ""
        for msg in conversation:
            role = msg.get("role", "")
            content = msg.get("content", "")
            thinking = msg.get("thinking", None)
            
            result += f"<|im_start|>{role}\n"
            if role == "assistant" and thinking:
                result += f"<think>\n{thinking}\n</think>\n"
            result += f"{content}<|im_end|>\n"
            
        return result.strip()

    def format_sharegpt(self, conversations: List[Dict[str, Any]]) -> str:
        """Format ShareGPT-style conversations (from/value dicts)."""
        # Map human -> user, gpt -> assistant, system -> system
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        standard_conv = []
        for turn in conversations:
            role = role_map.get(turn.get("from", "human"), "user")
            standard_conv.append({"role": role, "content": turn.get("value", "")})
        return self.format_conversation(standard_conv)
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format OpenAI-style messages (role/content dicts)."""
        return self.format_conversation(messages)
    
    def tokenize(self, text: str, max_length: int, truncation: bool = True) -> Dict[str, Any]:
        """Tokenize a formatted string."""
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding="max_length",
            return_tensors="pt"
        )
    
    def process_dataset(self, dataset: Any, max_seq_length: int, num_proc: int = 1) -> Any:
        """Apply formatting to a full dataset."""
        def map_fn(example: dict) -> dict:
            if "conversations" in example:
                text = self.format_sharegpt(example["conversations"])
            elif "messages" in example:
                text = self.format_messages(example["messages"])
            else:
                text = example.get("text", "")
            
            tokenized = self.tokenize(text, max_length=max_seq_length)
            return {k: v[0] if hasattr(v, "dim") and v.dim() > 1 else v for k, v in tokenized.items()}

        if hasattr(dataset, "map"):
            return dataset.map(map_fn, num_proc=num_proc)
        return dataset

    def __repr__(self) -> str:
        return f"ChatFormatter(format={self.format})"
