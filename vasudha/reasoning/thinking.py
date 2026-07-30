from typing import Tuple, Optional

class ThinkingMode:
    """Manages Qwen3-compatible thinking mode.
    
    Qwen3 uses <think> / </think> tokens to delineate internal reasoning.
    This class handles:
    - Injecting <think> into generation prompts
    - Extracting thinking from generated text
    - Suppressing thinking for fast inference
    """
    
    THINK_START = "<think>"
    THINK_END = "</think>"
    
    def __init__(self, tokenizer, enable: bool = True):
        self.tokenizer = tokenizer
        self.enable = enable
        
    def inject_thinking_prompt(self, prompt: str) -> str:
        """Append <think> to the prompt to trigger thinking mode."""
        if self.enable:
            return f"{prompt}\n{self.THINK_START}\n"
        return prompt
        
    def extract_thinking(self, generated_text: str) -> Tuple[str, str]:
        """Parse generated text into (thinking, response) parts."""
        if self.THINK_START in generated_text and self.THINK_END in generated_text:
            start_idx = generated_text.find(self.THINK_START) + len(self.THINK_START)
            end_idx = generated_text.find(self.THINK_END)
            thinking = generated_text[start_idx:end_idx].strip()
            response = generated_text[end_idx + len(self.THINK_END):].strip()
            return thinking, response
        elif self.THINK_START in generated_text:
            start_idx = generated_text.find(self.THINK_START) + len(self.THINK_START)
            return generated_text[start_idx:].strip(), ""
        return "", generated_text.strip()
        
    def strip_thinking(self, generated_text: str) -> str:
        """Remove <think>...</think> block from output (for fast mode)."""
        _, response = self.extract_thinking(generated_text)
        return response
        
    @property
    def think_token_id(self) -> Optional[int]:
        """Token ID for <think> if it exists in the tokenizer vocab."""
        if hasattr(self.tokenizer, 'convert_tokens_to_ids'):
            tok_id = self.tokenizer.convert_tokens_to_ids(self.THINK_START)
            if tok_id is not None and tok_id != self.tokenizer.unk_token_id:
                return tok_id
        return None
        
    def __repr__(self) -> str:
        return f"ThinkingMode(enable={self.enable}, think_token_id={self.think_token_id})"
