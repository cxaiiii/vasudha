from __future__ import annotations
from typing import Any, Optional
import logging
import os
import subprocess

logger = logging.getLogger(__name__)

class GGUFExporter:
    """Export VasudhaForCausalLM to GGUF format for llama.cpp."""
    
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        
    def export_to_safetensors(
        self,
        output_path: str,
        dtype: str = "float16",
    ) -> str:
        """Save model weights in SafeTensors format (intermediate step for llama.cpp)."""
        logger.info(f"Exporting model to safetensors at {output_path} with dtype {dtype}.")
        try:
            from safetensors.torch import save_model
        except ImportError:
            raise ImportError("safetensors package is required for export.")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_path, safe_serialization=True)
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(output_path)
            return output_path
        else:
            save_model(self.model, os.path.join(output_path, "model.safetensors"))
            return output_path
    
    def convert_to_gguf(
        self,
        output_path: str,
        quantization: str = "Q4_K_M",
        llama_cpp_path: Optional[str] = None,
    ) -> str:
        """Convert to GGUF using llama.cpp conversion script."""
        logger.info(f"Converting model to GGUF format (quant: {quantization}) at {output_path}.")
        if not llama_cpp_path:
            logger.warning("llama_cpp_path not provided. Cannot automatically convert. Falling back to instructions.")
            self.print_conversion_instructions()
            return ""
            
        script_path = os.path.join(llama_cpp_path, "convert-hf-to-gguf.py")
        if not os.path.exists(script_path):
            logger.error(f"Conversion script not found at {script_path}")
            self.print_conversion_instructions()
            return ""
            
        cmd = ["python", script_path, os.path.dirname(output_path), "--outfile", output_path, "--outtype", quantization]
        try:
            subprocess.run(cmd, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during GGUF conversion: {e}")
            self.print_conversion_instructions()
            return ""
    
    @staticmethod
    def print_conversion_instructions() -> None:
        """Print manual conversion instructions using Rich."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            
            instructions = (
                "1. Clone llama.cpp repository: `git clone https://github.com/ggerganov/llama.cpp`\n"
                "2. Install requirements: `pip install -r llama.cpp/requirements.txt`\n"
                "3. Export Vasudha to SafeTensors first: `exporter.export_to_safetensors('./model_hf')`\n"
                "4. Run conversion script: `python llama.cpp/convert-hf-to-gguf.py ./model_hf --outfile vasudha.gguf --outtype q8_0`\n"
                "5. Optional: quantize further using `./quantize vasudha.gguf vasudha-Q4_K_M.gguf Q4_K_M`"
            )
            console.print(Panel(instructions, title="GGUF Conversion Instructions"))
        except ImportError:
            print("GGUF Conversion Instructions:\n")
            print("1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
            print("2. Run conversion: python convert-hf-to-gguf.py /path/to/hf_model --outfile model.gguf")

    def __repr__(self) -> str:
        return f"GGUFExporter(model={self.model.__class__.__name__})"
