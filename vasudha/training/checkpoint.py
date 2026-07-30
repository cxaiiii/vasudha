from __future__ import annotations

import os
import shutil
from typing import Any, Optional
from vasudha.utils.logging import get_logger

logger = get_logger(__name__)

class CheckpointManager:
    """Checkpoint resume utilities for environment like Colab."""
    def __init__(self, output_dir: str, save_every_n_steps: int = 200):
        self.output_dir = output_dir
        self.save_every_n_steps = save_every_n_steps
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent valid checkpoint in output_dir."""
        if not os.path.exists(self.output_dir):
            return None
            
        checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
        if not checkpoints:
            return None
            
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        latest = os.path.join(self.output_dir, checkpoints[-1])
        logger.info(f"Found latest checkpoint at {latest}")
        return latest
    
    def save_training_state(self, trainer: Any, step: int) -> None:
        """Save trainer state including dataset position."""
        if step % self.save_every_n_steps == 0:
            logger.info(f"Saving checkpoint at step {step}")
            trainer.save_model(os.path.join(self.output_dir, f"checkpoint-{step}"))
            trainer.save_state()
    
    def load_training_state(self, trainer: Any, checkpoint_path: str) -> int:
        """Load trainer state. Returns step to resume from."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        # Note: Trainer._load_from_checkpoint is typically used.
        # This returns the integer step.
        try:
            step_str = os.path.basename(checkpoint_path).split("-")[-1]
            return int(step_str)
        except ValueError:
            return 0
    
    def cleanup_old_checkpoints(self, keep_n: int = 3) -> None:
        """Remove old checkpoints to free disk space."""
        if not os.path.exists(self.output_dir):
            return
            
        checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        
        if len(checkpoints) > keep_n:
            to_delete = checkpoints[:-keep_n]
            for ckpt in to_delete:
                path = os.path.join(self.output_dir, ckpt)
                logger.info(f"Deleting old checkpoint {path}")
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    logger.warning(f"Failed to delete {path}: {e}")
