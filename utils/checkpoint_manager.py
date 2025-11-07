#!/usr/bin/env python3
"""Checkpoint Manager for training resumption"""

import torch
import os
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage training checkpoints"""
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
    
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict,
        filename: Optional[str] = None
    ):
        """Save training checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.nn.Optimizer] = None,
        filename: Optional[str] = None
    ) -> Dict:
        """Load checkpoint"""
        if filename is None:
            # Load latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping last N"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoints) > self.keep_last_n:
            for checkpoint in checkpoints[:-self.keep_last_n]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")
