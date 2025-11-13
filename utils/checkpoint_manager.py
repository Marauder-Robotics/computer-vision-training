#!/usr/bin/env python3
"""Checkpoint Manager for training resumption with DO Spaces support"""

import torch
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage training checkpoints with DigitalOcean Spaces support"""
    
    def __init__(self, 
                 checkpoint_dir: str = "/datasets/marauder-do-bucket/training/checkpoints",
                 project_name: str = "marauder_cv",
                 keep_last_n: int = 5):
        """
        Initialize checkpoint manager
        
        Args:
            checkpoint_dir: Base directory for checkpoints (DO bucket path)
            project_name: Project name for organization
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir) / project_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.checkpoint_info_file = self.checkpoint_dir / "checkpoint_info.json"
        self.checkpoints = []
        self._load_checkpoint_info()
    
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict,
        step: Optional[int] = None,
        is_best: bool = False,
        filename: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Save training checkpoint to DO Spaces"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}"
            if step is not None:
                filename += f"_step_{step}"
            filename += ".pth"
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Track checkpoint
        self.checkpoints.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'timestamp': checkpoint['timestamp']
        })
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")
            
            # Save best model info
            best_info = {
                'epoch': epoch,
                'step': step,
                'metrics': metrics,
                'timestamp': checkpoint['timestamp']
            }
            with open(self.checkpoint_dir / 'best_model_info.json', 'w') as f:
                json.dump(best_info, f, indent=2, default=str)
        
        # Update latest checkpoint link
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        if latest_path.exists():
            latest_path.unlink()
        shutil.copy2(checkpoint_path, latest_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        self._save_checkpoint_info()
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        filename: Optional[str] = None,
        load_best: bool = False,
        map_location: str = 'cpu'
    ) -> Dict:
        """Load checkpoint from DO Spaces"""
        if load_best:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        elif filename is None:
            # Load latest checkpoint
            checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
            if not checkpoint_path.exists():
                checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
                if not checkpoints:
                    raise FileNotFoundError("No checkpoints found")
                checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint
    
    def resume_training(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: str = 'cpu'
    ) -> tuple:
        """
        Resume training from latest checkpoint
        
        Returns:
            epoch: Starting epoch
            step: Starting step  
            metrics: Last metrics
        """
        try:
            checkpoint = self.load_checkpoint(model, optimizer, map_location=map_location)
            epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
            step = checkpoint.get('step', 0)
            metrics = checkpoint.get('metrics', {})
            logger.info(f"Resuming from epoch {epoch}, step {step}")
            return epoch, step, metrics
        except FileNotFoundError:
            logger.info("No checkpoint found, starting from scratch")
            return 0, 0, {}
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping last N"""
        # Sort checkpoints by epoch and step
        self.checkpoints.sort(key=lambda x: (x['epoch'], x.get('step', 0)))
        
        # Keep best and latest always
        while len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            old_path = Path(old_checkpoint['path'])
            
            # Don't delete best or latest
            if old_path.exists() and 'best' not in old_path.name and 'latest' not in old_path.name:
                old_path.unlink()
                logger.debug(f"Removed old checkpoint: {old_path}")
    
    def _save_checkpoint_info(self):
        """Save checkpoint information to JSON"""
        info = {
            'checkpoints': self.checkpoints,
            'last_updated': datetime.now().isoformat(),
            'save_dir': str(self.checkpoint_dir)
        }
        with open(self.checkpoint_info_file, 'w') as f:
            json.dump(info, f, indent=2, default=str)
    
    def _load_checkpoint_info(self):
        """Load checkpoint information from JSON"""
        if self.checkpoint_info_file.exists():
            with open(self.checkpoint_info_file) as f:
                info = json.load(f)
                self.checkpoints = info.get('checkpoints', [])
