#!/usr/bin/env python3
"""Custom logger with W&B integration"""

import logging
import sys
from pathlib import Path
from typing import Optional
import wandb


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_wandb: bool = False
) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(metrics: dict, step: Optional[int] = None, use_wandb: bool = False):
    """Log metrics to W&B and console"""
    if use_wandb and wandb.run is not None:
        wandb.log(metrics, step=step)
    
    # Also log to console
    logger = logging.getLogger(__name__)
    metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    if step is not None:
        logger.info(f"Step {step} - {metrics_str}")
    else:
        logger.info(f"Metrics - {metrics_str}")
