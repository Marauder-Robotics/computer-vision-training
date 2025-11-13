#!/usr/bin/env python3
"""
Simple logging utility for Marauder CV Pipeline
Note: Main training logging handled by TrainingLogger (utils/training_logger.py)
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
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


def log_metrics(metrics: dict, step: Optional[int] = None):
    """Log metrics to console
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step/epoch number
    
    Note:
        For training metrics logging, use TrainingLogger instead
    """
    logger = logging.getLogger(__name__)
    metrics_str = ', '.join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                             for k, v in metrics.items()])
    if step is not None:
        logger.info(f"Step {step} - {metrics_str}")
    else:
        logger.info(f"Metrics - {metrics_str}")
