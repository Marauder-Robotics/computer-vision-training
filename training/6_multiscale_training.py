#!/usr/bin/env python3
"""
Week 5: Multi-Scale Training
Trains models with dynamic resolution to improve detection across different scales
"""

import os
import yaml
import torch
import logging
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiScaleTrainer:
    """Train YOLO models with multi-scale training"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.ms_config = self.config['multiscale_training']
        self.checkpoint_dir = Path(self.config['paths']['checkpoints']) / 'multiscale'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_multiscale_variant(self, base_model_path: str, variant_name: str) -> str:
        """
        Train a model with multi-scale augmentation
        
        Args:
            base_model_path: Path to ensemble variant model
            variant_name: Name of the variant
            
        Returns:
            Path to trained model
        """
        logger.info(f"\nTraining multi-scale {variant_name}")
        
        # Load base model
        model = YOLO(base_model_path)
        
        # Multi-scale training arguments
        train_args = {
            'data': str(Path(self.config['paths']['data_root']) / 'dataset.yaml'),
            'epochs': self.ms_config['epochs'],
            'batch': self.ms_config['batch_size'],
            'imgsz': max(self.ms_config['resolutions']),  # Use max resolution
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': self.config['general']['num_workers'],
            'project': str(self.checkpoint_dir),
            'name': f"multiscale_{variant_name}",
            'exist_ok': True,
            'pretrained': False,  # Already fine-tuned
            'optimizer': 'SGD',
            'lr0': 0.003,  # Lower LR for fine-tuning
            'lrf': 0.001,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 2.0,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            # Multi-scale augmentation
            'scale': self.ms_config['scale_variance'],
            'multi_scale': True,  # Enable dynamic image sizing
            'rect': False,  # Must be False for multi-scale
            'cache': False,
            'cos_lr': True,
            'amp': self.config['general']['mixed_precision'],
            'save_period': 10,
            'val': True,
        }
        
        # Train
        results = model.train(**train_args)
        
        best_model = self.checkpoint_dir / f"multiscale_{variant_name}" / 'weights' / 'best.pt'
        logger.info(f"Multi-scale {variant_name} saved to: {best_model}")
        
        return str(best_model)
    
    def train_all_multiscale(self) -> Dict[str, str]:
        """Train multi-scale versions of all ensemble variants"""
        logger.info("\n" + "="*80)
        logger.info("MULTI-SCALE TRAINING")
        logger.info("="*80 + "\n")
        
        # Get ensemble model paths
        ensemble_dir = Path(self.config['paths']['checkpoints']) / 'ensemble'
        variants = ['high_recall', 'balanced', 'high_precision']
        
        trained_models = {}
        
        for variant in variants:
            base_model = ensemble_dir / variant / 'weights' / 'best.pt'
            
            if not base_model.exists():
                logger.warning(f"Base model not found: {base_model}, skipping...")
                continue
                
            ms_model = self.train_multiscale_variant(str(base_model), variant)
            trained_models[variant] = ms_model
        
        # Save multiscale configuration
        ms_info = {
            'models': trained_models,
            'resolutions': self.ms_config['resolutions'],
            'test_scales': self.ms_config['test_scales']
        }
        
        config_path = self.checkpoint_dir / 'multiscale_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(ms_info, f)
        
        logger.info(f"\nMulti-scale configuration saved to: {config_path}")
        
        return trained_models


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-scale training')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    args = parser.parse_args()
    
    trainer = MultiScaleTrainer(args.config)
    trained_models = trainer.train_all_multiscale()
    
    logger.info("\n" + "="*80)
    logger.info("MULTI-SCALE TRAINING COMPLETE")
    logger.info("="*80)
    for variant, path in trained_models.items():
        logger.info(f"  {variant}: {path}")


if __name__ == '__main__':
    main()
