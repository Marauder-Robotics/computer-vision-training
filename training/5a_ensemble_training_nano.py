#!/usr/bin/env python3
"""
Week 5: Ensemble Training - Three Variants (Recall, Balanced, Precision)
Creates three specialized models optimized for different trade-offs
"""

import os
import sys
import yaml
import torch
import wandb
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any, List
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """Train three YOLO variants for ensemble"""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.ensemble_config = self.config['ensemble_training']
        self.general_config = self.config['general']
        
        # Setup paths
        self.checkpoint_dir = Path(self.config['paths']['checkpoints']) / 'ensemble'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize W&B if enabled
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                name=f"ensemble_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.ensemble_config
            )
    
    def train_variant(self, variant_name: str, variant_config: Dict[str, Any]) -> str:
        """
        Train a single ensemble variant
        
        Args:
            variant_name: Name of variant (high_recall, balanced, high_precision)
            variant_config: Configuration for this variant
            
        Returns:
            Path to best checkpoint
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {variant_name} variant")
        logger.info(f"{'='*80}\n")
        
        # Setup variant-specific paths
        variant_dir = self.checkpoint_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base model (critical species specialized model)
        base_model_path = self.ensemble_config['base_model']
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
            
        model = YOLO(base_model_path)
        logger.info(f"Loaded base model from {base_model_path}")
        
        # Prepare training arguments
        train_args = self._prepare_training_args(variant_name, variant_config, variant_dir)
        
        # Train model
        logger.info(f"Starting training with args:")
        for key, value in train_args.items():
            logger.info(f"  {key}: {value}")
            
        try:
            results = model.train(**train_args)
            logger.info(f"{variant_name} training completed successfully")
            
            # Get best model path
            best_model = variant_dir / 'weights' / 'best.pt'
            logger.info(f"Best model saved to: {best_model}")
            
            return str(best_model)
            
        except Exception as e:
            logger.error(f"Error training {variant_name}: {e}")
            raise
    
    def _prepare_training_args(
        self, 
        variant_name: str, 
        variant_config: Dict[str, Any],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Prepare training arguments for variant"""
        
        # Base arguments
        args = {
            'data': str(Path(self.config['paths']['data_root']) / 'dataset.yaml'),
            'epochs': self.ensemble_config['epochs'],
            'batch': self.ensemble_config['batch_size'],
            'imgsz': self.ensemble_config['imgsz'],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': self.general_config['num_workers'],
            'project': str(output_dir.parent),
            'name': variant_name,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'SGD',
            'verbose': True,
            'seed': self.general_config['seed'],
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'amp': self.general_config['mixed_precision'],
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': True,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_period': self.config['checkpointing']['save_interval'],
            'cache': False,
            'copy_paste': 0.0,
            'auto_augment': None,
            'erasing': 0.0,
            'crop_fraction': 1.0,
        }
        
        # Variant-specific adjustments
        if variant_name == 'high_recall':
            args.update({
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,  # Higher box loss weight
                'cls': 0.5,   # Lower classification loss
                'dfl': 1.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 10.0,  # More augmentation
                'translate': 0.2,
                'scale': 0.9,
                'shear': 5.0,
                'perspective': 0.001,
                'flipud': 0.1,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.15,  # More mixup for recall
            })
            
        elif variant_name == 'balanced':
            args.update({
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,  # Standard augmentation
            })
            
        elif variant_name == 'high_precision':
            args.update({
                'lr0': 0.005,  # Lower LR for precision
                'lrf': 0.005,
                'momentum': 0.95,
                'weight_decay': 0.001,  # Higher weight decay
                'warmup_epochs': 5.0,  # Longer warmup
                'warmup_momentum': 0.9,
                'warmup_bias_lr': 0.05,
                'box': 10.0,  # Much higher box loss
                'cls': 1.0,   # Higher classification loss
                'dfl': 2.0,
                'hsv_h': 0.01,  # Minimal augmentation
                'hsv_s': 0.5,
                'hsv_v': 0.3,
                'degrees': 0.0,
                'translate': 0.05,
                'scale': 0.3,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.3,
                'mosaic': 0.5,  # Less mosaic
                'mixup': 0.0,
            })
        
        return args
    
    def train_all_variants(self) -> Dict[str, str]:
        """Train all three ensemble variants"""
        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE TRAINING - Training all variants")
        logger.info("="*80 + "\n")
        
        variants = self.ensemble_config['variants']
        trained_models = {}
        
        for variant_name, variant_config in variants.items():
            try:
                model_path = self.train_variant(variant_name, variant_config)
                trained_models[variant_name] = model_path
                logger.info(f"\n✓ {variant_name} completed: {model_path}\n")
                
            except Exception as e:
                logger.error(f"\n✗ {variant_name} failed: {e}\n")
                raise
        
        # Save ensemble configuration
        ensemble_info = {
            'models': trained_models,
            'config': self.ensemble_config,
            'training_date': datetime.now().isoformat()
        }
        
        ensemble_config_path = self.checkpoint_dir / 'ensemble_config.yaml'
        with open(ensemble_config_path, 'w') as f:
            yaml.dump(ensemble_info, f)
        
        logger.info(f"\nEnsemble configuration saved to: {ensemble_config_path}")
        
        return trained_models
    
    def validate_ensemble(self, models: Dict[str, str]):
        """Validate all ensemble models"""
        logger.info("\n" + "="*80)
        logger.info("ENSEMBLE VALIDATION")
        logger.info("="*80 + "\n")
        
        results = {}
        
        for variant_name, model_path in models.items():
            logger.info(f"Validating {variant_name}...")
            
            model = YOLO(model_path)
            val_results = model.val(
                data=str(Path(self.config['paths']['data_root']) / 'dataset.yaml'),
                split='val',
                batch=self.ensemble_config['batch_size'],
                imgsz=self.ensemble_config['imgsz'],
                device='cuda' if torch.cuda.is_available() else 'cpu',
                verbose=True
            )
            
            results[variant_name] = {
                'mAP50': float(val_results.box.map50),
                'mAP50-95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr)
            }
            
            logger.info(f"{variant_name} results:")
            for metric, value in results[variant_name].items():
                logger.info(f"  {metric}: {value:.4f}")
            logger.info("")
        
        # Save validation results
        val_results_path = self.checkpoint_dir / 'validation_results.yaml'
        with open(val_results_path, 'w') as f:
            yaml.dump(results, f)
        
        logger.info(f"Validation results saved to: {val_results_path}")
        
        return results


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ensemble variants')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after training')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = EnsembleTrainer(args.config)
    
    # Train all variants
    trained_models = trainer.train_all_variants()
    
    # Validate if requested
    if args.validate:
        trainer.validate_ensemble(trained_models)
    
    logger.info("\n" + "="*80)
    logger.info("ENSEMBLE TRAINING COMPLETE")
    logger.info("="*80)
    logger.info("\nTrained models:")
    for variant, path in trained_models.items():
        logger.info(f"  {variant}: {path}")
    logger.info("")


if __name__ == '__main__':
    main()
