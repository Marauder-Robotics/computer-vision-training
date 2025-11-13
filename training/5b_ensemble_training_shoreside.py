#!/usr/bin/env python3
"""
Train Shore Models (YOLOv11x Ensemble)
======================================

Trains the YOLOv11x ensemble for shoreside deployment on GCP.
Uses dual ensemble architecture: 3x YOLOv8x (from nano) + 3x YOLOv11x (shore).

Architecture:
- High Recall variant (conf=0.15, iou=0.4)
- Balanced variant (conf=0.25, iou=0.5)
- High Precision variant (conf=0.35, iou=0.6)

Training uses the same data pipeline as nano models but optimized for:
- Maximum accuracy (no energy constraints)
- Real-time processing (10+ FPS on cloud GPUs)
- Cost efficiency on GCP infrastructure

Author: Marauder CV Team
Date: November 2025
"""

import os
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import torch
from ultralytics import YOLO
from utils.training_logger import TrainingLogger
from utils.checkpoint_manager import CheckpointManager


class ShoreModelTrainer:
    """
    Trains YOLOv11x models for shoreside deployment.
    
    Follows the same training pipeline as nano models:
    1. Load pretrained YOLOv11x base
    2. Train with critical species specialization
    3. Apply hard negative mining
    4. Optimize for recall/balanced/precision variants
    5. Export for GCP deployment
    
    Attributes:
        config: Training configuration
        variants: List of model variants to train
        output_dir: Output directory for checkpoints
        logger: Logging instance
    """
    
    def __init__(self, config_path: str):
        self.do_bucket_path = os.environ.get('DO_BUCKET_PATH', '/datasets/marauder-do-bucket')
        """
        Initialize shore model trainer.
        
        Args:
            config_path: Path to training configuration YAML
        """
        self.config = self._load_config(config_path)
        self.variants = ['recall', 'balanced', 'precision']
        self.output_dir = Path(self.config.get('output_dir', 'checkpoints/shore'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize training logger (replaces wandb)
        self._init_training_logger()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self):
        """Configure logging."""
        log_dir = Path(f'{self.do_bucket_path}/training/logs/shore_training')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'shore_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ShoreModelTrainer')
        self.logger.info(f"Logging to {log_file}")
    
    def _init_training_logger(self):
        """Initialize TrainingLogger for metrics tracking."""
        self.training_logger = TrainingLogger(
            project_name='marauder-cv-shore',
            run_name=f'shore_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            save_dir=f'{self.do_bucket_path}/training/logs',
            config=self.config
        )
        self.logger.info("Training logger initialized")
    
    def get_variant_config(self, variant: str) -> Dict:
        """
        Get hyperparameters for specific variant.
        
        Args:
            variant: One of 'recall', 'balanced', 'precision'
            
        Returns:
            Dictionary of hyperparameters optimized for the variant
        """
        base_config = {
            'epochs': self.config.get('epochs', 300),
            'batch_size': self.config.get('batch_size', 16),
            'imgsz': self.config.get('imgsz', 640),
            'device': self.config.get('device', 0),
            'workers': self.config.get('workers', 8),
            'patience': self.config.get('patience', 50),
            'save_period': self.config.get('save_period', 10),
            'cache': True,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': True,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True,
            'save': True,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'visualize': False,
            'augment': True,
            'agnostic_nms': False,
            'retina_masks': False,
            'format': 'onnx',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': None,
            'workspace': 4,
            'nms': True,
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
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
            'bgr': 0.0,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0,
        }
        
        # Variant-specific optimizations
        variant_configs = {
            'recall': {
                # Optimized for high recall - catch everything
                'conf': 0.15,  # Low confidence threshold
                'iou': 0.4,    # Low IoU threshold for NMS
                'max_det': 500,  # Allow many detections
                'augment': True,  # Enable test-time augmentation
                'label_smoothing': 0.1,  # Help with edge cases
                'cls': 0.3,  # Lower classification weight (prioritize detection)
                'box': 7.5,  # Standard box weight
                'dfl': 1.5,  # Standard DFL weight
                'hsv_h': 0.02,  # More color variation
                'hsv_s': 0.8,
                'hsv_v': 0.5,
                'mixup': 0.1,  # Add mixup for robustness
                'copy_paste': 0.1,  # Add copy-paste for rare species
                'mosaic': 1.0,  # Full mosaic augmentation
            },
            'balanced': {
                # Balanced precision and recall
                'conf': 0.25,  # Moderate confidence threshold
                'iou': 0.5,    # Standard IoU threshold
                'max_det': 300,  # Standard max detections
                'augment': True,
                'label_smoothing': 0.0,
                'cls': 0.5,  # Balanced classification weight
                'box': 7.5,
                'dfl': 1.5,
                'hsv_h': 0.015,  # Standard augmentation
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'mixup': 0.05,
                'copy_paste': 0.05,
                'mosaic': 1.0,
            },
            'precision': {
                # Optimized for high precision - only confident detections
                'conf': 0.35,  # High confidence threshold
                'iou': 0.6,    # High IoU threshold for NMS
                'max_det': 200,  # Fewer detections
                'augment': False,  # No test-time augmentation
                'label_smoothing': 0.0,
                'cls': 0.7,  # Higher classification weight
                'box': 7.5,
                'dfl': 1.5,
                'hsv_h': 0.01,  # Less augmentation
                'hsv_s': 0.6,
                'hsv_v': 0.3,
                'mixup': 0.0,  # No mixup
                'copy_paste': 0.0,  # No copy-paste
                'mosaic': 0.8,  # Reduced mosaic
            }
        }
        
        # Merge base config with variant-specific config
        config = {**base_config, **variant_configs[variant]}
        
        # Set project and name for this variant
        config['project'] = str(self.output_dir)
        config['name'] = variant
        
        return config
    
    def train_variant(
        self,
        variant: str,
        data_yaml: str,
        resume: bool = False,
        pretrained_weights: Optional[str] = None
    ) -> Tuple[YOLO, Dict]:
        """
        Train a single YOLOv11x variant.
        
        Args:
            variant: One of 'recall', 'balanced', 'precision'
            data_yaml: Path to dataset YAML file
            resume: Whether to resume training from checkpoint
            pretrained_weights: Path to pretrained weights (optional)
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Training Shore Model - {variant.upper()} Variant")
        self.logger.info(f"{'='*80}\n")
        
        # Get variant-specific configuration
        config = self.get_variant_config(variant)
        
        # Initialize model
        if resume:
            checkpoint_path = self.output_dir / variant / 'weights' / 'last.pt'
            if checkpoint_path.exists():
                self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                model = YOLO(str(checkpoint_path))
            else:
                self.logger.warning(f"No checkpoint found at {checkpoint_path}, starting fresh")
                model = YOLO('yolov11x.pt')
        elif pretrained_weights:
            self.logger.info(f"Loading pretrained weights: {pretrained_weights}")
            model = YOLO(pretrained_weights)
        else:
            self.logger.info("Starting from YOLOv11x pretrained weights")
            model = YOLO('yolov11x.pt')
        
        # Train model
        self.logger.info(f"Training with configuration:")
        self.logger.info(json.dumps(config, indent=2))
        
        try:
            results = model.train(
                data=data_yaml,
                **config
            )
            
            # Extract metrics
            metrics = {
                'variant': variant,
                'box_loss': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'cls_loss': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'dfl_loss': float(results.results_dict.get('train/box_loss', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
            }
            
            self.logger.info(f"\n{variant.upper()} Variant Training Complete!")
            self.logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
            
            # Log metrics to training logger
            self.training_logger.log({f'{variant}_{k}': v for k, v in metrics.items()})
            
            return model, metrics
            
        except Exception as e:
            self.logger.error(f"Error training {variant} variant: {str(e)}")
            raise
    
    def validate_model(
        self,
        model: YOLO,
        variant: str,
        data_yaml: str
    ) -> Dict:
        """
        Validate trained model on test set.
        
        Args:
            model: Trained YOLO model
            variant: Model variant name
            data_yaml: Path to dataset YAML
            
        Returns:
            Dictionary of validation metrics
        """
        self.logger.info(f"\nValidating {variant} variant...")
        
        # Get variant config for validation parameters
        config = self.get_variant_config(variant)
        
        try:
            results = model.val(
                data=data_yaml,
                conf=config['conf'],
                iou=config['iou'],
                max_det=config['max_det'],
                split='test',
                save_json=True,
                save_hybrid=True,
                plots=True
            )
            
            val_metrics = {
                'variant': variant,
                'box_P': float(results.box.p),
                'box_R': float(results.box.r),
                'box_mAP50': float(results.box.map50),
                'box_mAP50_95': float(results.box.map),
                'speed_preprocess': float(results.speed['preprocess']),
                'speed_inference': float(results.speed['inference']),
                'speed_postprocess': float(results.speed['postprocess']),
            }
            
            # Add per-class metrics for critical species
            critical_species_indices = self.config.get('critical_species_indices', [0, 1, 2, 3])
            for idx in critical_species_indices:
                if idx < len(results.box.maps):
                    val_metrics[f'class_{idx}_mAP50'] = float(results.box.maps[idx])
            
            self.logger.info(f"Validation metrics: {json.dumps(val_metrics, indent=2)}")
            
            self.training_logger.log({f'{variant}_val_{k}': v for k, v in val_metrics.items()})
            
            return val_metrics
            
        except Exception as e:
            self.logger.error(f"Error validating {variant} variant: {str(e)}")
            raise
    
    def export_model(
        self,
        model: YOLO,
        variant: str,
        formats: List[str] = ['onnx', 'torchscript']
    ):
        """
        Export model to deployment formats.
        
        Args:
            model: Trained YOLO model
            variant: Model variant name
            formats: List of export formats
        """
        self.logger.info(f"\nExporting {variant} variant to formats: {formats}")
        
        export_dir = self.output_dir / variant / 'export'
        export_dir.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            try:
                self.logger.info(f"Exporting to {fmt}...")
                exported_path = model.export(
                    format=fmt,
                    imgsz=self.config.get('imgsz', 640),
                    half=True,  # FP16 for faster inference on GCP
                    optimize=True,
                    simplify=True,
                    workspace=4,
                    nms=True
                )
                self.logger.info(f"Exported to: {exported_path}")
                
            except Exception as e:
                self.logger.error(f"Error exporting to {fmt}: {str(e)}")
    
    def create_ensemble_config(self):
        """
        Create ensemble configuration file for deployment.
        
        This config will be used by the shore inference pipeline to load
        and ensemble all 6 models (3x YOLOv8x + 3x YOLOv11x).
        """
        ensemble_config = {
            'yolov8_models': self.config.get('yolov8_models', [
                'checkpoints/ensemble/recall/best.pt',
                'checkpoints/ensemble/balanced/best.pt',
                'checkpoints/ensemble/precision/best.pt'
            ]),
            'yolov11_models': [
                str(self.output_dir / 'recall' / 'weights' / 'best.pt'),
                str(self.output_dir / 'balanced' / 'weights' / 'best.pt'),
                str(self.output_dir / 'precision' / 'weights' / 'best.pt')
            ],
            'voting_method': self.config.get('voting_method', 'weighted'),
            'weights': self.config.get('weights', [0.15, 0.2, 0.15, 0.2, 0.2, 0.1]),
            'conf_thresholds': {
                'yolov8': {
                    'recall': 0.15,
                    'balanced': 0.25,
                    'precision': 0.35
                },
                'yolov11': {
                    'recall': 0.15,
                    'balanced': 0.25,
                    'precision': 0.35
                }
            },
            'iou_thresholds': {
                'yolov8': {
                    'recall': 0.4,
                    'balanced': 0.5,
                    'precision': 0.6
                },
                'yolov11': {
                    'recall': 0.4,
                    'balanced': 0.5,
                    'precision': 0.6
                }
            },
            'nms_iou': 0.5,
            'max_detections': 300,
            'critical_species': {
                'ids': self.config.get('critical_species_indices', [0, 1, 2, 3]),
                'min_confidence': 0.25
            }
        }
        
        config_path = self.output_dir / 'ensemble_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(ensemble_config, f, default_flow_style=False)
        
        self.logger.info(f"Ensemble configuration saved to: {config_path}")
        
        return ensemble_config
    
    def train_all_variants(
        self,
        data_yaml: str,
        resume: bool = False
    ) -> Dict[str, Dict]:
        """
        Train all three variants (recall, balanced, precision).
        
        Args:
            data_yaml: Path to dataset YAML file
            resume: Whether to resume training from checkpoints
            
        Returns:
            Dictionary mapping variant names to their metrics
        """
        all_metrics = {}
        trained_models = {}
        
        for variant in self.variants:
            self.logger.info(f"\n{'#'*80}")
            self.logger.info(f"# Starting {variant.upper()} variant training")
            self.logger.info(f"{'#'*80}\n")
            
            try:
                # Train variant
                model, train_metrics = self.train_variant(
                    variant=variant,
                    data_yaml=data_yaml,
                    resume=resume
                )
                
                # Validate variant
                val_metrics = self.validate_model(
                    model=model,
                    variant=variant,
                    data_yaml=data_yaml
                )
                
                # Export variant
                self.export_model(
                    model=model,
                    variant=variant,
                    formats=['onnx', 'torchscript']
                )
                
                # Store results
                all_metrics[variant] = {
                    'train': train_metrics,
                    'validation': val_metrics
                }
                trained_models[variant] = model
                
                self.logger.info(f"\n{variant.upper()} variant complete!")
                
            except Exception as e:
                self.logger.error(f"Failed to train {variant} variant: {str(e)}")
                continue
        
        # Save all metrics
        metrics_path = self.output_dir / 'all_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        self.logger.info(f"\nAll metrics saved to: {metrics_path}")
        
        # Create ensemble configuration
        self.create_ensemble_config()
        
        # Generate summary report
        self._generate_summary_report(all_metrics)
        
        return all_metrics
    
    def _generate_summary_report(self, all_metrics: Dict[str, Dict]):
        """Generate a summary report of training results."""
        report_lines = [
            "\n" + "="*80,
            "SHORE MODEL TRAINING SUMMARY (YOLOv11x)",
            "="*80,
            "",
            f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Output directory: {self.output_dir}",
            "",
            "-"*80,
            "MODEL PERFORMANCE",
            "-"*80,
            ""
        ]
        
        # Add metrics for each variant
        for variant in self.variants:
            if variant in all_metrics:
                metrics = all_metrics[variant]
                val = metrics.get('validation', {})
                
                report_lines.extend([
                    f"{variant.upper()} Variant:",
                    f"  mAP50:       {val.get('box_mAP50', 0):.4f}",
                    f"  mAP50-95:    {val.get('box_mAP50_95', 0):.4f}",
                    f"  Precision:   {val.get('box_P', 0):.4f}",
                    f"  Recall:      {val.get('box_R', 0):.4f}",
                    f"  Inference:   {val.get('speed_inference', 0):.2f}ms",
                    ""
                ])
        
        # Add deployment information
        report_lines.extend([
            "-"*80,
            "DEPLOYMENT",
            "-"*80,
            "",
            "Model Locations:",
            f"  Recall:      {self.output_dir / 'recall' / 'weights' / 'best.pt'}",
            f"  Balanced:    {self.output_dir / 'balanced' / 'weights' / 'best.pt'}",
            f"  Precision:   {self.output_dir / 'precision' / 'weights' / 'best.pt'}",
            "",
            "Ensemble Config: checkpoints/shore/ensemble_config.yaml",
            "",
            "Next Steps:",
            "  1. Test ensemble inference with shore_inference.py",
            "  2. Deploy to GCP Vertex AI or Cloud Run",
            "  3. Configure auto-scaling based on load",
            "  4. Set up monitoring and alerting",
            "",
            "Dual Ensemble Architecture:",
            "  - 3x YOLOv8x models (from nano training)",
            "  - 3x YOLOv11x models (these shore models)",
            "  - Total: 6 models with weighted voting",
            "  - Expected performance: mAP50 0.75-0.80",
            "",
            "="*80,
            ""
        ])
        
        report = "\n".join(report_lines)
        
        # Print to console
        print(report)
        
        # Save to file
        report_path = self.output_dir / 'training_summary.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Summary report saved to: {report_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv11x models for shoreside deployment'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='config/dataset.yaml',
        help='Path to dataset YAML file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--variant',
        type=str,
        choices=['recall', 'balanced', 'precision', 'all'],
        default='all',
        help='Which variant to train'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ShoreModelTrainer(args.config)
    
    # Train models
    if args.variant == 'all':
        trainer.train_all_variants(
            data_yaml=args.data,
            resume=args.resume
        )
    else:
        # Train single variant
        model, train_metrics = trainer.train_variant(
            variant=args.variant,
            data_yaml=args.data,
            resume=args.resume
        )
        
        # Validate
        val_metrics = trainer.validate_model(
            model=model,
            variant=args.variant,
            data_yaml=args.data
        )
        
        # Export
        trainer.export_model(
            model=model,
            variant=args.variant
        )
    
    print("\n✅ Shore model training complete!")
    print(f"📁 Models saved to: {trainer.output_dir}")
    print(f"🚀 Ready for GCP deployment")


if __name__ == '__main__':
    main()
