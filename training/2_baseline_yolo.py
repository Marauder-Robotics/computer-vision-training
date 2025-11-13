#!/usr/bin/env python3
"""
Week 1: Baseline YOLO Training

Train baseline YOLOv8l model with SSL pretrained backbone on high-confidence Fathomnet data.
"""

import argparse
import os
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO
from utils.training_logger import TrainingLogger
from utils.checkpoint_manager import CheckpointManager


class BaselineYOLOTrainer:
    """Baseline YOLO training with SSL backbone"""
    
    def __init__(self, config_path: str):
        self.do_bucket_path = os.environ.get('DO_BUCKET_PATH', '/datasets/marauder-do-bucket')
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_2 = self.config['baseline_2']
        self.data_config = self.config['datasets']
        
        # Load species config
        species_config_path = Path(config_path).parent / 'species_config.yaml'
        with open(species_config_path, 'r') as f:
            self.species_config = yaml.safe_load(f)
    
    def prepare_dataset_yaml(self, data_dir: Path, output_path: Path):
        """
        Create dataset.yaml for YOLO training
        
        Args:
            data_dir: Root directory of dataset
            output_path: Path to save dataset.yaml
        """
        dataset_yaml = {
            'path': str(data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': self.species_config['num_classes'],
            'names': {}
        }
        
        # Build class names dictionary
        for category in ['critical', 'important', 'general']:
            for species in self.species_config['species'][category]['species_list']:
                class_id = species['id']
                common_name = species['common_name']
                dataset_yaml['names'][class_id] = common_name
        
        # Save
        with open(output_path, 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        print(f"Dataset config saved to {output_path}")
        return output_path
    
    def load_ssl_backbone(self, model: YOLO, ssl_weights_path: str) -> YOLO:
        """
        Load SSL pretrained backbone weights
        
        Args:
            model: YOLO model
            ssl_weights_path: Path to SSL backbone weights
            
        Returns:
            Model with loaded backbone
        """
        if not Path(ssl_weights_path).exists():
            print(f"Warning: SSL weights not found at {ssl_weights_path}")
            print("Training from scratch...")
            return model
        
        # Load SSL weights
        ssl_state = torch.load(ssl_weights_path, map_location='cpu')
        
        # Load into model backbone (requires matching architecture)
        try:
            model.model.backbone.load_state_dict(ssl_state, strict=False)
            print(f"Loaded SSL backbone from {ssl_weights_path}")
        except Exception as e:
            print(f"Warning: Could not load SSL weights: {e}")
            print("Training from scratch...")
        
        return model
    
    def train(
        self,
        data_yaml: str,
        output_dir: str,
        ssl_weights: str = None,
        resume: bool = False
    ):
        """
        Train baseline YOLO model
        
        Args:
            data_yaml: Path to dataset.yaml
            output_dir: Output directory for weights
            ssl_weights: Path to SSL pretrained weights
            resume: Resume from checkpoint
        """
        # Initialize training logger
        logger = TrainingLogger(
            project_name="marauder-cv",
            run_name="2-baseline-yolov8l",
            save_dir=f"{self.do_bucket_path}/training/logs",
            config=self.config_2
        )
        
        # Initialize model
        model_name = self.config_2['model'].lower()
        model = YOLO(f'{model_name}.pt')  # Start from official weights
        
        # Load SSL backbone if available
        if ssl_weights:
            model = self.load_ssl_backbone(model, ssl_weights)
        
        # Training parameters
        hyperparams = self.config_2['hyperparams']
        
        # Train
        results = model.train(
            data=data_yaml,
            epochs=hyperparams['epochs'],
            batch=hyperparams['batch_size'],
            imgsz=hyperparams['image_size'],
            lr0=hyperparams['learning_rate'],
            momentum=hyperparams['momentum'],
            weight_decay=hyperparams['weight_decay'],
            warmup_epochs=hyperparams['warmup_epochs'],
            patience=self.config_2['early_stopping']['patience'],
            
            # Augmentation
            mosaic=1.0,
            mixup=0.15,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            
            # Output
            project=output_dir,
            name='baseline_yolov8l',
            exist_ok=True,
            resume=resume,
            
            # Hardware
            device=0,  # GPU 0
            workers=8,
            
            # Logging
            verbose=True,
            plots=True
        )
        
        # Save final model
        output_path = Path(output_dir) / 'weights' / 'best.pt'
        print(f"\nTraining complete! Best model saved to {output_path}")
        
        # Log final metrics
        logger.log({
            'final_map50': results.results_dict['metrics/mAP50(B)'],
            'final_map50-95': results.results_dict['metrics/mAP50-95(B)'],
        })
        
        logger.finish()
        
        return results
    
    def validate(self, weights_path: str, data_yaml: str):
        """
        Validate trained model
        
        Args:
            weights_path: Path to model weights
            data_yaml: Path to dataset.yaml
        """
        model = YOLO(weights_path)
        
        results = model.val(
            data=data_yaml,
            batch=16,
            plots=True,
            verbose=True
        )
        
        print("\nValidation Results:")
        print(f"mAP@50: {results.results_dict['metrics/mAP50(B)']:.4f}")
        print(f"mAP@50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Train baseline YOLO model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to training config')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to prepared dataset')
    parser.add_argument('--output', type=str, default='outputs/2_baseline',
                        help='Output directory')
    parser.add_argument('--ssl-weights', type=str, default=None,
                        help='Path to SSL pretrained weights')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only run validation')
    parser.add_argument('--weights', type=str, default=None,
                        help='Weights for validation')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BaselineYOLOTrainer(args.config)
    
    # Prepare dataset.yaml
    data_yaml = Path(args.data_dir) / 'dataset.yaml'
    if not data_yaml.exists():
        data_yaml = trainer.prepare_dataset_yaml(
            Path(args.data_dir),
            data_yaml
        )
    
    if args.validate_only:
        if not args.weights:
            raise ValueError("--weights required for validation")
        trainer.validate(args.weights, str(data_yaml))
    else:
        trainer.train(
            data_yaml=str(data_yaml),
            output_dir=args.output,
            ssl_weights=args.ssl_weights,
            resume=args.resume
        )


if __name__ == '__main__':
    main()
