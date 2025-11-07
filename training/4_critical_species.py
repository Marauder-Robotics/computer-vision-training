#!/usr/bin/env python3
"""
Week 4: Critical Species Specialization

Train specialized model with 5x oversampling for critical species (IDs 0-19).
Focus on maximizing detection accuracy for high-priority marine species.
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
from typing import Dict, List
import shutil

import torch
from ultralytics import YOLO
import wandb
from tqdm import tqdm


class CriticalSpeciesTrainer:
    """Specialized training for critical species detection"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.critical_config = self.config.get('critical_species', {})
        self.critical_class_ids = list(range(20))  # IDs 0-19
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def create_oversampled_dataset(
        self,
        data_dir: str,
        output_dir: str,
        oversample_factor: int = 5
    ):
        """
        Create dataset with oversampled critical species
        
        Args:
            data_dir: Source dataset directory
            output_dir: Output directory for oversampled dataset
            oversample_factor: Multiplication factor for critical species
        """
        print(f"Creating oversampled dataset with {oversample_factor}x critical species...")
        
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        for split in ['train', 'val']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Process each split
        for split in ['train', 'val']:
            images_dir = data_path / split / 'images'
            labels_dir = data_path / split / 'labels'
            
            out_images_dir = output_path / split / 'images'
            out_labels_dir = output_path / split / 'labels'
            
            # Get all image files
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            critical_count = 0
            regular_count = 0
            
            for img_file in tqdm(image_files, desc=f"Processing {split}"):
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    continue
                
                # Check if contains critical species
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                has_critical = False
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        class_id = int(parts[0])
                        if class_id in self.critical_class_ids:
                            has_critical = True
                            break
                
                # Determine number of copies
                num_copies = oversample_factor if has_critical else 1
                
                # Copy files
                for copy_idx in range(num_copies):
                    if copy_idx == 0:
                        # Original
                        out_img = out_images_dir / img_file.name
                        out_lbl = out_labels_dir / label_file.name
                    else:
                        # Copies with suffix
                        out_img = out_images_dir / f"{img_file.stem}_copy{copy_idx}{img_file.suffix}"
                        out_lbl = out_labels_dir / f"{label_file.stem}_copy{copy_idx}.txt"
                    
                    shutil.copy2(img_file, out_img)
                    shutil.copy2(label_file, out_lbl)
                
                if has_critical:
                    critical_count += num_copies
                else:
                    regular_count += 1
            
            print(f"{split.capitalize()} - Critical: {critical_count}, Regular: {regular_count}")
        
        # Copy dataset.yaml
        src_yaml = data_path / 'dataset.yaml'
        if src_yaml.exists():
            dst_yaml = output_path / 'dataset.yaml'
            shutil.copy2(src_yaml, dst_yaml)
            
            # Update paths in yaml
            with open(dst_yaml, 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            yaml_content['path'] = str(output_path.absolute())
            
            with open(dst_yaml, 'w') as f:
                yaml.dump(yaml_content, f)
        
        print(f"Oversampled dataset created at {output_path}")
        return output_path
    
    def train(
        self,
        data_yaml: str,
        baseline_weights: str,
        output_dir: str,
        resume: bool = False
    ):
        """
        Train critical species specialized model
        
        Args:
            data_yaml: Path to dataset.yaml (oversampled)
            baseline_weights: Path to baseline model weights
            output_dir: Output directory
            resume: Resume from checkpoint
        """
        # Initialize wandb
        wandb.init(
            project="marauder-cv",
            name="4-critical-species",
            config=self.critical_config
        )
        
        # Load baseline model
        print(f"Loading baseline model from {baseline_weights}")
        model = YOLO(baseline_weights)
        
        # Training parameters
        epochs = self.critical_config.get('epochs', 200)
        batch_size = self.critical_config.get('batch_size', 16)
        learning_rate = self.critical_config.get('learning_rate', 0.005)
        freeze_backbone = self.critical_config.get('freeze_backbone', 10)
        
        # Train
        print("Starting critical species specialization training...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            lr0=learning_rate,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            patience=50,
            
            # Freeze backbone initially
            freeze=freeze_backbone,
            
            # Augmentation (moderate for specialized training)
            mosaic=0.8,
            mixup=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            
            # Output
            project=output_dir,
            name='critical_species',
            exist_ok=True,
            resume=resume,
            
            # Hardware
            device=0,
            workers=8,
            
            # Logging
            verbose=True,
            plots=True
        )
        
        # Save final model
        output_path = Path(output_dir) / 'weights' / 'best.pt'
        print(f"Training complete! Model saved to {output_path}")
        
        # Log metrics
        wandb.log({
            'final_map50': results.results_dict.get('metrics/mAP50(B)', 0),
            'final_map50-95': results.results_dict.get('metrics/mAP50-95(B)', 0)
        })
        
        wandb.finish()
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Critical species specialized training')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Training config path')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Source dataset directory')
    parser.add_argument('--baseline-weights', type=str, required=True,
                       help='Path to baseline model weights')
    parser.add_argument('--output', type=str, default='outputs/4_critical_species',
                       help='Output directory')
    parser.add_argument('--oversample-factor', type=int, default=5,
                       help='Oversampling factor for critical species')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    trainer = CriticalSpeciesTrainer(args.config)
    
    # Create oversampled dataset
    oversampled_dir = Path(args.output) / 'oversampled_dataset'
    trainer.create_oversampled_dataset(
        args.data_dir,
        str(oversampled_dir),
        oversample_factor=args.oversample_factor
    )
    
    # Train on oversampled dataset
    data_yaml = oversampled_dir / 'dataset.yaml'
    trainer.train(
        data_yaml=str(data_yaml),
        baseline_weights=args.baseline_weights,
        output_dir=args.output,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
