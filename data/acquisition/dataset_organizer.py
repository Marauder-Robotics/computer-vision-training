#!/usr/bin/env python3
"""
Dataset Organizer
Organizes images and labels into train/val/test splits
"""

import os
import yaml
import shutil
import random
from pathlib import Path
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Organize dataset into train/val/test splits"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_root = Path(self.config['paths']['data_root'])
        random.seed(self.config['general']['seed'])
    
    def organize_dataset(
        self,
        source_images: Path,
        source_labels: Path,
        output_dir: Path,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ):
        """
        Organize dataset into train/val/test splits
        
        Args:
            source_images: Source images directory
            source_labels: Source labels directory
            output_dir: Output directory
            split_ratios: (train, val, test) ratios
        """
        logger.info("Organizing dataset...")
        logger.info(f"  Source images: {source_images}")
        logger.info(f"  Source labels: {source_labels}")
        logger.info(f"  Output: {output_dir}")
        logger.info(f"  Split ratios: {split_ratios}")
        
        # Get all image files
        image_files = self._get_image_files(source_images)
        logger.info(f"Found {len(image_files)} images")
        
        # Shuffle
        random.shuffle(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        
        splits = {
            'train': image_files[:n_train],
            'val': image_files[n_train:n_train+n_val],
            'test': image_files[n_train+n_val:]
        }
        
        # Create split directories
        for split_name, files in splits.items():
            logger.info(f"\nProcessing {split_name} split ({len(files)} images)")
            
            split_img_dir = output_dir / 'images' / split_name
            split_lbl_dir = output_dir / 'labels' / split_name
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            for img_file in files:
                # Copy image
                dest_img = split_img_dir / img_file.name
                shutil.copy2(img_file, dest_img)
                
                # Copy label if exists
                label_file = source_labels / f"{img_file.stem}.txt"
                if label_file.exists():
                    dest_lbl = split_lbl_dir / label_file.name
                    shutil.copy2(label_file, dest_lbl)
        
        logger.info(f"\nDataset organized successfully!")
        logger.info(f"  Train: {len(splits['train'])} images")
        logger.info(f"  Val: {len(splits['val'])} images")
        logger.info(f"  Test: {len(splits['test'])} images")
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files recursively"""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for ext in extensions:
            image_files.extend(directory.rglob(f'*{ext}'))
            image_files.extend(directory.rglob(f'*{ext.upper()}'))
        
        return image_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize dataset')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--source-images', type=str, required=True)
    parser.add_argument('--source-labels', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--split', nargs=3, type=float, default=[0.7, 0.15, 0.15])
    
    args = parser.parse_args()
    
    organizer = DatasetOrganizer(args.config)
    organizer.organize_dataset(
        Path(args.source_images),
        Path(args.source_labels),
        Path(args.output),
        tuple(args.split)
    )


if __name__ == '__main__':
    main()
