#!/usr/bin/env python3
"""
Dataset Organizer for Marauder CV Pipeline
Organizes images from multiple sources (FathomNet, DeepFish, Marauder) 
Sorts by species priority and creates train/val/test splits
Works directly with DigitalOcean Spaces mounted bucket
"""

import os
import sys
import json
import yaml
import shutil
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Organize and split datasets for training"""
    
    def __init__(self,
                 bucket_path: str = "/datasets/marauder-do-bucket",
                 species_mapping_file: str = "config/species_mapping.yaml",
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Initialize dataset organizer
        
        Args:
            bucket_path: Path to DO bucket mount
            species_mapping_file: Path to species mapping YAML
            split_ratios: (train, val, test) split ratios
        """
        self.bucket_path = Path(bucket_path)
        self.images_root = self.bucket_path / "images"
        self.training_root = self.bucket_path / "training" / "datasets"
        
        # Data sources
        self.sources = {
            'fathomnet': self.images_root / 'fathomnet',
            'deepfish': self.images_root / 'deepfish',
            'marauder': self.images_root / 'marauder'
        }
        
        # Split ratios
        self.train_ratio, self.val_ratio, self.test_ratio = split_ratios
        
        # Load species mapping
        self.species_mapping = self._load_species_mapping(species_mapping_file)
        self.priority_mapping = self._create_priority_mapping()
        
        # Track files
        self.file_registry = defaultdict(lambda: {
            'images': [], 'labels': [],
            'critical': [], 'important': [], 'general': []
        })
        
        # Stats
        self.stats = {
            'total_images': 0,
            'labeled_images': 0,
            'unlabeled_images': 0,
            'by_source': {},
            'by_priority': {'critical': 0, 'important': 0, 'general': 0},
            'by_split': {'train': 0, 'val': 0, 'test': 0}
        }
    
    def _load_species_mapping(self, mapping_file: str) -> Dict:
        """Load species mapping from YAML"""
        mapping_path = Path(mapping_file)
        if mapping_path.exists():
            with open(mapping_path) as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Species mapping not found: {mapping_file}")
            return self._get_default_mapping()
    
    def _get_default_mapping(self) -> Dict:
        """Get default species mapping based on project specs"""
        return {
            'critical_species': [
                'pterois', 'lionfish',
                'strongylocentrotus purpuratus', 'purple sea urchin',
                'strongylocentrotus droebachiensis', 'green sea urchin',
                'mesocentrotus franciscanus', 'red sea urchin',
                'diadema antillarum', 'long-spined sea urchin',
                'haliotis sorenseni', 'white abalone',
                'haliotis rufescens', 'red abalone',
                'haliotis kamtschatkana', 'pinto abalone',
                'archosargus probatocephalus', 'sheepshead',
                'pugettia gracilis', 'kelp crab',
                'pugettia producta', 'kelp crab',
                'enteroctopus dofleini', 'giant pacific octopus',
                'oncorhynchus mykiss', 'trout',
                'pycnopodia helianthoides', 'sunflower sea star',
                'labridae', 'wrasse',
                'urosalpinx cinerea', 'oyster drills',
                'lutjanus campechanus', 'red snapper',
                'zostera marina', 'eel grass',
                'sebastes', 'rock fish'
            ],
            'important_species': [
                'epinephelidae', 'grouper',
                'ostreidae', 'oyster',
                'oncorhynchus', 'salmon',
                'asteroidea', 'sea star',
                'nudibranchia', 'sea slug',
                'anguilliformes', 'eel',
                'caridea', 'shrimp',
                'scyphozoa', 'jellyfish',
                'hippocampus', 'seahorse'
            ],
            'general_species': [
                'selachimorpha', 'shark',
                'chelonioidea', 'sea turtle',
                'otariinae', 'sea lion',
                'enhydra lutris', 'sea otter',
                'xiphias gladius', 'swordfish',
                'cetacea', 'whale',
                'fish', 'reef fish',
                'undefined', 'unknown'
            ]
        }
    
    def _create_priority_mapping(self) -> Dict[str, str]:
        """Create mapping from species name to priority category with both common and scientific names"""
        mapping = {}
        
        for priority in ['critical_species', 'important_species', 'general_species']:
            if priority in self.species_mapping:
                species_list = self.species_mapping[priority]
                category = priority.replace('_species', '')
                
                # Handle both paired (scientific, common) and unpaired formats
                for species in species_list:
                    # Normalize species name - handle spaces, underscores, hyphens
                    normalized = self._normalize_species_name(species)
                    mapping[normalized] = category
                    
                    # Also add variations
                    # Remove genus (first word) for broader matching
                    words = normalized.split()
                    if len(words) > 1:
                        # Add species epithet only
                        mapping[words[-1]] = category
                        # Add without spaces
                        mapping[''.join(words)] = category
        
        return mapping
    
    def _normalize_species_name(self, name: str) -> str:
        """Normalize species name for matching"""
        name = name.lower().strip()
        # Replace various separators with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        # Remove extra spaces
        name = ' '.join(name.split())
        return name
    
    def scan_source(self, source_name: str) -> Dict:
        """
        Scan a data source for images and labels
        
        Args:
            source_name: Name of the source (fathomnet, deepfish, marauder)
            
        Returns:
            Dictionary with file statistics
        """
        source_path = self.sources.get(source_name)
        if not source_path or not source_path.exists():
            logger.warning(f"Source not found: {source_name}")
            return {}
        
        logger.info(f"Scanning {source_name} at {source_path}")
        
        source_stats = {
            'images': 0,
            'labels': 0,
            'unlabeled': 0,
            'by_priority': {'critical': 0, 'important': 0, 'general': 0}
        }
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(source_path.rglob(f'*{ext}'))
            images.extend(source_path.rglob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(images)} images in {source_name}")
        
        # Check for labels directory
        label_dirs = list(source_path.rglob('labels'))
        labels_map = {}
        
        if label_dirs:
            # Build label mapping
            for label_dir in label_dirs:
                for label_file in label_dir.glob('*.txt'):
                    base_name = label_file.stem
                    labels_map[base_name] = label_file
        
        # Process each image
        for img_path in images:
            img_name = img_path.stem
            
            # Check for corresponding label
            has_label = img_name in labels_map
            
            # Determine priority based on path or filename
            priority = self._determine_priority(img_path, labels_map.get(img_name))
            
            # Register file
            self.file_registry[source_name]['images'].append(img_path)
            if has_label:
                self.file_registry[source_name]['labels'].append(labels_map[img_name])
                source_stats['labels'] += 1
            else:
                source_stats['unlabeled'] += 1
            
            self.file_registry[source_name][priority].append(img_path)
            source_stats['by_priority'][priority] += 1
            source_stats['images'] += 1
        
        # Update global stats
        self.stats['by_source'][source_name] = source_stats
        self.stats['total_images'] += source_stats['images']
        self.stats['labeled_images'] += source_stats['labels']
        self.stats['unlabeled_images'] += source_stats['unlabeled']
        
        for priority, count in source_stats['by_priority'].items():
            self.stats['by_priority'][priority] += count
        
        return source_stats
    
    def _determine_priority(self, img_path: Path, label_path: Optional[Path] = None) -> str:
        """
        Determine species priority from file path or label content using robust matching
        
        Args:
            img_path: Image file path
            label_path: Optional label file path
            
        Returns:
            Priority category: 'critical', 'important', or 'general'
        """
        # Normalize the full path for checking
        path_str = self._normalize_species_name(str(img_path))
        
        # Check against all species mappings with partial matching
        best_match_priority = None
        longest_match = 0
        
        for species_name, priority in self.priority_mapping.items():
            # Try exact match first
            if species_name in path_str:
                # Prefer longer matches (more specific)
                if len(species_name) > longest_match:
                    longest_match = len(species_name)
                    best_match_priority = priority
        
        if best_match_priority:
            return best_match_priority
        
        # Check label content if available for class-based priority
        if label_path and label_path.exists():
            try:
                with open(label_path) as f:
                    # YOLO format: class_id x y w h
                    # Check if any class IDs in the label correspond to critical/important species
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if parts:
                                class_id = int(parts[0])
                                # Based on project specs: 
                                # IDs 0-19 are critical species (20 critical species)
                                if class_id < 20:
                                    return 'critical'
                                # IDs 20-28 are important species (9 important species)
                                elif class_id < 29:
                                    return 'important'
                                # IDs 29+ are general species
            except Exception as e:
                logger.debug(f"Could not parse label file {label_path}: {e}")
        
        # Default to general if no match found
        return 'general'
    
    def organize_by_priority(self, output_dir: Optional[Path] = None, 
                            use_symlink: bool = True,
                            resume: bool = True) -> None:
        """
        Organize datasets by species priority with checkpoint support
        Creates structure:
        - critical/
            - labeled/
            - unlabeled/
        - important/
            - labeled/
            - unlabeled/
        - general/
            - labeled/
            - unlabeled/
            
        Args:
            output_dir: Output directory for organized data
            use_symlink: Use symlinks instead of copying (default True)
            resume: Resume from checkpoint if available (default True)
        """
        if output_dir is None:
            output_dir = self.training_root / "organized"
        
        output_dir = Path(output_dir)
        logger.info(f"Organizing datasets by priority to {output_dir}")
        
        # Checkpoint file for resuming
        checkpoint_file = output_dir / ".organize_checkpoint.json"
        processed_files = set()
        
        if resume and checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    checkpoint_data = json.load(f)
                    processed_files = set(checkpoint_data.get('processed_files', []))
                    logger.info(f"Resuming from checkpoint: {len(processed_files)} files already processed")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        total_processed = len(processed_files)
        
        for source_name, source_files in self.file_registry.items():
            if not source_files['images']:
                continue
            
            logger.info(f"Processing {source_name}")
            
            # Create label mapping
            labels_map = {}
            for label_path in source_files['labels']:
                labels_map[label_path.stem] = label_path
            
            # Process each priority category
            for priority in ['critical', 'important', 'general']:
                priority_images = source_files[priority]
                
                if not priority_images:
                    continue
                
                # Separate labeled and unlabeled
                labeled_dir = output_dir / priority / 'labeled'
                unlabeled_dir = output_dir / priority / 'unlabeled'
                labeled_dir.mkdir(parents=True, exist_ok=True)
                unlabeled_dir.mkdir(parents=True, exist_ok=True)
                
                # Create labels subdirectory for labeled data
                labels_dir = labeled_dir / 'labels'
                labels_dir.mkdir(exist_ok=True)
                
                for img_path in priority_images:
                    # Skip if already processed
                    file_key = f"{source_name}:{img_path.name}"
                    if file_key in processed_files:
                        continue
                    
                    img_name = img_path.stem
                    
                    # Generate unique name to avoid conflicts
                    unique_name = f"{source_name}_{img_name}_{hashlib.md5(str(img_path).encode()).hexdigest()[:8]}"
                    
                    if img_name in labels_map:
                        # Link labeled image and label
                        dest_img = labeled_dir / f"{unique_name}.jpg"
                        dest_label = labels_dir / f"{unique_name}.txt"
                        
                        self._copy_or_link(img_path, dest_img, use_symlink=use_symlink)
                        self._copy_or_link(labels_map[img_name], dest_label, use_symlink=use_symlink)
                    else:
                        # Link unlabeled image
                        dest_img = unlabeled_dir / f"{unique_name}.jpg"
                        self._copy_or_link(img_path, dest_img, use_symlink=use_symlink)
                    
                    # Mark as processed
                    processed_files.add(file_key)
                    total_processed += 1
                    
                    # Save checkpoint every 100 files
                    if total_processed % 100 == 0:
                        with open(checkpoint_file, 'w') as f:
                            json.dump({
                                'processed_files': list(processed_files),
                                'timestamp': datetime.now().isoformat()
                            }, f)
                        logger.info(f"Checkpoint saved: {total_processed} files processed")
        
        # Final checkpoint save
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'processed_files': list(processed_files),
                'timestamp': datetime.now().isoformat(),
                'completed': True
            }, f)
        
        logger.info(f"Organization by priority complete: {total_processed} files processed")
    
    def create_splits(self, 
                     input_dir: Optional[Path] = None,
                     output_dir: Optional[Path] = None,
                     stratify: bool = True,
                     use_symlink: bool = True,
                     resume: bool = True) -> None:
        """
        Create train/val/test splits with checkpoint support
        
        Args:
            input_dir: Input directory with organized data
            output_dir: Output directory for splits
            stratify: Whether to stratify by priority category
            use_symlink: Use symlinks instead of copying (default True)
            resume: Resume from checkpoint if available (default True)
        """
        if input_dir is None:
            input_dir = self.training_root / "organized"
        if output_dir is None:
            output_dir = self.training_root / "splits"
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        logger.info(f"Creating splits from {input_dir} to {output_dir}")
        
        # Checkpoint file
        checkpoint_file = output_dir / ".split_checkpoint.json"
        
        # Check if splits already exist and complete
        if resume and checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    checkpoint_data = json.load(f)
                    if checkpoint_data.get('completed'):
                        logger.info("Splits already completed, skipping...")
                        return
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        
        # Collect all labeled data
        all_files = []
        
        for priority in ['critical', 'important', 'general']:
            labeled_dir = input_dir / priority / 'labeled'
            if not labeled_dir.exists():
                continue
            
            # Get image-label pairs
            images = list(labeled_dir.glob('*.jpg')) + list(labeled_dir.glob('*.png'))
            labels_dir = labeled_dir / 'labels'
            
            for img_path in images:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    all_files.append({
                        'image': img_path,
                        'label': label_path,
                        'priority': priority
                    })
        
        if not all_files:
            logger.warning("No labeled data found for splitting")
            return
        
        logger.info(f"Found {len(all_files)} labeled image-label pairs")
        
        # Set random seed for reproducibility
        random.seed(42)
        random.shuffle(all_files)
        
        if stratify:
            # Group by priority
            by_priority = defaultdict(list)
            for item in all_files:
                by_priority[item['priority']].append(item)
            
            # Split each priority group
            train_files, val_files, test_files = [], [], []
            
            for priority, items in by_priority.items():
                n = len(items)
                n_train = int(n * self.train_ratio)
                n_val = int(n * self.val_ratio)
                
                train_files.extend(items[:n_train])
                val_files.extend(items[n_train:n_train + n_val])
                test_files.extend(items[n_train + n_val:])
                
                logger.info(f"{priority}: {n_train} train, {n_val} val, {n - n_train - n_val} test")
        else:
            # Simple split
            n = len(all_files)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            train_files = all_files[:n_train]
            val_files = all_files[n_train:n_train + n_val]
            test_files = all_files[n_train + n_val:]
        
        # Create split directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_dir = output_dir / split_name
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Creating {split_name} split with {len(files)} files")
            
            for i, item in enumerate(files):
                # Create symlinks to avoid duplication
                img_name = item['image'].name
                self._copy_or_link(item['image'], images_dir / img_name, use_symlink=use_symlink)
                self._copy_or_link(item['label'], labels_dir / f"{item['image'].stem}.txt", use_symlink=use_symlink)
                
                # Checkpoint every 500 files
                if (i + 1) % 500 == 0:
                    logger.info(f"  {split_name}: {i + 1}/{len(files)} files processed")
            
            self.stats['by_split'][split_name] = len(files)
        
        # Create YOLO dataset.yaml
        self._create_dataset_yaml(output_dir)
        
        # Save completion checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'completed': True,
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats['by_split']
            }, f)
        
        logger.info("Split creation complete")
    
    def _copy_or_link(self, src: Path, dst: Path, use_symlink: bool = True) -> None:
        """
        Create symlink or move file (no copying to avoid duplication)
        
        Args:
            src: Source file path
            dst: Destination file path  
            use_symlink: If True, create symlink (default); if False, move file
        """
        if dst.exists():
            return
        
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        if use_symlink:
            # Create relative symlink to save space
            try:
                # Calculate relative path from dst to src
                rel_path = os.path.relpath(src, dst.parent)
                os.symlink(rel_path, dst)
                logger.debug(f"Created symlink: {dst} -> {src}")
            except (OSError, NotImplementedError) as e:
                # If symlinks not supported, fall back to hard link
                logger.debug(f"Symlink failed, using hard link: {e}")
                try:
                    os.link(src, dst)
                except (OSError, NotImplementedError):
                    # Last resort: copy
                    logger.warning(f"Link failed, copying file: {src}")
                    shutil.copy2(src, dst)
        else:
            # Move file (restructure without duplication)
            shutil.move(str(src), str(dst))
    
    def _create_dataset_yaml(self, output_dir: Path) -> None:
        """Create YOLO dataset configuration file"""
        # Create class names list
        nc = len(self.priority_mapping)
        names = []
        
        # Add species in priority order
        for priority in ['critical_species', 'important_species', 'general_species']:
            if priority in self.species_mapping:
                species_list = self.species_mapping[priority]
                # Take common names (every other item)
                for i in range(1, len(species_list), 2):
                    names.append(species_list[i])
        
        # Ensure we have the right number of classes
        while len(names) < nc:
            names.append(f"class_{len(names)}")
        
        dataset_config = {
            'path': str(output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': nc,
            'names': names[:nc]
        }
        
        yaml_path = output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        logger.info(f"Created dataset.yaml at {yaml_path}")
    
    def generate_report(self, output_path: Optional[Path] = None) -> None:
        """Generate organization report"""
        if output_path is None:
            output_path = self.training_root / f"organization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'sources_scanned': list(self.stats['by_source'].keys()),
            'species_mapping': self.species_mapping,
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'output_structure': {
                'organized_by_priority': str(self.training_root / "organized"),
                'train_val_test_splits': str(self.training_root / "splits"),
                'dataset_yaml': str(self.training_root / "splits" / "dataset.yaml")
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {output_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("Dataset Organization Summary")
        print("="*50)
        print(f"Total Images: {self.stats['total_images']}")
        print(f"  - Labeled: {self.stats['labeled_images']}")
        print(f"  - Unlabeled: {self.stats['unlabeled_images']}")
        print("\nBy Priority:")
        for priority, count in self.stats['by_priority'].items():
            print(f"  - {priority.capitalize()}: {count}")
        print("\nBy Source:")
        for source, stats in self.stats['by_source'].items():
            print(f"  - {source}: {stats['images']} images ({stats['labels']} labeled)")
        print("\nBy Split:")
        for split, count in self.stats['by_split'].items():
            print(f"  - {split}: {count}")
        print("="*50 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Organize datasets for Marauder CV training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Scan all sources and generate report
            python dataset_organizer.py --scan-only
            
            # Organize by priority and create splits
            python dataset_organizer.py --all
            
            # Only organize, don't create splits
            python dataset_organizer.py --organize
            
            # Resume from previous run
            python dataset_organizer.py --all --resume
            
            # Use file copying instead of symlinks
            python dataset_organizer.py --all --no-symlink
                    """
                )
    parser.add_argument('--bucket-path', default='/datasets/marauder-do-bucket',
                       help='Path to DO bucket mount (default: /datasets/marauder-do-bucket)')
    parser.add_argument('--species-mapping', default='config/species_mapping.yaml',
                       help='Path to species mapping YAML file')
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan sources without organizing')
    parser.add_argument('--organize', action='store_true',
                       help='Organize by priority')
    parser.add_argument('--split', action='store_true',
                       help='Create train/val/test splits')
    parser.add_argument('--all', action='store_true',
                       help='Run all steps (scan, organize, split)')
    parser.add_argument('--no-symlink', action='store_true',
                       help='Use file copying instead of symlinks')
    parser.add_argument('--no-resume', dest='resume', action='store_false', default=True,
                       help='Do not resume from checkpoint')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        logger.error(f"Split ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)
    
    # Check bucket connection
    if not Path(args.bucket_path).exists():
        logger.error(f"Bucket not mounted at {args.bucket_path}")
        logger.error("Please ensure DigitalOcean Spaces bucket is mounted")
        sys.exit(1)
    
    # Initialize organizer
    logger.info("Initializing dataset organizer...")
    organizer = DatasetOrganizer(
        bucket_path=args.bucket_path,
        species_mapping_file=args.species_mapping,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio)
    )
    
    # Scan sources
    logger.info("="*60)
    logger.info("Scanning data sources...")
    logger.info("="*60)
    for source in ['fathomnet', 'deepfish', 'marauder']:
        try:
            stats = organizer.scan_source(source)
            if stats:
                logger.info(f"{source}: {stats['images']} images "
                          f"({stats['labels']} labeled, {stats['unlabeled']} unlabeled)")
        except Exception as e:
            logger.error(f"Error scanning {source}: {e}")
    
    if not args.scan_only:
        # Organize by priority
        if args.organize or args.all:
            logger.info("\n" + "="*60)
            logger.info("Organizing by priority...")
            logger.info("="*60)
            try:
                organizer.organize_by_priority(
                    use_symlink=not args.no_symlink,
                    resume=args.resume
                )
            except Exception as e:
                logger.error(f"Error organizing: {e}")
                raise
        
        # Create splits
        if args.split or args.all:
            logger.info("\n" + "="*60)
            logger.info("Creating train/val/test splits...")
            logger.info("="*60)
            try:
                organizer.create_splits(
                    use_symlink=not args.no_symlink,
                    resume=args.resume
                )
            except Exception as e:
                logger.error(f"Error creating splits: {e}")
                raise
    
    # Generate report
    logger.info("\n" + "="*60)
    logger.info("Generating report...")
    logger.info("="*60)
    organizer.generate_report()
    
    logger.info("\n" + "="*60)
    logger.info("Dataset organization complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
