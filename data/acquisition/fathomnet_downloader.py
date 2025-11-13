#!/usr/bin/env python3
"""
FathomNet Data Downloader with YOLO Format Conversion
Downloads images and bounding boxes from FathomNet API and converts to YOLO format
Saves directly to DigitalOcean Spaces bucket
"""

import os
import sys
import json
import logging
import tempfile
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import time
import yaml
from pathlib import Path

import requests
from fathomnet.api import boundingboxes, images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DO bucket path
DO_BUCKET_PATH = os.environ.get('DO_BUCKET_PATH', '/datasets/marauder-do-bucket')


@dataclass
class DownloadConfig:
    """Configuration for FathomNet downloader"""
    # Target species - can be None to download all
    target_concepts: List[str] = None
    
    # Resumption settings
    checkpoint_interval: int = 100
    batch_size: int = 100
    max_workers: int = 10
    
    # Paths
    bucket_path: str = DO_BUCKET_PATH
    images_dir: str = "images/fathomnet"
    labels_dir: str = "images/fathomnet/labels"
    
    # Species mapping file
    species_mapping_file: str = "config/species_mapping.yaml"


class FathomNetDownloader:
    """Enhanced FathomNet downloader with YOLO format conversion"""
    
    def __init__(self, config: DownloadConfig = None):
        self.config = config or DownloadConfig()
        
        # Setup paths
        self.images_path = Path(self.config.bucket_path) / self.config.images_dir
        self.labels_path = Path(self.config.bucket_path) / self.config.labels_dir
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.labels_path.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_downloaded': 0,
            'total_labels': 0,
            'total_failed': 0,
            'concepts_processed': [],
            'start_time': datetime.utcnow().isoformat()
        }
        
        # Load species mapping
        self.species_mapping = self._load_species_mapping()
        self.class_to_idx = self._create_class_mapping()
        
        # Checkpoint management
        self.checkpoint_file = self.images_path / "download_checkpoint.json"
        self.processed_concepts = set()
        self.processed_images = set()
        self._load_checkpoint()
    
    def _load_species_mapping(self) -> Dict:
        """Load species mapping from YAML file"""
        mapping_path = Path(self.config.species_mapping_file)
        if mapping_path.exists():
            with open(mapping_path) as f:
                return yaml.safe_load(f)
        else:
            # Create default mapping from project specs
            return self._create_default_mapping()
    
    def _create_default_mapping(self) -> Dict:
        """Create default species mapping based on project specifications"""
        mapping = {
            'critical_species': [
                'Pterois', 'Lionfish',
                'Strongylocentrotus purpuratus', 'Purple sea urchin',
                'Strongylocentrotus droebachiensis', 'Green sea urchin',
                'Mesocentrotus franciscanus', 'Red sea urchin',
                'Diadema antillarum', 'Long-spined sea urchin',
                'Haliotis sorenseni', 'White abalone',
                'Haliotis rufescens', 'Red abalone',
                'Haliotis kamtschatkana', 'Pinto abalone',
                'Archosargus probatocephalus', 'Sheepshead',
                'Pugettia gracilis', 'Kelp crab gracilis',
                'Pugettia producta', 'Kelp crab producta',
                'Enteroctopus dofleini', 'Giant pacific octopus',
                'Oncorhynchus mykiss', 'Trout',
                'Pycnopodia helianthoides', 'Sunflower sea star',
                'Labridae', 'Wrasse',
                'Urosalpinx cinerea', 'Oyster drills',
                'Lutjanus campechanus', 'Red snapper',
                'Zostera marina', 'Eel grass',
                'Sebastes', 'Rock fish'
            ],
            'important_species': [
                'Epinephelidae', 'Grouper',
                'Ostreidae', 'Oyster',
                'Oncorhynchus', 'Salmon',
                'Asteroidea', 'Sea star',
                'Nudibranchia', 'Sea slug',
                'Anguilliformes', 'Eel',
                'Caridea', 'Shrimp',
                'Scyphozoa', 'Jellyfish',
                'Hippocampus', 'Seahorse'
            ],
            'general_species': [
                'Selachimorpha', 'Shark',
                'Chelonioidea', 'Sea turtle',
                'Otariinae', 'Sea lion',
                'Enhydra lutris', 'Sea otter',
                'Xiphias gladius', 'Swordfish',
                'Cetacea', 'Whale',
                'Reef fish', 'Fish',
                'Undefined', 'Unknown'
            ]
        }
        
        # Save the default mapping
        os.makedirs('config', exist_ok=True)
        with open('config/species_mapping.yaml', 'w') as f:
            yaml.dump(mapping, f, default_flow_style=False)
        
        return mapping
    
    def _create_class_mapping(self) -> Dict[str, int]:
        """Create concept to class index mapping"""
        class_idx = {}
        idx = 0
        
        # Process each category in order (critical, important, general)
        for category in ['critical_species', 'important_species', 'general_species']:
            if category in self.species_mapping:
                # Process pairs (scientific name, common name)
                species_list = self.species_mapping[category]
                for i in range(0, len(species_list), 2):
                    if i + 1 < len(species_list):
                        scientific = species_list[i].lower()
                        common = species_list[i + 1].lower()
                        class_idx[scientific] = idx
                        class_idx[common] = idx
                        idx += 1
        
        return class_idx
    
    def get_all_concepts(self) -> List[str]:
        """Get all available concepts from FathomNet"""
        try:
            concepts = boundingboxes.find_concepts()
            if concepts:
                # Remove empty/invalid concepts
                to_be_removed = {'', ' ', None, 'None'}
                concepts = [c for c in concepts if c not in to_be_removed]
                logger.info(f"Found {len(concepts)} total concepts from FathomNet")
                return concepts
            else:
                logger.error("No concepts returned from FathomNet")
                return []
        except Exception as e:
            logger.error(f"Error fetching concepts: {e}")
            return []
    
    def get_concept_bounding_boxes(self, concept: str) -> List[Dict]:
        """Get all bounding boxes for a specific concept"""
        try:
            boxes = boundingboxes.find_by_concept(concept)
            if not boxes:
                return []
            
            # Convert to dict format
            box_dicts = []
            for box in boxes:
                try:
                    box_dict = vars(box) if hasattr(box, '__dict__') else box
                    if isinstance(box_dict, dict):
                        box_dicts.append(box_dict)
                except Exception as e:
                    logger.debug(f"Error converting box: {e}")
                    continue
            
            return box_dicts
        except Exception as e:
            logger.error(f"Error fetching bounding boxes for {concept}: {e}")
            return []
    
    def get_image_info(self, image_uuid: str) -> Optional[Dict]:
        """Get image metadata by UUID"""
        try:
            img = images.find_by_uuid(image_uuid)
            if img:
                img_dict = vars(img) if hasattr(img, '__dict__') else img
                return img_dict
            return None
        except Exception as e:
            logger.debug(f"Error fetching image {image_uuid}: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def download_image(self, img_info: Dict, output_path: Path) -> bool:
        """Download image from URL with retry logic"""
        url = img_info.get('url')
        if not url:
            return False
        
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Save image
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Optimize image
            with Image.open(output_path) as img:
                # Resize if too large
                max_size = (1920, 1080)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save with compression
                img.save(output_path, optimize=True, quality=85)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def get_class_id(self, concept: str) -> int:
        """Get class ID for a concept"""
        concept_lower = concept.lower()
        
        # Check direct mapping
        if concept_lower in self.class_to_idx:
            return self.class_to_idx[concept_lower]
        
        # Check partial matches
        for key, idx in self.class_to_idx.items():
            if key in concept_lower or concept_lower in key:
                return idx
        
        # Default to last class (undefined/unknown)
        return len(self.class_to_idx) - 1 if self.class_to_idx else 36
    
    def convert_to_yolo(self, boxes: List[Dict], img_info: Dict) -> List[str]:
        """Convert bounding boxes to YOLO format"""
        # Get image dimensions
        width = img_info.get('width', 0)
        height = img_info.get('height', 0)
        
        if width == 0 or height == 0:
            logger.warning(f"Invalid image dimensions for {img_info.get('uuid')}")
            return []
        
        yolo_lines = []
        for box in boxes:
            # Get bounding box coordinates
            x = box.get('x', 0)
            y = box.get('y', 0)
            w = box.get('width', 0)
            h = box.get('height', 0)
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Get class ID
            concept = box.get('concept', 'undefined')
            class_id = self.get_class_id(concept)
            
            # Convert to YOLO format (normalized center x, y, width, height)
            cx = (x + w/2) / width
            cy = (y + h/2) / height
            norm_w = w / width
            norm_h = h / height
            
            # Ensure values are in [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        return yolo_lines
    
    def process_concept(self, concept: str) -> Dict[str, int]:
        """Process a single concept - download images and create labels"""
        stats = {'images': 0, 'labels': 0, 'failed': 0}
        
        # Get bounding boxes for concept
        boxes = self.get_concept_bounding_boxes(concept)
        if not boxes:
            logger.warning(f"No bounding boxes found for {concept}")
            return stats
        
        # Group boxes by image UUID
        image_boxes = {}
        for box in boxes:
            img_uuid = box.get('image_uuid')
            if not img_uuid:
                continue
            if img_uuid not in image_boxes:
                image_boxes[img_uuid] = []
            image_boxes[img_uuid].append(box)
        
        logger.info(f"Processing {len(image_boxes)} images for {concept}")
        
        # Process each image
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for img_uuid, img_boxes in image_boxes.items():
                # Skip if already processed
                if img_uuid in self.processed_images:
                    continue
                
                # Get image info
                img_info = self.get_image_info(img_uuid)
                if not img_info or not img_info.get('url'):
                    continue
                
                # Submit processing task
                future = executor.submit(self.process_image, img_info, img_boxes)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['success']:
                        stats['images'] += 1
                        stats['labels'] += result['labels']
                        self.processed_images.add(result['uuid'])
                    else:
                        stats['failed'] += 1
                except Exception as e:
                    logger.error(f"Error processing future: {e}")
                    stats['failed'] += 1
        
        return stats
    
    def process_image(self, img_info: Dict, boxes: List[Dict]) -> Dict:
        """Process a single image - download and create label"""
        result = {'success': False, 'uuid': img_info['uuid'], 'labels': 0}
        
        # Define output paths with matching names
        safe_name = img_info['uuid'].replace('/', '_')
        img_path = self.images_path / f"{safe_name}.jpg"
        label_path = self.labels_path / f"{safe_name}.txt"
        
        try:
            # Download image if not exists
            if not img_path.exists():
                if not self.download_image(img_info, img_path):
                    return result
            
            # Create YOLO labels
            yolo_lines = self.convert_to_yolo(boxes, img_info)
            if yolo_lines:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                result['labels'] = len(yolo_lines)
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Error processing image {img_info['uuid']}: {e}")
        
        return result
    
    def run(self):
        """Main download execution"""
        logger.info("Starting FathomNet download")
        logger.info(f"Output directory: {self.images_path}")
        
        # Get target concepts
        if self.config.target_concepts:
            concepts = self.config.target_concepts
        else:
            concepts = self.get_all_concepts()
        
        if not concepts:
            logger.error("No concepts to process")
            return
        
        logger.info(f"Processing {len(concepts)} concepts")
        
        # Process each concept
        for i, concept in enumerate(concepts, 1):
            # Skip if already processed
            if concept in self.processed_concepts:
                logger.info(f"Skipping already processed concept: {concept}")
                continue
            
            logger.info(f"\n[{i}/{len(concepts)}] Processing concept: {concept}")
            
            try:
                # Process concept
                stats = self.process_concept(concept)
                
                # Update global stats
                self.stats['total_downloaded'] += stats['images']
                self.stats['total_labels'] += stats['labels']
                self.stats['total_failed'] += stats['failed']
                
                # Mark as processed
                self.processed_concepts.add(concept)
                self.stats['concepts_processed'].append(concept)
                
                # Save checkpoint periodically
                if i % self.config.checkpoint_interval == 0:
                    self.save_checkpoint()
                
                # Brief pause to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing concept {concept}: {e}")
                continue
        
        # Final save
        self.save_checkpoint()
        self.save_final_report()
        
        logger.info("\n" + "="*50)
        logger.info("Download completed!")
        logger.info(f"Total images: {self.stats['total_downloaded']}")
        logger.info(f"Total labels: {self.stats['total_labels']}")
        logger.info(f"Total failed: {self.stats['total_failed']}")
        logger.info(f"Concepts processed: {len(self.processed_concepts)}")
        logger.info("="*50)
    
    def save_checkpoint(self):
        """Save download checkpoint"""
        checkpoint = {
            'processed_concepts': list(self.processed_concepts),
            'processed_images': list(self.processed_images)[:1000],  # Save last 1000 to limit size
            'stats': self.stats,
            'timestamp': datetime.utcnow().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        logger.debug("Checkpoint saved")
    
    def _load_checkpoint(self):
        """Load checkpoint if exists"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    checkpoint = json.load(f)
                    self.processed_concepts = set(checkpoint.get('processed_concepts', []))
                    self.processed_images = set(checkpoint.get('processed_images', []))
                    self.stats.update(checkpoint.get('stats', {}))
                    logger.info(f"Loaded checkpoint with {len(self.processed_concepts)} processed concepts")
            except Exception as e:
                logger.warning(f"Error loading checkpoint: {e}")
    
    def save_final_report(self):
        """Save final download report"""
        report = {
            'config': asdict(self.config),
            'stats': self.stats,
            'end_time': datetime.utcnow().isoformat(),
            'class_mapping': {k: v for k, v in self.class_to_idx.items()},
            'total_concepts': len(self.processed_concepts),
            'total_images': self.stats['total_downloaded'],
            'total_labels': self.stats['total_labels']
        }
        
        report_path = self.images_path / f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Final report saved to: {report_path}")


def main():
    """Main entry point"""
    # Create configuration
    config = DownloadConfig()
    
    # Check DO bucket connection
    if not Path(DO_BUCKET_PATH).exists():
        logger.error(f"DigitalOcean bucket not mounted at {DO_BUCKET_PATH}")
        logger.info("Please mount the bucket or update the bucket_path in config")
        sys.exit(1)
    
    logger.info(f"Using bucket path: {DO_BUCKET_PATH}")
    
    # Initialize and run downloader
    downloader = FathomNetDownloader(config)
    
    try:
        downloader.run()
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        downloader.save_checkpoint()
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        downloader.save_checkpoint()
        sys.exit(1)


if __name__ == "__main__":
    main()
