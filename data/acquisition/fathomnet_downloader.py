#!/usr/bin/env python3
"""
FathomNet Data Downloader with YOLO Format Conversion
Downloads images and bounding boxes from FathomNet API and converts to YOLO format
Saves directly to DigitalOcean Spaces bucket
Based on provided functional fathomnet_download__1_.py
"""

import os
import sys
import json
import logging
import hashlib
import tempfile
import glob
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import time
import pandas as pd
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
    prefix: str = None
    
    # Resumption settings
    checkpoint_interval: int = 100
    state_file: str = "download_state.json"
    batch_size: int = 100
    
    # Paths
    bucket_path: str = DO_BUCKET_PATH
    images_dir: str = "images/fathomnet"
    labels_dir: str = "images/fathomnet/labels"


class DownloadManager:
    """Manages Data Download"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.base_path = Path(self.config.bucket_path) / self.config.images_dir
        self.labels_path = Path(self.config.bucket_path) / self.config.labels_dir
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.labels_path.mkdir(parents=True, exist_ok=True)
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def upload_image(self, prefix: str, img: Dict, safe_concept: str, metadata: Dict) -> bool:
        """Download image from URL and upload to DO bucket"""
        
        # Set paths for image and label
        concept_path = self.base_path / "concepts" / safe_concept
        concept_path.mkdir(parents=True, exist_ok=True)
        
        # Use uuid as filename for consistency
        base_name = img.get('uuid')
        image_path = concept_path / f"{base_name}.jpg"
        metadata_path = concept_path / f"{base_name}_metadata.json"
        label_path = self.labels_path / f"{base_name}.txt"
        
        image_url = img.get('url')
        try:
            # Check if already exists
            if image_path.exists():
                logger.debug(f"Image already exists: {image_path}")
                return True

            # Download image
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()

            # Write metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Write image
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # Compress image to reduce storage
            image_file = Image.open(image_path)
            image_file.save(image_path, optimize=True, quality=85)
            
            # Create YOLO label if bounding boxes exist
            if 'bounding_boxes' in img and img['bounding_boxes']:
                self.create_yolo_label(img, label_path, image_file.size)
                
            logger.info(f"Successfully downloaded: {image_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {image_path}: {e}")
            return False
    
    def create_yolo_label(self, img: Dict, label_path: Path, img_size: Tuple[int, int]):
        """Convert bounding boxes to YOLO format and save"""
        width, height = img_size
        yolo_annotations = []
        
        for box in img.get('bounding_boxes', []):
            # Get class ID from concept mapping
            concept = box.get('concept', 'undefined')
            class_id = self.get_class_id(concept)
            
            # Convert to YOLO format (normalized center x, y, width, height)
            x = box.get('x', 0)
            y = box.get('y', 0) 
            w = box.get('width', 0)
            h = box.get('height', 0)
            
            if w > 0 and h > 0:
                cx = (x + w/2) / width
                cy = (y + h/2) / height
                norm_w = w / width
                norm_h = h / height
                
                # Ensure values are in [0, 1]
                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))
                
                yolo_annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        # Save YOLO format labels
        if yolo_annotations:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    def get_class_id(self, concept: str) -> int:
        """Get class ID for concept based on species mapping"""
        # Load species mapping
        mapping_path = Path("config/species_mapping.yaml")
        if mapping_path.exists():
            with open(mapping_path) as f:
                species_mapping = yaml.safe_load(f)
                
            # Check each category
            for cat_id, (category, species_list) in enumerate(
                [(k, v) for k, v in species_mapping.items() if k != 'class_names']
            ):
                for species_info in species_list:
                    if isinstance(species_info, dict):
                        if (concept.lower() in species_info.get('common_name', '').lower() or
                            concept.lower() in species_info.get('scientific_name', '').lower() or
                            concept.lower() in species_info.get('fathomnet_concept', '').lower()):
                            return cat_id
        
        # Default to last class (undefined)
        return 36
    
    def save_checkpoint(self, state: Dict, prefix: str):
        """Save download state for resumption"""
        checkpoint_path = self.base_path / "checkpoint" / "state.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, prefix: str) -> Optional[Dict]:
        """Load previous download state if exists"""
        checkpoint_path = self.base_path / "checkpoint" / "state.json"
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    state = json.load(f)
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
                return state
            except:
                return None
        return None

    def list_existing_images(self, prefix: str, safe_concept: str) -> Set[str]:
        """Get set of already downloaded image UUIDs"""
        concept_path = self.base_path / "concepts" / safe_concept
        
        existing = set()
        if concept_path.exists():
            for img_file in concept_path.glob("*.jpg"):
                # Extract UUID from filename
                existing.add(img_file.stem)
        
        logger.info(f"Found {len(existing)} existing images for {safe_concept}")
        return existing


class FathomNetDownloader:
    """Main downloader class for FathomNet data"""

    def __init__(self, config: DownloadConfig = None):
        self.config = config or DownloadConfig()
        self.downloader = DownloadManager(self.config)
        self.stats = {
            'total_downloaded': 0,
            'total_failed': 0,
            'concepts_processed': [],
            'start_time': datetime.utcnow().isoformat()
        }
        
    def get_concepts(self):
        """Get all concepts from FathomNet"""
        concepts = boundingboxes.find_concepts()
        if concepts:
            to_be_removed = {'', ' ', None, 'None'}
            concepts = [c for c in concepts if c not in to_be_removed]
            logger.info(f"Found {len(concepts)} concepts")
        else:
            logger.error("Error getting concepts")
            return []
        return concepts
        
    def get_concept_images(self, concept: str) -> List[Dict]:
        """Fetch image metadata and bounding boxes for a concept"""
        all_images = []

        try:
            logger.info(f"Searching for images of: {concept}")
            
            # Get images with this concept
            concept_images = images.find_by_concept(concept)
            logger.info(f"Found {len(concept_images)} images for: {concept}")
            
            # Get bounding boxes for each image
            for img_data in concept_images:
                img_dict = vars(img_data) if hasattr(img_data, '__dict__') else img_data
                
                if img_dict and img_dict.get('url'):
                    # Get bounding boxes for this image
                    img_uuid = img_dict.get('uuid')
                    boxes = boundingboxes.find_by_image_uuid(img_uuid) if img_uuid else []
                    
                    # Convert boxes to dict format
                    box_dicts = []
                    for box in boxes:
                        box_dict = vars(box) if hasattr(box, '__dict__') else box
                        if box_dict.get('concept') == concept:
                            box_dicts.append(box_dict)
                    
                    all_images.append({
                        'uuid': img_uuid,
                        'url': img_dict.get('url'),
                        'concept': concept,
                        'bounding_boxes': box_dicts,
                        'metadata': {
                            'latitude': img_dict.get('latitude'),
                            'longitude': img_dict.get('longitude'),
                            'depth_meters': img_dict.get('depth_meters'),
                            'timestamp': str(img_dict.get('lastUpdatedTimestamp', '')),
                            'contributors': img_dict.get('contributorsEmail')
                        }
                    })

            return all_images

        except Exception as e:
            logger.error(f"Failed to query concept {concept}: {e}")
            return []

    def download_batch(self, images: List[Dict], prefix: str) -> Dict[str, int]:
        """Download a batch of images in parallel"""
        results = {'success': 0, 'failed': 0}

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}

            for img in images:
                # Create safe concept name
                safe_concept = img.get('concept', '').lower()
                safe_concept = safe_concept.replace(' ', '_').replace('/', '-')
                safe_concept = safe_concept.replace('(', '').replace(')', '')
                safe_concept = safe_concept.replace('.', '').replace('\\', '_')

                # Prepare metadata
                metadata = {
                    'fathomnet_uuid': img.get('uuid'),
                    'concept': img.get('concept'),
                    'source_url': img.get('url'),
                    'download_time': datetime.utcnow().isoformat(),
                    **img.get('metadata', {})
                }

                # Submit download task
                future = executor.submit(
                    self.downloader.upload_image,
                    prefix,
                    img,
                    safe_concept,
                    metadata
                )
                futures[future] = img.get('uuid')

            # Process results
            for future in as_completed(futures):
                uuid = futures[future]
                try:
                    if future.result():
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                except Exception as e:
                    logger.error(f"Download failed for {uuid}: {e}")
                    results['failed'] += 1

        return results

    def run(self):
        """Main download execution"""
        logger.info("Starting FathomNet download job")
        
        # Get all concepts or use target concepts
        if not self.config.target_concepts:
            self.config.target_concepts = self.get_concepts()
        
        logger.info(f"Processing {len(self.config.target_concepts)} concepts")

        # Load checkpoint if exists
        checkpoint = self.downloader.load_checkpoint(self.config.prefix)
        if checkpoint:
            self.stats = checkpoint.get('stats', self.stats)
            processed_concepts = set(checkpoint.get('processed_concepts', []))
            logger.info(f"Resuming from checkpoint. Already processed: {len(processed_concepts)} concepts")
        else:
            processed_concepts = set()

        total_downloaded = 0

        # Process each concept
        for concept in self.config.target_concepts:
            if concept in processed_concepts:
                logger.info(f"Skipping already processed: {concept}")
                continue

            # Create safe concept name
            safe_concept = concept.lower().replace(' ', '_').replace('/', '-')
            safe_concept = safe_concept.replace('(', '').replace(')', '').replace('.', '')
            
            # Get existing images
            existing_images = self.downloader.list_existing_images(self.config.prefix, safe_concept)
                                                                                    
            logger.info(f"\nProcessing concept: {concept}")
                                                                                    
            # Get images for this concept
            concept_images = self.get_concept_images(concept)

            # Filter out already downloaded
            new_images = [
                img for img in concept_images
                if img.get('uuid') not in existing_images
            ]

            if not new_images:
                logger.info(f"No new images to download for {concept}")
                processed_concepts.add(concept)
                continue

            logger.info(f"Downloading {len(new_images)} new images for {concept}")

            # Download in batches
            for i in range(0, len(new_images), self.config.batch_size):
                batch = new_images[i:i + self.config.batch_size]
                logger.info(f"Processing batch {i//self.config.batch_size + 1}: {len(batch)} images")

                results = self.download_batch(batch, self.config.prefix)

                self.stats['total_downloaded'] += results['success']
                self.stats['total_failed'] += results['failed']
                total_downloaded += results['success']

                # Save checkpoint periodically
                if total_downloaded % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(processed_concepts)

            processed_concepts.add(concept)
            self.stats['concepts_processed'] = list(processed_concepts)

            # Save checkpoint after each concept
            self.save_checkpoint(processed_concepts)

        # Final statistics
        self.stats['end_time'] = datetime.utcnow().isoformat()
        self.save_final_report()

        logger.info("\n" + "="*50)
        logger.info("Download job completed!")
        logger.info(f"Total downloaded: {self.stats['total_downloaded']}")
        logger.info(f"Total failed: {self.stats['total_failed']}")
        logger.info(f"Concepts processed: {len(processed_concepts)}")
        logger.info("="*50)

    def save_checkpoint(self, processed_concepts: Set[str]):
        """Save current progress"""
        checkpoint = {
            'stats': self.stats,
            'processed_concepts': list(processed_concepts),
            'timestamp': datetime.utcnow().isoformat()
        }
        self.downloader.save_checkpoint(checkpoint, self.config.prefix)

    def save_final_report(self):
        """Save final download report"""
        report = {
            'config': asdict(self.config) if hasattr(self.config, '__dict__') else {},
            'stats': self.stats,
            'timestamp': datetime.utcnow().isoformat()
        }

        report_dir = Path(DO_BUCKET_PATH) / self.config.images_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / f"download_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Final report saved: {report_path}")


def main():
    """Main entry point"""
    config = DownloadConfig(
        prefix=datetime.now().strftime('%Y_%m_%d')
    )
    
    downloader = FathomNetDownloader(config)
    
    try:
        downloader.run()
    except Exception as e:
        logger.error(f"Download job failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()True)
        
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
                # Filter out empty or invalid concepts
                valid_concepts = [c for c in concepts if c and c.strip()]
                logger.info(f"Found {len(valid_concepts)} valid concepts")
                return valid_concepts
            else:
                logger.error("No concepts retrieved from FathomNet")
                return []
        except Exception as e:
            logger.error(f"Error getting concepts: {e}")
            return []
    
    def get_concept_bounding_boxes(self, concept: str) -> List[Dict]:
        """Get bounding boxes for a specific concept"""
        try:
            # Get bounding boxes with limit
            boxes = boundingboxes.find_by_concept(concept, limit=10000)
            logger.info(f"Found {len(boxes)} bounding boxes for {concept}")
            
            # Convert to dict format
            box_data = []
            for box in boxes:
                try:
                    box_dict = {
                        'uuid': getattr(box, 'uuid', None),
                        'image_uuid': getattr(box, 'image_uuid', None),
                        'concept': getattr(box, 'concept', concept),
                        'x': getattr(box, 'x', 0),
                        'y': getattr(box, 'y', 0),
                        'width': getattr(box, 'width', 0),
                        'height': getattr(box, 'height', 0),
                        'observer': getattr(box, 'observer', 'unknown')
                    }
                    if box_dict['image_uuid']:
                        box_data.append(box_dict)
                except Exception as e:
                    logger.debug(f"Error processing box: {e}")
                    continue
            
            return box_data
        except Exception as e:
            logger.error(f"Error getting bounding boxes for {concept}: {e}")
            return []
    
    def get_image_info(self, image_uuid: str) -> Optional[Dict]:
        """Get image information by UUID"""
        try:
            img_info = images.find_by_uuid(image_uuid)
            if img_info:
                return {
                    'uuid': img_info.uuid,
                    'url': img_info.url,
                    'width': getattr(img_info, 'width', None),
                    'height': getattr(img_info, 'height', None),
                    'timestamp': getattr(img_info, 'timestamp', None),
                    'latitude': getattr(img_info, 'latitude', None),
                    'longitude': getattr(img_info, 'longitude', None),
                    'depth_meters': getattr(img_info, 'depth_meters', None)
                }
            return None
        except Exception as e:
            logger.debug(f"Error getting image info for {image_uuid}: {e}")
            return None
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
    def download_image(self, img_info: Dict, output_path: Path) -> bool:
        """Download image from URL"""
        try:
            response = requests.get(img_info['url'], timeout=30, stream=True)
            response.raise_for_status()
            
            # Save image temporarily
            temp_path = output_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Optimize image
            img = Image.open(temp_path)
            
            # Resize if too large
            if img.size[0] > self.config.max_image_size[0] or img.size[1] > self.config.max_image_size[1]:
                img.thumbnail(self.config.max_image_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save optimized image
            img.save(output_path, 'JPEG', quality=self.config.image_quality, optimize=True)
            
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to download image {img_info['uuid']}: {e}")
            return False
    
    def convert_to_yolo(self, boxes: List[Dict], img_info: Dict) -> List[str]:
        """Convert bounding boxes to YOLO format"""
        yolo_lines = []
        
        img_width = img_info.get('width', 1920)
        img_height = img_info.get('height', 1080)
        
        for box in boxes:
            # Get class index
            concept_lower = box['concept'].lower()
            class_idx = self.class_to_idx.get(concept_lower, -1)
            
            if class_idx == -1:
                # Try to find partial match
                for key, idx in self.class_to_idx.items():
                    if key in concept_lower or concept_lower in key:
                        class_idx = idx
                        break
            
            if class_idx == -1:
                # Default to undefined class
                class_idx = len(self.class_to_idx) - 1
            
            # Convert to YOLO format (normalized center coordinates)
            x_center = (box['x'] + box['width'] / 2) / img_width
            y_center = (box['y'] + box['height'] / 2) / img_height
            width = box['width'] / img_width
            height = box['height'] / img_height
            
            # Clamp values to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
        
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
            img_uuid = box['image_uuid']
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
    if not Path("/datasets/marauder-do-bucket").exists():
        logger.error("DigitalOcean bucket not mounted at /datasets/marauder-do-bucket")
        logger.info("Please mount the bucket or update the bucket_path in config")
        sys.exit(1)
    
    # Initialize and run downloader
    downloader = FathomNetDownloader(config)
    
    try:
        downloader.run()
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        downloader.save_checkpoint()
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        downloader.save_checkpoint()
        sys.exit(1)


if __name__ == "__main__":
    main()
