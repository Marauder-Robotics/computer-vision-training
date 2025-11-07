#!/usr/bin/env python3
"""
Fathomnet Data Acquisition
Downloads marine species images from Fathomnet API and converts to YOLO format
"""

import os
import yaml
import requests
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from PIL import Image
import io
import boto3
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FathomnetDownloader:
    """Download and process Fathomnet data"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load species mapping
        with open('config/species_mapping.yaml', 'r') as f:
            self.species_config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['paths']['fathomnet'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build concept mapping
        self.concept_to_class = self._build_concept_mapping()
        
        # S3/Spaces client (optional)
        self.s3_client = None
        if os.getenv('DO_SPACES_KEY'):
            self.s3_client = boto3.client(
                's3',
                endpoint_url=os.getenv('DO_SPACES_ENDPOINT'),
                aws_access_key_id=os.getenv('DO_SPACES_KEY'),
                aws_secret_access_key=os.getenv('DO_SPACES_SECRET'),
                region_name=os.getenv('DO_SPACES_REGION', 'nyc3')
            )
    
    def _build_concept_mapping(self) -> Dict[str, int]:
        """Build mapping from Fathomnet concepts to YOLO class IDs"""
        mapping = {}
        
        for category in ['critical', 'important', 'general']:
            for species in self.species_config['species'][category]:
                class_id = species['id']
                for concept in species['fathomnet_concepts']:
                    mapping[concept.lower()] = class_id
        
        return mapping
    
    def download_images(
        self,
        concepts: List[str],
        images_per_concept: int = 1000,
        max_workers: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Download images for specified concepts
        
        Args:
            concepts: List of Fathomnet concept names
            images_per_concept: Max images per concept
            max_workers: Parallel download workers
            
        Returns:
            List of downloaded image metadata
        """
        logger.info(f"Downloading images for {len(concepts)} concepts")
        
        all_images = []
        
        for concept in concepts:
            logger.info(f"\nProcessing concept: {concept}")
            
            # Query Fathomnet API
            images = self._query_fathomnet(concept, images_per_concept)
            logger.info(f"  Found {len(images)} images")
            
            if not images:
                continue
            
            # Download images in parallel
            download_dir = self.output_dir / 'images' / concept.replace(' ', '_')
            download_dir.mkdir(parents=True, exist_ok=True)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._download_image, img, download_dir): img
                    for img in images
                }
                
                for future in tqdm(as_completed(futures), total=len(images), desc=f"  Downloading"):
                    result = future.result()
                    if result:
                        all_images.append(result)
        
        logger.info(f"\nDownloaded {len(all_images)} images total")
        return all_images
    
    def _query_fathomnet(self, concept: str, limit: int) -> List[Dict]:
        """Query Fathomnet API for images"""
        base_url = "https://fathomnet.org/api/v1/images/find"
        
        params = {
            'concept': concept,
            'limit': limit,
            'offset': 0
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to query {concept}: {e}")
            return []
    
    def _download_image(self, image_data: Dict, output_dir: Path) -> Optional[Dict]:
        """Download single image and metadata"""
        try:
            image_url = image_data.get('url')
            if not image_url:
                return None
            
            # Download image
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Open and compress to JPG
            img = Image.open(io.BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save
            image_id = image_data.get('id', str(hash(image_url)))
            output_path = output_dir / f"{image_id}.jpg"
            img.save(output_path, 'JPEG', quality=75, optimize=True)
            
            return {
                'path': str(output_path),
                'metadata': image_data,
                'width': img.width,
                'height': img.height
            }
            
        except Exception as e:
            logger.debug(f"Failed to download image: {e}")
            return None
    
    def convert_to_yolo(self, images: List[Dict]) -> int:
        """Convert annotations to YOLO format"""
        logger.info("\nConverting annotations to YOLO format")
        
        labels_dir = self.output_dir / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        converted = 0
        
        for img in tqdm(images, desc="Converting"):
            try:
                # Get annotations from metadata
                annotations = img['metadata'].get('annotations', [])
                if not annotations:
                    continue
                
                # Get image dimensions
                img_width = img['width']
                img_height = img['height']
                
                # Convert annotations
                yolo_annotations = []
                
                for ann in annotations:
                    concept = ann.get('concept', '').lower()
                    
                    # Map to class ID
                    class_id = self.concept_to_class.get(concept)
                    if class_id is None:
                        continue
                    
                    # Get bounding box
                    bbox = ann.get('boundingBox')
                    if not bbox:
                        continue
                    
                    x = bbox.get('x', 0)
                    y = bbox.get('y', 0)
                    w = bbox.get('width', 0)
                    h = bbox.get('height', 0)
                    
                    # Convert to YOLO format (normalized center coordinates)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width_norm = w / img_width
                    height_norm = h / img_height
                    
                    # Clamp to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width_norm = max(0, min(1, width_norm))
                    height_norm = max(0, min(1, height_norm))
                    
                    yolo_annotations.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                    )
                
                if yolo_annotations:
                    # Save label file
                    image_path = Path(img['path'])
                    label_path = labels_dir / image_path.parent.name / f"{image_path.stem}.txt"
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    
                    converted += 1
                    
            except Exception as e:
                logger.debug(f"Failed to convert annotations: {e}")
        
        logger.info(f"Converted {converted} images to YOLO format")
        return converted
    
    def create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training"""
        dataset_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 36,
            'names': self._get_class_names()
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f)
        
        logger.info(f"Created dataset.yaml: {yaml_path}")
    
    def _get_class_names(self) -> List[str]:
        """Get ordered list of class names"""
        all_species = []
        
        for category in ['critical', 'important', 'general']:
            for species in self.species_config['species'][category]:
                all_species.append((species['id'], species['common_name']))
        
        all_species.sort(key=lambda x: x[0])
        return [name for _, name in all_species]


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download Fathomnet data')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--concepts', nargs='+', help='Specific concepts to download')
    parser.add_argument('--limit', type=int, default=1000, help='Images per concept')
    
    args = parser.parse_args()
    
    downloader = FathomnetDownloader(args.config)
    
    # Download all concepts if not specified
    if not args.concepts:
        args.concepts = list(downloader.concept_to_class.keys())
    
    # Download
    images = downloader.download_images(args.concepts, args.limit)
    
    # Convert to YOLO
    if images:
        downloader.convert_to_yolo(images)
        downloader.create_dataset_yaml()
    
    logger.info("\nDownload complete!")


if __name__ == '__main__':
    main()
