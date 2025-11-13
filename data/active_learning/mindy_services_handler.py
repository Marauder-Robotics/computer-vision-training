#!/usr/bin/env python3
"""
Mindy Services Handler
Export/import annotations for external labeling
"""

import os
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MindyServicesHandler:
    """Handle annotation export/import for Mindy Services"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def export_for_annotation(
        self,
        image_paths: List[str],
        output_dir: Path,
        include_guidelines: bool = True
    ):
        """Export images and metadata for annotation"""
        logger.info(f"Exporting {len(image_paths)} images for annotation")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        # Copy images
        for img_path in image_paths:
            shutil.copy2(img_path, images_dir / Path(img_path).name)
        
        # Create COCO format metadata
        coco_data = self._create_coco_format(image_paths)
        with open(output_dir / 'annotations.json', 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        # Create guidelines
        if include_guidelines:
            self._create_guidelines(output_dir)
        
        # Create ZIP
        shutil.make_archive(str(output_dir), 'zip', output_dir)
        logger.info(f"Export package created: {output_dir}.zip")
    
    def import_annotations(
        self,
        coco_json_path: str,
        output_labels_dir: Path
    ) -> int:
        """Import COCO annotations and convert to YOLO format"""
        logger.info("Importing annotations from COCO format")
        
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to YOLO format
        converted = 0
        for img in coco_data['images']:
            img_id = img['id']
            img_width = img['width']
            img_height = img['height']
            
            # Get annotations for this image
            annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
            
            if not annotations:
                continue
            
            # Convert to YOLO format
            yolo_lines = []
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                yolo_lines.append(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Save label file
            label_path = output_labels_dir / f"{Path(img['file_name']).stem}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            converted += 1
        
        logger.info(f"Imported {converted} annotated images")
        return converted
    
    def _create_coco_format(self, image_paths: List[str]) -> Dict:
        """Create COCO format metadata"""
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories (species)
        with open('config/species_mapping.yaml', 'r') as f:
            species_config = yaml.safe_load(f)
        
        for category in ['critical', 'important', 'general']:
            for species in species_config['species'][category]:
                coco_data['categories'].append({
                    'id': species['id'],
                    'name': species['common_name'],
                    'supercategory': category
                })
        
        # Add images
        for idx, img_path in enumerate(image_paths):
            from PIL import Image
            img = Image.open(img_path)
            coco_data['images'].append({
                'id': idx,
                'file_name': Path(img_path).name,
                'width': img.width,
                'height': img.height
            })
        
        return coco_data
    
    def _create_guidelines(self, output_dir: Path):
        """Create annotation guidelines document"""
        guidelines = """
# Marauder CV - Annotation Guidelines

## General Instructions
1. Draw tight bounding boxes around each organism
2. Only label organisms with >30% visibility
3. Each individual organism should be labeled separately

## Handling Uncertainty
- Most certain: Use specific species name
- Uncertain: Use genus or family name
- Very uncertain: Use broad category (e.g., "Reef Fish")

## Quality Checklist
- Tight boxes around organisms
- Partial visibility included (>30%)
- Occlusion handled appropriately
- No boxes extending beyond image boundaries

## Critical Species (Priority)
These species require extra attention and double-checking:
- Lionfish
- All urchin species
- All abalone species
- Giant Pacific Octopus
- Sunflower Sea Star

Please contact us if you have any questions.
"""
        
        with open(output_dir / 'ANNOTATION_GUIDELINES.md', 'w') as f:
            f.write(guidelines)
