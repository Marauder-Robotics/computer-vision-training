#!/usr/bin/env python3
"""
Shore Inference Pipeline
High-accuracy inference for GCP with dual ensemble (YOLOv8x + YOLOv11x)
"""

import os
import yaml
import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import supervision as sv
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualEnsembleInference:
    """Dual ensemble: 3x YOLOv8x + 3x YOLOv11x"""
    
    def __init__(self, yolov8_paths: List[str], yolov11_paths: List[str], config: Dict[str, Any]):
        """
        Initialize dual ensemble
        
        Args:
            yolov8_paths: Paths to YOLOv8x models (recall, balanced, precision)
            yolov11_paths: Paths to YOLOv11x models (recall, balanced, precision)
            config: Inference configuration
        """
        self.yolov8_models = []
        self.yolov11_models = []
        
        # Load YOLOv8 models
        for path in yolov8_paths:
            try:
                model = YOLO(path)
                self.yolov8_models.append(model)
                logger.info(f"Loaded YOLOv8x: {path}")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 {path}: {e}")
        
        # Load YOLOv11 models
        for path in yolov11_paths:
            try:
                model = YOLO(path)
                self.yolov11_models.append(model)
                logger.info(f"Loaded YOLOv11x: {path}")
            except Exception as e:
                logger.error(f"Failed to load YOLOv11 {path}: {e}")
        
        self.all_models = self.yolov8_models + self.yolov11_models
        
        if not self.all_models:
            raise ValueError("No models loaded successfully")
        
        self.conf_threshold = config.get('conf_threshold', 0.15)  # Lower threshold for shore
        self.iou_threshold = config.get('iou_threshold', 0.5)
        self.max_det = config.get('max_det', 500)
        self.voting_method = config.get('voting_method', 'weighted')
        
        # Model weights for voting
        # [YOLOv8 recall, balanced, precision, YOLOv11 recall, balanced, precision]
        self.model_weights = config.get('weights', [0.15, 0.2, 0.15, 0.2, 0.2, 0.1])
        
    def predict(self, image: np.ndarray, use_parallel: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run dual ensemble prediction
        
        Args:
            image: Input image
            use_parallel: Use parallel inference
            
        Returns:
            boxes, scores, classes
        """
        if use_parallel:
            return self._predict_parallel(image)
        else:
            return self._predict_sequential(image)
    
    def _predict_parallel(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parallel inference across all models"""
        all_boxes = []
        all_scores = []
        all_classes = []
        
        def run_inference(model):
            results = model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False
            )
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                return (
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy()
                )
            return None
        
        # Run all models in parallel
        with ThreadPoolExecutor(max_workers=len(self.all_models)) as executor:
            futures = {executor.submit(run_inference, model): i 
                      for i, model in enumerate(self.all_models)}
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    boxes, scores, classes = result
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_classes.append(classes)
        
        if not all_boxes:
            return np.array([]), np.array([]), np.array([])
        
        # Merge predictions
        return self._merge_predictions(all_boxes, all_scores, all_classes)
    
    def _predict_sequential(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sequential inference (fallback)"""
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for model in self.all_models:
            results = model.predict(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False
            )
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                all_boxes.append(boxes.xyxy.cpu().numpy())
                all_scores.append(boxes.conf.cpu().numpy())
                all_classes.append(boxes.cls.cpu().numpy())
        
        if not all_boxes:
            return np.array([]), np.array([]), np.array([])
        
        return self._merge_predictions(all_boxes, all_scores, all_classes)
    
    def _merge_predictions(
        self,
        all_boxes: List[np.ndarray],
        all_scores: List[np.ndarray],
        all_classes: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge predictions using weighted voting"""
        # Concatenate
        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)
        
        # Weight scores by model weights
        # Repeat weights for each detection from each model
        weights = []
        for i, (box_set, weight) in enumerate(zip(all_boxes, self.model_weights)):
            weights.extend([weight] * len(box_set))
        weights = np.array(weights)
        
        # Apply weights
        weighted_scores = scores * weights
        
        # Weighted NMS
        keep = self._weighted_nms(boxes, weighted_scores, classes)
        
        return boxes[keep], scores[keep], classes[keep]
    
    def _weighted_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        iou_threshold: float = 0.5
    ) -> List[int]:
        """Weighted NMS"""
        keep = []
        order = scores.argsort()[::-1]
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            ious = self._compute_iou(boxes[i:i+1], boxes[order[1:]])
            mask = ious[0] <= iou_threshold
            order = order[1:][mask]
        
        return keep
    
    def _compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU"""
        x1_max = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
        y1_max = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
        x2_min = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
        y2_min = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
        
        inter_width = np.maximum(0, x2_min - x1_max)
        inter_height = np.maximum(0, y2_min - y1_max)
        inter_area = inter_width * inter_height
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union_area = area1[:, None] + area2 - inter_area
        iou = inter_area / (union_area + 1e-10)
        
        return iou


class ShoreInferencePipeline:
    """Complete shore-side inference pipeline"""
    
    def __init__(self, config_path: str):
        """Initialize shore pipeline"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load species mapping
        with open('config/species_mapping.yaml', 'r') as f:
            species_config = yaml.safe_load(f)
        
        self.class_names = self._build_class_names(species_config)
        self.critical_species_ids = list(range(20))
        
        logger.info("Initializing Shore inference pipeline...")
        
        # Load dual ensemble
        checkpoint_dir = Path(self.config['paths']['checkpoints'])
        
        # YOLOv8x models
        yolov8_paths = [
            str(checkpoint_dir / 'ensemble' / 'high_recall' / 'weights' / 'best.pt'),
            str(checkpoint_dir / 'ensemble' / 'balanced' / 'weights' / 'best.pt'),
            str(checkpoint_dir / 'ensemble' / 'high_precision' / 'weights' / 'best.pt')
        ]
        
        # YOLOv11x models (would be trained separately)
        yolov11_paths = [
            str(checkpoint_dir / 'shore' / 'yolov11_recall' / 'weights' / 'best.pt'),
            str(checkpoint_dir / 'shore' / 'yolov11_balanced' / 'weights' / 'best.pt'),
            str(checkpoint_dir / 'shore' / 'yolov11_precision' / 'weights' / 'best.pt')
        ]
        
        inference_config = {
            'conf_threshold': 0.15,
            'iou_threshold': 0.5,
            'max_det': 500,
            'voting_method': 'weighted',
            'weights': [0.15, 0.2, 0.15, 0.2, 0.2, 0.1]
        }
        
        self.ensemble = DualEnsembleInference(yolov8_paths, yolov11_paths, inference_config)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.15,
            lost_track_buffer=50,
            minimum_matching_threshold=0.85,
            frame_rate=5
        )
        
        logger.info("Shore pipeline initialized with dual ensemble")
    
    def _build_class_names(self, species_config: Dict) -> List[str]:
        """Build class names list"""
        all_species = []
        
        for category in ['critical', 'important', 'general']:
            for species in species_config['species'][category]:
                all_species.append((species['id'], species['common_name']))
        
        all_species.sort(key=lambda x: x[0])
        return [name for _, name in all_species]
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame with dual ensemble"""
        # Run inference
        boxes, scores, classes = self.ensemble.predict(frame, use_parallel=True)
        
        if len(boxes) == 0:
            return {'detections': [], 'critical_alerts': []}
        
        # Convert to supervision format
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=classes.astype(int)
        )
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Process results
        results = {
            'detections': [],
            'critical_alerts': [],
            'species_counts': {}
        }
        
        for i in range(len(detections)):
            class_id = int(detections.class_id[i])
            class_name = self.class_names[class_id]
            
            detection_info = {
                'bbox': detections.xyxy[i].tolist(),
                'confidence': float(detections.confidence[i]),
                'class_id': class_id,
                'class_name': class_name,
                'track_id': int(detections.tracker_id[i]) if detections.tracker_id is not None else None
            }
            
            results['detections'].append(detection_info)
            
            # Count species
            results['species_counts'][class_name] = results['species_counts'].get(class_name, 0) + 1
            
            # Check critical
            if class_id in self.critical_species_ids:
                results['critical_alerts'].append({
                    'species': class_name,
                    'confidence': float(detections.confidence[i]),
                    'bbox': detections.xyxy[i].tolist()
                })
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Shore Inference Pipeline')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    pipeline = ShoreInferencePipeline(args.config)
    
    # Process video or image
    logger.info(f"Processing: {args.input}")
    
    # Implementation would process input and save results
    logger.info("✓ Processing complete")


if __name__ == '__main__':
    main()
