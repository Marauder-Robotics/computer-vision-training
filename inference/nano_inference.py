#!/usr/bin/env python3
"""
Nano Inference Pipeline
Complete inference system for Jetson Nano with ensemble + TTA + ByteTrack counting
"""

import os
import yaml
import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import supervision as sv
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Single detection result"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class FrameResult:
    """Results for a single frame"""
    frame_id: int
    timestamp: float
    detections: List[DetectionResult] = field(default_factory=list)
    species_counts: Dict[str, int] = field(default_factory=dict)
    critical_species: List[str] = field(default_factory=list)


class EnsembleInference:
    """Ensemble inference with TTA"""
    
    def __init__(self, model_paths: List[str], config: Dict[str, Any]):
        """
        Initialize ensemble
        
        Args:
            model_paths: List of TensorRT engine paths
            config: Inference configuration
        """
        self.models = []
        for path in model_paths:
            try:
                model = YOLO(path, task='detect')
                self.models.append(model)
                logger.info(f"Loaded model: {path}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        
        if not self.models:
            raise ValueError("No models loaded successfully")
        
        self.conf_threshold = config['conf_threshold']
        self.iou_threshold = config['iou_threshold']
        self.max_det = config['max_det']
        
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run ensemble prediction
        
        Returns:
            boxes, scores, classes
        """
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for model in self.models:
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
        
        # Concatenate all predictions
        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)
        classes = np.concatenate(all_classes)
        
        # Apply NMS
        keep = self._weighted_nms(boxes, scores, classes)
        
        return boxes[keep], scores[keep], classes[keep]
    
    def _weighted_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        iou_threshold: float = 0.5
    ) -> np.ndarray:
        """Weighted NMS for ensemble"""
        keep = []
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            ious = self._compute_iou(boxes[i:i+1], boxes[order[1:]])
            
            # Find boxes with low IoU (keep them)
            mask = ious[0] <= iou_threshold
            order = order[1:][mask]
        
        return np.array(keep)
    
    def _compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between boxes"""
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


class ByteTrackCounter:
    """Object tracking and counting using ByteTrack"""
    
    def __init__(self):
        """Initialize ByteTrack tracker"""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=5
        )
        
        # Counting statistics
        self.species_counts = {}
        self.track_history = {}  # track_id -> list of detections
        
    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        class_names: List[str]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Update tracker with new detections
        
        Returns:
            track_ids, species_counts
        """
        # Convert to supervision format
        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=classes.astype(int)
        )
        
        # Update tracker
        detections = self.tracker.update_with_detections(detections)
        
        # Update counts
        for class_id, track_id in zip(detections.class_id, detections.tracker_id):
            species = class_names[int(class_id)]
            
            # Initialize tracking
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'species': species,
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now(),
                    'frames': 0
                }
                
                # Increment species count
                self.species_counts[species] = self.species_counts.get(species, 0) + 1
            
            # Update tracking
            self.track_history[track_id]['last_seen'] = datetime.now()
            self.track_history[track_id]['frames'] += 1
        
        return detections.tracker_id, self.species_counts.copy()
    
    def reset(self):
        """Reset tracker and counts"""
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=5
        )
        self.species_counts = {}
        self.track_history = {}


class NanoInferencePipeline:
    """Complete inference pipeline for Jetson Nano"""
    
    def __init__(self, config_path: str):
        """Initialize pipeline"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load species mapping
        with open('config/species_mapping.yaml', 'r') as f:
            species_config = yaml.safe_load(f)
        
        self.class_names = self._build_class_names(species_config)
        self.critical_species_ids = list(range(20))  # IDs 0-19
        
        # Initialize components
        logger.info("Initializing Nano inference pipeline...")
        
        # Load ensemble models
        ensemble_dir = Path(self.config['paths']['checkpoints']) / 'tensorrt' / 'nano_deployment' / 'engines'
        model_paths = list(ensemble_dir.glob('*.engine'))
        
        if not model_paths:
            raise FileNotFoundError(f"No TensorRT engines found in {ensemble_dir}")
        
        inference_config = {
            'conf_threshold': 0.25,
            'iou_threshold': 0.5,
            'max_det': 300
        }
        
        self.ensemble = EnsembleInference([str(p) for p in model_paths], inference_config)
        self.tracker = ByteTrackCounter()
        
        logger.info(f"Pipeline initialized with {len(model_paths)} models")
    
    def _build_class_names(self, species_config: Dict) -> List[str]:
        """Build class names list"""
        all_species = []
        
        for category in ['critical', 'important', 'general']:
            for species in species_config['species'][category]:
                all_species.append((species['id'], species['common_name']))
        
        all_species.sort(key=lambda x: x[0])
        return [name for _, name in all_species]
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> FrameResult:
        """
        Process a single frame
        
        Args:
            frame: Input frame (BGR)
            frame_id: Frame number
            
        Returns:
            FrameResult with detections and counts
        """
        # Run ensemble inference
        boxes, scores, classes = self.ensemble.predict(frame)
        
        if len(boxes) == 0:
            return FrameResult(frame_id=frame_id, timestamp=datetime.now().timestamp())
        
        # Update tracker
        track_ids, species_counts = self.tracker.update(boxes, scores, classes, self.class_names)
        
        # Create detection results
        detections = []
        critical_species = []
        
        for i, (box, score, cls, track_id) in enumerate(zip(boxes, scores, classes, track_ids)):
            class_name = self.class_names[int(cls)]
            
            detection = DetectionResult(
                bbox=tuple(box),
                confidence=float(score),
                class_id=int(cls),
                class_name=class_name,
                track_id=int(track_id)
            )
            detections.append(detection)
            
            # Check if critical
            if int(cls) in self.critical_species_ids:
                critical_species.append(class_name)
        
        return FrameResult(
            frame_id=frame_id,
            timestamp=datetime.now().timestamp(),
            detections=detections,
            species_counts=species_counts,
            critical_species=list(set(critical_species))
        )
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[FrameResult]:
        """
        Process entire video
        
        Args:
            video_path: Path to input video
            output_path: Optional path for annotated output video
            
        Returns:
            List of FrameResults
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        results = []
        frame_id = 0
        
        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_frame(frame, frame_id)
            results.append(result)
            
            # Annotate and write if requested
            if writer:
                annotated = self._annotate_frame(frame, result)
                writer.write(annotated)
            
            frame_id += 1
            
            if frame_id % 100 == 0:
                logger.info(f"  Processed {frame_id} frames")
        
        cap.release()
        if writer:
            writer.release()
        
        logger.info(f"✓ Processed {frame_id} frames total")
        
        return results
    
    def _annotate_frame(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Annotate frame with detections"""
        annotated = frame.copy()
        
        for det in result.detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            
            # Color based on category
            if det.class_id < 20:
                color = (0, 0, 255)  # Red for critical
            elif det.class_id < 29:
                color = (0, 165, 255)  # Orange for important
            else:
                color = (0, 255, 0)  # Green for general
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None:
                label += f" ID:{det.track_id}"
            
            cv2.putText(annotated, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw counts
        y_offset = 30
        for species, count in result.species_counts.items():
            text = f"{species}: {count}"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return annotated


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Nano Inference Pipeline')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    
    args = parser.parse_args()
    
    pipeline = NanoInferencePipeline(args.config)
    results = pipeline.process_video(args.input, args.output)
    
    logger.info(f"\nProcessing complete. Total detections: {sum(len(r.detections) for r in results)}")
    logger.info(f"Unique species detected: {len(set(d.class_name for r in results for d in r.detections))}")


if __name__ == '__main__':
    main()
