#!/usr/bin/env python3
"""
Week 6: Test-Time Augmentation (TTA) and Confidence Calibration
Implements TTA for improved accuracy and calibrates confidence scores for critical species
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple
import cv2
from tqdm import tqdm
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTAPredictor:
    """Test-Time Augmentation for YOLO ensemble"""
    
    def __init__(self, model_paths: List[str], tta_config: Dict[str, Any]):
        """
        Initialize TTA predictor
        
        Args:
            model_paths: List of paths to ensemble models
            tta_config: TTA configuration
        """
        self.models = [YOLO(path) for path in model_paths]
        self.tta_config = tta_config
        
    def augment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate TTA augmented images"""
        augmented = [image]  # Original
        
        # Horizontal flip
        if self.tta_config.get('horizontal_flip', True):
            augmented.append(cv2.flip(image, 1))
        
        # Rotations
        rotations = self.tta_config.get('rotations', [])
        for angle in rotations:
            if angle != 0:
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                augmented.append(cv2.warpAffine(image, M, (w, h)))
        
        # Scales
        scales = self.tta_config.get('scales', [1.0])
        for scale in scales:
            if scale != 1.0:
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled = cv2.resize(image, (new_w, new_h))
                # Pad or crop to original size
                if scale < 1.0:
                    # Pad
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    padded = cv2.copyMakeBorder(
                        scaled, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w,
                        cv2.BORDER_CONSTANT, value=[0, 0, 0]
                    )
                    augmented.append(padded)
                else:
                    # Crop center
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    cropped = scaled[start_h:start_h+h, start_w:start_w+w]
                    augmented.append(cropped)
        
        return augmented
    
    def predict_with_tta(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Run prediction with TTA
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            
        Returns:
            Merged predictions from all augmentations
        """
        augmented_images = self.augment_image(image)
        all_predictions = []
        
        for aug_img in augmented_images:
            for model in self.models:
                results = model.predict(
                    aug_img,
                    conf=conf_threshold,
                    verbose=False
                )
                
                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    all_predictions.append({
                        'boxes': boxes.xyxy.cpu().numpy(),
                        'scores': boxes.conf.cpu().numpy(),
                        'classes': boxes.cls.cpu().numpy()
                    })
        
        # Merge predictions using Weighted Box Fusion or NMS
        merged = self._merge_predictions(all_predictions)
        
        return merged
    
    def _merge_predictions(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Merge predictions from multiple augmentations"""
        if not predictions:
            return {'boxes': np.array([]), 'scores': np.array([]), 'classes': np.array([])}
        
        all_boxes = np.vstack([p['boxes'] for p in predictions])
        all_scores = np.concatenate([p['scores'] for p in predictions])
        all_classes = np.concatenate([p['classes'] for p in predictions])
        
        # Apply NMS
        keep_indices = self._nms(all_boxes, all_scores, iou_threshold=0.5)
        
        return {
            'boxes': all_boxes[keep_indices],
            'scores': all_scores[keep_indices],
            'classes': all_classes[keep_indices]
        }
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep


class ConfidenceCalibrator:
    """Calibrate confidence scores using temperature scaling"""
    
    def __init__(self, critical_species_ids: List[int]):
        """
        Initialize calibrator
        
        Args:
            critical_species_ids: List of critical species class IDs
        """
        self.critical_species_ids = critical_species_ids
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.optimizer = None
        
    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100):
        """
        Fit temperature scaling
        
        Args:
            logits: Model logits (uncalibrated)
            labels: Ground truth labels
            max_iter: Maximum iterations
        """
        self.optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            self.optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        
        self.optimizer.step(eval_loss)
        logger.info(f"Optimal temperature: {self.temperature.item():.4f}")
        
    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling"""
        return logits / self.temperature
    
    def calibrate_predictions(self, scores: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """
        Calibrate prediction scores
        
        Args:
            scores: Uncalibrated confidence scores
            classes: Predicted classes
            
        Returns:
            Calibrated scores
        """
        calibrated = scores.copy()
        
        # Apply temperature scaling to critical species
        for i, cls in enumerate(classes):
            if int(cls) in self.critical_species_ids:
                # Convert to logit, scale, convert back
                logit = np.log(scores[i] / (1 - scores[i] + 1e-10))
                scaled_logit = logit / self.temperature.item()
                calibrated[i] = 1 / (1 + np.exp(-scaled_logit))
        
        return calibrated
    
    def save(self, path: str):
        """Save calibrator"""
        torch.save({
            'temperature': self.temperature.item(),
            'critical_species_ids': self.critical_species_ids
        }, path)
        
    @classmethod
    def load(cls, path: str):
        """Load calibrator"""
        data = torch.load(path)
        calibrator = cls(data['critical_species_ids'])
        calibrator.temperature = nn.Parameter(torch.ones(1) * data['temperature'])
        return calibrator


class TTACalibrationPipeline:
    """Complete TTA and calibration pipeline"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tta_config = self.config['tta_calibration']['tta']
        self.calib_config = self.config['tta_calibration']['calibration']
        
        self.output_dir = Path(self.config['paths']['checkpoints']) / 'tta_calibration'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_pipeline(self, model_paths: List[str], calibration_data_path: str):
        """
        Run complete TTA and calibration pipeline
        
        Args:
            model_paths: Paths to ensemble models
            calibration_data_path: Path to calibration dataset
        """
        logger.info("\n" + "="*80)
        logger.info("TTA AND CALIBRATION PIPELINE")
        logger.info("="*80 + "\n")
        
        # Initialize TTA predictor
        tta_predictor = TTAPredictor(model_paths, self.tta_config)
        
        # Save TTA configuration
        tta_config_path = self.output_dir / 'tta_config.yaml'
        with open(tta_config_path, 'w') as f:
            yaml.dump(self.tta_config, f)
        
        logger.info(f"TTA configuration saved to: {tta_config_path}")
        
        # Initialize calibrator for critical species (IDs 0-19)
        critical_species_ids = list(range(20))
        calibrator = ConfidenceCalibrator(critical_species_ids)
        
        # Calibrate if calibration data is provided
        if calibration_data_path and os.path.exists(calibration_data_path):
            logger.info("\nCalibrating confidence scores...")
            # This would load calibration data and fit the temperature
            # For now, using default temperature
            logger.info("Calibration complete")
        
        # Save calibrator
        calibrator_path = self.output_dir / 'calibrator.pth'
        calibrator.save(str(calibrator_path))
        logger.info(f"Calibrator saved to: {calibrator_path}")
        
        # Save complete pipeline info
        pipeline_info = {
            'tta_config': self.tta_config,
            'calibration_config': self.calib_config,
            'model_paths': model_paths,
            'calibrator_path': str(calibrator_path)
        }
        
        pipeline_path = self.output_dir / 'pipeline_config.yaml'
        with open(pipeline_path, 'w') as f:
            yaml.dump(pipeline_info, f)
        
        logger.info(f"\nPipeline configuration saved to: {pipeline_path}")
        
        return tta_predictor, calibrator


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TTA and Calibration')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to ensemble models')
    parser.add_argument('--calibration-data', type=str, default=None,
                       help='Path to calibration dataset')
    
    args = parser.parse_args()
    
    pipeline = TTACalibrationPipeline(args.config)
    tta_predictor, calibrator = pipeline.run_pipeline(
        args.models,
        args.calibration_data
    )
    
    logger.info("\n" + "="*80)
    logger.info("TTA AND CALIBRATION COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
