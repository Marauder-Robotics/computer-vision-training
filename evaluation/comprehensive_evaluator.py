#!/usr/bin/env python3
"""
Complete Evaluation Suite
Evaluates model performance: mAP, counting accuracy, energy consumption
"""

import os
import yaml
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from ultralytics import YOLO
import psutil
import time
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class mAPCalculator:
    """Calculate mean Average Precision"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.predictions = []
        self.ground_truths = []
    
    def add_batch(self, pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes):
        """Add batch of predictions and ground truths"""
        self.predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'classes': pred_classes
        })
        self.ground_truths.append({
            'boxes': gt_boxes,
            'classes': gt_classes
        })
    
    def compute_map(self, iou_thresholds=[0.5, 0.75]) -> Dict[str, float]:
        """Compute mAP at different IoU thresholds"""
        results = {}
        
        for iou_threshold in iou_thresholds:
            ap_per_class = []
            
            for class_id in range(self.num_classes):
                # Get predictions and GTs for this class
                class_preds = []
                class_gts = []
                
                for pred, gt in zip(self.predictions, self.ground_truths):
                    # Filter by class
                    pred_mask = pred['classes'] == class_id
                    gt_mask = gt['classes'] == class_id
                    
                    if pred_mask.any():
                        class_preds.append({
                            'boxes': pred['boxes'][pred_mask],
                            'scores': pred['scores'][pred_mask]
                        })
                    
                    if gt_mask.any():
                        class_gts.append({
                            'boxes': gt['boxes'][gt_mask]
                        })
                
                if not class_gts:
                    continue
                
                # Compute AP for this class
                ap = self._compute_ap(class_preds, class_gts, iou_threshold)
                ap_per_class.append(ap)
            
            # Compute mAP
            if ap_per_class:
                results[f'mAP@{iou_threshold}'] = np.mean(ap_per_class)
            else:
                results[f'mAP@{iou_threshold}'] = 0.0
        
        # mAP@[0.5:0.95]
        iou_range = np.arange(0.5, 1.0, 0.05)
        maps_range = []
        for iou in iou_range:
            ap_per_class = []
            for class_id in range(self.num_classes):
                class_preds = []
                class_gts = []
                
                for pred, gt in zip(self.predictions, self.ground_truths):
                    pred_mask = pred['classes'] == class_id
                    gt_mask = gt['classes'] == class_id
                    
                    if pred_mask.any():
                        class_preds.append({
                            'boxes': pred['boxes'][pred_mask],
                            'scores': pred['scores'][pred_mask]
                        })
                    
                    if gt_mask.any():
                        class_gts.append({'boxes': gt['boxes'][gt_mask]})
                
                if class_gts:
                    ap = self._compute_ap(class_preds, class_gts, iou)
                    ap_per_class.append(ap)
            
            if ap_per_class:
                maps_range.append(np.mean(ap_per_class))
        
        if maps_range:
            results['mAP@[0.5:0.95]'] = np.mean(maps_range)
        else:
            results['mAP@[0.5:0.95]'] = 0.0
        
        return results
    
    def _compute_ap(self, predictions, ground_truths, iou_threshold):
        """Compute AP for single class"""
        if not predictions or not ground_truths:
            return 0.0
        
        # Combine all predictions
        all_boxes = []
        all_scores = []
        for pred in predictions:
            all_boxes.append(pred['boxes'])
            all_scores.append(pred['scores'])
        
        if not all_boxes:
            return 0.0
        
        all_boxes = np.vstack(all_boxes)
        all_scores = np.concatenate(all_scores)
        
        # Sort by score
        order = all_scores.argsort()[::-1]
        all_boxes = all_boxes[order]
        all_scores = all_scores[order]
        
        # Count ground truths
        num_gts = sum(len(gt['boxes']) for gt in ground_truths)
        
        # Compute precision-recall curve
        tp = np.zeros(len(all_boxes))
        fp = np.zeros(len(all_boxes))
        
        gt_matched = [set() for _ in ground_truths]
        
        for pred_idx, pred_box in enumerate(all_boxes):
            best_iou = 0
            best_gt_idx = -1
            best_img_idx = -1
            
            # Find best matching GT
            for img_idx, gt in enumerate(ground_truths):
                if not len(gt['boxes']):
                    continue
                    
                ious = self._compute_iou(pred_box[None, :], gt['boxes'])
                max_iou = ious.max()
                
                if max_iou > best_iou:
                    best_iou = max_iou
                    best_gt_idx = ious.argmax()
                    best_img_idx = img_idx
            
            # Check if match
            if best_iou >= iou_threshold:
                if best_gt_idx not in gt_matched[best_img_idx]:
                    tp[pred_idx] = 1
                    gt_matched[best_img_idx].add(best_gt_idx)
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / num_gts
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP (area under PR curve)
        recall = np.concatenate([[0], recall, [1]])
        precision = np.concatenate([[0], precision, [0]])
        
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        
        indices = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
        
        return ap
    
    def _compute_iou(self, boxes1, boxes2):
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


class CountingAccuracyEvaluator:
    """Evaluate counting accuracy"""
    
    def __init__(self):
        self.predictions = []
        self.ground_truths = []
    
    def add_sample(self, pred_count: int, gt_count: int):
        """Add count sample"""
        self.predictions.append(pred_count)
        self.ground_truths.append(gt_count)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute counting metrics"""
        preds = np.array(self.predictions)
        gts = np.array(self.ground_truths)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(preds - gts))
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((preds - gts) ** 2))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((preds - gts) / (gts + 1e-10))) * 100
        
        # Accuracy within threshold
        acc_1 = np.mean(np.abs(preds - gts) <= 1) * 100  # Within 1
        acc_5 = np.mean(np.abs(preds - gts) <= 5) * 100  # Within 5
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Accuracy@1': acc_1,
            'Accuracy@5': acc_5
        }


class EnergyProfiler:
    """Profile energy consumption"""
    
    def __init__(self):
        self.measurements = []
    
    def start_measurement(self):
        """Start energy measurement"""
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent(interval=None)
    
    def end_measurement(self, num_inferences: int):
        """End measurement and record"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Estimate power consumption (simplified)
        # For Jetson Nano: assume 5-15W during inference
        avg_power_watts = 10  # Conservative estimate
        energy_wh = (avg_power_watts * duration) / 3600
        
        self.measurements.append({
            'duration_seconds': duration,
            'num_inferences': num_inferences,
            'energy_wh': energy_wh,
            'energy_per_inference_j': (energy_wh * 3600) / num_inferences
        })
    
    def get_daily_estimate(self, inferences_per_day: int) -> Dict[str, float]:
        """Estimate daily energy consumption"""
        if not self.measurements:
            return {}
        
        avg_energy_per_inf = np.mean([m['energy_per_inference_j'] for m in self.measurements])
        daily_wh = (avg_energy_per_inf * inferences_per_day) / 3600
        
        return {
            'inferences_per_day': inferences_per_day,
            'daily_energy_wh': daily_wh,
            'avg_energy_per_inference_j': avg_energy_per_inf
        }


class ComprehensiveEvaluator:
    """Complete evaluation pipeline"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['paths']['outputs']) / 'evaluation'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.map_calculator = mAPCalculator(36)  # 36 species
        self.counting_evaluator = CountingAccuracyEvaluator()
        self.energy_profiler = EnergyProfiler()
    
    def evaluate_model(
        self,
        model_path: str,
        test_data_path: str,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate a single model
        
        Args:
            model_path: Path to model
            test_data_path: Path to test dataset
            model_name: Name for logging
            
        Returns:
            Evaluation results
        """
        logger.info(f"\nEvaluating {model_name}...")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Test data: {test_data_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Standard YOLO evaluation
        logger.info("  Running YOLO validation...")
        val_results = model.val(
            data=test_data_path,
            split='test',
            batch=16,
            imgsz=640,
            verbose=True
        )
        
        # Extract metrics
        results = {
            'model_name': model_name,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'mAP50': float(val_results.box.map50),
                'mAP50-95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
                'f1': 2 * (float(val_results.box.mp) * float(val_results.box.mr)) / 
                     (float(val_results.box.mp) + float(val_results.box.mr) + 1e-10)
            },
            'per_class_ap': val_results.box.ap.tolist() if hasattr(val_results.box, 'ap') else []
        }
        
        # Save results
        results_path = self.output_dir / f"{model_name}_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"  Results saved: {results_path}")
        logger.info(f"  mAP50: {results['metrics']['mAP50']:.4f}")
        logger.info(f"  mAP50-95: {results['metrics']['mAP50-95']:.4f}")
        logger.info(f"  Precision: {results['metrics']['precision']:.4f}")
        logger.info(f"  Recall: {results['metrics']['recall']:.4f}")
        
        return results
    
    def evaluate_ensemble(
        self,
        model_paths: List[str],
        test_data_path: str,
        ensemble_name: str = "ensemble"
    ) -> Dict[str, Any]:
        """Evaluate ensemble of models"""
        logger.info(f"\n" + "="*80)
        logger.info(f"EVALUATING ENSEMBLE: {ensemble_name}")
        logger.info("="*80 + "\n")
        
        all_results = []
        
        for i, model_path in enumerate(model_paths):
            variant_name = f"{ensemble_name}_variant_{i+1}"
            results = self.evaluate_model(model_path, test_data_path, variant_name)
            all_results.append(results)
        
        # Aggregate results
        ensemble_results = {
            'ensemble_name': ensemble_name,
            'num_models': len(model_paths),
            'individual_results': all_results,
            'aggregate_metrics': {
                'mean_mAP50': np.mean([r['metrics']['mAP50'] for r in all_results]),
                'mean_mAP50-95': np.mean([r['metrics']['mAP50-95'] for r in all_results]),
                'mean_precision': np.mean([r['metrics']['precision'] for r in all_results]),
                'mean_recall': np.mean([r['metrics']['recall'] for r in all_results]),
                'std_mAP50': np.std([r['metrics']['mAP50'] for r in all_results]),
            }
        }
        
        # Save ensemble results
        results_path = self.output_dir / f"{ensemble_name}_ensemble_evaluation.json"
        with open(results_path, 'w') as f:
            json.dump(ensemble_results, f, indent=2)
        
        logger.info(f"\nEnsemble evaluation complete:")
        logger.info(f"  Mean mAP50: {ensemble_results['aggregate_metrics']['mean_mAP50']:.4f}")
        logger.info(f"  Std mAP50: {ensemble_results['aggregate_metrics']['std_mAP50']:.4f}")
        
        return ensemble_results
    
    def profile_energy(
        self,
        model_path: str,
        num_inferences: int = 1000,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Profile energy consumption"""
        logger.info(f"\nProfiling energy for {model_name}...")
        
        model = YOLO(model_path)
        
        # Create dummy input
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Profile
        self.energy_profiler.start_measurement()
        
        for i in range(num_inferences):
            _ = model.predict(dummy_input, verbose=False)
            if (i + 1) % 100 == 0:
                logger.info(f"  {i+1}/{num_inferences} inferences")
        
        self.energy_profiler.end_measurement(num_inferences)
        
        # Daily estimate (4 cameras, 30s/30min, 5 FPS, 3 model ensemble)
        inferences_per_day = 4 * 90 * 24 * 5 * 3  # 129,600
        daily_estimate = self.energy_profiler.get_daily_estimate(inferences_per_day)
        
        energy_results = {
            'model_name': model_name,
            'measurements': self.energy_profiler.measurements,
            'daily_estimate': daily_estimate
        }
        
        # Save
        results_path = self.output_dir / f"{model_name}_energy_profile.json"
        with open(results_path, 'w') as f:
            json.dump(energy_results, f, indent=2)
        
        logger.info(f"  Daily energy estimate: {daily_estimate['daily_energy_wh']:.2f} Wh")
        logger.info(f"  Energy per inference: {daily_estimate['avg_energy_per_inference_j']:.4f} J")
        
        return energy_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--name', type=str, default='model')
    parser.add_argument('--profile-energy', action='store_true')
    
    args = parser.parse_args()
    
    evaluator = ComprehensiveEvaluator(args.config)
    
    # Evaluate model
    results = evaluator.evaluate_model(args.model, args.test_data, args.name)
    
    # Profile energy if requested
    if args.profile_energy:
        energy_results = evaluator.profile_energy(args.model, model_name=args.name)
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
