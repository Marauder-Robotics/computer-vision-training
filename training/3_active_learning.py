#!/usr/bin/env python3
"""
Week 2: Active Learning

Run inference on unlabeled data, score uncertainty, and select samples for annotation.
Exports selected images for Mindy Services annotation.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

import torch
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
import wandb


class UncertaintyScorer:
    """Calculate uncertainty scores for active learning sample selection"""
    
    def __init__(self):
        self.metrics = ['entropy', 'least_confidence', 'margin']
    
    def entropy(self, confidences: np.ndarray) -> float:
        """
        Calculate entropy-based uncertainty
        
        Args:
            confidences: Array of confidence scores
            
        Returns:
            Entropy score (higher = more uncertain)
        """
        if len(confidences) == 0:
            return 0.0
        
        # Normalize
        probs = confidences / (confidences.sum() + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return entropy
    
    def least_confidence(self, confidences: np.ndarray) -> float:
        """
        Calculate least confidence uncertainty
        
        Args:
            confidences: Array of confidence scores
            
        Returns:
            Least confidence score (higher = more uncertain)
        """
        if len(confidences) == 0:
            return 1.0
        
        max_conf = np.max(confidences)
        return 1.0 - max_conf
    
    def margin_sampling(self, confidences: np.ndarray) -> float:
        """
        Calculate margin sampling uncertainty
        
        Args:
            confidences: Array of confidence scores
            
        Returns:
            Margin score (lower = more uncertain)
        """
        if len(confidences) < 2:
            return 0.0
        
        sorted_conf = np.sort(confidences)[::-1]
        margin = sorted_conf[0] - sorted_conf[1]
        
        return 1.0 - margin
    
    def combined_score(self, confidences: np.ndarray) -> float:
        """
        Calculate combined uncertainty score
        
        Args:
            confidences: Array of confidence scores
            
        Returns:
            Combined score (higher = more uncertain)
        """
        entropy_score = self.entropy(confidences)
        lc_score = self.least_confidence(confidences)
        margin_score = self.margin_sampling(confidences)
        
        # Weighted combination
        combined = 0.4 * entropy_score + 0.3 * lc_score + 0.3 * margin_score
        
        return combined


class ActiveLearner:
    """Active Learning for sample selection"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.al_config = self.config.get('active_learning', {})
        self.scorer = UncertaintyScorer()
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def run_inference(
        self,
        model_path: str,
        image_dir: str,
        conf_threshold: float = 0.25
    ) -> Dict[str, Dict]:
        """
        Run inference on unlabeled images
        
        Args:
            model_path: Path to trained model
            image_dir: Directory with images
            conf_threshold: Confidence threshold
            
        Returns:
            Dictionary mapping image paths to predictions
        """
        print("Running inference on unlabeled data...")
        
        # Load model
        model = YOLO(model_path)
        
        # Collect images
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(Path(image_dir).rglob(ext))
        
        print(f"Found {len(image_paths)} images")
        
        # Run inference
        results_dict = {}
        
        for img_path in tqdm(image_paths, desc="Inference"):
            try:
                results = model.predict(
                    str(img_path),
                    conf=conf_threshold,
                    iou=0.45,
                    verbose=False
                )
                
                if len(results) > 0:
                    result = results[0]
                    
                    # Extract predictions
                    boxes = result.boxes
                    confidences = boxes.conf.cpu().numpy() if boxes is not None else np.array([])
                    classes = boxes.cls.cpu().numpy() if boxes is not None else np.array([])
                    
                    results_dict[str(img_path)] = {
                        'confidences': confidences.tolist(),
                        'classes': classes.tolist(),
                        'num_detections': len(confidences)
                    }
                else:
                    results_dict[str(img_path)] = {
                        'confidences': [],
                        'classes': [],
                        'num_detections': 0
                    }
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        return results_dict
    
    def score_samples(self, predictions: Dict[str, Dict]) -> Dict[str, float]:
        """
        Score samples by uncertainty
        
        Args:
            predictions: Prediction dictionary from run_inference
            
        Returns:
            Dictionary mapping image paths to uncertainty scores
        """
        print("Calculating uncertainty scores...")
        
        scores = {}
        
        for img_path, pred in predictions.items():
            confidences = np.array(pred['confidences'])
            
            # Calculate uncertainty score
            if len(confidences) > 0:
                score = self.scorer.combined_score(confidences)
            else:
                # No detections = very uncertain
                score = 1.0
            
            scores[img_path] = score
        
        return scores
    
    def select_critical_species(
        self,
        predictions: Dict[str, Dict],
        critical_class_ids: List[int]
    ) -> List[str]:
        """
        Select images containing critical species
        
        Args:
            predictions: Prediction dictionary
            critical_class_ids: List of critical species class IDs (0-19)
            
        Returns:
            List of image paths with critical species
        """
        critical_images = []
        
        for img_path, pred in predictions.items():
            classes = pred['classes']
            
            # Check if any critical species detected
            if any(cls in critical_class_ids for cls in classes):
                critical_images.append(img_path)
        
        return critical_images
    
    def select_samples(
        self,
        scores: Dict[str, float],
        predictions: Dict[str, Dict],
        target_count: int = 2000,
        critical_class_ids: List[int] = None
    ) -> List[str]:
        """
        Select samples for annotation
        
        Args:
            scores: Uncertainty scores
            predictions: Prediction dictionary
            target_count: Target number of samples
            critical_class_ids: List of critical species IDs
            
        Returns:
            List of selected image paths
        """
        print(f"Selecting {target_count} samples...")
        
        selected = []
        
        # First, select all critical species images
        if critical_class_ids:
            critical_images = self.select_critical_species(predictions, critical_class_ids)
            selected.extend(critical_images[:target_count // 2])  # Up to 50% critical
            print(f"Selected {len(selected)} critical species images")
        
        # Sort remaining by uncertainty
        remaining_images = [img for img in scores.keys() if img not in selected]
        sorted_images = sorted(
            remaining_images,
            key=lambda x: scores[x],
            reverse=True
        )
        
        # Select most uncertain
        needed = target_count - len(selected)
        selected.extend(sorted_images[:needed])
        
        print(f"Total selected: {len(selected)} samples")
        
        return selected
    
    def export_for_annotation(
        self,
        selected_images: List[str],
        output_dir: str
    ):
        """
        Export selected images for annotation
        
        Args:
            selected_images: List of selected image paths
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save image list
        list_file = output_path / 'selected_images.txt'
        with open(list_file, 'w') as f:
            for img_path in selected_images:
                f.write(f"{img_path}\n")
        
        print(f"Exported image list to {list_file}")
        
        # Create export package using Mindy Services handler
        from data.active_learning.mindy_services_handler import MindyServicesHandler
        
        handler = MindyServicesHandler('config/dataset_config.yaml')
        zip_path = handler.export_for_annotation(
            selected_images,
            str(output_path)
        )
        
        print(f"Export package created: {zip_path}")
        
        return zip_path
    
    def save_results(
        self,
        predictions: Dict[str, Dict],
        scores: Dict[str, float],
        selected_images: List[str],
        output_dir: str
    ):
        """Save active learning results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        pred_file = output_path / 'predictions.json'
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Save scores
        scores_file = output_path / 'uncertainty_scores.json'
        with open(scores_file, 'w') as f:
            json.dump(scores, f, indent=2)
        
        # Save selected images
        selected_file = output_path / 'selected_images.json'
        with open(selected_file, 'w') as f:
            json.dump(selected_images, f, indent=2)
        
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Active Learning for sample selection')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Training config path')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained baseline model')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory with unlabeled images')
    parser.add_argument('--output', type=str, default='outputs/3_active_learning',
                       help='Output directory')
    parser.add_argument('--target-count', type=int, default=2000,
                       help='Target number of samples to select')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold for inference')
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(
        project="marauder-cv",
        name="3-active-learning"
    )
    
    # Create learner
    learner = ActiveLearner(args.config)
    
    # Run inference
    predictions = learner.run_inference(
        args.model,
        args.image_dir,
        conf_threshold=args.conf_threshold
    )
    
    # Score samples
    scores = learner.score_samples(predictions)
    
    # Critical species IDs (0-19)
    critical_class_ids = list(range(20))
    
    # Select samples
    selected_images = learner.select_samples(
        scores,
        predictions,
        target_count=args.target_count,
        critical_class_ids=critical_class_ids
    )
    
    # Export for annotation
    learner.export_for_annotation(selected_images, args.output)
    
    # Save results
    learner.save_results(predictions, scores, selected_images, args.output)
    
    # Log statistics
    wandb.log({
        'total_images': len(predictions),
        'selected_images': len(selected_images),
        'avg_uncertainty': np.mean(list(scores.values())),
        'max_uncertainty': np.max(list(scores.values()))
    })
    
    wandb.finish()
    
    print("\nActive learning complete!")
    print(f"Selected {len(selected_images)} images for annotation")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
