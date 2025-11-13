#!/usr/bin/env python3
"""
Week 2: Active Learning - Enhanced with Advanced Sampling Strategies

Features:
- Additional uncertainty metrics (variation ratio, Bayesian, predictive entropy)
- Diversity-based sampling using clustering
- Multi-criteria scoring with weighted combinations
- Ensemble uncertainty support
- Better visualization and analysis
- Efficient batch processing
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
from tqdm import tqdm
from utils.training_logger import TrainingLogger


class AdvancedUncertaintyScorer:
    """
    Advanced uncertainty scoring for active learning
    
    Implements multiple uncertainty metrics:
    - Entropy-based uncertainty
    - Least confidence
    - Margin sampling
    - Variation ratio (for ensembles)
    - Bayesian uncertainty (MC dropout)
    - Predictive entropy
    """
    
    def __init__(self, do_bucket_path: str = None):
        self.do_bucket_path = do_bucket_path or os.environ.get('DO_BUCKET_PATH', '/datasets/marauder-do-bucket')
        self.metrics = [
            'entropy', 
            'least_confidence', 
            'margin',
            'variation_ratio',
            'bayesian',
            'predictive_entropy'
        ]
    
    def entropy(self, confidences: np.ndarray) -> float:
        """
        Calculate entropy-based uncertainty
        
        Higher entropy = more uncertain
        
        Args:
            confidences: Array of confidence scores [N]
            
        Returns:
            Entropy score
        """
        if len(confidences) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = confidences / (confidences.sum() + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        return entropy
    
    def least_confidence(self, confidences: np.ndarray) -> float:
        """
        Calculate least confidence uncertainty
        
        Args:
            confidences: Array of confidence scores [N]
            
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
        
        Difference between top two predictions
        
        Args:
            confidences: Array of confidence scores [N]
            
        Returns:
            Margin score (lower = more uncertain)
        """
        if len(confidences) < 2:
            return 0.0
        
        sorted_conf = np.sort(confidences)[::-1]
        margin = sorted_conf[0] - sorted_conf[1]
        
        return 1.0 - margin  # Invert so higher = more uncertain
    
    def variation_ratio(self, predictions: List[np.ndarray]) -> float:
        """
        Calculate variation ratio for ensemble predictions
        
        Measures disagreement among ensemble members
        
        Args:
            predictions: List of prediction arrays from different models
            
        Returns:
            Variation ratio (higher = more disagreement)
        """
        if len(predictions) == 0:
            return 0.0
        
        # Find most common prediction
        predictions_array = np.array(predictions)
        mode_count = np.max(np.bincount(predictions_array.flatten().astype(int)))
        
        variation = 1.0 - (mode_count / len(predictions))
        
        return variation
    
    def bayesian_uncertainty(
        self, 
        model: YOLO, 
        image_path: str,
        num_samples: int = 10,
        dropout_rate: float = 0.1
    ) -> float:
        """
        Calculate Bayesian uncertainty using MC Dropout
        
        Args:
            model: YOLO model
            image_path: Path to image
            num_samples: Number of forward passes
            dropout_rate: Dropout probability
            
        Returns:
            Bayesian uncertainty score
        """
        # Enable dropout during inference
        def enable_dropout(m):
            if type(m) == torch.nn.Dropout:
                m.train()
                m.p = dropout_rate
        
        model.model.apply(enable_dropout)
        
        # Multiple forward passes
        predictions = []
        for _ in range(num_samples):
            results = model.predict(image_path, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                confs = results[0].boxes.conf.cpu().numpy()
                predictions.append(confs)
        
        # Calculate variance across predictions
        if len(predictions) == 0:
            return 0.0
        
        # Flatten and calculate std
        all_confs = np.concatenate(predictions)
        uncertainty = np.std(all_confs) if len(all_confs) > 0 else 0.0
        
        return uncertainty
    
    def predictive_entropy(self, ensemble_predictions: List[Dict]) -> float:
        """
        Calculate predictive entropy across ensemble
        
        Args:
            ensemble_predictions: List of prediction dicts from ensemble models
            
        Returns:
            Predictive entropy score
        """
        if len(ensemble_predictions) == 0:
            return 0.0
        
        # Aggregate class probabilities
        class_probs = defaultdict(list)
        for pred in ensemble_predictions:
            if 'boxes' in pred and len(pred['boxes']) > 0:
                for cls, conf in zip(pred['classes'], pred['confidences']):
                    class_probs[cls].append(conf)
        
        # Calculate mean probabilities
        mean_probs = []
        for cls, confs in class_probs.items():
            mean_probs.append(np.mean(confs))
        
        if len(mean_probs) == 0:
            return 0.0
        
        mean_probs = np.array(mean_probs)
        mean_probs = mean_probs / (mean_probs.sum() + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        
        return entropy
    
    def score_image(
        self,
        predictions: Dict,
        ensemble_predictions: Optional[List[Dict]] = None,
        model: Optional[YOLO] = None,
        image_path: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate all uncertainty scores for an image
        
        Args:
            predictions: Single model prediction dict
            ensemble_predictions: Optional ensemble predictions
            model: Optional model for Bayesian uncertainty
            image_path: Optional image path for Bayesian uncertainty
            weights: Optional weights for combining metrics
            
        Returns:
            Dictionary of uncertainty scores
        """
        scores = {}
        
        # Extract confidences
        if 'boxes' in predictions and len(predictions['boxes']) > 0:
            confidences = np.array(predictions['confidences'])
        else:
            confidences = np.array([])
        
        # Basic metrics
        scores['entropy'] = self.entropy(confidences)
        scores['least_confidence'] = self.least_confidence(confidences)
        scores['margin'] = self.margin_sampling(confidences)
        
        # Ensemble metrics
        if ensemble_predictions:
            # Variation ratio
            all_classes = []
            for pred in ensemble_predictions:
                if 'boxes' in pred and len(pred['boxes']) > 0:
                    all_classes.extend(pred['classes'])
            scores['variation_ratio'] = self.variation_ratio(all_classes) if all_classes else 0.0
            
            # Predictive entropy
            scores['predictive_entropy'] = self.predictive_entropy(ensemble_predictions)
        else:
            scores['variation_ratio'] = 0.0
            scores['predictive_entropy'] = 0.0
        
        # Bayesian uncertainty (expensive, use sparingly)
        if model and image_path:
            scores['bayesian'] = self.bayesian_uncertainty(model, image_path, num_samples=5)
        else:
            scores['bayesian'] = 0.0
        
        # Calculate weighted combined score
        if weights is None:
            weights = {
                'entropy': 0.3,
                'least_confidence': 0.2,
                'margin': 0.2,
                'variation_ratio': 0.15,
                'predictive_entropy': 0.1,
                'bayesian': 0.05
            }
        
        scores['combined'] = sum(scores[k] * weights.get(k, 0) for k in scores.keys() if k != 'combined')
        
        return scores


class DiversitySampler:
    """
    Diversity-based sampling using clustering
    
    Ensures selected samples are diverse and representative
    """
    
    def __init__(self, n_clusters: int = 50, feature_dim: int = 2048):
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
    
    def extract_features(
        self, 
        model: YOLO, 
        image_paths: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract features from images using model backbone
        
        Args:
            model: YOLO model
            image_paths: List of image paths
            batch_size: Batch size for feature extraction
            
        Returns:
            Feature matrix [N, D]
        """
        features = []
        
        # Extract features in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            
            for img_path in batch_paths:
                try:
                    # Use model's forward pass to get features
                    results = model.predict(img_path, verbose=False)
                    
                    # Get features from last layer before detection head
                    # This is a simplified version; actual implementation would need model internals
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        # Use bounding box features as proxy
                        boxes = results[0].boxes.xywhn.cpu().numpy()  # Normalized boxes
                        confs = results[0].boxes.conf.cpu().numpy()
                        
                        # Create feature vector from detections
                        feat = np.concatenate([
                            boxes.flatten()[:min(20, len(boxes.flatten()))],  # First 20 box coords
                            confs.flatten()[:min(10, len(confs.flatten()))]   # First 10 confidences
                        ])
                        
                        # Pad to fixed size
                        feat = np.pad(feat, (0, max(0, 30 - len(feat))), mode='constant')[:30]
                        features.append(feat)
                    else:
                        # No detections - use zero feature
                        features.append(np.zeros(30))
                        
                except Exception as e:
                    print(f"Error extracting features from {img_path}: {e}")
                    features.append(np.zeros(30))
        
        features = np.array(features)
        
        return features
    
    def cluster_samples(
        self, 
        features: np.ndarray, 
        n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """
        Cluster samples using MiniBatchKMeans
        
        Args:
            features: Feature matrix [N, D]
            n_clusters: Number of clusters (uses self.n_clusters if None)
            
        Returns:
            Cluster assignments [N]
        """
        n_clusters = n_clusters or self.n_clusters
        n_clusters = min(n_clusters, len(features))  # Can't have more clusters than samples
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Cluster
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1000,
            random_state=42,
            n_init=3
        )
        
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        return cluster_labels, kmeans
    
    def select_diverse_samples(
        self,
        uncertainty_scores: Dict[str, float],
        features: np.ndarray,
        n_samples: int,
        cluster_labels: np.ndarray
    ) -> List[str]:
        """
        Select diverse samples based on uncertainty and cluster distribution
        
        Args:
            uncertainty_scores: Dict mapping image paths to scores
            features: Feature matrix
            n_samples: Number of samples to select
            cluster_labels: Cluster assignments
            
        Returns:
            List of selected image paths
        """
        image_paths = list(uncertainty_scores.keys())
        
        # Calculate samples per cluster (proportional to cluster size)
        unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
        samples_per_cluster = {}
        
        for cluster_id, count in zip(unique_clusters, cluster_counts):
            samples_per_cluster[cluster_id] = max(1, int(n_samples * count / len(image_paths)))
        
        # Normalize to exactly n_samples
        total_samples = sum(samples_per_cluster.values())
        if total_samples != n_samples:
            # Adjust largest cluster
            max_cluster = max(samples_per_cluster, key=samples_per_cluster.get)
            samples_per_cluster[max_cluster] += (n_samples - total_samples)
        
        # Select most uncertain samples from each cluster
        selected = []
        
        for cluster_id in unique_clusters:
            # Get images in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_paths = [path for i, path in enumerate(image_paths) if cluster_mask[i]]
            
            # Sort by uncertainty
            cluster_scores = [(path, uncertainty_scores[path]) for path in cluster_paths]
            cluster_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top K from this cluster
            k = samples_per_cluster[cluster_id]
            selected.extend([path for path, _ in cluster_scores[:k]])
        
        return selected[:n_samples]


class ActiveLearnerV2:
    """Enhanced active learning with advanced sampling"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.al_config = self.config.get('active_learning', {})
        self.do_bucket_path = os.environ.get('DO_BUCKET_PATH', '/datasets/marauder-do-bucket')
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Uncertainty scorer
        self.uncertainty_scorer = AdvancedUncertaintyScorer(self.do_bucket_path)
        
        # Diversity sampler
        self.diversity_sampler = DiversitySampler(
            n_clusters=self.al_config.get('n_clusters', 50)
        )
    
    def run_inference(
        self,
        model_path: str,
        image_paths: List[str],
        batch_size: int = 32
    ) -> Dict[str, Dict]:
        """
        Run inference on images and collect predictions
        
        Args:
            model_path: Path to trained model
            image_paths: List of image paths
            batch_size: Batch size for inference
            
        Returns:
            Dictionary mapping image paths to predictions
        """
        model = YOLO(model_path)
        model.to(self.device)
        
        predictions = {}
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Running inference"):
            batch_paths = image_paths[i:i+batch_size]
            
            for img_path in batch_paths:
                try:
                    results = model.predict(img_path, verbose=False)
                    
                    if len(results) > 0:
                        boxes = results[0].boxes
                        
                        pred_dict = {
                            'boxes': boxes.xyxy.cpu().numpy() if len(boxes) > 0 else np.array([]),
                            'confidences': boxes.conf.cpu().numpy() if len(boxes) > 0 else np.array([]),
                            'classes': boxes.cls.cpu().numpy().astype(int) if len(boxes) > 0 else np.array([]),
                        }
                        
                        predictions[img_path] = pred_dict
                    else:
                        predictions[img_path] = {
                            'boxes': np.array([]),
                            'confidences': np.array([]),
                            'classes': np.array([])
                        }
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    predictions[img_path] = {
                        'boxes': np.array([]),
                        'confidences': np.array([]),
                        'classes': np.array([])
                    }
        
        return predictions, model
    
    def run_ensemble_inference(
        self,
        model_paths: List[str],
        image_paths: List[str],
        batch_size: int = 32
    ) -> Tuple[Dict[str, List[Dict]], List[YOLO]]:
        """
        Run ensemble inference with multiple models
        
        Args:
            model_paths: List of model paths
            image_paths: List of image paths
            batch_size: Batch size
            
        Returns:
            Tuple of (ensemble predictions dict, list of models)
        """
        models = [YOLO(path) for path in model_paths]
        for model in models:
            model.to(self.device)
        
        ensemble_predictions = defaultdict(list)
        
        for model_idx, model in enumerate(models):
            print(f"Running inference with model {model_idx + 1}/{len(models)}")
            
            for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Model {model_idx+1}"):
                batch_paths = image_paths[i:i+batch_size]
                
                for img_path in batch_paths:
                    try:
                        results = model.predict(img_path, verbose=False)
                        
                        if len(results) > 0:
                            boxes = results[0].boxes
                            
                            pred_dict = {
                                'boxes': boxes.xyxy.cpu().numpy() if len(boxes) > 0 else np.array([]),
                                'confidences': boxes.conf.cpu().numpy() if len(boxes) > 0 else np.array([]),
                                'classes': boxes.cls.cpu().numpy().astype(int) if len(boxes) > 0 else np.array([]),
                            }
                            
                            ensemble_predictions[img_path].append(pred_dict)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        return dict(ensemble_predictions), models
    
    def select_samples(
        self,
        unlabeled_dir: str,
        output_dir: str,
        model_path: str,
        n_samples: int = 2000,
        use_ensemble: bool = False,
        ensemble_models: Optional[List[str]] = None,
        use_diversity: bool = True
    ):
        """
        Select samples for annotation using advanced strategies
        
        Args:
            unlabeled_dir: Directory with unlabeled images
            output_dir: Output directory for selected samples
            model_path: Path to baseline model
            n_samples: Number of samples to select
            use_ensemble: Whether to use ensemble predictions
            ensemble_models: List of ensemble model paths
            use_diversity: Whether to use diversity sampling
        """
        # Initialize logger
        logger = TrainingLogger(
            project_name="marauder-cv",
            run_name="3-active-learning",
            save_dir=f"{self.do_bucket_path}/training/logs",
            config=self.al_config
        )
        
        print("Collecting unlabeled images...")
        unlabeled_path = Path(unlabeled_dir)
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend([str(p) for p in unlabeled_path.rglob(ext)])
        
        print(f"Found {len(image_paths)} unlabeled images")
        
        # Run inference
        if use_ensemble and ensemble_models:
            print("Running ensemble inference...")
            ensemble_preds, models = self.run_ensemble_inference(ensemble_models, image_paths)
            # Use first model for feature extraction
            model = models[0]
        else:
            print("Running single model inference...")
            predictions, model = self.run_inference(model_path, image_paths)
            ensemble_preds = None
        
        # Calculate uncertainty scores
        print("Calculating uncertainty scores...")
        uncertainty_scores = {}
        
        for img_path in tqdm(image_paths, desc="Scoring uncertainty"):
            if ensemble_preds:
                # Use ensemble predictions
                preds = ensemble_preds.get(img_path, [])
                if preds:
                    scores = self.uncertainty_scorer.score_image(
                        predictions=preds[0],  # Use first model's predictions as base
                        ensemble_predictions=preds
                    )
                else:
                    scores = {'combined': 0.0}
            else:
                # Use single model predictions
                pred = predictions.get(img_path, {'boxes': [], 'confidences': [], 'classes': []})
                scores = self.uncertainty_scorer.score_image(predictions=pred)
            
            uncertainty_scores[img_path] = scores['combined']
        
        # Select samples
        if use_diversity:
            print("Extracting features for diversity sampling...")
            features = self.diversity_sampler.extract_features(model, image_paths, batch_size=32)
            
            print("Clustering samples...")
            cluster_labels, _ = self.diversity_sampler.cluster_samples(features)
            
            print("Selecting diverse samples...")
            selected_paths = self.diversity_sampler.select_diverse_samples(
                uncertainty_scores,
                features,
                n_samples,
                cluster_labels
            )
        else:
            # Simple uncertainty-based selection
            sorted_paths = sorted(uncertainty_scores.items(), key=lambda x: x[1], reverse=True)
            selected_paths = [path for path, _ in sorted_paths[:n_samples]]
        
        # Export selected samples
        print(f"Exporting {len(selected_paths)} selected samples...")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy/symlink images
        for img_path in tqdm(selected_paths, desc="Exporting"):
            img_file = Path(img_path)
            dest = output_path / img_file.name
            
            try:
                # Use symlink to save space
                if not dest.exists():
                    os.symlink(img_path, dest)
            except:
                # Fallback to copy
                import shutil
                shutil.copy(img_path, dest)
        
        # Save selection metadata
        metadata = {
            'n_samples': len(selected_paths),
            'total_unlabeled': len(image_paths),
            'use_ensemble': use_ensemble,
            'use_diversity': use_diversity,
            'selected_paths': selected_paths,
            'uncertainty_scores': {path: float(score) for path, score in uncertainty_scores.items() if path in selected_paths}
        }
        
        metadata_path = output_path / 'selection_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata: {metadata_path}")
        
        # Log statistics
        selected_scores = [uncertainty_scores[path] for path in selected_paths]
        logger.log({
            'n_selected': len(selected_paths),
            'mean_uncertainty': np.mean(selected_scores),
            'std_uncertainty': np.std(selected_scores),
            'min_uncertainty': np.min(selected_scores),
            'max_uncertainty': np.max(selected_scores)
        })
        
        logger.finish()
        
        print(f"Selected {len(selected_paths)} samples saved to {output_dir}")
        return selected_paths


def main():
    parser = argparse.ArgumentParser(description='Active Learning with Advanced Sampling')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Training config path')
    parser.add_argument('--unlabeled-dir', type=str, required=True,
                       help='Directory with unlabeled images')
    parser.add_argument('--output', type=str,
                       default='/datasets/marauder-do-bucket/training/active_learning/selected',
                       help='Output directory for selected samples')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to baseline model')
    parser.add_argument('--n-samples', type=int, default=2000,
                       help='Number of samples to select')
    parser.add_argument('--use-ensemble', action='store_true',
                       help='Use ensemble predictions')
    parser.add_argument('--ensemble-models', type=str, nargs='+',
                       help='Paths to ensemble models')
    parser.add_argument('--no-diversity', action='store_true',
                       help='Disable diversity sampling')
    
    args = parser.parse_args()
    
    learner = ActiveLearnerV2(args.config)
    learner.select_samples(
        unlabeled_dir=args.unlabeled_dir,
        output_dir=args.output,
        model_path=args.model,
        n_samples=args.n_samples,
        use_ensemble=args.use_ensemble,
        ensemble_models=args.ensemble_models,
        use_diversity=not args.no_diversity
    )


if __name__ == '__main__':
    main()
