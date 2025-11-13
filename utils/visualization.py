#!/usr/bin/env python3
"""Visualization utilities"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    class_names: List[str],
    conf_threshold: float = 0.25
) -> np.ndarray:
    """Draw bounding boxes on image"""
    result = image.copy()
    
    for box, score, cls in zip(boxes, scores, classes):
        if score < conf_threshold:
            continue
        
        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names[int(cls)]
        
        # Color based on class
        if int(cls) < 20:
            color = (0, 0, 255)  # Red for critical
        elif int(cls) < 29:
            color = (0, 165, 255)  # Orange for important
        else:
            color = (0, 255, 0)  # Green for general
        
        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1-20), (x1+w, y1), color, -1)
        cv2.putText(result, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics: dict,
    save_path: str
):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss')
    axes[0, 0].plot(val_losses, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Loss Curves')
    
    # mAP
    if 'map' in metrics:
        axes[0, 1].plot(metrics['map'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mAP')
        axes[0, 1].set_title('mAP')
    
    # Precision/Recall
    if 'precision' in metrics and 'recall' in metrics:
        axes[1, 0].plot(metrics['precision'], label='Precision')
        axes[1, 0].plot(metrics['recall'], label='Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].set_title('Precision & Recall')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
