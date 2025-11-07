#!/usr/bin/env python3
"""
Hybrid Image Preprocessor
CLAHE, dehazing, color correction for underwater images
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class HybridPreprocessor:
    """Hybrid preprocessing for underwater images"""
    
    def __init__(
        self,
        use_clahe: bool = True,
        use_dehaze: bool = True,
        use_color_correction: bool = True
    ):
        self.use_clahe = use_clahe
        self.use_dehaze = use_dehaze
        self.use_color_correction = use_color_correction
        
        # CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply hybrid preprocessing"""
        result = image.copy()
        
        if self.use_color_correction:
            result = self._color_correction(result)
        
        if self.use_dehaze:
            result = self._dehaze(result)
        
        if self.use_clahe:
            result = self._apply_clahe(result)
        
        return result
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to LAB color space"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _dehaze(self, image: np.ndarray) -> np.ndarray:
        """Simple dehazing using dark channel prior"""
        # Simplified version
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_channel = cv2.erode(gray, np.ones((15, 15)))
        
        # Estimate atmospheric light
        flat_image = image.reshape(-1, 3)
        flat_dark = dark_channel.flatten()
        indices = np.argsort(flat_dark)[-int(0.001 * len(flat_dark)):]
        atmospheric_light = np.mean(flat_image[indices], axis=0)
        
        # Transmission estimation
        transmission = 1 - 0.95 * (dark_channel / max(1, np.max(dark_channel)))
        transmission = cv2.max(transmission, 0.1)
        
        # Recover scene radiance
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            result[:,:,i] = (image[:,:,i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _color_correction(self, image: np.ndarray) -> np.ndarray:
        """White balance and color correction"""
        # Simple gray world assumption
        result = image.astype(np.float32)
        
        for i in range(3):
            avg = np.mean(result[:,:,i])
            result[:,:,i] = result[:,:,i] * (128.0 / max(1, avg))
        
        return np.clip(result, 0, 255).astype(np.uint8)
