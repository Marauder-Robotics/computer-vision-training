#!/usr/bin/env python3
"""
Enhanced Hybrid Image Preprocessor

Features:
- Checkpoint support for large datasets
- Batch processing with progress tracking
- DO Spaces bucket integration
- Required preprocessing (not optional)
- Metadata and statistics tracking
- Memory-efficient processing
- Error handling and recovery
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import hashlib
from datetime import datetime


class HybridPreprocessor:
    """
    Enhanced hybrid preprocessing for underwater images
    
    REQUIRED preprocessing pipeline:
    1. Color correction (white balance)
    2. Dehazing (dark channel prior)
    3. CLAHE (contrast enhancement)
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        batch_size: int = 100,
        save_stats: bool = True,
        quality: int = 95
    ):
        """
        Initialize preprocessor
        
        Args:
            checkpoint_dir: Directory for checkpoint files (DO bucket path)
            batch_size: Number of images to process before checkpointing
            save_stats: Whether to save preprocessing statistics
            quality: JPEG quality for saving (95 recommended)
        """
        # DO Bucket path
        self.do_bucket_path = os.environ.get('DO_BUCKET_PATH', '/datasets/marauder-do-bucket')
        
        # Checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = f'{self.do_bucket_path}/training/preprocessing/checkpoints'
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.checkpoint_dir / 'preprocess_checkpoint.json'
        
        # Processing parameters (REQUIRED - all enabled)
        self.use_clahe = True  # REQUIRED
        self.use_dehaze = True  # REQUIRED
        self.use_color_correction = True  # REQUIRED
        
        # Batch processing
        self.batch_size = batch_size
        self.save_stats = save_stats
        self.quality = quality
        
        # CLAHE instance
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'processing_times': [],
            'start_time': None,
            'end_time': None
        }
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply REQUIRED hybrid preprocessing pipeline
        
        Pipeline order:
        1. Color correction
        2. Dehazing
        3. CLAHE
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image (BGR format)
        """
        result = image.copy()
        
        # Step 1: Color correction (REQUIRED)
        result = self._color_correction(result)
        
        # Step 2: Dehazing (REQUIRED)
        result = self._dehaze(result)
        
        # Step 3: CLAHE (REQUIRED)
        result = self._apply_clahe(result)
        
        return result
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to LAB color space"""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"CLAHE failed: {e}, returning original")
            return image
    
    def _dehaze(self, image: np.ndarray) -> np.ndarray:
        """
        Dehazing using dark channel prior
        
        Improved version with better atmospheric light estimation
        """
        try:
            # Convert to float for calculations
            img_float = image.astype(np.float32) / 255.0
            
            # Calculate dark channel
            min_channel = np.min(img_float, axis=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dark_channel = cv2.erode(min_channel, kernel)
            
            # Estimate atmospheric light (top 0.1% brightest pixels in dark channel)
            flat_img = img_float.reshape(-1, 3)
            flat_dark = dark_channel.ravel()
            num_pixels = flat_dark.size
            num_brightest = int(max(num_pixels * 0.001, 1))
            indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
            atmospheric_light = np.max(flat_img[indices], axis=0)
            
            # Estimate transmission map
            omega = 0.95  # Keep some haze for natural look
            transmission = 1 - omega * dark_channel
            transmission = np.maximum(transmission, 0.1)  # Lower bound
            
            # Recover scene radiance
            transmission_3d = transmission[:, :, np.newaxis]
            recovered = (img_float - atmospheric_light) / transmission_3d + atmospheric_light
            recovered = np.clip(recovered * 255, 0, 255).astype(np.uint8)
            
            return recovered
            
        except Exception as e:
            print(f"Dehazing failed: {e}, returning original")
            return image
    
    def _color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced white balance and color correction
        
        Uses LAB color space for better underwater color correction
        """
        try:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Correct color cast
            avg_a = np.average(a)
            avg_b = np.average(b)
            
            # Shift towards neutral
            a = a - ((avg_a - 128) * (l / 255.0) * 1.1)
            b = b - ((avg_b - 128) * (l / 255.0) * 1.1)
            
            # Merge and convert back
            lab = cv2.merge([l, a.astype(np.uint8), b.astype(np.uint8)])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return result
            
        except Exception as e:
            print(f"Color correction failed: {e}, returning original")
            return image
    
    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        resume: bool = True,
        copy_labels: bool = True
    ) -> Dict:
        """
        Process entire dataset with checkpoint support
        
        Args:
            input_dir: Input directory with images
            output_dir: Output directory for preprocessed images
            resume: Whether to resume from checkpoint
            copy_labels: Whether to copy label files
            
        Returns:
            Statistics dictionary
        """
        print(f"Starting REQUIRED preprocessing pipeline...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(input_path.rglob(ext)))
        
        print(f"Found {len(all_images)} images to process")
        
        # Load checkpoint if resuming
        processed_files = set()
        if resume and self.checkpoint_file.exists():
            checkpoint = self.load_checkpoint()
            if checkpoint:
                processed_files = set(checkpoint.get('processed_files', []))
                self.stats = checkpoint.get('stats', self.stats)
                print(f"Resuming: {len(processed_files)} already processed")
        
        # Start time
        self.stats['start_time'] = datetime.utcnow().isoformat()
        
        # Process images in batches
        batch = []
        for img_path in tqdm(all_images, desc="Preprocessing images"):
            # Skip if already processed
            if str(img_path) in processed_files:
                continue
            
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"Failed to load: {img_path}")
                    self.stats['failed'] += 1
                    continue
                
                # Process image
                import time
                start_time = time.time()
                processed = self.process(image)
                processing_time = time.time() - start_time
                
                # Create output path (preserve directory structure)
                rel_path = img_path.relative_to(input_path)
                out_path = output_path / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save processed image
                cv2.imwrite(str(out_path), processed, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                
                # Copy label file if exists
                if copy_labels:
                    label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
                    if label_path.exists():
                        out_label_path = out_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
                        out_label_path.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy(label_path, out_label_path)
                
                # Update stats
                self.stats['successful'] += 1
                self.stats['total_processed'] += 1
                self.stats['processing_times'].append(processing_time)
                processed_files.add(str(img_path))
                
                # Batch checkpoint
                batch.append(str(img_path))
                if len(batch) >= self.batch_size:
                    self.save_checkpoint(processed_files)
                    batch = []
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                self.stats['failed'] += 1
                continue
        
        # Final checkpoint
        if batch:
            self.save_checkpoint(processed_files)
        
        # End time
        self.stats['end_time'] = datetime.utcnow().isoformat()
        
        # Calculate statistics
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            total_time = sum(self.stats['processing_times'])
        else:
            avg_time = 0
            total_time = 0
        
        summary = {
            'total_images': len(all_images),
            'processed': self.stats['total_processed'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'avg_processing_time': avg_time,
            'total_processing_time': total_time,
            'start_time': self.stats['start_time'],
            'end_time': self.stats['end_time']
        }
        
        # Save statistics
        if self.save_stats:
            stats_path = output_path / 'preprocessing_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved statistics: {stats_path}")
        
        print(f"\nPreprocessing complete!")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Average time per image: {avg_time:.3f}s")
        
        return summary
    
    def save_checkpoint(self, processed_files: set):
        """Save checkpoint to DO bucket"""
        checkpoint = {
            'processed_files': list(processed_files),
            'stats': self.stats,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint from DO bucket"""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
        
        return None
    
    def clear_checkpoint(self):
        """Clear checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print(f"Cleared checkpoint: {self.checkpoint_file}")


def main():
    """CLI for preprocessing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Image Preprocessor with Checkpoint Support')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for preprocessed images')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Checkpoint save frequency')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh (ignore checkpoint)')
    parser.add_argument('--no-labels', action='store_true',
                       help='Do not copy label files')
    parser.add_argument('--quality', type=int, default=95,
                       help='JPEG quality (default: 95)')
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Custom checkpoint directory')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = HybridPreprocessor(
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        quality=args.quality
    )
    
    # Clear checkpoint if not resuming
    if args.no_resume:
        preprocessor.clear_checkpoint()
    
    # Process dataset
    stats = preprocessor.process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        resume=not args.no_resume,
        copy_labels=not args.no_labels
    )
    
    print("\n" + "="*50)
    print("Preprocessing Summary:")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    main()
