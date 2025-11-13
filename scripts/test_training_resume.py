#!/usr/bin/env python3
"""
Marauder CV Project - Training Resume Testing Script
Tests checkpoint saving/loading and training state restoration

Usage:
    python scripts/test_training_resume.py [--quick]
    python scripts/test_training_resume.py --test-script 1_ssl_pretrain
"""

import os
import sys
import json
import torch
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.checkpoint_manager import CheckpointManager


class TrainingResumeTest:
    """Test suite for training resume functionality"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
        
    def log(self, message: str, level: str = "info"):
        """Log a message"""
        if not self.verbose and level == "debug":
            return
            
        colors = {
            "info": "\033[0m",      # Default
            "success": "\033[0;32m", # Green
            "error": "\033[0;31m",   # Red
            "warning": "\033[1;33m", # Yellow
        }
        reset = "\033[0m"
        
        if level in colors:
            print(f"{colors[level]}{message}{reset}")
        else:
            print(message)
    
    def test_result(self, test_name: str, passed: bool, details: str = ""):
        """Record and report test result"""
        self.test_results.append({
            "name": test_name,
            "passed": passed,
            "details": details
        })
        
        if passed:
            self.tests_passed += 1
            self.log(f"✓ {test_name}", "success")
        else:
            self.tests_failed += 1
            self.log(f"✗ {test_name}", "error")
            if details:
                self.log(f"  Details: {details}", "error")
    
    def test_checkpoint_manager_init(self) -> bool:
        """Test CheckpointManager initialization"""
        self.log("\n=== Test 1: CheckpointManager Initialization ===")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_dir = Path(tmpdir) / "checkpoints"
                
                # Test initialization
                manager = CheckpointManager(checkpoint_dir, max_checkpoints=3)
                
                # Verify directory was created
                if not checkpoint_dir.exists():
                    self.test_result(
                        "CheckpointManager init",
                        False,
                        "Checkpoint directory not created"
                    )
                    return False
                
                # Verify attributes
                if manager.max_checkpoints != 3:
                    self.test_result(
                        "CheckpointManager init",
                        False,
                        "max_checkpoints not set correctly"
                    )
                    return False
                
                self.test_result("CheckpointManager init", True)
                return True
                
        except Exception as e:
            self.test_result("CheckpointManager init", False, str(e))
            return False
    
    def test_checkpoint_save_and_load(self) -> bool:
        """Test checkpoint saving and loading"""
        self.log("\n=== Test 2: Checkpoint Save/Load ===")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_dir = Path(tmpdir) / "checkpoints"
                manager = CheckpointManager(checkpoint_dir, max_checkpoints=3)
                
                # Create mock model state
                mock_state = {
                    'epoch': 10,
                    'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
                    'optimizer_state_dict': {'lr': 0.001, 'momentum': 0.9},
                    'scheduler_state_dict': {'last_epoch': 10},
                    'best_metric': 0.95,
                    'training_args': {'batch_size': 32, 'learning_rate': 0.001}
                }
                
                # Save checkpoint
                save_path = manager.save_checkpoint(
                    epoch=10,
                    model_state=mock_state['model_state_dict'],
                    optimizer_state=mock_state['optimizer_state_dict'],
                    scheduler_state=mock_state['scheduler_state_dict'],
                    metric=mock_state['best_metric'],
                    training_args=mock_state['training_args'],
                    is_best=True
                )
                
                if not save_path.exists():
                    self.test_result(
                        "Checkpoint save",
                        False,
                        f"Checkpoint not saved to {save_path}"
                    )
                    return False
                
                self.log("  ✓ Checkpoint saved", "success")
                
                # Load checkpoint
                loaded_state = manager.load_checkpoint(save_path)
                
                if loaded_state is None:
                    self.test_result(
                        "Checkpoint load",
                        False,
                        "Loaded state is None"
                    )
                    return False
                
                # Verify loaded state matches saved state
                if loaded_state['epoch'] != mock_state['epoch']:
                    self.test_result(
                        "Checkpoint load",
                        False,
                        f"Epoch mismatch: {loaded_state['epoch']} != {mock_state['epoch']}"
                    )
                    return False
                
                if loaded_state['best_metric'] != mock_state['best_metric']:
                    self.test_result(
                        "Checkpoint load",
                        False,
                        f"Metric mismatch: {loaded_state['best_metric']} != {mock_state['best_metric']}"
                    )
                    return False
                
                self.log("  ✓ Checkpoint loaded", "success")
                self.log("  ✓ State verification passed", "success")
                
                self.test_result("Checkpoint save/load", True)
                return True
                
        except Exception as e:
            self.test_result("Checkpoint save/load", False, str(e))
            return False
    
    def test_best_checkpoint_tracking(self) -> bool:
        """Test best checkpoint tracking"""
        self.log("\n=== Test 3: Best Checkpoint Tracking ===")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_dir = Path(tmpdir) / "checkpoints"
                manager = CheckpointManager(checkpoint_dir, max_checkpoints=3)
                
                # Save multiple checkpoints with different metrics
                metrics = [0.7, 0.85, 0.92, 0.88, 0.95]  # 0.95 is best
                
                for epoch, metric in enumerate(metrics):
                    manager.save_checkpoint(
                        epoch=epoch,
                        model_state={'weights': torch.randn(5, 5)},
                        optimizer_state={'lr': 0.001},
                        metric=metric,
                        is_best=(metric == max(metrics))
                    )
                
                # Check if best.pt exists
                best_path = checkpoint_dir / "best.pt"
                if not best_path.exists():
                    self.test_result(
                        "Best checkpoint tracking",
                        False,
                        "best.pt not created"
                    )
                    return False
                
                # Load best checkpoint
                best_state = manager.load_checkpoint(best_path)
                
                # Verify it's the checkpoint with highest metric
                if abs(best_state['best_metric'] - 0.95) > 1e-6:
                    self.test_result(
                        "Best checkpoint tracking",
                        False,
                        f"Best metric should be 0.95, got {best_state['best_metric']}"
                    )
                    return False
                
                self.log(f"  ✓ Best checkpoint (metric=0.95) correctly tracked", "success")
                self.test_result("Best checkpoint tracking", True)
                return True
                
        except Exception as e:
            self.test_result("Best checkpoint tracking", False, str(e))
            return False
    
    def test_checkpoint_rotation(self) -> bool:
        """Test checkpoint rotation (max_checkpoints limit)"""
        self.log("\n=== Test 4: Checkpoint Rotation ===")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_dir = Path(tmpdir) / "checkpoints"
                manager = CheckpointManager(checkpoint_dir, max_checkpoints=3)
                
                # Save 5 checkpoints (should keep only 3 + best)
                for epoch in range(5):
                    manager.save_checkpoint(
                        epoch=epoch,
                        model_state={'weights': torch.randn(3, 3)},
                        optimizer_state={'lr': 0.001},
                        metric=0.8 + epoch * 0.01,
                        is_best=(epoch == 4)
                    )
                
                # Count checkpoint files (excluding best.pt)
                checkpoints = list(checkpoint_dir.glob("epoch_*.pt"))
                
                if len(checkpoints) > 3:
                    self.test_result(
                        "Checkpoint rotation",
                        False,
                        f"Expected max 3 checkpoints, found {len(checkpoints)}"
                    )
                    return False
                
                # Verify best.pt still exists
                best_path = checkpoint_dir / "best.pt"
                if not best_path.exists():
                    self.test_result(
                        "Checkpoint rotation",
                        False,
                        "best.pt was deleted during rotation"
                    )
                    return False
                
                self.log(f"  ✓ Kept {len(checkpoints)} checkpoints (max=3)", "success")
                self.log(f"  ✓ best.pt preserved", "success")
                
                self.test_result("Checkpoint rotation", True)
                return True
                
        except Exception as e:
            self.test_result("Checkpoint rotation", False, str(e))
            return False
    
    def test_resume_from_checkpoint(self) -> bool:
        """Test resuming training from checkpoint"""
        self.log("\n=== Test 5: Resume from Checkpoint ===")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_dir = Path(tmpdir) / "checkpoints"
                manager = CheckpointManager(checkpoint_dir, max_checkpoints=3)
                
                # Simulate training session 1
                initial_state = {
                    'epoch': 15,
                    'model_state_dict': {'layer1': torch.randn(5, 5)},
                    'optimizer_state_dict': {'lr': 0.001, 'step': 1500},
                    'scheduler_state_dict': {'last_epoch': 15},
                    'best_metric': 0.89,
                    'random_state': torch.get_rng_state(),
                    'training_args': {'batch_size': 32}
                }
                
                checkpoint_path = manager.save_checkpoint(
                    epoch=initial_state['epoch'],
                    model_state=initial_state['model_state_dict'],
                    optimizer_state=initial_state['optimizer_state_dict'],
                    scheduler_state=initial_state['scheduler_state_dict'],
                    metric=initial_state['best_metric'],
                    training_args=initial_state['training_args'],
                    is_best=True
                )
                
                self.log(f"  ✓ Saved checkpoint at epoch {initial_state['epoch']}", "success")
                
                # Simulate training session 2 (resume)
                resumed_state = manager.load_checkpoint(checkpoint_path)
                
                # Verify all state was restored
                checks = [
                    (resumed_state['epoch'] == initial_state['epoch'], "Epoch"),
                    (resumed_state['best_metric'] == initial_state['best_metric'], "Metric"),
                    ('model_state_dict' in resumed_state, "Model state"),
                    ('optimizer_state_dict' in resumed_state, "Optimizer state"),
                    ('scheduler_state_dict' in resumed_state, "Scheduler state"),
                    ('training_args' in resumed_state, "Training args"),
                ]
                
                all_passed = True
                for passed, name in checks:
                    if passed:
                        self.log(f"  ✓ {name} restored", "success")
                    else:
                        self.log(f"  ✗ {name} NOT restored", "error")
                        all_passed = False
                
                if not all_passed:
                    self.test_result(
                        "Resume from checkpoint",
                        False,
                        "Some state was not restored"
                    )
                    return False
                
                self.test_result("Resume from checkpoint", True)
                return True
                
        except Exception as e:
            self.test_result("Resume from checkpoint", False, str(e))
            return False
    
    def test_preprocessing_checkpoint(self) -> bool:
        """Test preprocessing checkpoint format"""
        self.log("\n=== Test 6: Preprocessing Checkpoint Format ===")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_file = Path(tmpdir) / "preprocess_checkpoint.json"
                
                # Mock preprocessing state
                preprocess_state = {
                    'processed_files': [
                        'image_001.jpg',
                        'image_002.jpg',
                        'image_003.jpg'
                    ],
                    'total_processed': 3,
                    'total_failed': 0,
                    'last_file': 'image_003.jpg',
                    'timestamp': '2025-11-11T12:00:00'
                }
                
                # Save checkpoint
                with open(checkpoint_file, 'w') as f:
                    json.dump(preprocess_state, f, indent=2)
                
                self.log("  ✓ Preprocessing checkpoint saved", "success")
                
                # Load and verify
                with open(checkpoint_file, 'r') as f:
                    loaded_state = json.load(f)
                
                if loaded_state['total_processed'] != 3:
                    self.test_result(
                        "Preprocessing checkpoint",
                        False,
                        "State not restored correctly"
                    )
                    return False
                
                if len(loaded_state['processed_files']) != 3:
                    self.test_result(
                        "Preprocessing checkpoint",
                        False,
                        "Processed files list incorrect"
                    )
                    return False
                
                self.log("  ✓ Preprocessing state verified", "success")
                self.test_result("Preprocessing checkpoint", True)
                return True
                
        except Exception as e:
            self.test_result("Preprocessing checkpoint", False, str(e))
            return False
    
    def run_all_tests(self) -> bool:
        """Run all tests"""
        self.log("=" * 60)
        self.log("Marauder CV - Training Resume Tests")
        self.log("=" * 60)
        
        # Run tests
        self.test_checkpoint_manager_init()
        self.test_checkpoint_save_and_load()
        self.test_best_checkpoint_tracking()
        self.test_checkpoint_rotation()
        self.test_resume_from_checkpoint()
        self.test_preprocessing_checkpoint()
        
        # Print summary
        self.log("\n" + "=" * 60)
        self.log("Test Results Summary")
        self.log("=" * 60)
        self.log(f"Passed: {self.tests_passed}", "success")
        self.log(f"Failed: {self.tests_failed}", "error")
        self.log("")
        
        if self.tests_failed == 0:
            self.log("✓ All tests passed!", "success")
            return True
        else:
            self.log("✗ Some tests failed", "error")
            self.log("\nFailed tests:", "error")
            for result in self.test_results:
                if not result['passed']:
                    self.log(f"  - {result['name']}", "error")
                    if result['details']:
                        self.log(f"    {result['details']}", "error")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test training resume functionality for Marauder CV project"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (skip slow tests)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = TrainingResumeTest(verbose=args.verbose)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
