#!/usr/bin/env python3
"""
Week 6: TensorRT Export
Exports trained models to TensorRT format for efficient inference on Jetson Nano
"""

import os
import yaml
import torch
import logging
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorRTExporter:
    """Export YOLO models to TensorRT format"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.trt_config = self.config['tensorrt_export']
        self.output_dir = Path(self.config['paths']['checkpoints']) / 'tensorrt'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_model(
        self,
        model_path: str,
        output_name: str,
        imgsz: int = 640
    ) -> str:
        """
        Export a single model to TensorRT
        
        Args:
            model_path: Path to PyTorch model
            output_name: Name for exported model
            imgsz: Input image size
            
        Returns:
            Path to exported TensorRT engine
        """
        logger.info(f"\nExporting {output_name} to TensorRT...")
        logger.info(f"  Source: {model_path}")
        logger.info(f"  Precision: {self.trt_config['precision']}")
        logger.info(f"  Image size: {imgsz}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = YOLO(model_path)
        
        # Export to TensorRT
        try:
            # First export to ONNX as intermediate
            onnx_path = str(self.output_dir / f"{output_name}.onnx")
            logger.info(f"  Exporting to ONNX (intermediate)...")
            
            model.export(
                format='onnx',
                imgsz=imgsz,
                simplify=True,
                opset=17,
                dynamic=False
            )
            
            # Then export to TensorRT
            logger.info(f"  Exporting to TensorRT...")
            trt_path = model.export(
                format='engine',
                imgsz=imgsz,
                half=self.trt_config['precision'] == 'fp16',
                int8=self.trt_config['precision'] == 'int8',
                workspace=self.trt_config['workspace_size'],
                batch=self.trt_config['max_batch_size'],
                simplify=True,
                device=0,  # Use GPU 0
                verbose=True
            )
            
            logger.info(f"  ✓ TensorRT engine saved: {trt_path}")
            
            # Save export metadata
            export_info = {
                'source_model': str(model_path),
                'tensorrt_engine': str(trt_path),
                'precision': self.trt_config['precision'],
                'imgsz': imgsz,
                'max_batch_size': self.trt_config['max_batch_size'],
                'workspace_size_gb': self.trt_config['workspace_size']
            }
            
            info_path = self.output_dir / f"{output_name}_export_info.yaml"
            with open(info_path, 'w') as f:
                yaml.dump(export_info, f)
            
            return str(trt_path)
            
        except Exception as e:
            logger.error(f"  ✗ Export failed: {e}")
            raise
    
    def export_ensemble(self, ensemble_models: Dict[str, str]) -> Dict[str, str]:
        """
        Export all ensemble models
        
        Args:
            ensemble_models: Dictionary of variant names to model paths
            
        Returns:
            Dictionary of variant names to TensorRT engine paths
        """
        logger.info("\n" + "="*80)
        logger.info("TENSORRT EXPORT - Ensemble Models")
        logger.info("="*80 + "\n")
        
        exported_models = {}
        imgsz = 640  # Standard size for Nano
        
        for variant_name, model_path in ensemble_models.items():
            try:
                trt_path = self.export_model(
                    model_path,
                    f"nano_ensemble_{variant_name}",
                    imgsz=imgsz
                )
                exported_models[variant_name] = trt_path
                
            except Exception as e:
                logger.error(f"Failed to export {variant_name}: {e}")
        
        # Save ensemble export configuration
        ensemble_config = {
            'models': exported_models,
            'precision': self.trt_config['precision'],
            'imgsz': imgsz,
            'dla_core': self.trt_config['dla_core']
        }
        
        config_path = self.output_dir / 'nano_ensemble_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(ensemble_config, f)
        
        logger.info(f"\nEnsemble export configuration saved: {config_path}")
        
        return exported_models
    
    def validate_exported_models(self, exported_models: Dict[str, str]):
        """
        Validate exported TensorRT models
        
        Args:
            exported_models: Dictionary of variant names to engine paths
        """
        logger.info("\n" + "="*80)
        logger.info("VALIDATING TENSORRT MODELS")
        logger.info("="*80 + "\n")
        
        validation_results = {}
        
        for variant_name, engine_path in exported_models.items():
            logger.info(f"Validating {variant_name}...")
            
            try:
                # Load and test the engine
                model = YOLO(engine_path, task='detect')
                
                # Run a test prediction
                # Note: This requires a test image
                test_img_path = 'test_image.jpg'  # Placeholder
                
                if os.path.exists(test_img_path):
                    results = model.predict(test_img_path, verbose=False)
                    validation_results[variant_name] = {
                        'status': 'success',
                        'engine_path': engine_path
                    }
                    logger.info(f"  ✓ Validation successful")
                else:
                    validation_results[variant_name] = {
                        'status': 'no_test_image',
                        'engine_path': engine_path
                    }
                    logger.info(f"  ⚠ No test image available, skipped inference test")
                
            except Exception as e:
                validation_results[variant_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'engine_path': engine_path
                }
                logger.error(f"  ✗ Validation failed: {e}")
        
        # Save validation results
        val_path = self.output_dir / 'validation_results.yaml'
        with open(val_path, 'w') as f:
            yaml.dump(validation_results, f)
        
        logger.info(f"\nValidation results saved: {val_path}")
        
        return validation_results
    
    def generate_deployment_package(self, exported_models: Dict[str, str]):
        """
        Generate deployment package for Jetson Nano
        
        Args:
            exported_models: Dictionary of exported model paths
        """
        logger.info("\n" + "="*80)
        logger.info("GENERATING DEPLOYMENT PACKAGE")
        logger.info("="*80 + "\n")
        
        deploy_dir = self.output_dir / 'nano_deployment'
        deploy_dir.mkdir(exist_ok=True)
        
        # Copy TensorRT engines
        engines_dir = deploy_dir / 'engines'
        engines_dir.mkdir(exist_ok=True)
        
        for variant, engine_path in exported_models.items():
            if os.path.exists(engine_path):
                import shutil
                dest = engines_dir / os.path.basename(engine_path)
                shutil.copy2(engine_path, dest)
                logger.info(f"  Copied {variant}: {dest.name}")
        
        # Create deployment configuration
        deploy_config = {
            'models': {
                variant: f"engines/{os.path.basename(path)}"
                for variant, path in exported_models.items()
            },
            'precision': self.trt_config['precision'],
            'imgsz': 640,
            'energy_budget': {
                'max_daily_wh': 40,
                'target_fps': 5,
                'cameras': 4,
                'capture_duration': 30,
                'captures_per_hour': 2
            },
            'inference': {
                'conf_threshold': 0.25,
                'iou_threshold': 0.5,
                'max_det': 300
            }
        }
        
        config_path = deploy_dir / 'deployment_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(deploy_config, f)
        
        logger.info(f"\nDeployment package created: {deploy_dir}")
        logger.info(f"  Configuration: {config_path}")
        logger.info(f"  Engines: {engines_dir}")
        
        return str(deploy_dir)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Export models to TensorRT')
    parser.add_argument('--config', type=str, default='config/training_config.yaml')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to models to export')
    parser.add_argument('--names', nargs='+', required=True,
                       help='Names for exported models')
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported models')
    parser.add_argument('--package', action='store_true',
                       help='Create deployment package')
    
    args = parser.parse_args()
    
    if len(args.models) != len(args.names):
        raise ValueError("Number of models and names must match")
    
    exporter = TensorRTExporter(args.config)
    
    # Export models
    exported_models = {}
    for model_path, name in zip(args.models, args.names):
        try:
            trt_path = exporter.export_model(model_path, name)
            exported_models[name] = trt_path
        except Exception as e:
            logger.error(f"Failed to export {name}: {e}")
    
    # Validate if requested
    if args.validate and exported_models:
        exporter.validate_exported_models(exported_models)
    
    # Create deployment package if requested
    if args.package and exported_models:
        exporter.generate_deployment_package(exported_models)
    
    logger.info("\n" + "="*80)
    logger.info("TENSORRT EXPORT COMPLETE")
    logger.info("="*80)
    for name, path in exported_models.items():
        logger.info(f"  {name}: {path}")


if __name__ == '__main__':
    main()
