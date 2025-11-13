"""
Training Module - Complete training pipeline ( 1-6)
Includes SSL, baseline, active learning, specialization, ensemble, TTA, and TensorRT export
"""

__version__ = "1.0.0"
__all__ = [
    "1_ssl_pretrain",
    "2_baseline_yolo", 
    "3_active_learning",
    "4_critical_species",
    "5a_ensemble_training_nano",
    "5b_ensemble_training_shoreside",
    "6_multiscale_training",
    "7_tta_calibration",
    "8_tensorrt_export"
]

# Training configuration defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 300
DEFAULT_IMG_SIZE = 640
DEFAULT_DEVICE = "cuda"
