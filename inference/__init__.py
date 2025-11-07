"""
Inference Module - Nano and Shore inference pipelines
Includes ByteTrack counting and ensemble prediction
"""
from pathlib import Path

__version__ = "1.0.0"
__all__ = ["nano_inference", "shore_inference"]

# Inference defaults
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_MAX_DETECTIONS = 300
