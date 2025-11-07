"""
Marauder CV Pipeline - Data Module
Handles data acquisition, preprocessing, and active learning
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ["acquisition", "preprocessing", "active_learning"]

# Data paths
DATA_ROOT = Path(__file__).parent.parent / "data"

# Dataset statistics
FATHOMNET_IMAGES_EXPECTED = 280000
DEEPFISH_IMAGES_EXPECTED = 40000
MARAUDER_IMAGES_EXPECTED = 850
