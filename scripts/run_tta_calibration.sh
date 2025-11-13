#!/bin/bash
set -e

# Script: run_tta_calibration.sh
# Purpose: Run Step 7 - Test Time Augmentation & Calibration
# Usage: ./run_tta_calibration.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Apply Test Time Augmentation (TTA) and calibrate confidence scores"
    echo "Optimizes thresholds for critical species detection"
    echo ""
    echo "Options are passed directly to 7_tta_calibration.py"
    echo "Common options:"
    echo "  --config            Config file (default: config/training_config.yaml)"
    echo "  --models            Comma-separated paths to models (required)"
    echo "  --val-dataset       Validation dataset path (required)"
    echo "  --output            Output directory for calibrated models"
    echo "  --tta-transforms    TTA transformations to apply"
    echo ""
    echo "Example:"
    echo "  $0 --models \"model1.pt,model2.pt,model3.pt\" \\"
    echo "     --val-dataset /datasets/marauder-do-bucket/training/datasets/val \\"
    echo "     --output /datasets/marauder-do-bucket/training/models/calibrated"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="training/7_tta_calibration.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the training script
echo "======================================"
echo "Step 7: TTA & Calibration"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
