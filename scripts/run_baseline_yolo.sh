#!/bin/bash
set -e

# Script: run_baseline_yolo.sh
# Purpose: Run Step 2 - Baseline YOLO Training
# Usage: ./run_baseline_yolo.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Train baseline YOLO model on labeled FathomNet dataset"
    echo "Uses SSL backbone as initial input for increased accuracy"
    echo ""
    echo "Options are passed directly to 2_baseline_yolo.py"
    echo "Common options:"
    echo "  --config        Config file (default: config/training_config.yaml)"
    echo "  --data-yaml     YOLO data config file (required)"
    echo "  --output        Output directory for models"
    echo "  --ssl-backbone  Path to SSL pretrained backbone (optional)"
    echo "  --resume        Resume from checkpoint"
    echo ""
    echo "Example:"
    echo "  $0 --data-yaml /datasets/marauder-do-bucket/training/datasets/data.yaml \\"
    echo "     --output /datasets/marauder-do-bucket/training/models/baseline \\"
    echo "     --ssl-backbone /datasets/marauder-do-bucket/training/models/ssl/ssl_backbone_final.pt"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="training/2_baseline_yolo.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the training script
echo "======================================"
echo "Step 2: Baseline YOLO Training"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
