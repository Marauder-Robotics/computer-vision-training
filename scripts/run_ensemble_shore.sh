#!/bin/bash
set -e

# Script: run_ensemble_shore.sh
# Purpose: Run Step 5b - Ensemble Training for Shore-based Deployment
# Usage: ./run_ensemble_shore.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Train 6-model ensemble for shore-based processing (GCP)"
    echo "Combines 3x YOLOv8x + 3x YOLOv11x models for maximum accuracy"
    echo ""
    echo "Options are passed directly to 5b_ensemble_training_shoreside.py"
    echo "Common options:"
    echo "  --config        Config file (default: config/training_config.yaml)"
    echo "  --data-yaml     YOLO data config file (required)"
    echo "  --output        Output directory for ensemble models"
    echo "  --base-model    Base specialized model to start from"
    echo "  --resume        Resume from checkpoint"
    echo ""
    echo "Example:"
    echo "  $0 --data-yaml /datasets/marauder-do-bucket/training/datasets/data.yaml \\"
    echo "     --output /datasets/marauder-do-bucket/training/models/ensemble_shore \\"
    echo "     --base-model /datasets/marauder-do-bucket/training/models/critical/critical_species_final.pt"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="training/5b_ensemble_training_shoreside.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the training script
echo "======================================"
echo "Step 5b: Ensemble Training (Shore)"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
