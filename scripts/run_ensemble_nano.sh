#!/bin/bash
set -e

# Script: run_ensemble_nano.sh
# Purpose: Run Step 5a - Ensemble Training for Nano Deployment
# Usage: ./run_ensemble_nano.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Train 3-model ensemble optimized for Jetson Nano deployment"
    echo "Creates high precision, balanced, and high recall variants"
    echo ""
    echo "Options are passed directly to 5a_ensemble_training_nano.py"
    echo "Common options:"
    echo "  --config        Config file (default: config/training_config.yaml)"
    echo "  --data-yaml     YOLO data config file (required)"
    echo "  --output        Output directory for ensemble models"
    echo "  --base-model    Base specialized model to start from"
    echo "  --resume        Resume from checkpoint"
    echo ""
    echo "Example:"
    echo "  $0 --data-yaml /datasets/marauder-do-bucket/training/datasets/data.yaml \\"
    echo "     --output /datasets/marauder-do-bucket/training/models/ensemble_nano \\"
    echo "     --base-model /datasets/marauder-do-bucket/training/models/critical/critical_species_final.pt"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="training/5a_ensemble_training_nano.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the training script
echo "======================================"
echo "Step 5a: Ensemble Training (Nano)"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
