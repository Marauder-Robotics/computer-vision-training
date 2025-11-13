#!/bin/bash
set -e

# Script: run_critical_species.sh
# Purpose: Run Step 4 - Critical Species Specialization
# Usage: ./run_critical_species.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Train specialized model for critical species detection"
    echo "Includes oversampling and hard negative mining"
    echo ""
    echo "Options are passed directly to 4_critical_species.py"
    echo "Common options:"
    echo "  --config              Config file (default: config/training_config.yaml)"
    echo "  --data-dir            Source dataset directory (required)"
    echo "  --output              Output directory for models"
    echo "  --oversample          Create oversampled dataset"
    echo "  --oversample-factor   Oversampling multiplier (default: 5)"
    echo "  --train               Run training (use with --oversample or after)"
    echo "  --base-model          Base YOLO model to start from"
    echo ""
    echo "Example:"
    echo "  # Step 1: Create oversampled dataset"
    echo "  $0 --data-dir /datasets/marauder-do-bucket/training/datasets/organized \\"
    echo "     --oversample --oversample-factor 5"
    echo ""
    echo "  # Step 2: Train specialized model"
    echo "  $0 --data-dir /datasets/marauder-do-bucket/training/datasets/organized \\"
    echo "     --train --base-model /datasets/marauder-do-bucket/training/models/baseline/best.pt"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="training/4_critical_species.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the training script
echo "======================================"
echo "Step 4: Critical Species Specialization"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
