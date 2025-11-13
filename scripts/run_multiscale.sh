#!/bin/bash
set -e

# Script: run_multiscale.sh
# Purpose: Run Step 6 - Multi-scale Training
# Usage: ./run_multiscale.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Apply dynamic resolution training to ensemble models"
    echo "Improves detection across multiple scales and distances"
    echo ""
    echo "Options are passed directly to 6_multiscale_training.py"
    echo "Common options:"
    echo "  --config            Config file (default: config/training_config.yaml)"
    echo "  --data-yaml         YOLO data config file (required)"
    echo "  --ensemble-models   Comma-separated paths to ensemble models (required)"
    echo "  --output            Output directory for multi-scale models"
    echo "  --resume            Resume from checkpoint"
    echo ""
    echo "Example:"
    echo "  $0 --data-yaml /datasets/marauder-do-bucket/training/datasets/data.yaml \\"
    echo "     --ensemble-models \"model1.pt,model2.pt,model3.pt\" \\"
    echo "     --output /datasets/marauder-do-bucket/training/models/multiscale"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="training/6_multiscale_training.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the training script
echo "======================================"
echo "Step 6: Multi-scale Training"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
