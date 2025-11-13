#!/bin/bash
set -e

# Script: run_active_learning.sh
# Purpose: Run Step 3 - Active Learning Sample Selection
# Usage: ./run_active_learning.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Run inference on unlabeled dataset and select samples for annotation"
    echo "Uses uncertainty and diversity sampling strategies"
    echo ""
    echo "Options are passed directly to 3_active_learning.py"
    echo "Common options:"
    echo "  --config           Config file (default: config/training_config.yaml)"
    echo "  --unlabeled-dir    Directory with unlabeled images (required)"
    echo "  --model            Trained YOLO model path (required)"
    echo "  --output           Output directory for selected samples"
    echo "  --num-samples      Number of samples to select"
    echo "  --strategy         Sampling strategy (entropy|least_confidence|margin)"
    echo ""
    echo "Example:"
    echo "  $0 --unlabeled-dir /datasets/marauder-do-bucket/images/marauder \\"
    echo "     --model /datasets/marauder-do-bucket/training/models/baseline/best.pt \\"
    echo "     --output /datasets/marauder-do-bucket/training/active_learning/selected \\"
    echo "     --num-samples 2000"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="training/3_active_learning.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the training script
echo "======================================"
echo "Step 3: Active Learning"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
