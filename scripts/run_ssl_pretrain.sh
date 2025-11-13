#!/bin/bash
set -e

# Script: run_ssl_pretrain.sh
# Purpose: Run Step 1 - Self-Supervised Learning (SSL) Pretraining
# Usage: ./run_ssl_pretrain.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Run SSL pretraining with MoCo V3 on unlabeled underwater images"
    echo ""
    echo "Options are passed directly to 1_ssl_pretrain.py"
    echo "Common options:"
    echo "  --config        Config file (default: config/training_config.yaml)"
    echo "  --data-dir      Directory with unlabeled images (required)"
    echo "  --output        Output directory for models"
    echo "  --resume        Resume from checkpoint"
    echo ""
    echo "Example:"
    echo "  $0 --data-dir /datasets/marauder-do-bucket/images/deepfish \\"
    echo "     --output /datasets/marauder-do-bucket/training/models/ssl"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="training/1_ssl_pretrain.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the training script
echo "======================================"
echo "Step 1: SSL Pretraining"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
