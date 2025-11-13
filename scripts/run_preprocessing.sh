#!/bin/bash
set -e

# Script: run_preprocessing.sh
# Purpose: Run hybrid preprocessing on image datasets
# Usage: ./run_preprocessing.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Run hybrid image preprocessing for Marauder CV training"
    echo ""
    echo "Options are passed directly to hybrid_preprocessor.py"
    echo "Common options:"
    echo "  --input-dir     Input directory with raw images"
    echo "  --output-dir    Output directory for processed images"
    echo "  --config        Config file (default: config/training_config.yaml)"
    echo "  --resume        Resume from checkpoint"
    echo ""
    echo "Example:"
    echo "  $0 --input-dir /datasets/marauder-do-bucket/images/fathomnet \\"
    echo "     --output-dir /datasets/marauder-do-bucket/training/datasets/processed"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="data/preprocessing/hybrid_preprocessor.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Preprocessing script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the preprocessing script
echo "======================================"
echo "Marauder CV - Hybrid Preprocessing"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
