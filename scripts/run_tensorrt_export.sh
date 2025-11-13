#!/bin/bash
set -e

# Script: run_tensorrt_export.sh
# Purpose: Run Step 8 - TensorRT Export for Nano Deployment
# Usage: ./run_tensorrt_export.sh [options]

# Help text
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Export trained models to TensorRT format for Jetson Nano deployment"
    echo "Optimizes models for maximum inference speed and efficiency"
    echo ""
    echo "Options are passed directly to 8_tensorrt_export.py"
    echo "Common options:"
    echo "  --config        Config file (default: config/training_config.yaml)"
    echo "  --models        Comma-separated paths to models (required)"
    echo "  --output        Output directory for TensorRT engines"
    echo "  --precision     Precision mode (fp32|fp16|int8, default: fp16)"
    echo "  --batch-size    Batch size for optimization (default: 1)"
    echo ""
    echo "Example:"
    echo "  $0 --models \"model1.pt,model2.pt,model3.pt\" \\"
    echo "     --output /datasets/marauder-do-bucket/training/models/tensorrt \\"
    echo "     --precision fp16 --batch-size 1"
    echo ""
    echo "Note: This should be run on the Jetson Nano or similar ARM device"
    exit 0
fi

# Check if Python script exists
SCRIPT_PATH="training/8_tensorrt_export.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found: $SCRIPT_PATH"
    echo "Please run from project root directory"
    exit 1
fi

# Run the training script
echo "======================================"
echo "Step 8: TensorRT Export"
echo "======================================"
echo ""
python "$SCRIPT_PATH" "$@"
