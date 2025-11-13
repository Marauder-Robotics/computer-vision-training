#!/bin/bash
#
# Master Training Script - Marauder CV Pipeline
# Runs complete training pipeline
#

set -e  # Exit on error

echo "=========================================="
echo "Marauder CV Pipeline - Complete Training"
echo "=========================================="
echo ""

# Configuration
CONFIG_FILE="${1:-config/training_config.yaml}"
DATA_DIR="${notebooks/marauder-do-bucket/images/deepfish}"
SKIP_SSL="${SKIP_SSL:-false}"
SKIP_ACTIVE_LEARNING="${SKIP_ACTIVE_LEARNING:-false}"

echo "Configuration: $CONFIG_FILE"
echo "Skip SSL: $SKIP_SSL"
echo "Skip Active Learning: $SKIP_ACTIVE_LEARNING"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p checkpoints logs outputs data

# ====== WEEK 1: SSL PRETRAINING ======
if [ "$SKIP_SSL" != "true" ]; then
    echo ""
    echo "=========================================="
    echo "WEEK 1: SSL Pretraining (MoCo V3)"
    echo "=========================================="
    python training/1_ssl_pretrain.py --config "$CONFIG_FILE" --data-dir 2>&1 | tee logs/1_ssl.log
    
    if [ $? -eq 0 ]; then
        echo "✓ SSL pretraining completed"
    else
        echo "✗ SSL pretraining failed"
        exit 1
    fi
else
    echo "Skipping SSL pretraining..."
fi