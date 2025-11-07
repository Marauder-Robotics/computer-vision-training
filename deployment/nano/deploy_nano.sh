#!/bin/bash
# Deploy to Jetson Nano

set -e

echo "=========================================="
echo "Marauder CV - Jetson Nano Deployment"
echo "=========================================="

# Check if TensorRT engines exist
ENGINE_DIR="../../checkpoints/tensorrt/nano_deployment/engines"

if [ ! -d "$ENGINE_DIR" ]; then
    echo "ERROR: TensorRT engines not found at $ENGINE_DIR"
    echo "Please run: python training/8_tensorrt_export.py --package"
    exit 1
fi

# Create deployment directory
DEPLOY_DIR="/opt/marauder-cv"
echo "Creating deployment directory: $DEPLOY_DIR"
sudo mkdir -p $DEPLOY_DIR/{models,config,logs}

# Copy TensorRT engines
echo "Copying TensorRT engines..."
sudo cp -r $ENGINE_DIR/* $DEPLOY_DIR/models/

# Copy configuration
echo "Copying configuration..."
sudo cp ../../config/*.yaml $DEPLOY_DIR/config/

# Copy inference script
echo "Copying inference script..."
sudo cp ../../inference/nano_inference.py $DEPLOY_DIR/

# Setup Python environment
echo "Setting up Python environment..."
pip3 install -r ../../requirements.txt --user

# Create systemd service
echo "Creating systemd service..."
cat << 'SERVICEEOF' | sudo tee /etc/systemd/system/marauder-cv.service
[Unit]
Description=Marauder CV Inference Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/opt/marauder-cv
ExecStart=/usr/bin/python3 /opt/marauder-cv/nano_inference.py --source 0
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Enable and start service
echo "Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable marauder-cv
sudo systemctl start marauder-cv

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "Service status: sudo systemctl status marauder-cv"
echo "View logs: sudo journalctl -u marauder-cv -f"
echo "=========================================="
