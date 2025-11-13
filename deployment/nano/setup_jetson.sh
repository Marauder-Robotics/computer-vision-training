#!/bin/bash
# Initial Jetson Nano setup

set -e

echo "=========================================="
echo "Jetson Nano Initial Setup"
echo "=========================================="

# Update system
echo "Updating system..."
sudo apt-get update
sudo apt-get upgrade -y

# Install dependencies
echo "Installing dependencies..."
sudo apt-get install -y python3-pip python3-dev build-essential

# Install Python packages
echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install numpy opencv-python pyyaml tqdm

# Setup power mode
echo "Setting power mode to MAXN (15W)..."
sudo nvpmodel -m 0
sudo jetson_clocks

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Copy your TensorRT engines to this device"
echo "2. Run ./deploy_nano.sh"
echo "=========================================="
