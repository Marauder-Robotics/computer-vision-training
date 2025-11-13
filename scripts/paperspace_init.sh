#!/bin/bash
"""
Paperspace Gradient Initialization Script
Sets up environment, clones repo, mounts DO bucket, and prepares for training
Designed for JupyterLab + Terminal environment on Paperspace free tier
"""

set -e  # Exit on error

echo "==========================================="
echo "Marauder CV Training - Paperspace Setup"
echo "==========================================="

# Configuration
GITHUB_REPO="https://github.com/Marauder-Robotics/computer-vision-training.git"
PROJECT_NAME="computer-vision-training"
DO_BUCKET_MOUNT="/datasets/marauder-do-bucket"
PYTHON_VERSION="3.10"
VENV_NAME="marauder_cv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[STATUS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running in Paperspace
if [ ! -d "/notebooks" ] && [ ! -d "/storage" ]; then
    print_warning "This script is designed for Paperspace Gradient environment"
fi

# Step 1: System Updates
print_status "Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    screen \
    htop \
    nvtop \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev

# Step 2: Check DigitalOcean Spaces mount
print_status "Checking DigitalOcean Spaces mount..."
if [ -d "$DO_BUCKET_MOUNT" ]; then
    print_status "DO bucket mounted at $DO_BUCKET_MOUNT"
    ls -la $DO_BUCKET_MOUNT
else
    print_error "DO bucket not mounted at $DO_BUCKET_MOUNT"
    print_warning "Please mount your DigitalOcean Space as a data source in Paperspace"
    echo "Instructions:"
    echo "1. Go to Paperspace Gradient console"
    echo "2. Add data source -> S3-compatible storage"
    echo "3. Mount at: /datasets/marauder-do-bucket"
    exit 1
fi

# Step 3: Clone or update GitHub repository
print_status "Setting up GitHub repository..."
if [ -d "$HOME/$PROJECT_NAME" ]; then
    print_status "Repository exists, pulling latest changes..."
    cd "$HOME/$PROJECT_NAME"
    git fetch origin
    git reset --hard origin/main
    git pull origin main
else
    print_status "Cloning repository..."
    cd "$HOME"
    git clone "$GITHUB_REPO"
    cd "$PROJECT_NAME"
fi

# Step 4: Create Python virtual environment
print_status "Setting up Python virtual environment..."
cd "$HOME/$PROJECT_NAME"

# Remove old venv if exists
if [ -d "venv" ]; then
    rm -rf venv
fi

# Create new virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Step 5: Install Python dependencies
print_status "Installing Python dependencies..."

# Install PyTorch with CUDA support (for A4000/A6000 GPUs)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Install FathomNet separately (sometimes has issues)
pip install --upgrade lxml
pip install fathomnet

# Install the project in development mode
pip install -e .

# Step 6: Setup directory structure in DO bucket
print_status "Setting up directory structure in DO bucket..."
mkdir -p $DO_BUCKET_MOUNT/images/fathomnet
mkdir -p $DO_BUCKET_MOUNT/images/deepfish  
mkdir -p $DO_BUCKET_MOUNT/images/marauder
mkdir -p $DO_BUCKET_MOUNT/training/checkpoints
mkdir -p $DO_BUCKET_MOUNT/training/logs
mkdir -p $DO_BUCKET_MOUNT/training/datasets
mkdir -p $DO_BUCKET_MOUNT/training/models

# Step 7: Create environment file
print_status "Creating environment configuration..."
cat > "$HOME/$PROJECT_NAME/.env" <<EOF
# DigitalOcean Spaces Configuration
DO_BUCKET_PATH=$DO_BUCKET_MOUNT
IMAGES_PATH=$DO_BUCKET_MOUNT/images
TRAINING_PATH=$DO_BUCKET_MOUNT/training
CHECKPOINT_PATH=$DO_BUCKET_MOUNT/training/checkpoints
LOG_PATH=$DO_BUCKET_MOUNT/training/logs

# Training Configuration
BATCH_SIZE=16
NUM_WORKERS=4
DEVICE=cuda
MIXED_PRECISION=true

# Paperspace Configuration
MAX_RUNTIME_HOURS=5.5
CHECKPOINT_INTERVAL=100
LOG_INTERVAL=10
EOF

# Step 8: Create screen configuration
print_status "Setting up screen for long-running processes..."
cat > "$HOME/.screenrc" <<EOF
# Screen configuration for training
startup_message off
defscrollback 10000
hardstatus alwayslastline
hardstatus string '%{= kG}[ %{G}%H %{g}][%= %{= kw}%?%-Lw%?%{r}(%{W}%n*%f%t%?(%u)%?%{r})%{w}%?%+Lw%?%?%= %{g}][%{B} %Y-%m-%d %{W}%c %{g}]'
EOF

# Step 9: Create helper scripts
print_status "Creating helper scripts..."

# Create data download script
cat > "$HOME/$PROJECT_NAME/download_data.sh" <<'EOF'
#!/bin/bash
source venv/bin/activate
echo "Starting data download..."

# Download FathomNet data
echo "Downloading FathomNet data..."
python data/acquisition/fathomnet_downloader.py

# Download DeepFish data
echo "Downloading DeepFish data..."
python data/acquisition/deepfish_downloader.py

# Organize datasets
echo "Organizing datasets..."
python data/acquisition/dataset_organizer.py --all

echo "Data download complete!"
EOF
chmod +x "$HOME/$PROJECT_NAME/download_data.sh"

# Create training start script
cat > "$HOME/$PROJECT_NAME/start_training.sh" <<'EOF'
#!/bin/bash
source venv/bin/activate

# Check for resume flag
if [ "$1" == "--resume" ]; then
    echo "Resuming training from checkpoint..."
    RESUME_FLAG="--resume"
else
    echo "Starting new training..."
    RESUME_FLAG=""
fi

# Start training in screen session
screen -dmS training bash -c "
    source venv/bin/activate
    python training/train_all.py $RESUME_FLAG 2>&1 | tee -a $DO_BUCKET_PATH/training/logs/training_$(date +%Y%m%d_%H%M%S).log
"

echo "Training started in screen session 'training'"
echo "Use 'screen -r training' to attach"
echo "Use 'Ctrl+A, D' to detach"
EOF
chmod +x "$HOME/$PROJECT_NAME/start_training.sh"

# Create monitoring script
cat > "$HOME/$PROJECT_NAME/monitor_training.sh" <<'EOF'
#!/bin/bash
source venv/bin/activate

echo "Training Monitor"
echo "================"
echo ""

# Check if training is running
if screen -list | grep -q "training"; then
    echo "✓ Training is running in screen session"
else
    echo "✗ No training session found"
fi

echo ""
echo "GPU Usage:"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "Latest checkpoint:"
ls -lah $DO_BUCKET_PATH/training/checkpoints/ | tail -5

echo ""
echo "Latest logs:"
tail -20 $DO_BUCKET_PATH/training/logs/$(ls -t $DO_BUCKET_PATH/training/logs/ | head -1)

echo ""
echo "Disk usage:"
df -h $DO_BUCKET_PATH
EOF
chmod +x "$HOME/$PROJECT_NAME/monitor_training.sh"

# Step 10: Test GPU availability
print_status "Testing GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('✗ No CUDA devices found')
"

# Step 11: Create Jupyter kernel
print_status "Creating Jupyter kernel..."
python -m ipykernel install --user --name=$VENV_NAME --display-name="Marauder CV"

# Step 12: Final instructions
echo ""
echo "==========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Download data:        ./download_data.sh"
echo "2. Start training:       ./start_training.sh"
echo "3. Resume training:      ./start_training.sh --resume"
echo "4. Monitor training:     ./monitor_training.sh"
echo "5. Attach to training:   screen -r training"
echo ""
echo "Jupyter kernel '$VENV_NAME' has been created."
echo "Select it in JupyterLab: Kernel -> Change Kernel -> Marauder CV"
echo ""
echo "Environment activated. To reactivate later:"
echo "  cd ~/$PROJECT_NAME && source venv/bin/activate"
echo ""
print_status "Ready for training!"
