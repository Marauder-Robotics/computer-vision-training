# Quick Start Guide

Get up and running with the Marauder CV Pipeline in 15 minutes!

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ (or macOS for development)
- **Python**: 3.10+
- **GPU**: CUDA-capable GPU (for training)
- **RAM**: 16GB+ (32GB recommended for training)
- **Storage**: 500GB+ for datasets

### Accounts Needed
- Weights & Biases (free tier)
- DigitalOcean Spaces or AWS S3 (optional)
- Google Cloud Platform (for shore deployment)

## Installation (5 minutes)

### Step 1: Clone and Setup Environment

```bash
# Extract the package
tar -xzf marauder-cv-pipeline-merged-final.tar.gz
cd marauder-cv-pipeline-merged

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required Variables**:
```bash
WANDB_API_KEY=your_wandb_key
DO_SPACES_KEY=your_spaces_key  # Optional
DO_SPACES_SECRET=your_spaces_secret  # Optional
```

### Step 3: Verify Installation

```bash
# Test imports
python -c "from ultralytics import YOLO; import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.0+, CUDA: True
```

## Quick Data Download (10 minutes)

### Option A: Sample Dataset (Fastest)

Download a small sample for testing:

```bash
# Create data directory
mkdir -p data/sample

# Download ~1000 images for testing
python data/acquisition/fathomnet_downloader.py \
    --config config/training_config.yaml \
    --concepts Pterois "Purple sea urchin" \
    --limit 500

# Organize dataset
python data/acquisition/dataset_organizer.py \
    --source-images data/fathomnet/images \
    --source-labels data/fathomnet/labels \
    --output data/organized \
    --split 0.7 0.15 0.15
```

### Option B: Full Dataset (Recommended)

Download complete Fathomnet data (~280K images):

```bash
# This will take 2-4 hours depending on connection
python data/acquisition/fathomnet_downloader.py \
    --config config/training_config.yaml

# Organize
python data/acquisition/dataset_organizer.py \
    --source-images data/fathomnet/images \
    --source-labels data/fathomnet/labels \
    --output data/organized
```

## Quick Training (1-2 hours with sample data)

### Train Ensemble Models

```bash
# Train ensemble (Week 5)
python training/5a_ensemble_training_nano.py \
    --config config/training_config.yaml

# Expected time: 
# - Sample data: 30-60 minutes
# - Full data: 1-2 days

# Train multi-scale (Week 5)
python training/6_multiscale_training.py \
    --config config/training_config.yaml
```

### Apply TTA and Export

```bash
# Get ensemble model paths
MODELS=$(find checkpoints/ensemble -name "best.pt" -type f)

# Apply TTA and calibration
python training/7_tta_calibration.py \
    --config config/training_config.yaml \
    --models $MODELS

# Export to TensorRT (for Nano)
python training/8_tensorrt_export.py \
    --config config/training_config.yaml \
    --models $MODELS \
    --names high_recall balanced high_precision \
    --validate \
    --package
```

## Quick Inference Test

### Test on Video

```bash
# Download a test video or use your own
# wget https://example.com/underwater_test.mp4 -O test_video.mp4

# Run Nano inference
python inference/nano_inference.py \
    --config config/training_config.yaml \
    --input test_video.mp4 \
    --output results_annotated.mp4

# Run Shore inference
python inference/shore_inference.py \
    --config config/training_config.yaml \
    --input test_video.mp4 \
    --output results.json
```

### View Results

```bash
# Open annotated video
vlc results_annotated.mp4  # or your video player

# View JSON results
cat results.json | python -m json.tool
```

## Quick Evaluation

```bash
# Evaluate a model
python evaluation/comprehensive_evaluator.py \
    --config config/training_config.yaml \
    --model checkpoints/ensemble/balanced/best.pt \
    --test-data data/organized/dataset.yaml \
    --name balanced_eval \
    --profile-energy

# View results
cat outputs/evaluation/balanced_eval_evaluation.json
```

## Quick Deployment to Jetson Nano

### Prerequisites on Jetson

```bash
# On Jetson Nano
sudo apt-get update
sudo apt-get install python3-pip

# Install JetPack 5.0+ (includes TensorRT)
# Follow: https://developer.nvidia.com/embedded/jetpack
```

### Deploy Models

```bash
# Copy deployment package to Jetson
scp -r checkpoints/tensorrt/nano_deployment jetson@JETSON_IP:/tmp/

# SSH into Jetson
ssh jetson@JETSON_IP

# Run setup
cd /tmp/nano_deployment
../../../deployment/nano/setup_jetson.sh

# Deploy
../../../deployment/nano/deploy_nano.sh

# Check status
sudo systemctl status marauder-cv
```

## Common Commands Cheat Sheet

### Data
```bash
# Download Fathomnet data
python data/acquisition/fathomnet_downloader.py --config config/training_config.yaml

# Organize dataset
python data/acquisition/dataset_organizer.py --source-images data/fathomnet/images --source-labels data/fathomnet/labels --output data/organized
```

### Training
```bash
# Train ensemble
python training/5a_ensemble_training_nano.py --config config/training_config.yaml

# Train multi-scale
python training/6_multiscale_training.py --config config/training_config.yaml

# Export to TensorRT
python training/8_tensorrt_export.py --models checkpoints/ensemble/*.pt --names high_recall balanced high_precision --package
```

### Inference
```bash
# Nano inference
python inference/nano_inference.py --input video.mp4 --output results.mp4

# Shore inference
python inference/shore_inference.py --input video.mp4 --output results.json
```

### Evaluation
```bash
# Evaluate model
python evaluation/comprehensive_evaluator.py --model checkpoints/ensemble/balanced/best.pt --test-data data/organized/dataset.yaml --name my_model
```

### Deployment
```bash
# Setup Jetson
./deployment/nano/setup_jetson.sh

# Deploy to Jetson
./deployment/nano/deploy_nano.sh

# Deploy to GCP
python deployment/gcp/deploy_gcp.py --project-id YOUR_PROJECT --model-path checkpoints/ensemble/
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config
nano config/training_config.yaml
# Change: batch_size: 16 → batch_size: 8
```

### Slow Training
```bash
# Enable mixed precision
nano config/training_config.yaml
# Ensure: mixed_precision: true

# Increase workers
# Change: num_workers: 4 → num_workers: 8
```

### Low mAP
```bash
# More training epochs
nano config/training_config.yaml
# Change: epochs: 150 → epochs: 300

# Better augmentation
# Adjust augmentation parameters in config
```

### High Energy Consumption
```bash
# Use smaller model
nano config/training_config.yaml
# Change: model_size: yolov8l → yolov8m

# Reduce FPS
# Change target_fps in config
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.10+
```

## Next Steps

### For Training
1. Review [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed training pipeline
2. Review [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) for architecture
3. Customize hyperparameters in `config/training_config.yaml`

### For Deployment
1. Review [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. Setup monitoring and logging
3. Configure alerting system

### For Development
1. Review [ARCHITECTURE.md](ARCHITECTURE.md)
2. Explore code in `training/` and `inference/`
3. Run tests and benchmarks

## Getting Help

### Documentation
- **README.md**: Complete project overview
- **ARCHITECTURE.md**: System architecture
- **TRAINING_GUIDE.md**: Detailed training instructions
- **DEPLOYMENT_GUIDE.md**: Deployment walkthrough

### Common Issues
- Check PROJECT_STATUS.md for known issues
- Review logs in `logs/` directory
- Check W&B dashboard for training metrics

### Support
- GitHub Issues: [repository-url]/issues
- Documentation: `docs/` directory
- Email: support@marauder-project.org

## Success Checklist

- [ ] Environment setup complete
- [ ] Dependencies installed
- [ ] Data downloaded
- [ ] Dataset organized
- [ ] Training started
- [ ] Models trained
- [ ] TensorRT export complete
- [ ] Inference tested
- [ ] Evaluation run
- [ ] Results reviewed

**You're ready to go! 🚀**

---

**Estimated Time**: 
- Setup: 5 minutes
- Data download (sample): 10 minutes
- Training (sample): 1 hour
- Inference test: 5 minutes
- **Total**: ~1.5 hours for complete workflow with sample data

**Next**: Review [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for complete training pipeline details.
