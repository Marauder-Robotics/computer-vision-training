# Training Guide

Complete guide for training the Marauder CV Pipeline from data acquisition to deployment-ready models.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Step 0: Preprocessing (REQUIRED)](#step-0-preprocessing-required)
5. [Week 1: Foundation Models](#week-1-foundation-models)
6. [Week 2: Active Learning](#week-2-active-learning)
7. [Week 3: Annotation](#week-3-annotation)
8. [Week 4: Specialization](#week-4-specialization)
9. [Week 5: Ensemble Training](#week-5-ensemble-training)
10. [Week 6: Optimization & Export](#week-6-optimization--export)
11. [Testing & Validation](#testing--validation)
12. [Troubleshooting](#troubleshooting)
13. [Best Practices](#best-practices)

---

## Prerequisites

### Hardware Requirements

#### Training (Paperspace or Cloud Platform)
- **GPU**: NVIDIA A4000 (16GB), A6000 (48GB), or better
- **RAM**: 32GB minimum, 64GB recommended
- **Storage**: 1TB SSD (for datasets and checkpoints)
- **Network**: High-speed internet for data download

#### Development/Testing
- **CPU**: 8+ cores
- **RAM**: 16GB minimum
- **GPU**: Optional but recommended for local testing

### Software Requirements

```bash
# Operating System
Ubuntu 20.04 LTS or newer (22.04 recommended)

# Python
Python 3.10 or newer

# CUDA (for GPU training)
CUDA 11.8 or newer
cuDNN 8.6+

# Git
git version 2.25+
```

### Required Skills

- Basic Linux command line
- Python fundamentals
- Understanding of computer vision concepts
- Familiarity with YOLO models (helpful but not required)

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/Marauder-Robotics/computer-vision-training.git
cd computer-vision-training
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Install the project in development mode
pip install -e .
```

### 4. Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# Run validation tests
./scripts/test_data_pipeline.sh --quick
python scripts/validate_species_mapping.py
```

### 5. Configure DigitalOcean Spaces (for Paperspace)

If using Paperspace with DO Spaces:

```bash
# Mount DO Spaces bucket at /datasets/marauder-do-bucket
# This is done through Paperspace data sources configuration

# Verify mount
ls -la /datasets/marauder-do-bucket
```

### 6. Set Environment Variables

```bash
# Create .env file
cat > .env << 'EOF'
# DigitalOcean Spaces Configuration
DO_BUCKET_PATH=/datasets/marauder-do-bucket
IMAGES_PATH=/datasets/marauder-do-bucket/images
TRAINING_PATH=/datasets/marauder-do-bucket/training

# Training Configuration
BATCH_SIZE=32
NUM_WORKERS=4
DEVICE=cuda
MIXED_PRECISION=true
EOF

# Load environment variables
source .env
```

---

## Data Preparation

### 1. Create Directory Structure

```bash
# Create local directories
mkdir -p data/raw data/processed checkpoints logs outputs

# Create DO Spaces directories (if using cloud storage)
mkdir -p /datasets/marauder-do-bucket/images/{fathomnet,deepfish,marauder}
mkdir -p /datasets/marauder-do-bucket/training/{checkpoints,logs,datasets,models}
```

### 2. Download Datasets

#### FathomNet Dataset

```bash
python data/acquisition/fathomnet_downloader.py \
  --output /datasets/marauder-do-bucket/images/fathomnet \
  --max-images 50000 \
  --resume
```

**Expected Output**: ~10,000-50,000 labeled underwater images  
**Duration**: 2-6 hours depending on network speed

#### DeepFish Dataset

```bash
python data/acquisition/deepfish_downloader.py \
  --output /datasets/marauder-do-bucket/images/deepfish \
  --resume
```

**Expected Output**: ~40,000 underwater images across 20 terrains  
**Duration**: 1-3 hours

#### Marauder Dataset

Upload your custom Marauder images to:
```
/datasets/marauder-do-bucket/images/marauder/
```

### 3. Organize Datasets

```bash
python data/acquisition/dataset_organizer.py \
  --input-dirs /datasets/marauder-do-bucket/images/fathomnet \
               /datasets/marauder-do-bucket/images/deepfish \
               /datasets/marauder-do-bucket/images/marauder \
  --output /datasets/marauder-do-bucket/training/datasets/organized \
  --config config/species_mapping.yaml \
  --all
```

**What This Does**:
- Organizes images by priority (critical/important/general)
- Creates train/val/test splits (70/20/10)
- Generates YOLO format labels
- Creates symbolic links to save space
- Generates `data.yaml` for YOLO training

**Expected Output**:
```
organized/
├── critical/      # 20 critical species
├── important/     # 9 important species
├── general/       # 7 general categories
├── ssl/           # Unlabeled for SSL pretraining
└── data.yaml      # YOLO configuration file
```

---

## Step 0: Preprocessing (REQUIRED)

### Purpose

Apply hybrid preprocessing to enhance underwater image quality before training. This is a **required** step that significantly improves model performance.

### Features

- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Dehazing for water clarity
- Color correction for underwater lighting
- White balance adjustment
- Checkpoint support for large datasets

### Run Preprocessing

```bash
# Process all datasets
./scripts/run_preprocessing.sh \
  --input-dir /datasets/marauder-do-bucket/images/fathomnet \
  --output-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/fathomnet \
  --config config/training_config.yaml

./scripts/run_preprocessing.sh \
  --input-dir /datasets/marauder-do-bucket/images/deepfish \
  --output-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/deepfish

./scripts/run_preprocessing.sh \
  --input-dir /datasets/marauder-do-bucket/images/marauder \
  --output-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/marauder
```

### Resume If Interrupted

```bash
# Preprocessing saves checkpoints every 1000 images
./scripts/run_preprocessing.sh \
  --input-dir /datasets/marauder-do-bucket/images/fathomnet \
  --output-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/fathomnet \
  --resume
```

### Verify Preprocessing

```bash
# Check output directory
ls -lh /datasets/marauder-do-bucket/training/datasets/preprocessed/fathomnet

# Verify image quality (view a few processed images)
# Use your favorite image viewer or:
python -c "from PIL import Image; Image.open('path/to/processed/image.jpg').show()"
```

**Expected Duration**: 2-4 hours for 50,000 images  
**Checkpoint Frequency**: Every 1,000 images

---

## Week 1: Foundation Models

Week 1 focuses on building the foundation models: SSL pretraining and baseline YOLO.

### Step 1: SSL Pretraining

#### Purpose

Use self-supervised learning (MoCo V3) to learn robust underwater features without labels.

#### Features

- Multi-GPU/DDP support
- Mixed precision training (AMP)
- Underwater-specific augmentations
- Model EMA for better generalization
- Warmup + cosine annealing schedule

#### Run SSL Pretraining

```bash
./scripts/run_ssl_pretrain.sh \
  --data-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/deepfish \
  --output /datasets/marauder-do-bucket/training/models/ssl \
  --config config/training_config.yaml \
  --epochs 100 \
  --batch-size 256
```

#### Multi-GPU Training

```bash
# Training will automatically use all available GPUs
# To specify GPUs:
CUDA_VISIBLE_DEVICES=0,1 ./scripts/run_ssl_pretrain.sh \
  --data-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/deepfish \
  --output /datasets/marauder-do-bucket/training/models/ssl \
  --epochs 100 \
  --batch-size 256
```

#### Monitor Training

```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi

# View logs
tail -f /datasets/marauder-do-bucket/training/logs/ssl_pretrain.log
```

#### Resume If Interrupted

```bash
./scripts/run_ssl_pretrain.sh \
  --data-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/deepfish \
  --output /datasets/marauder-do-bucket/training/models/ssl \
  --resume /datasets/marauder-do-bucket/training/checkpoints/ssl/checkpoint_latest.pt
```

**Expected Duration**: 12-24 hours (A4000/A6000)  
**Expected Output**: `ssl_backbone_final.pt` (pretrained ResNet50 backbone)  
**Checkpoint Frequency**: Every 10 epochs

---

### Step 2: Baseline YOLO Training

#### Purpose

Train initial detection model on labeled FathomNet data using the SSL backbone.

#### Run Baseline Training

```bash
./scripts/run_baseline_yolo.sh \
  --data-yaml /datasets/marauder-do-bucket/training/datasets/organized/data.yaml \
  --output /datasets/marauder-do-bucket/training/models/baseline \
  --ssl-backbone /datasets/marauder-do-bucket/training/models/ssl/ssl_backbone_final.pt \
  --config config/training_config.yaml \
  --epochs 100
```

#### Training Without SSL Backbone (Not Recommended)

```bash
# If SSL pretraining failed or you want to skip it:
./scripts/run_baseline_yolo.sh \
  --data-yaml /datasets/marauder-do-bucket/training/datasets/organized/data.yaml \
  --output /datasets/marauder-do-bucket/training/models/baseline \
  --epochs 100
```

#### Monitor Training

```bash
# View training progress
tail -f /datasets/marauder-do-bucket/training/logs/baseline_yolo.log

# Check validation metrics
# Metrics are logged every epoch
```

**Expected Duration**: 6-12 hours  
**Expected Output**: `best.pt` (best model by validation mAP)  
**Checkpoint Frequency**: Every epoch, keeps best 5

---

## Week 2: Active Learning

### Step 3: Active Learning Sample Selection

#### Purpose

Select the most valuable unlabeled images for annotation using uncertainty and diversity sampling.

#### Features

- 6 uncertainty metrics (entropy, confidence, margin, variation ratio, Bayesian, predictive entropy)
- Cluster-based diversity sampling
- Ensemble support for better uncertainty estimates
- Multi-criteria scoring

#### Run Active Learning

```bash
./scripts/run_active_learning.sh \
  --unlabeled-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/marauder \
  --model /datasets/marauder-do-bucket/training/models/baseline/best.pt \
  --output /datasets/marauder-do-bucket/training/active_learning/selected \
  --config config/training_config.yaml \
  --num-samples 2000 \
  --strategy entropy
```

#### Using Ensemble for Better Selection

```bash
# If you have multiple trained models, use ensemble:
./scripts/run_active_learning.sh \
  --unlabeled-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/marauder \
  --model /datasets/marauder-do-bucket/training/models/baseline/best.pt \
  --output /datasets/marauder-do-bucket/training/active_learning/selected \
  --num-samples 2000 \
  --use-ensemble
```

#### Review Selected Samples

```bash
# Check selected images
ls -lh /datasets/marauder-do-bucket/training/active_learning/selected/

# View uncertainty scores
cat /datasets/marauder-do-bucket/training/active_learning/selected/uncertainty_scores.json
```

**Expected Duration**: 2-4 hours  
**Expected Output**: 2,000 high-uncertainty, diverse images for annotation  
**Checkpoint Frequency**: Saves inference progress

---

## Week 3: Annotation

### External Annotation Service

#### Recommended Provider: Mindy Services

1. **Export Selected Images**
```bash
# Package selected images for annotation
cd /datasets/marauder-do-bucket/training/active_learning/selected
tar -czf images_for_annotation.tar.gz images/
```

2. **Upload to Annotation Service**
- Upload `images_for_annotation.tar.gz`
- Provide annotation guidelines (see below)
- Specify YOLO format output

3. **Annotation Guidelines**

Provide these instructions to the annotation service:

```
MARAUDER CV ANNOTATION GUIDELINES

1. FORMAT: YOLO bounding boxes
   - One .txt file per image
   - Format: class_id x_center y_center width height
   - Coordinates normalized 0.0-1.0

2. BOUNDING BOXES:
   - Draw tight boxes around each organism
   - Only annotate organisms with >30% visibility
   - Annotate only the visible portion
   - Each individual organism gets separate box

3. CLASS ASSIGNMENT:
   - Most certain: Use species-level label
   - Somewhat certain: Use genus or family
   - Uncertain: Use broad category (e.g., "fish")
   - See species_mapping.yaml for class IDs

4. QUALITY CHECKS:
   - Double-check all critical species (IDs 0-19)
   - Verify no duplicate boxes on same organism
   - Ensure boxes don't extend beyond image boundaries

5. SPECIAL CASES:
   - Occluded organisms: Label visible portion only
   - Schooling fish: Label each individual if possible
   - Partial organisms: Label if >30% visible
```

4. **Expected Turnaround**: 3-5 business days  
5. **Expected Cost**: $0.10-0.20 per image (~$200-400 for 2,000 images)

#### Receive and Validate Annotations

```bash
# Download annotated dataset from service
# Extract to organized directory
tar -xzf annotated_images.tar.gz -C /datasets/marauder-do-bucket/training/datasets/annotated/

# Validate annotations
python scripts/validate_annotations.py \
  --images /datasets/marauder-do-bucket/training/datasets/annotated/images \
  --labels /datasets/marauder-do-bucket/training/datasets/annotated/labels \
  --config config/species_mapping.yaml
```

---

## Week 4: Specialization

### Step 4: Critical Species Training

#### Purpose

Train specialized model with focus on critical species, reducing false positives.

#### Features

- Focal loss for class imbalance
- Advanced hard negative mining
- Intelligent augmentation preserving species features
- Multi-stage training (standard → mine → fine-tune)
- 5x oversampling for critical species

#### Stage 1: Create Oversampled Dataset

```bash
./scripts/run_critical_species.sh \
  --config config/training_config.yaml \
  --data-dir /datasets/marauder-do-bucket/training/datasets/organized \
  --output /datasets/marauder-do-bucket/training/models/critical \
  --oversample \
  --oversample-factor 5
```

**What This Does**:
- Identifies critical species images
- Creates 5 augmented versions of each critical species image
- Preserves non-critical species balance

**Duration**: 1-2 hours

#### Stage 2: Train Specialized Model

```bash
./scripts/run_critical_species.sh \
  --config config/training_config.yaml \
  --data-dir /datasets/marauder-do-bucket/training/datasets/organized \
  --output /datasets/marauder-do-bucket/training/models/critical \
  --train \
  --base-model /datasets/marauder-do-bucket/training/models/baseline/best.pt \
  --epochs 50
```

**Training Stages**:
1. **Standard Training** (20 epochs): Train on oversampled dataset
2. **Hard Negative Mining** (20 epochs): Mine and retrain on false positives
3. **Fine-tuning** (10 epochs): Final refinement

**Expected Duration**: 8-16 hours  
**Expected Output**: `critical_species_final.pt`  
**Checkpoint Frequency**: Each stage saves checkpoints

#### Monitor Training Progress

```bash
# Watch training logs
tail -f /datasets/marauder-do-bucket/training/logs/critical_species.log

# View hard negative mining results
cat /datasets/marauder-do-bucket/training/models/critical/hard_negatives/statistics.json
```

---

## Week 5: Ensemble Training

### Step 6a: Nano Ensemble

#### Purpose

Train 3-variant ensemble optimized for Jetson Nano deployment (energy-efficient).

#### Variants

1. **High Recall**: Optimized for sensitivity (minimize missed detections)
2. **Balanced**: Best overall F1 score
3. **High Precision**: Optimized for accuracy (minimize false positives)

#### Run Nano Ensemble Training

```bash
./scripts/run_ensemble_nano.sh \
  --data-yaml /datasets/marauder-do-bucket/training/datasets/organized/data.yaml \
  --output /datasets/marauder-do-bucket/training/models/ensemble_nano \
  --base-model /datasets/marauder-do-bucket/training/models/critical/critical_species_final.pt \
  --config config/training_config.yaml
```

**Expected Duration**: 12-18 hours  
**Expected Output**: 
- `model_recall.pt` - High recall variant
- `model_balanced.pt` - Balanced variant
- `model_precision.pt` - High precision variant

**Energy Budget**: 14.4-18 Wh/day @ 5 FPS

---

### Step 6b: Shore Ensemble

#### Purpose

Train 6-variant dual ensemble for maximum accuracy on shore-based processing (GCP).

#### Variants

- 3x YOLOv8x models: recall/balanced/precision
- 3x YOLOv11x models: recall/balanced/precision

#### Run Shore Ensemble Training

```bash
./scripts/run_ensemble_shore.sh \
  --data-yaml /datasets/marauder-do-bucket/training/datasets/organized/data.yaml \
  --output /datasets/marauder-do-bucket/training/models/ensemble_shore \
  --base-model /datasets/marauder-do-bucket/training/models/critical/critical_species_final.pt \
  --config config/training_config.yaml
```

**Expected Duration**: 16-24 hours  
**Expected Output**: 6 models (3x YOLOv8x + 3x YOLOv11x)  
**Accuracy**: +5-10% mAP over nano ensemble

---

## Week 6: Optimization & Export

### Step 7: Multi-Scale Training

#### Purpose

Improve detection across multiple scales and object sizes through dynamic resolution training.

#### Run Multi-Scale Training

```bash
./scripts/run_multiscale.sh \
  --data-yaml /datasets/marauder-do-bucket/training/datasets/organized/data.yaml \
  --ensemble-models /datasets/marauder-do-bucket/training/models/ensemble_nano/*.pt \
  --output /datasets/marauder-do-bucket/training/models/multiscale \
  --config config/training_config.yaml
```

**Features**:
- Dynamic resolution (480-768px)
- Scale-specific augmentations
- Progressive scaling schedule

**Expected Duration**: 8-12 hours  
**Improvement**: +3-5% mAP on small objects

---

### Step 8: TTA & Calibration

#### Purpose

Apply test-time augmentation and calibrate confidence scores for reliable predictions.

#### Run TTA & Calibration

```bash
./scripts/run_tta_calibration.sh \
  --models /datasets/marauder-do-bucket/training/models/multiscale/*.pt \
  --val-dataset /datasets/marauder-do-bucket/training/datasets/organized/val \
  --output /datasets/marauder-do-bucket/training/models/calibrated \
  --config config/training_config.yaml
```

**Features**:
- TTA: Flip, rotate, scale augmentations
- Temperature scaling for critical species
- Threshold optimization
- Confidence calibration

**Expected Duration**: 4-6 hours  
**Improvement**: +2-3% mAP, better confidence scores

---

### Step 9: TensorRT Export

#### Purpose

Export models to TensorRT format for optimized inference on Jetson Nano.

#### Requirements

**IMPORTANT**: This step must be run on the target Jetson Nano device for optimal compatibility.

#### Run TensorRT Export (on Jetson Nano)

```bash
# Copy models to Jetson Nano first
scp /datasets/marauder-do-bucket/training/models/calibrated/*.pt nano@jetson-ip:/home/nano/models/

# SSH into Jetson Nano
ssh nano@jetson-ip

# Run export
./scripts/run_tensorrt_export.sh \
  --models /home/nano/models/*.pt \
  --output /home/nano/models/tensorrt \
  --precision fp16 \
  --batch-size 1
```

**Expected Duration**: 2-4 hours  
**Expected Output**: `.engine` files optimized for Jetson Nano  
**Speedup**: 2-3x faster inference vs PyTorch

---

## Testing & Validation

### Run Validation Suite

```bash
# Test data pipeline
./scripts/test_data_pipeline.sh --full

# Test training resumption
python scripts/test_training_resume.py

# Validate species mapping
python scripts/validate_species_mapping.py --strict
```

### Validate Model Performance

```bash
# Run comprehensive evaluation
python evaluation/comprehensive_evaluator.py \
  --model /datasets/marauder-do-bucket/training/models/calibrated/best.pt \
  --data-yaml /datasets/marauder-do-bucket/training/datasets/organized/data.yaml \
  --output /datasets/marauder-do-bucket/evaluation/results
```

**Metrics Evaluated**:
- mAP@0.5 and mAP@0.5:0.95
- Precision, Recall, F1 per class
- Counting accuracy
- Inference speed (FPS)
- Energy consumption (for nano models)

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms**: Training crashes with "RuntimeError: CUDA out of memory"

**Solutions**:
```bash
# Reduce batch size
./scripts/run_baseline_yolo.sh --batch-size 16  # Default is 32

# Use gradient accumulation
./scripts/run_ssl_pretrain.sh --batch-size 64 --accumulate 4  # Effective batch = 256

# Enable mixed precision (if not already enabled)
# Check config/training_config.yaml: mixed_precision: true
```

#### 2. Training Stops After 6 Hours (Paperspace)

**Symptoms**: Training stops at 6-hour mark on Paperspace free tier

**Solutions**:
```bash
# All scripts auto-save checkpoints
# Simply resume with --resume flag:
./scripts/run_ssl_pretrain.sh \
  --data-dir /path/to/data \
  --resume /datasets/marauder-do-bucket/training/checkpoints/ssl/checkpoint_latest.pt
```

#### 3. Slow Data Loading

**Symptoms**: GPU utilization <50%, data loading is bottleneck

**Solutions**:
```bash
# Increase num_workers in config
# Edit config/training_config.yaml:
# num_workers: 8  # Increase from default 4

# Use faster storage (SSD vs HDD)
# Preload data to RAM if possible
```

#### 4. Poor Validation Performance

**Symptoms**: Low mAP or high false positive rate

**Solutions**:
1. **Check Preprocessed Images**: Ensure preprocessing completed successfully
2. **Validate Labels**: Use `scripts/validate_annotations.py` to check label quality
3. **Adjust Learning Rate**: Reduce if training is unstable
4. **Increase Training Time**: Some models need more epochs
5. **Check Class Balance**: Use oversampling for underrepresented classes

#### 5. Models Not Converging

**Symptoms**: Loss plateaus early or doesn't decrease

**Solutions**:
```bash
# Use SSL pretrained backbone
./scripts/run_baseline_yolo.sh \
  --ssl-backbone /path/to/ssl_backbone_final.pt

# Adjust learning rate
# Edit config/training_config.yaml
# learning_rate: 0.001  # Try 0.0001 if unstable

# Check data augmentation
# Too much augmentation can hurt performance
```

#### 6. Annotation Quality Issues

**Symptoms**: Model performs poorly despite good training metrics

**Solutions**:
1. **Manual Review**: Spot-check 100-200 random annotations
2. **Validate with Script**: Run `scripts/validate_annotations.py`
3. **Check Class Distribution**: Ensure critical species are well-represented
4. **Verify Bounding Boxes**: Should be tight around organisms

---

## Best Practices

### 1. Checkpoint Management

```bash
# Always enable checkpoints for long runs
# Checkpoints are automatic in all scripts

# Keep important checkpoints
cp /datasets/marauder-do-bucket/training/checkpoints/ssl/best.pt \
   /datasets/marauder-do-bucket/training/checkpoints/ssl/best_backup_$(date +%Y%m%d).pt

# Clean old checkpoints periodically
# Keep last 5 checkpoints only (automatic in most scripts)
```

### 2. Experiment Tracking

```bash
# Use descriptive output directories
./scripts/run_baseline_yolo.sh \
  --output /datasets/marauder-do-bucket/training/models/baseline_exp1_lr001

# Document changes in a log file
echo "$(date): Experiment 1 - Reduced LR to 0.001" >> training_log.txt

# Save configuration with each experiment
cp config/training_config.yaml \
   /datasets/marauder-do-bucket/training/models/baseline_exp1_lr001/config.yaml
```

### 3. Resource Management

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Use screen or tmux for long runs
screen -S training
./scripts/run_ssl_pretrain.sh --data-dir /path/to/data
# Ctrl+A, D to detach
# screen -r training to reattach

# Check disk space regularly
df -h /datasets/marauder-do-bucket
```

### 4. Data Management

```bash
# Use symbolic links to save space
# dataset_organizer.py does this automatically

# Compress old logs
gzip /datasets/marauder-do-bucket/training/logs/*.log

# Archive completed experiments
tar -czf baseline_exp1.tar.gz /datasets/marauder-do-bucket/training/models/baseline_exp1/
```

### 5. Validation Strategy

```bash
# Validate after each major step
python evaluation/comprehensive_evaluator.py \
  --model /path/to/model.pt \
  --data-yaml /path/to/data.yaml

# Keep validation set separate
# Never train on validation data

# Use stratified splits
# Ensure all species are represented in val/test sets
```

### 6. Documentation

```bash
# Document each experiment
cat >> EXPERIMENT_LOG.md << 'EOF'
## Experiment 5: Critical Species with Focal Loss
**Date**: 2025-11-12
**Goal**: Reduce false positives on critical species
**Changes**: Added focal loss (α=0.25, γ=2.0), increased oversampling to 5x
**Results**: mAP 0.78 (+0.05), FP rate -30%
**Model**: /datasets/marauder-do-bucket/training/models/critical_exp5/
EOF
```

---

## Next Steps

After completing all training steps:

1. **Evaluate Final Models**
```bash
python evaluation/comprehensive_evaluator.py \
  --model /datasets/marauder-do-bucket/training/models/calibrated/best.pt \
  --data-yaml /datasets/marauder-do-bucket/training/datasets/organized/data.yaml
```

2. **Deploy to Nano**
```bash
# See DEPLOYMENT_GUIDE.md for detailed instructions
cd deployment/nano
./deploy_to_nano.sh
```

3. **Deploy to Shore (GCP)**
```bash
# See DEPLOYMENT_GUIDE.md for detailed instructions
cd deployment/gcp
./deploy_to_gcp.sh
```

4. **Run Inference**
```bash
# Nano inference
python inference/nano_inference.py --input video.mp4 --output results.mp4

# Shore inference
python inference/shore_inference.py --input video.mp4 --output results.mp4
```

---

## Additional Resources

- **Architecture Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Pipeline Overview**: [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md)
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Paperspace Setup**: [PAPERSPACE_GUIDE.md](PAPERSPACE_GUIDE.md)

---
