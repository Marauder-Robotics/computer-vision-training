# Training Guide

Complete guide for training the Marauder CV Pipeline from data acquisition to deployment-ready models.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Week 1: Foundation](#week-1-foundation)
4. [Week 2: Active Learning](#week-2-active-learning)
5. [Week 3: Annotation](#week-3-annotation)
6. [Week 4: Specialization](#week-4-specialization)
7. [Week 5: Ensemble & Multi-Scale](#week-5-ensemble--multi-scale)
8. [Week 6: Optimization & Export](#week-6-optimization--export)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Prerequisites

### Hardware Requirements

#### Training (Paperspace or Similar)
- **GPU**: A4000 (16GB), A6000 (48GB), or better
- **RAM**: 32GB+ (64GB recommended)
- **Storage**: 1TB SSD
- **Network**: High-speed for data download

#### Development
- **CPU**: 8+ cores
- **RAM**: 16GB minimum
- **GPU**: Optional but recommended for testing

### Software Requirements

```bash
# Operating System
Ubuntu 20.04 LTS or newer

# Python
Python 3.10 or newer

# CUDA (for GPU training)
CUDA 11.8 or newer
cuDNN 8.6+

# Required packages (see requirements.txt)
PyTorch 2.0+
Ultralytics YOLO 8.0+
OpenCV 4.8+
```

### Account Setup

1. **Weights & Biases**
   ```bash
   wandb login YOUR_API_KEY
   ```

2. **Cloud Storage** (DigitalOcean Spaces or AWS S3)
   ```bash
   export DO_SPACES_KEY=your_key
   export DO_SPACES_SECRET=your_secret
   ```

3. **Paperspace** (if using)
   ```bash
   paperspace-node login
   ```

---

## Data Preparation

### Step 1: Download Fathomnet Data

```bash
# Full dataset (~280K images, 2-4 hours)
python data/acquisition/fathomnet_downloader.py \
    --config config/training_config.yaml \
    --limit 1000

# Specific concepts only
python data/acquisition/fathomnet_downloader.py \
    --config config/training_config.yaml \
    --concepts "Pterois" "Purple sea urchin" "Abalone" \
    --limit 500
```

**Output**:
- Images: `data/fathomnet/images/`
- Labels: `data/fathomnet/labels/`
- Metadata: `data/fathomnet/metadata.json`

### Step 2: Add Deepfish Dataset

```bash
# Download Deepfish from https://alzayats.github.io/DeepFish/
# Extract to data/deepfish/

# Expected structure:
# data/deepfish/
#   ├── images/
#   └── (no labels - used for SSL only)
```

### Step 3: Add Marauder Dataset

```bash
# Place your own underwater footage
mkdir -p data/marauder/images
# Copy your images here
```

### Step 4: Organize Dataset

```bash
python data/acquisition/dataset_organizer.py \
    --source-images data/fathomnet/images \
    --source-labels data/fathomnet/labels \
    --output data/organized \
    --split 0.7 0.15 0.15
```

**Output Structure**:
```
data/organized/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

---

## Week 1: Foundation

### SSL Pretraining (Optional but Recommended)

**Purpose**: Learn underwater visual features without labels

```bash
python training/1_ssl_pretrain.py \
    --config config/training_config.yaml
```

**Configuration** (`config/training_config.yaml`):
```yaml
ssl_pretraining:
  enabled: true
  model: "moco_v3"
  backbone: "resnet50"
  epochs: 100
  batch_size: 256
  learning_rate: 0.03
```

**Expected Output**:
- Checkpoint: `checkpoints/ssl/best_moco.pth`
- Training time: 1-2 days (full dataset)
- W&B logs: Training curves, representations

### Baseline YOLO Training

**Purpose**: Train initial detection model

```bash
python training/1_baseline_yolo.py \
    --config config/training_config.yaml
```

**Configuration**:
```yaml
baseline_yolo:
  model_size: "yolov8m"  # or yolov8l
  use_ssl_backbone: true
  epochs: 300
  batch_size: 16
  learning_rate: 0.01
  warmup_epochs: 3
```

**Expected Output**:
- Model: `checkpoints/baseline/best.pt`
- Metrics: mAP50 ~0.55-0.60
- Training time: 2-3 days

**Monitoring**:
```bash
# Watch training progress
tail -f logs/1_baseline.log

# View W&B dashboard
wandb sync
```

---

## Week 2: Active Learning

### Run Inference on Unlabeled Data

```bash
python training/3_active_learning.py \
    --config config/training_config.yaml
```

**What it does**:
1. Loads baseline model
2. Runs inference on unlabeled pool
3. Calculates uncertainty scores
4. Selects most uncertain 2000 images
5. Exports annotation package

**Configuration**:
```yaml
active_learning:
  strategy: "uncertainty_sampling"
  num_samples: 2000
  min_confidence: 0.3
  max_confidence: 0.7
```

**Expected Output**:
- Package: `outputs/active_learning/annotation_package.zip`
- Contains:
  - Images for annotation
  - COCO format metadata
  - Annotation guidelines
  - Thumbnails

---

## Week 3: Annotation

### Send to Mindy Services

```bash
# Package is ready at outputs/active_learning/annotation_package.zip
# Send to Mindy Services via their platform
```

### Wait for Annotations

**Timeline**: 1-3 days depending on service agreement

### Import Annotations

```bash
# Once received, import annotations
python -c "
from data.active_learning.mindy_services_handler import MindyServicesHandler
handler = MindyServicesHandler('config/training_config.yaml')
handler.import_annotations(
    'path/to/annotations.json',
    'data/annotated/labels'
)
"
```

---

## Week 4: Specialization

### Critical Species Training

**Purpose**: Improve detection of critical species (lionfish, urchins, abalone, etc.)

```bash
python training/4_critical_species.py \
    --config config/training_config.yaml
```

**Configuration**:
```yaml
critical_species:
  base_model: "checkpoints/baseline/best.pt"
  oversampling:
    enabled: true
    critical_species_ratio: 3.0
  hard_negative_mining:
    enabled: true
    iterations: 3
```

**What happens**:
1. **Oversampling**: Critical species repeated 3x in training
2. **Hard Negative Mining**:
   - Iteration 1: Train → Find false positives → Add to dataset
   - Iteration 2: Train → Find false positives → Add to dataset
   - Iteration 3: Train → Final model

**Expected Output**:
- Model: `checkpoints/critical_species/best.pt`
- Metrics: mAP50 ~0.62-0.65
- Critical species recall: >0.70
- Training time: 1.5-2 days

---

## Week 5: Ensemble & Multi-Scale

### Ensemble Training

**Purpose**: Create 3 complementary models

```bash
python training/5a_ensemble_training_nano.py \
    --config config/training_config.yaml \
    --validate
```

**Three Variants**:

1. **High Recall** (maximize detections)
   - Low confidence threshold: 0.15
   - Heavy augmentation
   - Focus: Don't miss anything

2. **Balanced** (balanced performance)
   - Medium confidence: 0.25
   - Standard augmentation
   - Focus: Good overall performance

3. **High Precision** (minimize false positives)
   - High confidence: 0.4
   - Light augmentation
   - Focus: Only confident detections

**Expected Output**:
- Models:
  - `checkpoints/ensemble/high_recall/best.pt`
  - `checkpoints/ensemble/balanced/best.pt`
  - `checkpoints/ensemble/high_precision/best.pt`
- Training time: 3-4 days (all variants)

### Multi-Scale Training

**Purpose**: Improve detection at different scales

```bash
python training/6_multiscale_training.py \
    --config config/training_config.yaml
```

**Configuration**:
```yaml
multiscale_training:
  resolutions: [480, 512, 576, 640, 704, 768]
  scale_variance: 0.5
  epochs: 100
```

**Expected Output**:
- Models with "_multiscale" suffix
- Better small object detection
- Training time: 2-3 days

---

## Week 6: Optimization & Export

### Test-Time Augmentation & Calibration

```bash
# Get ensemble model paths
MODELS=$(find checkpoints/ensemble -name "best.pt" -type f | tr '\n' ' ')

python training/7_tta_calibration.py \
    --config config/training_config.yaml \
    --models $MODELS \
    --calibration-data data/organized/val
```

**What it does**:
1. **TTA**: Test with flips, rotations, scales
2. **Calibration**: Adjust confidence scores
3. **Threshold Optimization**: Find optimal thresholds per class

**Expected Output**:
- Config: `checkpoints/tta_calibration/pipeline_config.yaml`
- Calibrator: `checkpoints/tta_calibration/calibrator.pth`
- Improved mAP: +0.02-0.05

### TensorRT Export

```bash
python training/8_tensorrt_export.py \
    --config config/training_config.yaml \
    --models checkpoints/ensemble/high_recall/best.pt \
            checkpoints/ensemble/balanced/best.pt \
            checkpoints/ensemble/high_precision/best.pt \
    --names high_recall balanced high_precision \
    --validate \
    --package
```

**What it does**:
1. Export PyTorch → ONNX → TensorRT
2. FP16 quantization
3. Optimization for Jetson Nano
4. Validation on test set
5. Package for deployment

**Expected Output**:
- Engines: `checkpoints/tensorrt/nano_deployment/engines/*.engine`
- Config: `checkpoints/tensorrt/nano_deployment/deployment_config.yaml`
- Package ready for Nano deployment

---

## Troubleshooting

### CUDA Out of Memory

**Problem**: GPU memory exceeded during training

**Solutions**:
```bash
# 1. Reduce batch size
nano config/training_config.yaml
# Change: batch_size: 16 → batch_size: 8

# 2. Reduce image size
# Change: imgsz: 640 → imgsz: 512

# 3. Use gradient accumulation
# Add: gradient_accumulation_steps: 2
```

### Slow Training

**Problem**: Training taking too long

**Solutions**:
```bash
# 1. Enable mixed precision
# Ensure in config: mixed_precision: true

# 2. Increase workers
# Change: num_workers: 4 → num_workers: 8

# 3. Use SSD for data storage
# Move data to faster drive

# 4. Use smaller dataset for testing
python data/acquisition/fathomnet_downloader.py --limit 1000
```

### Low mAP

**Problem**: Model accuracy below target

**Solutions**:
```bash
# 1. More training epochs
# Change: epochs: 150 → epochs: 300

# 2. Better augmentation
# Increase augmentation strength in config

# 3. More training data
# Download full Fathomnet dataset

# 4. Check data quality
python -c "
from pathlib import Path
labels = list(Path('data/organized/labels/train').glob('*.txt'))
print(f'Training labels: {len(labels)}')
# Should be >10,000 for good performance
"
```

### High Energy Consumption

**Problem**: Nano energy budget exceeded

**Solutions**:
```bash
# 1. Use smaller model
# Change: model_size: yolov8l → yolov8m

# 2. Reduce FPS
# Change: target_fps: 5 → target_fps: 3

# 3. Reduce capture duration
# Change: capture_duration: 30 → capture_duration: 20
```

### Training Crash/Interruption

**Problem**: Training stopped unexpectedly

**Solution**: Resume from checkpoint
```bash
# Training automatically resumes if checkpoint exists
python training/5a_ensemble_training_nano.py --config config/training_config.yaml

# Or manually specify checkpoint
python training/5a_ensemble_training_nano.py \
    --config config/training_config.yaml \
    --resume checkpoints/ensemble/high_recall/last.pt
```

---

## Best Practices

### Data Management

1. **Version Control**: Track dataset versions
   ```bash
   # Create dataset manifest
   find data/organized -type f > data/dataset_v1_manifest.txt
   ```

2. **Data Validation**: Check before training
   ```bash
   # Validate dataset
   python -c "
   from ultralytics import YOLO
   YOLO().val(data='data/organized/dataset.yaml', split='val')
   "
   ```

3. **Backup**: Regular backups of checkpoints
   ```bash
   # Sync to cloud storage
   rclone sync checkpoints/ remote:marauder-checkpoints/
   ```

### Training Optimization

1. **Use Mixed Precision**: 30-50% faster
   ```yaml
   general:
     mixed_precision: true
   ```

2. **Optimal Batch Size**: Find maximum that fits in memory
   ```bash
   # Start with 16, double until OOM, then use previous
   ```

3. **Learning Rate Finder**: Find optimal LR
   ```bash
   # Plot learning rate curve
   # Choose LR at steepest descent
   ```

4. **Early Stopping**: Save time on plateaus
   ```yaml
   baseline_yolo:
     patience: 50  # Stop if no improvement in 50 epochs
   ```

### Monitoring

1. **W&B Dashboard**: Monitor all experiments
   ```bash
   wandb login
   # View at wandb.ai
   ```

2. **TensorBoard**: Alternative monitoring
   ```bash
   tensorboard --logdir runs/
   ```

3. **Log Analysis**: Check logs regularly
   ```bash
   tail -f logs/5a_ensemble.log
   grep "mAP" logs/5a_ensemble.log
   ```

### Experiment Tracking

1. **Name Experiments**: Clear naming convention
   ```yaml
   logging:
     wandb:
       name: "ensemble_v1_highrecall_20231106"
   ```

2. **Tag Experiments**: Use tags for organization
   ```python
   wandb.init(tags=["ensemble", "week5", "high-recall"])
   ```

3. **Compare Runs**: Use W&B compare feature

### Checkpoint Management

1. **Keep Best Models**: Don't overwrite
   ```yaml
   checkpointing:
     save_best_only: false
     keep_last_n: 3
   ```

2. **Test Before Overwrite**: Validate new models
   ```bash
   python evaluation/comprehensive_evaluator.py \
       --model checkpoints/new_model.pt \
       --test-data data/organized/dataset.yaml
   ```

---

## Training Timeline Summary

| Phase | Duration (Sample) | Duration (Full) |
|-------|-------------------|-----------------|
| Data Download | 30 min | 4 hours |
| Week 1: SSL | 2 hours | 1 day |
| Week 1: Baseline | 3 hours | 2 days |
| Week 2: Active | 30 min | 4 hours |
| Week 3: Annotation | - | 1-3 days |
| Week 4: Specialization | 2 hours | 1.5 days |
| Week 5: Ensemble | 4 hours | 3 days |
| Week 5: Multi-Scale | 2 hours | 2 days |
| Week 6: TTA/Export | 1 hour | 3 hours |
| **Total** | **~15 hours** | **~2 weeks** |

---

## Next Steps

After completing training:

1. **Evaluate Models**: [Use evaluation suite](QUICKSTART.md#quick-evaluation)
2. **Deploy to Nano**: [See deployment guide](DEPLOYMENT_GUIDE.md#jetson-nano)
3. **Deploy to GCP**: [See deployment guide](DEPLOYMENT_GUIDE.md#google-cloud-platform)
4. **Monitor Production**: Setup monitoring and alerting

---

**See Also**:
- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) - Visual pipeline diagrams
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment instructions
