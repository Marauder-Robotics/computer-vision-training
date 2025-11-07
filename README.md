# Marauder CV Pipeline

**Complete Computer Vision System for Marine Species Detection, Tracking, and Counting**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.0+-00FF00.svg)](https://github.com/ultralytics/ultralytics)

---

## 🌊 Overview

The Marauder CV Pipeline is a production-ready computer vision system designed for real-time marine species detection and counting in underwater environments. The system deploys two model variants:

1. **Nano Model**: Energy-efficient ensemble (3x YOLOv8) running on Jetson Orin Nano 8GB for real-time filtering
2. **Shore Model**: High-accuracy dual ensemble (3x YOLOv8x + 3x YOLOv11x) running on Google Cloud Platform

### Key Features

- 🎯 **Species Detection**: Critical, Important, General
- 🔄 **Advanced Ensemble**: 3-variant models (recall, balanced, precision)
- 🚀 **ByteTrack Integration**: Accurate object tracking and counting
- ⚡ **Energy Optimized**: ~14.4-18 Wh/day on Jetson Nano
- 📊 **Test-Time Augmentation**: Improved accuracy through TTA
- 🎓 **Self-Supervised Learning**: MoCo V3 pretraining
- 🔧 **TensorRT Export**: FP16 optimization for edge deployment
- 📈 **Complete Evaluation**: mAP, counting accuracy, energy profiling

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Documentation](#documentation)
- [Species List](#species-list)

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repository-url>
cd marauder-cv-pipeline
pip install -r requirements.txt

# 2. Download Fathomnet data
python data/acquisition/fathomnet_downloader.py --config config/training_config.yaml

# 3. Train models (Week 1-6)
./scripts/train_all.sh

# 4. Export for deployment
python training/week6_tensorrt_export.py --models checkpoints/ensemble/*.pt

# 5. Run inference
python inference/nano_inference.py --input video.mp4 --output results.mp4
```

---

## 💻 System Requirements

### Training (Paperspace)
- **GPU**: A4000, A6000, or better
- **RAM**: 32GB+
- **Storage**: 500GB+ (for datasets)
- **OS**: Ubuntu 20.04+

### Deployment - Nano
- **Device**: Jetson Orin Nano 8GB
- **Power**: 40 Wh/day budget
- **Storage**: 64GB+ SD card
- **JetPack**: 5.0+

### Deployment - Shore
- **Platform**: Google Cloud Platform
- **Service**: Vertex AI or Compute Engine
- **GPU**: T4 or better
- **RAM**: 16GB+

---

## 📦 Installation

### Development Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Jetson Nano Setup

```bash
# Flash JetPack 5.0+
# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip
pip3 install -r requirements.txt

# Install TensorRT (included in JetPack)
# Deploy models
cd deployment/nano
./deploy_nano.sh
```

---

## 📁 Project Structure

```
marauder-cv-pipeline/
├── config/                          # Configuration files
│   ├── species_mapping.yaml         # 36 species with Fathomnet mapping
│   ├── training_config.yaml         # Complete training configuration
│   └── inference_config.yaml        # Inference settings
├── data/                            # Data acquisition and processing
│   ├── acquisition/
│   │   ├── fathomnet_downloader.py  # Automated Fathomnet API downloader
│   │   └── dataset_organizer.py     # Dataset organization
│   ├── preprocessing/
│   │   ├── hybrid_preprocessor.py   # CLAHE, dehazing, color correction
│   │   └── augmentation_pipeline.py # Training augmentations
│   └── active_learning/
│       ├── mindy_services_handler.py # Annotation export/import
│       └── active_learner.py         # Uncertainty sampling
├── training/                        # Complete training pipeline
│   ├── 1_ssl_pretrain.py        # MoCo V3 self-supervised learning
│   ├── 2_baseline_yolo.py       # Baseline YOLOv8 training
│   ├── 2_active_learning.py     # Active learning iteration
│   ├── 4_critical_species.py    # Critical species specialization
│   ├── 5a_ensemble_training_nano.py   # 3-variant ensemble for nano
│   ├── 5b_ensemble_training_shoreside.py   # 6-variant ensemble for gcp compute
│   ├── 6_multiscale_training.py # Multi-scale training
│   ├── 7_tta_calibration.py     # TTA and confidence calibration
│   └── 8_tensorrt_export.py     # TensorRT FP16 export
├── inference/                       # Inference pipelines
│   ├── nano_inference.py            # Jetson Nano pipeline
│   ├── shore_inference.py           # GCP dual ensemble pipeline
│   └── bytetrack_counter.py         # Object tracking and counting
├── evaluation/                      # Evaluation suite
│   ├── comprehensive_evaluator.py   # mAP, counting, energy
│   └── metrics_calculator.py        # Custom metrics
├── deployment/                      # Deployment scripts
│   ├── nano/                        # Jetson Nano deployment
│   │   ├── deploy_nano.sh          # Nano deployment script
│   │   └── setup_jetson.sh         # Initial setup
│   └── gcp/                         # Google Cloud deployment
│       ├── deploy_gcp.py           # GCP deployment
│       └── vertex_ai_setup.py      # Vertex AI configuration
├── utils/                           # Utilities
│   ├── logger.py                    # Logging setup
│   └── visualization.py             # Visualization tools
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # System architecture
│   ├── TRAINING_GUIDE.md            # Training guide
│   └── DEPLOYMENT_GUIDE.md          # Deployment guide
├── requirements.txt                 # Python dependencies
└── setup.py                         # Package setup
```

---

## 🎓 Training Pipeline

The training pipeline consists of 6 weeks of progressive model improvement:

### Week 1: Foundation
- **SSL Pretraining**: MoCo V3 on 50,000+ unlabeled underwater images
- **Baseline YOLO**: YOLOv8m/l trained on 10,000+ labeled Fathomnet images

### Week 2: Active Learning
- Inference on 2,000+ uncertain images
- Mindy Services annotation export
- Uncertainty-based sample selection

### Week 3: Annotation
- External annotation via Mindy Services
- COCO to YOLO format conversion

### Week 4: Specialization
- Critical species oversampling (3x)
- Hard negative mining (3 iterations)
- False positive optimization

### Week 5: Ensemble
- **Ensemble Training**: 3 variants (recall, balanced, precision)
- **Multi-Scale Training**: Dynamic resolution (480-768)
- **Weighted NMS**: Optimized box fusion

### Week 6: Optimization
- **TTA**: Test-time augmentation
- **Calibration**: Temperature scaling for critical species
- **TensorRT Export**: FP16 for Nano deployment

### Training Commands

```bash
# Week 1: SSL + Baseline
python training/1_ssl_pretrain.py --config config/training_config.yaml
python training/2_baseline_yolo.py --config config/training_config.yaml

# Week 2: Active Learning
python training/3_active_learning.py --config config/training_config.yaml

# Week 4: Critical Species
python training/4_critical_species.py --config config/training_config.yaml

# Week 5: Ensemble + Multi-Scale
python training/5a_ensemble_training_nano.py --config config/training_config.yaml
python training/6_multiscale_training.py --config config/training_config.yaml

# Week 6: TTA + Export
python training/7_tta_calibration.py --models checkpoints/ensemble/*.pt
python training/8_tensorrt_export.py --models checkpoints/ensemble/*.pt --package
```

---

## 🔮 Inference

### Nano Inference (Edge)

```bash
# Single video processing
python inference/nano_inference.py \
    --config config/training_config.yaml \
    --input video.mp4 \
    --output annotated_output.mp4

# Real-time from camera
python inference/nano_inference.py \
    --config config/training_config.yaml \
    --source 0  # Camera index
```

### Shore Inference (Cloud)

```bash
# High-accuracy dual ensemble
python inference/shore_inference.py \
    --config config/training_config.yaml \
    --input video.mp4 \
    --output results.json

# Batch processing
python inference/shore_inference.py \
    --config config/training_config.yaml \
    --input-dir /path/to/videos/ \
    --output-dir /path/to/results/
```

---

## 📊 Evaluation

### Model Evaluation

```bash
# Evaluate single model
python evaluation/comprehensive_evaluator.py \
    --model checkpoints/ensemble/balanced/best.pt \
    --test-data config/dataset.yaml \
    --name balanced_model

# Evaluate ensemble
python evaluation/comprehensive_evaluator.py \
    --models checkpoints/ensemble/*/best.pt \
    --test-data config/dataset.yaml \
    --name nano_ensemble

# Energy profiling
python evaluation/comprehensive_evaluator.py \
    --model checkpoints/ensemble/balanced/best.pt \
    --profile-energy \
    --name balanced_energy
```

### Metrics

- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: Mean Average Precision across IoU 0.5-0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **MAE**: Mean Absolute Error for counting
- **Energy**: Wh per day, J per inference

---

## 🚢 Deployment

### Jetson Nano Deployment

```bash
cd deployment/nano

# 1. Setup Jetson (one-time)
./setup_jetson.sh

# 2. Deploy models
./deploy_nano.sh

# 3. Run inference service
python nano_service.py --port 8080
```

### GCP Deployment

```bash
cd deployment/gcp

# 1. Setup GCP (one-time)
python vertex_ai_setup.py --project-id YOUR_PROJECT_ID

# 2. Deploy to Vertex AI
python deploy_gcp.py --model-path checkpoints/shore/

# 3. Test endpoint
python test_endpoint.py --endpoint ENDPOINT_URL
```

---

## 📚 Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System architecture and design
- **[TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**: Detailed training instructions
- **[DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)**: Deployment walkthrough

---

## ⚡ Energy Budget Estimate

**Target**: 40 Wh/day for CV inference

### Nano Configuration
- **Cameras**: 4x Barlus cameras
- **Capture**: 30 seconds, twice per hour
- **FPS**: 5 frames per second
- **Ensemble**: 3 models
- **Daily Inferences**: 129,600
- **Energy**: 14.4 Wh/day (YOLOv8m) - 18.0 Wh/day (YOLOv8l)

---

## 🎯 Performance Targets

| Metric | Nano (Edge) | Shore (Cloud) |
|--------|-------------|---------------|
| mAP50 | >0.65 | >0.75 |
| mAP50-95 | >0.45 | >0.55 |
| FPS | 5 | 10+ |
| Energy | <18 Wh/day | N/A |
| Latency | <200ms | <100ms |

---
