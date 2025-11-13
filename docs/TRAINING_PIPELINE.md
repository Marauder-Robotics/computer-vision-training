# Training Pipeline

## Overview

The Marauder CV training pipeline consists of 9 steps (including Step 0: Preprocessing) executed sequentially over 6 weeks. All steps support checkpoint resumption for Paperspace's 6-hour time limit.

### Enhanced Features

Key training improvements for production-ready models:

- **Step 1 (SSL)**: Multi-GPU DDP, underwater augmentation, mixed precision, EMA (+15-20% performance)
- **Step 3 (Active Learning)**: 6 uncertainty metrics, cluster-based diversity, full ensemble (+30% efficiency)
- **Step 4 (Critical Species)**: Focal loss, advanced hard negative mining, 3-stage training (+10-15% precision)

## Complete Pipeline Flow

```
DATA ACQUISITION → PREPROCESSING (Step 0) → TRAINING (Steps 1-8) → EXPORT (Step 9)
```

### Training Pipeline Stages

```
┌──────────────────────────────────────────────────────────────────┐
│ STEP 0: PREPROCESSING (REQUIRED)                                │
│ Input: Raw images from FathomNet, DeepFish, Marauder           │
│ Output: Preprocessed images with enhanced underwater features   │
│ Duration: 2-4 hours                                             │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ STEP 1: SSL PRETRAINING (Week 1)                               │
│ Input: 50K+ unlabeled underwater images                        │
│ Output: Pretrained backbone                                     │
│ Duration: 12-24 hours                                           │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ STEP 2: BASELINE YOLO (Week 1)                                 │
│ Input: 10K+ labeled images + SSL backbone                      │
│ Output: Baseline detection model                               │
│ Duration: 6-12 hours                                            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ STEP 3: ACTIVE LEARNING (Week 2)                               │
│ Input: Baseline model + unlabeled images                       │
│ Output: 2,000 uncertain images for annotation                  │
│ Duration: 2-4 hours                                             │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ STEP 4: ANNOTATION (Week 3)                                    │
│ Provider: Mindy Services                                        │
│ Input: 2,000 selected images                                   │
│ Output: Annotated YOLO labels                                   │
│ Duration: 3-5 days                                              │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│ STEP 5: CRITICAL SPECIES SPECIALIZATION (Week 4)               │
│ Input: Baseline + critical species images                      │
│ Output: Specialized critical species model                     │
│ Duration: 8-16 hours                                            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐   ┌─────────────────────┐
│ STEP 6a: NANO       │   │ STEP 6b: SHORE      │
│ ENSEMBLE (Week 5)   │   │ ENSEMBLE (Week 5)   │
│ 3x YOLOv8m/l        │   │ 3x YOLOv8x +        │
│ Duration: 12-18 hrs │   │ 3x YOLOv11x         │
│                     │   │ Duration: 16-24 hrs │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           └─────────┬───────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────────┐
│ STEP 7: MULTI-SCALE TRAINING (Week 6)                          │
│ Input: Ensemble models                                          │
│ Output: Multi-resolution models                                │
│ Duration: 8-12 hours                                            │
└────────────────────┬─────────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────────┐
│ STEP 8: TTA & CALIBRATION (Week 6)                             │
│ Input: Multi-scale models                                       │
│ Output: Calibrated models with TTA                             │
│ Duration: 4-6 hours                                             │
└────────────────────┬─────────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────────┐
│ STEP 9: TENSORRT EXPORT (Week 6)                               │
│ Input: Calibrated models                                        │
│ Output: TensorRT engines for Jetson Nano                       │
│ Duration: 2-4 hours                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete Pipeline Execution

### Option A: Full Pipeline (Automated)

```bash
# Run entire pipeline from start to finish
python training/train_all.py
```

### Option B: Individual Steps (Recommended)

Use the convenient wrapper scripts for each step:

```bash
# Step 0: Preprocessing
./scripts/run_preprocessing.sh --input /datasets/marauder-do-bucket/images/fathomnet \
                                --output /datasets/marauder-do-bucket/training/datasets/preprocessed

# Step 1: SSL Pretraining
./scripts/run_ssl_pretrain.sh --data-dir /datasets/marauder-do-bucket/training/datasets/preprocessed/deepfish

# Step 2: Baseline YOLO
./scripts/run_baseline_yolo.sh --data-yaml /datasets/marauder-do-bucket/training/datasets/data.yaml

# Step 3: Active Learning
./scripts/run_active_learning.sh --unlabeled-dir /datasets/marauder-do-bucket/images/marauder \
                                  --model /datasets/marauder-do-bucket/training/models/baseline/best.pt

# Step 4: Annotation (External - Mindy Services)

# Step 5: Critical Species
./scripts/run_critical_species.sh --data-dir /datasets/marauder-do-bucket/training/datasets/organized \
                                   --oversample --train

# Step 6a: Nano Ensemble
./scripts/run_ensemble_nano.sh --data-yaml /datasets/marauder-do-bucket/training/datasets/data.yaml

# Step 6b: Shore Ensemble
./scripts/run_ensemble_shore.sh --data-yaml /datasets/marauder-do-bucket/training/datasets/data.yaml

# Step 7: Multi-scale
./scripts/run_multiscale.sh --data-yaml /datasets/marauder-do-bucket/training/datasets/data.yaml

# Step 8: TTA & Calibration
./scripts/run_tta_calibration.sh --models /datasets/marauder-do-bucket/training/models/multiscale/*.pt

# Step 9: TensorRT Export (on Jetson Nano)
./scripts/run_tensorrt_export.sh --models /datasets/marauder-do-bucket/training/models/calibrated/*.pt
```

---

## Checkpoint & Resume Support

All training steps support checkpoint resumption for Paperspace's 6-hour limit:

```bash
# Training will automatically save checkpoints
./scripts/run_ssl_pretrain.sh --data-dir /path/to/data

# Resume from last checkpoint
./scripts/run_ssl_pretrain.sh --data-dir /path/to/data \
  --resume /datasets/marauder-do-bucket/training/checkpoints/ssl/checkpoint_latest.pt
```

---

## Timeline Summary

| Week | Steps | Duration | Focus |
|------|-------|----------|-------|
| Week 0 | Preprocessing | 2-4 hours | Data preparation |
| Week 1 | SSL + Baseline | 18-36 hours | Foundation models |
| Week 2 | Active Learning | 2-4 hours | Sample selection |
| Week 3 | Annotation | 3-5 days | External labeling |
| Week 4 | Critical Species | 8-16 hours | Specialization |
| Week 5 | Ensemble Training | 28-42 hours | Production models |
| Week 6 | Multi-scale + TTA + Export | 14-22 hours | Optimization |

**Total Training Time**: ~70-120 hours GPU time over 6 weeks

---

## Best Practices

1. **Start with Preprocessing**: Always run Step 0 preprocessing before training
2. **Use Checkpoints**: Enable checkpoint saving for all long-running steps
3. **Monitor GPU Usage**: Use `nvidia-smi` to track utilization
4. **Validate Continuously**: Run validation after each major step
5. **Keep Backups**: Copy important checkpoints to DO Spaces regularly
6. **Use Screen/Tmux**: Run long training sessions in persistent terminals
7. **Test Incrementally**: Validate models before proceeding to next step
8. **Document Changes**: Keep notes on hyperparameter modifications

---

## Troubleshooting

### Training Crashes After 6 Hours

**Problem**: Paperspace free tier has 6-hour limit  
**Solution**: All scripts auto-save checkpoints. Resume with `--resume` flag

### Out of Memory

**Problem**: GPU OOM during training  
**Solution**: Reduce batch size or enable gradient accumulation

### Poor Validation Results

**Problem**: Model not learning effectively  
**Solution**: 
- Check preprocessed images are correct
- Verify label quality and format
- Adjust learning rate or augmentation strength

### Slow Preprocessing

**Problem**: Preprocessing takes too long  
**Solution**: Use checkpoint resume and process in batches

---

For more details on individual steps, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
