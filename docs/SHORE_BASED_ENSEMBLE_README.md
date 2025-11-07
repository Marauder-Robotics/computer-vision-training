# Shore Models Training (YOLOv11x Ensemble)

## Overview

The `train_shore_models.py` script trains the YOLOv11x ensemble for shoreside deployment on Google Cloud Platform. This is part of the dual ensemble architecture that combines:
- **3x YOLOv8x models** (from nano training - already trained)
- **3x YOLOv11x models** (trained by this script)

Total: **6 models** with weighted voting for maximum accuracy.

## Architecture

The shore model is optimized for cloud deployment with:
- **No energy constraints** (unlike nano model)
- **Maximum accuracy** (mAP50 target: 0.75-0.80)
- **Real-time processing** (10+ FPS on cloud GPUs)
- **Cost efficiency** on GCP infrastructure

## Model Variants

Three YOLOv11x variants are trained, each optimized for different use cases:

### 1. High Recall Variant
- **Confidence threshold:** 0.15 (low)
- **IoU threshold:** 0.4 (low)
- **Purpose:** Catch all potential detections, minimize false negatives
- **Use case:** Initial screening, critical species detection

### 2. Balanced Variant
- **Confidence threshold:** 0.25 (medium)
- **IoU threshold:** 0.5 (standard)
- **Purpose:** Balance precision and recall
- **Use case:** General purpose detection, counting

### 3. High Precision Variant
- **Confidence threshold:** 0.35 (high)
- **IoU threshold:** 0.6 (high)
- **Purpose:** Only confident detections, minimize false positives
- **Use case:** Species identification, reporting

## Installation

```bash
# Install dependencies
pip install ultralytics torch torchvision wandb pyyaml

# Verify CUDA is available (recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Train All Variants (Recommended)

```bash
python train_shore_models.py \
    --config config/training_config.yaml \
    --data config/dataset.yaml
```

### Train Single Variant

```bash
# Train only recall variant
python train_shore_models.py \
    --config config/training_config.yaml \
    --data config/dataset.yaml \
    --variant recall

# Train only balanced variant
python train_shore_models.py \
    --variant balanced

# Train only precision variant
python train_shore_models.py \
    --variant precision
```

### Resume Training from Checkpoint

```bash
python train_shore_models.py \
    --config config/training_config.yaml \
    --data config/dataset.yaml \
    --resume
```

## Configuration

Edit `config/training_config.yaml` to customize training:

```yaml
# Training parameters
epochs: 300
batch_size: 16  # Adjust based on GPU memory
imgsz: 640
device: 0  # GPU device number
workers: 8
patience: 50

# Output
output_dir: checkpoints/shore

# YOLOv8x models (from nano training)
yolov8_models:
  - checkpoints/ensemble/recall/best.pt
  - checkpoints/ensemble/balanced/best.pt
  - checkpoints/ensemble/precision/best.pt

# Ensemble configuration
voting_method: weighted  # weighted, majority, unanimous
weights: [0.15, 0.2, 0.15, 0.2, 0.2, 0.1]  # 6 models

# Critical species
critical_species_indices: [0, 1, 2, 3]

# Logging
logging:
  wandb:
    enabled: true
    project: marauder-cv-shore
    entity: your-entity  # Optional
```

## Output Structure

After training, the following structure is created:

```
checkpoints/shore/
├── recall/
│   ├── weights/
│   │   ├── best.pt          # Best model
│   │   └── last.pt          # Latest checkpoint
│   ├── export/              # ONNX, TorchScript exports
│   └── results.png          # Training curves
├── balanced/
│   └── ...
├── precision/
│   └── ...
├── ensemble_config.yaml     # Dual ensemble configuration
├── all_metrics.json         # All training metrics
└── training_summary.txt     # Human-readable summary
```

## Training Timeline

On NVIDIA A6000 GPU:
- **Per variant:** ~24-36 hours (300 epochs)
- **All three variants:** ~3-4 days total
- **Validation:** ~1 hour per variant
- **Export:** ~15 minutes per variant

## Expected Performance

Based on the project specifications:

| Metric | Target | YOLOv11x Expected |
|--------|--------|-------------------|
| mAP50 | 0.75-0.80 | 0.76-0.82 |
| mAP50-95 | 0.55-0.60 | 0.57-0.63 |
| Precision | 0.70+ | 0.72-0.78 |
| Recall | 0.70+ | 0.74-0.80 |
| Inference Speed | 10+ FPS | 12-15 FPS |

## Ensemble Inference

After training, use the ensemble for inference:

```python
from inference.shore_inference import ShoreEnsemble

# Initialize dual ensemble (6 models total)
ensemble = ShoreEnsemble(
    config_path='checkpoints/shore/ensemble_config.yaml'
)

# Run inference
results = ensemble.predict(
    source='path/to/video.mp4',
    save=True,
    conf=0.25
)
```

## GCP Deployment

### 1. Export Models

Models are automatically exported to ONNX and TorchScript during training.

### 2. Upload to GCP

```bash
# Upload to Cloud Storage
gsutil -m cp -r checkpoints/shore/ gs://your-bucket/models/

# Or use the included deployment script
python deployment/gcp_deploy.py \
    --model-path checkpoints/shore \
    --bucket your-bucket \
    --region us-central1
```

### 3. Deploy to Vertex AI

```bash
# Create endpoint
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name=marauder-shore-ensemble

# Deploy models
python deployment/vertex_ai_deploy.py \
    --endpoint-id YOUR_ENDPOINT_ID \
    --models checkpoints/shore/ensemble_config.yaml
```

### 4. Configure Auto-scaling

```yaml
# deployment/autoscaling_config.yaml
min_replicas: 1
max_replicas: 10
target_cpu_utilization: 70
target_throughput: 100  # requests per second
```

## Monitoring

### Weights & Biases

Training metrics are automatically logged to W&B:
- Training/validation loss curves
- mAP metrics per epoch
- Per-class performance
- Learning rate schedule
- GPU utilization

View dashboard: `https://wandb.ai/your-entity/marauder-cv-shore`

### Local Logs

```bash
# View training logs
tail -f logs/shore_training/shore_training_*.log

# View tensorboard
tensorboard --logdir checkpoints/shore/
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
python train_shore_models.py --config config/training_config.yaml

# Edit config/training_config.yaml:
batch_size: 8  # Default is 16
```

### Slow Training

```bash
# Enable mixed precision (FP16)
# Already enabled by default with amp: true

# Use multiple GPUs
device: 0,1,2,3  # Use 4 GPUs

# Reduce workers if CPU bottleneck
workers: 4  # Default is 8
```

### Model Not Converging

```bash
# Increase warmup epochs
warmup_epochs: 5.0  # Default is 3.0

# Adjust learning rate
lr0: 0.0005  # Default is 0.001
lrf: 0.005   # Default is 0.01

# Enable more augmentation
mosaic: 1.0
mixup: 0.15
```

### Resume Failed Training

```bash
# The script automatically saves checkpoints every 10 epochs
python train_shore_models.py \
    --config config/training_config.yaml \
    --data config/dataset.yaml \
    --resume
```

## Integration with Nano Models

The shore models work in conjunction with nano models:

1. **Nano (YOLOv8m)** on Jetson: 
   - Filters video underwater (energy-constrained)
   - Sends only important segments to shore
   - 5 FPS, ~14-18 Wh/day

2. **Shore (YOLOv11x + YOLOv8x)**:
   - Re-analyzes important segments with higher accuracy
   - No energy constraints
   - 10+ FPS, cloud-scale processing

3. **Dual Ensemble**:
   - Combines 6 models for maximum accuracy
   - Weighted voting (recall: 0.15, balanced: 0.2, precision: 0.15, etc.)
   - Critical species get priority thresholds

## Comparison: Nano vs Shore

| Feature | Nano (YOLOv8m) | Shore (YOLOv11x) |
|---------|----------------|-------------------|
| Model Size | Medium (25M params) | Extra Large (100M+ params) |
| Platform | Jetson Orin Nano | GCP GPU (T4/V100/A100) |
| Energy | 14-18 Wh/day | Unlimited |
| FPS | 5 | 10-15 |
| mAP50 | 0.65-0.70 | 0.76-0.82 |
| Latency | <200ms | <100ms |
| Purpose | Filter/screen | Verify/analyze |

## Next Steps

After training shore models:

1. ✅ **Test ensemble inference**
   ```bash
   python inference/shore_inference.py \
       --input test_video.mp4 \
       --config checkpoints/shore/ensemble_config.yaml
   ```

2. ✅ **Evaluate on test set**
   ```bash
   python evaluation/comprehensive_evaluator.py \
       --shore-models checkpoints/shore/ensemble_config.yaml \
       --test-data config/dataset.yaml
   ```

3. ✅ **Deploy to GCP**
   ```bash
   python deployment/gcp_deploy.py \
       --model-path checkpoints/shore \
       --bucket marauder-cv-models
   ```

4. ✅ **Set up monitoring**
   ```bash
   python deployment/setup_monitoring.py \
       --endpoint-id YOUR_ENDPOINT_ID
   ```

5. ✅ **Configure alerting**
   - Set up alerts for:
     - Critical species detections
     - Model degradation
     - API errors
     - Cost thresholds

## Support

For issues or questions:
- Check the logs: `logs/shore_training/`
- Review metrics: `checkpoints/shore/all_metrics.json`
- Read summary: `checkpoints/shore/training_summary.txt`
- Check W&B dashboard

## References

- **Project Document**: `Marauder_Nano_CV_Overview.docx`
- **Training Pipeline**: Weeks 1-6 as outlined in project doc
- **Ultralytics YOLOv11**: https://docs.ultralytics.com/models/yolo11/
- **GCP Vertex AI**: https://cloud.google.com/vertex-ai/docs

## License

Part of the Marauder CV Pipeline project.
