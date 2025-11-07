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
    python training/1_ssl_pretrain.py --config "$CONFIG_FILE" 2>&1 | tee logs/1_ssl.log
    
    if [ $? -eq 0 ]; then
        echo "✓ SSL pretraining completed"
    else
        echo "✗ SSL pretraining failed"
        exit 1
    fi
else
    echo "Skipping SSL pretraining..."
fi

# ====== WEEK 1: BASELINE YOLO ======
echo ""
echo "=========================================="
echo "WEEK 1: Baseline YOLO Training"
echo "=========================================="
python training/2_baseline_yolo.py --config "$CONFIG_FILE" 2>&1 | tee logs/2_baseline.log

if [ $? -eq 0 ]; then
    echo "✓ Baseline YOLO training completed"
else
    echo "✗ Baseline YOLO training failed"
    exit 1
fi

# ====== WEEK 2: ACTIVE LEARNING ======
if [ "$SKIP_ACTIVE_LEARNING" != "true" ]; then
    echo ""
    echo "=========================================="
    echo "WEEK 2: Active Learning"
    echo "=========================================="
    python training/3_active_learning.py --config "$CONFIG_FILE" 2>&1 | tee logs/3_active_learning.log
    
    if [ $? -eq 0 ]; then
        echo "✓ Active learning completed"
        echo ""
        echo "=========================================="
        echo "ACTION REQUIRED: WEEK 3"
        echo "=========================================="
        echo "1. Send generated annotation package to Mindy Services"
        echo "2. Wait for annotations to be completed"
        echo "3. Import annotations back into the pipeline"
        echo "4. Continue with week 4 trianing"
        echo "=========================================="
        exit 0
    else
        echo "✗ Active learning failed"
        exit 1
    fi
else
    echo "Skipping active learning..."
fi

# ====== WEEK 4: CRITICAL SPECIES SPECIALIZATION ======
echo ""
echo "=========================================="
echo "WEEK 4: Critical Species Specialization"
echo "=========================================="
python training/4_critical_species.py --config "$CONFIG_FILE" 2>&1 | tee logs/4_critical_species.log

if [ $? -eq 0 ]; then
    echo "✓ Critical species specialization completed"
else
    echo "✗ Critical species specialization failed"
    exit 1
fi

# ====== WEEK 5: ENSEMBLE TRAINING ======
echo ""
echo "=========================================="
echo "WEEK 5: Ensemble Training (3 Variants)"
echo "=========================================="
python training/5a_ensemble_training.py --config "$CONFIG_FILE" --validate 2>&1 | tee logs/5a_ensemble.log

if [ $? -eq 0 ]; then
    echo "✓ Ensemble training completed"
else
    echo "✗ Ensemble training failed"
    exit 1
fi

# ====== WEEK 5: MULTI-SCALE TRAINING ======
echo ""
echo "=========================================="
echo "WEEK 5: Multi-Scale Training"
echo "=========================================="
python training/6_multiscale_training.py --config "$CONFIG_FILE" 2>&1 | tee logs/6_multiscale.log

if [ $? -eq 0 ]; then
    echo "✓ Multi-scale training completed"
else
    echo "✗ Multi-scale training failed"
    exit 1
fi

# ====== WEEK 6: TTA & CALIBRATION ======
echo ""
echo "=========================================="
echo "WEEK 6: TTA & Calibration"
echo "=========================================="

ENSEMBLE_MODELS=$(find checkpoints/ensemble -name "best.pt" -type f)
if [ -z "$ENSEMBLE_MODELS" ]; then
    echo "ERROR: No ensemble models found"
    exit 1
fi

python training/7_tta_calibration.py \
    --config "$CONFIG_FILE" \
    --models $ENSEMBLE_MODELS \
    2>&1 | tee logs/7_tta_calibration.log

if [ $? -eq 0 ]; then
    echo "✓ TTA and calibration completed"
else
    echo "✗ TTA and calibration failed"
    exit 1
fi

# ====== WEEK 6: TENSORRT EXPORT ======
echo ""
echo "=========================================="
echo "WEEK 6: TensorRT Export"
echo "=========================================="

MODEL_NAMES=("high_recall" "balanced" "high_precision")
MODEL_PATHS=()
for name in "${MODEL_NAMES[@]}"; do
    path="checkpoints/ensemble/$name/weights/best.pt"
    if [ -f "$path" ]; then
        MODEL_PATHS+=("$path")
    fi
done

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "ERROR: No models found for export"
    exit 1
fi

python training/8_tensorrt_export.py \
    --config "$CONFIG_FILE" \
    --models "${MODEL_PATHS[@]}" \
    --names "${MODEL_NAMES[@]}" \
    --validate \
    --package \
    2>&1 | tee logs/8_tensorrt_export.log

if [ $? -eq 0 ]; then
    echo "✓ TensorRT export completed"
else
    echo "✗ TensorRT export failed"
    exit 1
fi

# ====== FINAL EVALUATION ======
echo ""
echo "=========================================="
echo "FINAL EVALUATION"
echo "=========================================="

# Evaluate each ensemble variant
for model_path in "${MODEL_PATHS[@]}"; do
    model_name=$(basename $(dirname $(dirname "$model_path")))
    echo "Evaluating $model_name..."
    
    python evaluation/comprehensive_evaluator.py \
        --config "$CONFIG_FILE" \
        --model "$model_path" \
        --test-data config/dataset.yaml \
        --name "$model_name" \
        --profile-energy
done

# ====== COMPLETION SUMMARY ======
echo ""
echo "=========================================="
echo "TRAINING PIPELINE COMPLETE! ✓"
echo "=========================================="
echo ""
echo "Generated Artifacts:"
echo "  Checkpoints: checkpoints/"
echo "  Logs: logs/"
echo "  TensorRT Engines: checkpoints/tensorrt/nano_deployment/"
echo "  Evaluation Results: outputs/evaluation/"
echo ""
echo "Next Steps:"
echo "  1. Review evaluation results in outputs/evaluation/"
echo "  2. Deploy to Jetson Nano: cd deployment/nano && ./deploy_nano.sh"
echo "  3. Deploy to GCP: cd deployment/gcp && python deploy_gcp.py"
echo "  4. Run inference: python inference/nano_inference.py --input video.mp4"
echo ""
echo "=========================================="
