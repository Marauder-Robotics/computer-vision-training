# Training Pipeline Diagram

## Complete Training Flow (Week 1-6)

```mermaid
graph TB
    Start[Start Training Pipeline] --> Data[Data Acquisition]
    
    Data --> |280K images| Fathomnet[Fathomnet Download]
    Data --> |40K images| Deepfish[Deepfish Dataset]
    Data --> |850 images| Marauder[Marauder Dataset]
    
    Fathomnet --> Preprocess[Preprocessing & Organization]
    Deepfish --> Preprocess
    Marauder --> Preprocess
    
    Preprocess --> Split[Train/Val/Test Split<br/>70%/15%/15%]
    
    Split --> Week1{Week 1}
    
    Week1 --> SSL[SSL Pretraining<br/>MoCo V3<br/>50K+ images]
    Week1 --> Baseline[Baseline YOLO<br/>YOLOv8m/l<br/>10K+ labeled]
    
    SSL --> |Pretrained<br/>Backbone| Baseline
    
    Baseline --> Week2[Week 2: Active Learning]
    
    Week2 --> Inference1[Inference on<br/>Uncertain Images]
    Inference1 --> Score[Score & Select<br/>2000 images]
    Score --> Export[Export to<br/>Mindy Services]
    
    Export --> Week3[Week 3: Wait for<br/>Annotations]
    
    Week3 --> Import[Import Annotated<br/>Images]
    
    Import --> Week4{Week 4}
    
    Week4 --> Oversample[Critical Species<br/>Oversampling 3x]
    Week4 --> HardNeg[Hard Negative<br/>Mining 3 iterations]
    
    Oversample --> Specialized[Critical Species<br/>Specialized Model]
    HardNeg --> Specialized
    
    Specialized --> Week5{Week 5}
    
    Week5 --> Ens1[Train Recall<br/>Variant]
    Week5 --> Ens2[Train Balanced<br/>Variant]
    Week5 --> Ens3[Train Precision<br/>Variant]
    
    Ens1 --> MS1[Multi-Scale<br/>Training]
    Ens2 --> MS2[Multi-Scale<br/>Training]
    Ens3 --> MS3[Multi-Scale<br/>Training]
    
    MS1 --> Week6{Week 6}
    MS2 --> Week6
    MS3 --> Week6
    
    Week6 --> TTA[Test-Time<br/>Augmentation]
    TTA --> Calib[Confidence<br/>Calibration]
    Calib --> TRT[TensorRT<br/>Export FP16]
    
    TRT --> Nano[Nano Models<br/>Ready]
    
    Week5 --> YV11[Train YOLOv11x<br/>Variants]
    YV11 --> Shore[Shore Models<br/>Ready]
    
    Nano --> Deploy1[Deploy to<br/>Jetson Nano]
    Shore --> Deploy2[Deploy to<br/>GCP]
    
    Deploy1 --> End[Ready for<br/>Production]
    Deploy2 --> End
```

## Detailed Week-by-Week Flow

### Week 1: Foundation

```mermaid
graph LR
    subgraph "SSL Pretraining"
        A[Unlabeled Images<br/>50K+] --> B[Data Augmentation<br/>Flip, Rotate, Color]
        B --> C[MoCo V3<br/>Contrastive Learning]
        C --> D[Pretrained Backbone<br/>Underwater Features]
    end
    
    subgraph "Baseline Training"
        D --> E[YOLOv8m/l<br/>with SSL Backbone]
        F[Labeled Images<br/>10K+] --> E
        E --> G[Baseline Model<br/>mAP ~0.55]
    end
```

### Week 2: Active Learning

```mermaid
graph TB
    A[Baseline Model] --> B{Run Inference}
    C[Unlabeled Pool<br/>2000+ images] --> B
    
    B --> D[Calculate Uncertainty<br/>Confidence < 0.7]
    D --> E[Score Images<br/>Entropy, Margin]
    E --> F[Select Top 2000<br/>Most Uncertain]
    
    F --> G[Export COCO Format]
    G --> H[Add Thumbnails]
    H --> I[Create Annotation<br/>Package ZIP]
    
    I --> J[Send to<br/>Mindy Services]
```

### Week 3: Annotation Wait

```mermaid
graph LR
    A[Mindy Services] --> B[Human Annotators]
    B --> C[Quality Control]
    C --> D[COCO Annotations]
    D --> E[Convert to<br/>YOLO Format]
    E --> F[Import & Validate]
    F --> G[Ready for Week 4]
```

### Week 4: Specialization

```mermaid
graph TB
    A[Baseline Model] --> B{Critical Species?}
    
    B --> |Yes| C[Oversample 3x]
    B --> |No| D[Standard Sampling]
    
    C --> E[Specialized Training]
    D --> E
    
    E --> F[Specialized Model v1]
    
    F --> G[Run Inference<br/>Find False Positives]
    G --> H{Iteration < 3?}
    
    H --> |Yes| I[Hard Negative Mining]
    I --> J[Add to Training Set]
    J --> E
    
    H --> |No| K[Final Specialized<br/>Model]
```

### Week 5: Ensemble & Multi-Scale

```mermaid
graph TB
    subgraph "Ensemble Training"
        A[Specialized Model] --> B[Recall Variant<br/>Low threshold 0.15]
        A --> C[Balanced Variant<br/>Medium threshold 0.25]
        A --> D[Precision Variant<br/>High threshold 0.4]
        
        B --> E[Heavy Augmentation]
        C --> F[Medium Augmentation]
        D --> G[Light Augmentation]
    end
    
    subgraph "Multi-Scale Training"
        E --> H[Dynamic Resolution<br/>480-768px]
        F --> H
        G --> H
        
        H --> I[Recall MS]
        H --> J[Balanced MS]
        H --> K[Precision MS]
    end
    
    I --> L[3-Model Ensemble]
    J --> L
    K --> L
```

### Week 6: Optimization & Export

```mermaid
graph TB
    A[3-Model Ensemble] --> B{Test-Time Augmentation}
    
    B --> C[Horizontal Flip]
    B --> D[Rotations 0,90,180,270]
    B --> E[Scales 0.9,1.0,1.1]
    
    C --> F[Merge Predictions<br/>Weighted NMS]
    D --> F
    E --> F
    
    F --> G[Calibration Dataset<br/>1000 images]
    G --> H[Temperature Scaling<br/>Critical Species]
    
    H --> I[Calibrated Ensemble]
    
    I --> J[Export to ONNX]
    J --> K[Convert to TensorRT<br/>FP16]
    
    K --> L[Validate on Nano]
    L --> M{Accuracy OK?}
    
    M --> |Yes| N[Package for Deployment]
    M --> |No| O[Adjust & Re-export]
    O --> K
    
    N --> P[Nano Deployment<br/>Package]
```

## Training Data Distribution

### By Dataset

```mermaid
pie title Training Data Distribution
    "Fathomnet (Labeled)" : 280000
    "Deepfish (SSL Only)" : 40000
    "Marauder (Priority)" : 850
    "Active Learning" : 2000
```

### By Species Category

```mermaid
pie title Species Distribution
    "Critical (0-19)" : 55
    "Important (20-28)" : 25
    "General (29-35)" : 20
```

## Model Evolution Tracking

### Performance Over Weeks

```
Week 1 (Baseline):     mAP50 = 0.55, Recall = 0.50
Week 2 (Active):       mAP50 = 0.58, Recall = 0.53
Week 4 (Specialized):  mAP50 = 0.62, Recall = 0.58
Week 5 (Ensemble):     mAP50 = 0.68, Recall = 0.65
Week 6 (TTA):          mAP50 = 0.70, Recall = 0.68
```

### Energy Consumption Evolution

```
Baseline YOLOv8x:      24.3 Wh/day (too high)
After Multi-Scale:     21.5 Wh/day (still high)
YOLOv8l:              18.0 Wh/day (acceptable)
YOLOv8m:              14.4 Wh/day (optimal)
```

## Training Scripts Overview

### Execution Order

```bash
# Week 1: Foundation
./scripts/train_all.sh               # Master script (recommended)
# OR manually:
python training/1_ssl_pretrain.py
python training/2_baseline_yolo.py

# Week 2: Active Learning
python training/3_active_learning.py
# → Generates annotation package for Mindy Services

# Week 3: Wait for annotations
# → External process

# Week 4: Specialization
python training/4_critical_species.py

# Week 5: Ensemble
python training/5a_ensemble_training_nano.py
python training/6_multiscale_training.py

# Week 6: Optimization
python training/7_tta_calibration.py --models checkpoints/ensemble/*.pt
python training/8_tensorrt_export.py --models checkpoints/ensemble/*.pt --package
```

## Configuration Hierarchy

```mermaid
graph TB
    A[training_config.yaml] --> B[General Settings]
    A --> C[Week-Specific Settings]
    
    B --> D[device: cuda]
    B --> E[batch_size: 16]
    B --> F[num_workers: 4]
    
    C --> G[ssl_pretraining]
    C --> H[baseline_yolo]
    C --> I[active_learning]
    C --> J[critical_species]
    C --> K[ensemble_training]
    C --> L[multiscale_training]
    C --> M[tta_calibration]
    C --> N[tensorrt_export]
    
    G --> O[MoCo V3 params]
    H --> P[YOLO params]
    K --> Q[3 variant configs]
```

## Checkpoint Management

```mermaid
graph LR
    A[Training Start] --> B{Checkpoint Exists?}
    
    B --> |Yes| C[Resume from Checkpoint]
    B --> |No| D[Train from Scratch]
    
    C --> E[Training Loop]
    D --> E
    
    E --> F[Save Every 10 Epochs]
    F --> G[Keep Last 3]
    
    G --> H{Training Complete?}
    H --> |No| E
    H --> |Yes| I[Save Best Model]
```

## Parallel Training Strategy

### Multi-GPU Training (if available)

```mermaid
graph TB
    A[Data Batch] --> B[Split Across GPUs]
    
    B --> C[GPU 0<br/>Batch 0-3]
    B --> D[GPU 1<br/>Batch 4-7]
    B --> E[GPU 2<br/>Batch 8-11]
    B --> F[GPU 3<br/>Batch 12-15]
    
    C --> G[Gradients]
    D --> G
    E --> G
    F --> G
    
    G --> H[Aggregate & Update]
    H --> I[Synchronized Weights]
```

## Training Time Estimates

### With Paperspace A6000

| Week | Task | Time (Sample) | Time (Full) |
|------|------|---------------|-------------|
| 1 | SSL | 2 hours | 1 day |
| 1 | Baseline | 3 hours | 2 days |
| 2 | Active | 30 min | 4 hours |
| 3 | Annotation | - | 1-3 days |
| 4 | Specialization | 2 hours | 1.5 days |
| 5 | Ensemble | 4 hours | 3 days |
| 5 | Multi-Scale | 2 hours | 2 days |
| 6 | TTA/Export | 1 hour | 3 hours |
| **Total** | **~15 hours** | **~2 weeks** |

## Resource Requirements

### GPU Memory Usage

```
SSL Pretraining:    12-16 GB
Baseline Training:  10-14 GB
Ensemble Training:  12-16 GB per variant
Multi-Scale:        14-18 GB
TTA:                6-8 GB
```

### Storage Requirements

```
Raw Data:           500 GB
Preprocessed:       200 GB
Checkpoints:        50 GB
Logs:               5 GB
Total:              ~755 GB
```

---

**See Also**:
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training instructions
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
