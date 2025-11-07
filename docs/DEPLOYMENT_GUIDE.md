# Deployment Guide

Complete guide for deploying the Marauder CV Pipeline to production environments.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Jetson Nano Deployment](#jetson-nano-deployment)
3. [Google Cloud Platform Deployment](#google-cloud-platform-deployment)
4. [Production Monitoring](#production-monitoring)
5. [Maintenance](#maintenance)
6. [Troubleshooting](#troubleshooting)

---

## Deployment Overview

### Deployment Architecture

```
┌─────────────────────────────────────────┐
│         Buoy (Edge Device)              │
│  ┌───────────────────────────────────┐  │
│  │   Jetson Orin Nano 8GB            │  │
│  │   - TensorRT Models (FP16)        │  │
│  │   - Energy: 14.4-18 Wh/day        │  │
│  │   - FPS: 5 (4 cameras)            │  │
│  └───────────────────────────────────┘  │
│              │                           │
│         Filtered Data                    │
└──────────────┼──────────────────────────┘
               │
         WiFi/Bluetooth/Iridium
               │
               ▼
┌─────────────────────────────────────────┐
│      Shore (Cloud Infrastructure)       │
│  ┌───────────────────────────────────┐  │
│  │   GCP Vertex AI                   │  │
│  │   - Dual Ensemble (6 models)      │  │
│  │   - High Accuracy                 │  │
│  │   - FPS: 10+                      │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Prerequisites Checklist

- [ ] Trained and exported models
- [ ] TensorRT engines for Nano
- [ ] Cloud storage configured (S3/Spaces)
- [ ] GCP account setup (for shore)
- [ ] Monitoring tools ready
- [ ] Network connectivity tested

---

## Jetson Nano Deployment

### Hardware Setup

#### Required Components
- Jetson Orin Nano 8GB Developer Kit
- 64GB+ microSD card (or NVMe SSD)
- 5V 4A power supply (or PoE)
- 4x Barlus cameras
- Network adapter (WiFi/Ethernet)

#### Recommended Accessories
- Heatsink with fan
- Real-time clock (RTC) module
- UPS/battery backup

### Software Installation

#### Step 1: Flash JetPack

```bash
# Download JetPack 5.0+ from NVIDIA
# Use NVIDIA SDK Manager or SD card image

# Verify installation
cat /etc/nv_tegra_release
# Expected: R35 (release), REVISION: 1.0+
```

#### Step 2: Initial Setup

```bash
# Run setup script
cd deployment/nano
chmod +x setup_jetson.sh
./setup_jetson.sh

# This will:
# - Update system packages
# - Install dependencies
# - Set power mode to MAXN (15W)
# - Configure jetson_clocks
```

#### Step 3: Transfer Models

**Option A: Direct Copy**
```bash
# From training machine
scp -r checkpoints/tensorrt/nano_deployment jetson@JETSON_IP:/tmp/

# On Jetson
ssh jetson@JETSON_IP
```

**Option B: Download from Cloud**
```bash
# On Jetson
wget https://your-storage/nano_deployment.tar.gz
tar -xzf nano_deployment.tar.gz
```

#### Step 4: Deploy

```bash
# On Jetson
cd deployment/nano
chmod +x deploy_nano.sh
sudo ./deploy_nano.sh

# This will:
# - Create /opt/marauder-cv directory
# - Copy TensorRT engines
# - Copy configuration files
# - Install Python dependencies
# - Create systemd service
# - Enable and start service
```

#### Step 5: Verify Deployment

```bash
# Check service status
sudo systemctl status marauder-cv

# View logs
sudo journalctl -u marauder-cv -f

# Test inference
python3 /opt/marauder-cv/nano_inference.py --input test.mp4
```

### Configuration

#### Deployment Configuration

Edit `/opt/marauder-cv/deployment_config.yaml`:

```yaml
models:
  high_recall: engines/nano_ensemble_high_recall.engine
  balanced: engines/nano_ensemble_balanced.engine
  high_precision: engines/nano_ensemble_high_precision.engine

inference:
  conf_threshold: 0.25
  iou_threshold: 0.5
  max_det: 300

energy_budget:
  max_daily_wh: 40
  target_fps: 5
  cameras: 4
  capture_duration: 30
  captures_per_hour: 2

cameras:
  camera_0:
    source: "/dev/video0"
    resolution: [1920, 1080]
    enabled: true
  camera_1:
    source: "/dev/video1"
    resolution: [1920, 1080]
    enabled: true
  camera_2:
    source: "/dev/video2"
    resolution: [1920, 1080]
    enabled: true
  camera_3:
    source: "/dev/video3"
    resolution: [1920, 1080]
    enabled: true

network:
  primary: "wifi"  # wifi, ethernet, bluetooth
  fallback: "bluetooth"
  upload_interval: 3600  # seconds

storage:
  local_cache: "/opt/marauder-cv/cache"
  max_cache_gb: 10
  retention_days: 7

alerts:
  critical_species: true
  email_enabled: false
  webhook_url: null
```

### Power Management

#### Power Mode Configuration

```bash
# Check current power mode
sudo nvpmodel -q

# Set to MAXN (15W) - maximum performance
sudo nvpmodel -m 0

# Set to 10W mode (power saving)
sudo nvpmodel -m 1

# Lock clocks for consistent performance
sudo jetson_clocks

# Check power consumption
sudo tegrastats
```

#### Energy Budget Monitoring

```bash
# Monitor real-time power
python3 << 'EOF'
import subprocess
import time

while True:
    result = subprocess.run(['tegrastats'], capture_output=True, text=True)
    # Parse and log power consumption
    time.sleep(1)
EOF
```

### Camera Setup

#### Test Cameras

```bash
# List video devices
ls /dev/video*

# Test camera 0
gst-launch-1.0 v4l2src device=/dev/video0 ! video/x-raw,width=1920,height=1080 ! xvimagesink

# Capture test frame
gst-launch-1.0 v4l2src device=/dev/video0 num-buffers=1 ! jpegenc ! filesink location=test.jpg
```

#### Camera Configuration

Edit `/opt/marauder-cv/camera_config.yaml`:

```yaml
cameras:
  - id: 0
    name: "forward"
    device: "/dev/video0"
    resolution: [1920, 1080]
    fps: 30
    format: "MJPEG"
    
  - id: 1
    name: "starboard"
    device: "/dev/video1"
    resolution: [1920, 1080]
    fps: 30
    format: "MJPEG"
```

### Network Configuration

#### WiFi Setup

```bash
# Configure WiFi
sudo nmcli device wifi connect SSID password PASSWORD

# Set static IP (optional)
sudo nmcli connection modify CONNECTION_NAME ipv4.addresses 192.168.1.100/24
sudo nmcli connection modify CONNECTION_NAME ipv4.gateway 192.168.1.1
sudo nmcli connection modify CONNECTION_NAME ipv4.dns "8.8.8.8 8.8.4.4"
```

#### Bluetooth Setup

```bash
# Enable Bluetooth
sudo systemctl start bluetooth
sudo systemctl enable bluetooth

# Pair device
bluetoothctl
> scan on
> pair XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX
> connect XX:XX:XX:XX:XX:XX
```

### Service Management

#### Systemd Service

Service file at `/etc/systemd/system/marauder-cv.service`:

```ini
[Unit]
Description=Marauder CV Inference Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/opt/marauder-cv
ExecStart=/usr/bin/python3 /opt/marauder-cv/nano_inference.py --source 0
Restart=on-failure
RestartSec=10
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
```

#### Service Commands

```bash
# Start service
sudo systemctl start marauder-cv

# Stop service
sudo systemctl stop marauder-cv

# Restart service
sudo systemctl restart marauder-cv

# Enable on boot
sudo systemctl enable marauder-cv

# Disable on boot
sudo systemctl disable marauder-cv

# View status
sudo systemctl status marauder-cv

# View logs
sudo journalctl -u marauder-cv -f

# View last 100 lines
sudo journalctl -u marauder-cv -n 100
```

---

## Google Cloud Platform Deployment

### Prerequisites

#### GCP Account Setup

```bash
# Install gcloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable compute.googleapis.com
```

#### Authentication

```bash
# Create service account
gcloud iam service-accounts create marauder-cv \
    --display-name="Marauder CV Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:marauder-cv@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create key
gcloud iam service-accounts keys create key.json \
    --iam-account=marauder-cv@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

### Model Upload

#### Prepare Model Package

```bash
# Create model directory structure
mkdir -p gcp_deployment/models
cp checkpoints/ensemble/*/best.pt gcp_deployment/models/
cp checkpoints/shore/yolov11x/*/best.pt gcp_deployment/models/

# Create requirements file
cat > gcp_deployment/requirements.txt << EOF
torch==2.0.0
ultralytics==8.0.196
opencv-python==4.8.0
numpy==1.24.0
EOF

# Create predictor script
# (This would be a custom prediction script for Vertex AI)
```

#### Upload to Cloud Storage

```bash
# Create GCS bucket
gsutil mb -l us-central1 gs://marauder-cv-models/

# Upload models
gsutil -m cp -r gcp_deployment/* gs://marauder-cv-models/

# Verify upload
gsutil ls gs://marauder-cv-models/
```

### Vertex AI Deployment

#### Deploy Model

```bash
# Use deployment script
python deployment/gcp/deploy_gcp.py \
    --project-id YOUR_PROJECT_ID \
    --region us-central1 \
    --model-path gs://marauder-cv-models/ \
    --endpoint-name marauder-cv-shore

# Expected output:
# ✓ Model uploaded: projects/.../models/...
# ✓ Endpoint created: projects/.../endpoints/...
# ✓ Model deployed to endpoint
```

#### Manual Deployment (Alternative)

```python
from google.cloud import aiplatform

aiplatform.init(project='YOUR_PROJECT_ID', location='us-central1')

# Upload model
model = aiplatform.Model.upload(
    display_name='marauder-cv-shore',
    artifact_uri='gs://marauder-cv-models/',
    serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-13:latest'
)

# Create endpoint
endpoint = aiplatform.Endpoint.create(display_name='marauder-cv-shore')

# Deploy
model.deploy(
    endpoint=endpoint,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=3
)
```

### Testing Deployment

#### Test Endpoint

```python
from google.cloud import aiplatform
import base64

endpoint = aiplatform.Endpoint('projects/.../endpoints/...')

# Prepare test image
with open('test_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make prediction
response = endpoint.predict(instances=[{'image': image_data}])

print(response.predictions)
```

### Auto-scaling Configuration

```python
# Configure auto-scaling
endpoint.update(
    min_replica_count=1,
    max_replica_count=10,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4'
)
```

---

## Production Monitoring

### Metrics to Monitor

#### Nano Metrics
- Inference latency (ms)
- Energy consumption (W)
- CPU/GPU utilization (%)
- Memory usage (GB)
- Storage usage (GB)
- Network bandwidth (MB/s)
- Detection counts per species
- False positive rate
- Camera health

#### Shore Metrics
- Request latency (ms)
- Throughput (requests/s)
- Model accuracy (mAP)
- Endpoint availability (%)
- Cost per prediction ($)
- Error rate (%)

### Logging Setup

#### Nano Logging

```python
# /opt/marauder-cv/logging_config.yaml
version: 1
formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /opt/marauder-cv/logs/marauder-cv.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: detailed
  console:
    class: logging.StreamHandler
    formatter: detailed
loggers:
  marauder_cv:
    level: INFO
    handlers: [file, console]
```

#### Cloud Logging

```bash
# Enable Cloud Logging
gcloud logging write marauder-cv-logs "Deployment started" \
    --severity=INFO

# View logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Endpoint" \
    --limit=50
```

### Alerting

#### Setup Alerts

```bash
# Create alert for high error rate
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="High Error Rate" \
    --condition-display-name="Error Rate > 5%" \
    --condition-threshold-value=0.05
```

---

## Maintenance

### Regular Tasks

#### Daily
- Check service status
- Review logs for errors
- Monitor energy consumption
- Verify detection counts

#### Weekly
- Clear old logs and cache
- Review performance metrics
- Check model accuracy drift
- Backup critical data

#### Monthly
- Update system packages
- Review and optimize configuration
- Analyze long-term trends
- Plan model updates

### Update Procedures

#### Model Updates

```bash
# On training machine
python training/8_tensorrt_export.py --models checkpoints/new_model.pt

# Transfer to Nano
scp checkpoints/tensorrt/new_model.engine jetson@JETSON_IP:/tmp/

# On Nano
sudo systemctl stop marauder-cv
sudo cp /tmp/new_model.engine /opt/marauder-cv/models/
sudo systemctl start marauder-cv
```

#### Software Updates

```bash
# On Nano
sudo apt-get update
sudo apt-get upgrade

# Update Python packages
pip3 install --upgrade -r /opt/marauder-cv/requirements.txt

# Restart service
sudo systemctl restart marauder-cv
```

---

## Troubleshooting

### Nano Issues

#### Service Won't Start

```bash
# Check logs
sudo journalctl -u marauder-cv -n 100

# Common issues:
# 1. CUDA not available
nvidia-smi  # Should show GPU

# 2. Missing models
ls /opt/marauder-cv/models/

# 3. Permission issues
sudo chown -R jetson:jetson /opt/marauder-cv/
```

#### High Power Consumption

```bash
# Check current consumption
sudo tegrastats

# Reduce power mode
sudo nvpmodel -m 1  # 10W mode

# Disable unused features
sudo systemctl disable bluetooth
```

#### Camera Not Detected

```bash
# Check camera connection
ls /dev/video*

# Test camera
v4l2-ctl --list-devices

# Check camera permissions
sudo usermod -a -G video jetson
```

### GCP Issues

#### Deployment Failed

```bash
# Check deployment status
gcloud ai models list
gcloud ai endpoints list

# View detailed logs
gcloud logging read "resource.type=aiplatform.googleapis.com/Model" \
    --limit=100
```

#### High Latency

```bash
# Check endpoint metrics
gcloud ai endpoints describe ENDPOINT_ID

# Increase replicas
# Update auto-scaling configuration
```

---

**See Also**:
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training instructions
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
