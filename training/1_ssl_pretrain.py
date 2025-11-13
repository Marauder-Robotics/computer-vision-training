#!/usr/bin/env python3
"""
Week 1: SSL Pretraining - Enhanced with Advanced Features

Features:
- Multi-GPU/DDP support
- Underwater-specific augmentations (CLAHE, dehazing, color correction)
- Mixed precision training (AMP)
- Warmup + cosine annealing LR schedule
- Model EMA (Exponential Moving Average)
- Better checkpoint management
- Memory-efficient data loading
- Advanced monitoring and logging
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from utils.training_logger import TrainingLogger
from utils.checkpoint_manager import CheckpointManager


class UnderwaterAugmentation:
    """Underwater-specific augmentation techniques"""
    
    def __init__(self, apply_clahe: bool = True, apply_dehaze: bool = True):
        self.apply_clahe = apply_clahe
        self.apply_dehaze = apply_dehaze
    
    def clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def white_balance(self, img: np.ndarray) -> np.ndarray:
        """Simple white balance correction"""
        result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    
    def dehaze(self, img: np.ndarray, omega: float = 0.95) -> np.ndarray:
        """Simple dehazing using dark channel prior"""
        img_float = img.astype(np.float32) / 255.0
        
        # Dark channel
        min_channel = np.min(img_float, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(min_channel, kernel)
        
        # Atmospheric light
        flat_img = img_float.reshape(-1, 3)
        flat_dark = dark_channel.ravel()
        num_pixels = flat_dark.size
        num_brightest = int(max(num_pixels * 0.001, 1))
        indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
        atmospheric_light = np.max(flat_img[indices], axis=0)
        
        # Transmission
        transmission = 1 - omega * dark_channel
        transmission = np.maximum(transmission, 0.1)
        
        # Recover
        transmission = transmission[:, :, np.newaxis]
        recovered = (img_float - atmospheric_light) / transmission + atmospheric_light
        recovered = np.clip(recovered * 255, 0, 255).astype(np.uint8)
        
        return recovered
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply underwater augmentation pipeline"""
        img_np = np.array(img)
        
        # Randomly apply techniques
        if self.apply_clahe and np.random.rand() < 0.5:
            img_np = self.clahe(img_np)
        
        if self.apply_dehaze and np.random.rand() < 0.3:
            img_np = self.dehaze(img_np)
        
        if np.random.rand() < 0.4:
            img_np = self.white_balance(img_np)
        
        return Image.fromarray(img_np)


class MoCoV3Enhanced(nn.Module):
    """
    Enhanced MoCo V3 with improvements:
    - Better momentum update strategy
    - Larger projection head
    - Support for different backbone architectures
    """
    
    def __init__(
        self,
        base_encoder: nn.Module,
        dim: int = 256,
        mlp_dim: int = 4096,
        temperature: float = 0.2,
        momentum: float = 0.999,
        use_momentum_schedule: bool = True
    ):
        super().__init__()
        
        self.temperature = temperature
        self.base_momentum = momentum
        self.use_momentum_schedule = use_momentum_schedule
        self.momentum = momentum
        
        # Query encoder
        self.encoder_q = base_encoder
        
        # Momentum encoder
        self.encoder_k = self._copy_encoder(base_encoder)
        
        # Disable gradients for momentum encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            encoder_dim = base_encoder(dummy_input).shape[1]
        
        # Enhanced projection heads (3-layer MLP)
        self.projector_q = self._build_projector(encoder_dim, mlp_dim, dim, num_layers=3)
        self.projector_k = self._build_projector(encoder_dim, mlp_dim, dim, num_layers=3)
        
        # Copy projector params
        for param_q, param_k in zip(self.projector_q.parameters(),
                                     self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Predictor (2-layer MLP)
        self.predictor = self._build_predictor(dim, mlp_dim, dim)
    
    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        """Create a copy of the encoder"""
        import copy
        encoder_copy = copy.deepcopy(encoder)
        return encoder_copy
    
    def _build_projector(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int,
        num_layers: int = 3
    ) -> nn.Module:
        """Build enhanced MLP projector"""
        layers = []
        
        # First layer
        layers.extend([
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        ])
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        # Output layer
        layers.extend([
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        ])
        
        return nn.Sequential(*layers)
    
    def _build_predictor(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        """Build MLP predictor"""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def update_momentum(self, epoch: int, max_epochs: int):
        """Update momentum value with cosine schedule"""
        if self.use_momentum_schedule:
            # Cosine schedule: gradually increase momentum
            self.momentum = 1 - (1 - self.base_momentum) * (np.cos(np.pi * epoch / max_epochs) + 1) / 2
    
    @torch.no_grad()
    def _update_momentum_encoder(self):
        """Update momentum encoder with EMA of query encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        for param_q, param_k in zip(self.projector_q.parameters(),
                                     self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss with improved numerical stability
        """
        # Normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # Gather all keys from all GPUs (if using DDP)
        if dist.is_initialized():
            k = concat_all_gather(k)
        
        # Positive logits: [N, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: [N, K]
        l_neg = torch.einsum('nc,ck->nk', [q, k.T])
        
        # Logits: [N, K+1]
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels: positives are the 0th column
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        """
        Forward pass with optional momentum update
        
        Args:
            x1: First augmented view
            x2: Second augmented view
            epoch: Current epoch (for momentum scheduling)
            
        Returns:
            Contrastive loss
        """
        # Query features
        q1 = self.encoder_q(x1)
        q1 = self.projector_q(q1)
        q1 = self.predictor(q1)
        
        q2 = self.encoder_q(x2)
        q2 = self.projector_q(q2)
        q2 = self.predictor(q2)
        
        # Key features (no gradient)
        with torch.no_grad():
            self._update_momentum_encoder()
            
            k1 = self.encoder_k(x1)
            k1 = self.projector_k(k1)
            
            k2 = self.encoder_k(x2)
            k2 = self.projector_k(k2)
        
        # Compute loss (symmetric)
        loss = (self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)) / 2
        
        return loss


class UnderwaterSSLDatasetV2(Dataset):
    """Enhanced dataset with underwater augmentation and better error handling"""
    
    def __init__(
        self, 
        image_paths: List[str], 
        transform=None,
        underwater_aug: Optional[UnderwaterAugmentation] = None,
        cache_size: int = 0
    ):
        self.image_paths = image_paths
        self.transform = transform
        self.underwater_aug = underwater_aug
        
        # Simple cache for frequently accessed images
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        
        # Check cache
        if img_path in self.cache:
            img = self.cache[img_path].copy()
        else:
            try:
                img = Image.open(img_path).convert('RGB')
                
                # Update cache if enabled
                if self.cache_size > 0:
                    self.access_count[img_path] = self.access_count.get(img_path, 0) + 1
                    if len(self.cache) < self.cache_size:
                        self.cache[img_path] = img.copy()
                    elif self.access_count[img_path] > 5:  # Popular image
                        # Evict least accessed image
                        least_accessed = min(self.cache.keys(), 
                                           key=lambda k: self.access_count.get(k, 0))
                        del self.cache[least_accessed]
                        self.cache[img_path] = img.copy()
                        
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Return black image as fallback
                img = Image.new('RGB', (640, 640))
        
        # Apply underwater augmentation
        if self.underwater_aug:
            img = self.underwater_aug(img)
        
        # Apply standard augmentation
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1 = img2 = transforms.ToTensor()(img)
        
        return img1, img2


class ModelEMA:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


class SSLTrainerV2:
    """Enhanced SSL Pretraining Trainer with advanced features"""
    
    def __init__(self, config_path: str, local_rank: int = -1):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ssl_config = self.config.get('ssl_pretraining', {})
        self.local_rank = local_rank
        
        # Setup distributed training
        self.distributed = local_rank != -1
        if self.distributed:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        
        self.device = torch.device(f'cuda:{local_rank}' if local_rank != -1 else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # DO Bucket path
        self.do_bucket_path = os.environ.get('DO_BUCKET_PATH', '/datasets/marauder-do-bucket')
        
        # Training parameters
        self.epochs = self.ssl_config.get('epochs', 100)
        self.batch_size = self.ssl_config.get('batch_size', 256) // self.world_size
        self.learning_rate = self.ssl_config.get('learning_rate', 0.03) * self.world_size  # Linear scaling
        self.weight_decay = self.ssl_config.get('weight_decay', 0.0001)
        self.momentum = self.ssl_config.get('momentum', 0.9)
        self.warmup_epochs = self.ssl_config.get('warmup_epochs', 10)
        
        # Advanced features
        self.use_amp = self.ssl_config.get('use_amp', True)
        self.use_ema = self.ssl_config.get('use_ema', True)
        self.ema_decay = self.ssl_config.get('ema_decay', 0.999)
        
        # Checkpoint settings
        self.save_frequency = self.ssl_config.get('checkpoint', {}).get('save_frequency', 10)
        self.checkpoint_dir = Path(f'{self.do_bucket_path}/training/checkpoints/ssl')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_augmentation_pipeline(self) -> transforms.Compose:
        """Get enhanced augmentation pipeline for SSL"""
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),  # Useful for underwater
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(15)
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_lr_scheduler(self, optimizer, warmup_steps: int, total_steps: int):
        """Get learning rate scheduler with warmup + cosine annealing"""
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def collect_image_paths(self, data_dir: str) -> List[str]:
        """Collect all image paths for SSL training"""
        image_paths = []
        data_path = Path(data_dir)
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend([str(p) for p in data_path.rglob(ext)])
        
        if self.rank == 0:
            print(f"Found {len(image_paths)} images for SSL training")
        
        return image_paths
    
    def train(
        self,
        data_dir: str,
        output_dir: str,
        resume_from: str = None
    ):
        """
        Train SSL model with enhanced features and checkpoint support
        """
        # Initialize TrainingLogger (only on rank 0)
        if self.rank == 0:
            self.logger = TrainingLogger(
                project_name="marauder-cv",
                run_name="1-ssl-moco-v3-enhanced",
                save_dir=f"{self.do_bucket_path}/training/logs",
                config=self.ssl_config
            )
            
            # Initialize CheckpointManager
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=str(self.checkpoint_dir),
                project_name="ssl_pretrain",
                keep_last_n=5
            )
        
        # Collect images
        image_paths = self.collect_image_paths(data_dir)
        
        # Create dataset and dataloader
        transform = self.get_augmentation_pipeline()
        underwater_aug = UnderwaterAugmentation(apply_clahe=True, apply_dehaze=True)
        
        dataset = UnderwaterSSLDatasetV2(
            image_paths, 
            transform=transform,
            underwater_aug=underwater_aug,
            cache_size=1000  # Cache 1000 most accessed images
        )
        
        # Distributed sampler
        sampler = DistributedSampler(dataset) if self.distributed else None
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        
        # Create base encoder (ResNet50)
        from torchvision.models import resnet50
        base_encoder = resnet50(pretrained=False)
        base_encoder.fc = nn.Identity()  # Remove classification head
        
        # Create MoCo model
        model = MoCoV3Enhanced(
            base_encoder=base_encoder,
            dim=self.ssl_config.get('projection_dim', 256),
            temperature=self.ssl_config.get('temperature', 0.2),
            momentum=self.ssl_config.get('momentum', 0.999),
            use_momentum_schedule=True
        ).to(self.device)
        
        # Wrap with DDP if distributed
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        
        # Model EMA
        ema = ModelEMA(model_without_ddp, decay=self.ema_decay) if self.use_ema else None
        
        # Optimizer (LARS would be better but using SGD for simplicity)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = self.epochs * len(dataloader)
        warmup_steps = self.warmup_epochs * len(dataloader)
        scheduler = self.get_lr_scheduler(optimizer, warmup_steps, total_steps)
        
        # Mixed precision scaler
        scaler = GradScaler() if self.use_amp else None
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from and Path(resume_from).exists():
            if self.rank == 0:
                print(f"Resuming from checkpoint: {resume_from}")
            
            checkpoint_data = torch.load(resume_from, map_location=self.device)
            
            if checkpoint_data:
                model_without_ddp.load_state_dict(checkpoint_data['model_state_dict'])
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint_data:
                    scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                
                if scaler and 'scaler_state_dict' in checkpoint_data:
                    scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
                
                if ema and 'ema_state_dict' in checkpoint_data:
                    ema.shadow = checkpoint_data['ema_state_dict']
                
                start_epoch = checkpoint_data['epoch'] + 1
                
                if self.rank == 0:
                    print(f"Resumed from epoch {start_epoch}")
        
        # Training loop
        model.train()
        best_loss = float('inf')
        
        for epoch in range(start_epoch, self.epochs):
            if self.distributed:
                sampler.set_epoch(epoch)
            
            # Update momentum schedule
            model_without_ddp.update_momentum(epoch, self.epochs)
            
            epoch_loss = 0
            
            if self.rank == 0:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            else:
                pbar = dataloader
            
            for batch_idx, (x1, x2) in enumerate(pbar):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        loss = model(x1, x2, epoch=epoch)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model(x1, x2, epoch=epoch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Update EMA
                if ema:
                    ema.update()
                
                # Update learning rate
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                
                if self.rank == 0:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'lr': optimizer.param_groups[0]['lr'],
                        'momentum': model_without_ddp.momentum
                    })
                    
                    # Log metrics
                    step = epoch * len(dataloader) + batch_idx
                    self.logger.log({
                        'batch_loss': loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'momentum': model_without_ddp.momentum
                    }, step=step)
            
            # Epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            
            if self.rank == 0:
                print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
                
                self.logger.log({
                    'epoch': epoch + 1,
                    'epoch_loss': avg_loss
                })
                
                # Save checkpoint
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                
                if (epoch + 1) % self.save_frequency == 0 or is_best:
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': model_without_ddp.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'metrics': {'loss': avg_loss, 'best_loss': best_loss}
                    }
                    
                    if scaler:
                        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                    
                    if ema:
                        checkpoint_data['ema_state_dict'] = ema.shadow
                    
                    checkpoint_manager.save_checkpoint(
                        epoch=epoch,
                        model=model_without_ddp,
                        optimizer=optimizer,
                        metrics={'loss': avg_loss, 'best_loss': best_loss},
                        is_best=is_best,
                        additional_info=checkpoint_data
                    )
                    print(f"Saved checkpoint at epoch {epoch+1}")
        
        # Save final model (use EMA if available)
        if self.rank == 0:
            final_path = Path(output_dir) / 'ssl_backbone_final.pt'
            final_path.parent.mkdir(parents=True, exist_ok=True)
            
            if ema:
                ema.apply_shadow()
                torch.save(model_without_ddp.encoder_q.state_dict(), final_path)
                ema.restore()
            else:
                torch.save(model_without_ddp.encoder_q.state_dict(), final_path)
            
            print(f"Saved final SSL backbone: {final_path}")
            self.logger.finish()
        
        # Cleanup distributed
        if self.distributed:
            dist.destroy_process_group()
        
        return model


# Utility function for DDP
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    if not dist.is_initialized():
        return tensor
    
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather, dim=0)
    return output


def main():
    parser = argparse.ArgumentParser(description='SSL Pretraining with Enhanced Features')
    parser.add_argument('--config', type=str, default='config/training_config.yaml', 
                       help='Training config path')
    parser.add_argument('--data-dir', type=str, required=True, 
                       help='Directory with unlabeled images')
    parser.add_argument('--output', type=str, 
                       default='/datasets/marauder-do-bucket/training/models/ssl',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Resume from checkpoint')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    trainer = SSLTrainerV2(args.config, local_rank=args.local_rank)
    trainer.train(args.data_dir, args.output, resume_from=args.resume)


if __name__ == '__main__':
    main()
