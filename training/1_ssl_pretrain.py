#!/usr/bin/env python3
"""
Week 1: SSL Pretraining with MoCo V3

Self-supervised learning pretraining on 50,000+ unlabeled underwater images.
Uses MoCo V3 (Momentum Contrast v3) to learn underwater feature representations.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import wandb
from tqdm import tqdm


class MoCoV3(nn.Module):
    """
    MoCo V3 implementation for self-supervised learning
    
    Based on: "An Empirical Study of Training Self-Supervised Vision Transformers"
    https://arxiv.org/abs/2104.02057
    """
    
    def __init__(
        self,
        base_encoder: nn.Module,
        dim: int = 256,
        mlp_dim: int = 4096,
        temperature: float = 0.2,
        momentum: float = 0.999
    ):
        super().__init__()
        
        self.temperature = temperature
        self.momentum = momentum
        
        # Query encoder
        self.encoder_q = base_encoder
        
        # Momentum encoder
        self.encoder_k = base_encoder
        
        # Copy query params to key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                     self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Projection heads
        self.projector_q = self._build_projector(2048, mlp_dim, dim)
        self.projector_k = self._build_projector(2048, mlp_dim, dim)
        
        # Copy projector params
        for param_q, param_k in zip(self.projector_q.parameters(),
                                     self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Predictor
        self.predictor = self._build_predictor(dim, mlp_dim, dim)
    
    def _build_projector(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        """Build MLP projector"""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        )
    
    def _build_predictor(self, in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
        """Build MLP predictor"""
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    
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
        Compute contrastive loss
        
        Args:
            q: Query features [N, D]
            k: Key features [N, D]
            
        Returns:
            Loss value
        """
        # Normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # Positive logits: [N, 1]
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: [N, N-1]
        l_neg = torch.einsum('nc,ck->nk', [q, k.T])
        
        # Remove positive pairs from negatives
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.temperature
        
        # Labels: positives are the first column
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x1: First augmented view
            x2: Second augmented view
            
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


class UnderwaterSSLDataset(Dataset):
    """Dataset for self-supervised learning on underwater images"""
    
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image as fallback
            img = Image.new('RGB', (640, 640))
        
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        else:
            img1 = img2 = transforms.ToTensor()(img)
        
        return img1, img2


class SSLTrainer:
    """SSL Pretraining Trainer"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.ssl_config = self.config.get('ssl_pretraining', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.epochs = self.ssl_config.get('epochs', 100)
        self.batch_size = self.ssl_config.get('batch_size', 256)
        self.learning_rate = self.ssl_config.get('learning_rate', 0.03)
        self.weight_decay = self.ssl_config.get('weight_decay', 0.0001)
        self.momentum = self.ssl_config.get('momentum', 0.9)
        
        # Checkpoint settings
        self.save_frequency = self.ssl_config.get('checkpoint', {}).get('save_frequency', 10)
        self.checkpoint_dir = Path('checkpoints/ssl')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_augmentation_pipeline(self) -> transforms.Compose:
        """Get augmentation pipeline for SSL"""
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def collect_image_paths(self, data_dir: str) -> List[str]:
        """Collect all image paths for SSL training"""
        image_paths = []
        data_path = Path(data_dir)
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend([str(p) for p in data_path.rglob(ext)])
        
        print(f"Found {len(image_paths)} images for SSL training")
        return image_paths
    
    def train(
        self,
        data_dir: str,
        output_dir: str,
        resume_from: str = None
    ):
        """
        Train SSL model
        
        Args:
            data_dir: Directory containing unlabeled images
            output_dir: Output directory for checkpoints
            resume_from: Path to checkpoint to resume from
        """
        # Initialize wandb
        wandb.init(
            project="marauder-cv",
            name="1-ssl-moco-v3",
            config=self.ssl_config
        )
        
        # Collect images
        image_paths = self.collect_image_paths(data_dir)
        
        # Create dataset and dataloader
        transform = self.get_augmentation_pipeline()
        dataset = UnderwaterSSLDataset(image_paths, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )
        
        # Create base encoder (ResNet50)
        from torchvision.models import resnet50
        base_encoder = resnet50(pretrained=False)
        base_encoder.fc = nn.Identity()  # Remove classification head
        
        # Create MoCo model
        model = MoCoV3(
            base_encoder=base_encoder,
            dim=self.ssl_config.get('projection_dim', 256),
            temperature=self.ssl_config.get('temperature', 0.2),
            momentum=self.ssl_config.get('momentum', 0.999)
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=0
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            checkpoint = torch.load(resume_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}")
        
        # Training loop
        model.train()
        for epoch in range(start_epoch, self.epochs):
            epoch_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch_idx, (x1, x2) in enumerate(pbar):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                # Forward pass
                loss = model(x1, x2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Log to wandb
                wandb.log({
                    'batch_loss': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
            
            wandb.log({
                'epoch': epoch + 1,
                'epoch_loss': avg_loss
            })
            
            # Update scheduler
            scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.save_frequency == 0:
                checkpoint_path = self.checkpoint_dir / f'ssl_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save final model
        final_path = Path(output_dir) / 'ssl_backbone_final.pt'
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder_q.state_dict(), final_path)
        print(f"Saved final SSL backbone: {final_path}")
        
        wandb.finish()
        
        return model


def main():
    parser = argparse.ArgumentParser(description='SSL Pretraining with MoCo V3')
    parser.add_argument('--config', type=str, default='config/training_config.yaml', help='Training config path')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with unlabeled images')
    parser.add_argument('--output', type=str, default='outputs/1_ssl', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    trainer = SSLTrainer(args.config)
    trainer.train(args.data_dir, args.output, resume_from=args.resume)


if __name__ == '__main__':
    main()
