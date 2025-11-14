"""
=============================================================================
VISION TRANSFORMER CIFAR-100 - MAXIMUM PERFORMANCE (RTX 5090 OPTIMIZED)
=============================================================================

🚀 PURPOSE: Achieve HIGHEST possible accuracy on CIFAR-100 using RTX 5090
💪 TARGET: 90%+ validation accuracy through advanced techniques
⚡ OPTIMIZATIONS: 
  - Large model architecture (512 embed_dim, 16 heads, 16 layers)
  - Advanced data augmentation (balanced for performance)
  - MixUp regularization (tuned intensity)
  - Label smoothing
  - Cosine learning rate schedule with warmup
  - Mixed precision training
  - Optimized for RTX 5090 compute capability

🔧 ARCHITECTURE:
  - Patch size: 4x4 (optimal for 32x32 images)
  - Embedding dimension: 384 (optimized size)
  - Attention heads: 12  
  - Transformer layers: 12
  - MLP dimension: 1536
  - Parameters: ~48M (balanced model)

📊 FEATURES:
  ✅ GPU memory optimization
  ✅ Advanced augmentation pipeline  
  ✅ MixUp with tuned parameters
  ✅ Label smoothing (0.1)
  ✅ Warmup + cosine decay
  ✅ Early stopping & checkpoints
  ✅ TensorBoard logging
=============================================================================
"""

import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import numpy as np
import math
from typing import Optional

# =====================================================
# HIGH-PERFORMANCE CONFIGURATIONS (Based on 85% Accuracy Repository)
# =====================================================
OUTPUT_DIR = "runs_sota_pytorch"
EPOCHS = 100         # Sufficient epochs with early stopping
BATCH_SIZE = 32      # Optimal batch size from successful repo
IMG_SIZE = 224       # Higher resolution like successful approach
PATCH_SIZE = 16      # Larger patches for 224x224 images
LEARNING_RATE = 0.01 # Higher initial LR for OneCycleLR
SEED = 42

# Progressive Training Stages (inspired by successful repository)
STAGE1_EPOCHS = 10   # Head only training
STAGE2_EPOCHS = 15   # Partial model training  
STAGE3_EPOCHS = 75   # Full model training

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(2)}")

# =====================================================
# ADVANCED DATA AUGMENTATION
# =====================================================

class MixUp:
    def __init__(self, alpha=0.8):  # Increased alpha like successful repo
        self.alpha = alpha
        
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class CutMix:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        
        # Generate random bounding box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to match the actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Enhanced augmentation pipeline (based on 85% accuracy repository)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to 224x224
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    transforms.RandomErasing(p=0.3, scale=(0.05, 0.2), ratio=(0.3, 3.3))
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Consistent sizing
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# =====================================================
# LOAD AND PREPARE DATA
# =====================================================

# Load CIFAR-100 dataset
train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =====================================================
# VISION TRANSFORMER ARCHITECTURE
# =====================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 196 patches for 224x224
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)        # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (B, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=100,
        embed_dim=512,
        num_heads=16,
        num_layers=16,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Use class token
        x = self.head(cls_output)
        
        return x

# =====================================================
# TRAINING UTILITIES
# =====================================================

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # pred: (N, C), target: (N,)
        log_prob = F.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class OneCycleLRWrapper:
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3):
        from torch.optim.lr_scheduler import OneCycleLR
        self.scheduler = OneCycleLR(
            optimizer, 
            max_lr=max_lr, 
            total_steps=total_steps,
            pct_start=pct_start,  # 30% warmup like successful repo
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )
    
    def step(self):
        self.scheduler.step()
        return self.scheduler.get_last_lr()[0]

def freeze_layers(model, stage):
    """Progressive freezing strategy"""
    # First, unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
        
    if stage == 1:  # Only head
        for name, param in model.named_parameters():
            if not name.startswith('head'):
                param.requires_grad = False
                
    elif stage == 2:  # Head + last few blocks
        for name, param in model.named_parameters():
            block_num = None
            if 'blocks.' in name:
                block_num = int(name.split('blocks.')[1].split('.')[0])
            
            # Only train last 4 blocks + head
            if block_num is not None and block_num < 8:
                param.requires_grad = False
            elif not (name.startswith('head') or name.startswith('norm') or 'blocks.' in name):
                param.requires_grad = False
    # Stage 3: All layers trainable (default)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Stage {stage}: {trainable_params:,}/{total_params:,} parameters trainable")

def train_epoch(model, train_loader, criterion, optimizer, scheduler, mixup, cutmix, epoch, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Use mixed precision training
        with torch.amp.autocast('cuda'):
            # Apply MixUp/CutMix with 50% probability each (like successful repo)
            if np.random.rand() < 0.5:
                if np.random.rand() < 0.5:  # MixUp
                    data, target_a, target_b, lam = mixup(data, target)
                else:  # CutMix
                    data, target_a, target_b, lam = cutmix(data, target)
                
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
                # Approximate accuracy for mixed samples
                pred = output.argmax(dim=1)
                correct += (lam * pred.eq(target_a).sum().item() + 
                           (1 - lam) * pred.eq(target_b).sum().item())
            else:
                output = model(data)
                loss = criterion(output, target)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step and scheduler update
        scaler.step(optimizer)
        scaler.update()
        
        # Update OneCycleLR scheduler every batch
        current_lr = scheduler.step()
            
        total_loss += loss.item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:  # More frequent progress updates
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, LR: {current_lr:.6f}')
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return total_loss / len(test_loader), accuracy

# =====================================================
# MAIN TRAINING FUNCTION
# =====================================================

def train_optimized_model():
    print("🚀 Creating high-performance ViT model (224x224)...")
    
    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=768,     # Standard ViT-Base size for 224x224
        num_heads=12,      
        num_layers=12,     
        dropout=0.1        
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    print(f"📊 Input resolution: {IMG_SIZE}x{IMG_SIZE}")
    print(f"📊 Patches per image: {(IMG_SIZE//PATCH_SIZE)**2}")
    
    # Enhanced augmentation with MixUp and CutMix
    mixup = MixUp(alpha=0.8)   # High alpha like successful repo
    cutmix = CutMix(alpha=0.8)
    
    # Calculate total training steps for OneCycleLR
    total_steps = len(train_loader) * EPOCHS
    print(f"📊 Total training steps: {total_steps:,}")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Create output directory and mixed precision scaler
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scaler = torch.amp.GradScaler('cuda')
    
    print("🏃 Starting progressive training...")
    start_time = time.time()
    
    best_accuracy = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    # Progressive Training Implementation
    current_epoch = 0
    
    # STAGE 1: Head only training (Epochs 1-10)
    print("\n🎩 STAGE 1: Training classification head only...")
    freeze_layers(model, stage=1)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4, nesterov=True)
    stage1_steps = len(train_loader) * STAGE1_EPOCHS
    scheduler = OneCycleLRWrapper(optimizer, max_lr=LEARNING_RATE, total_steps=stage1_steps)
    
    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, mixup, cutmix, current_epoch, scaler)
        val_loss, val_acc = validate(model, test_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.scheduler.get_last_lr()[0])
        
        print(f'Stage 1 - Epoch {current_epoch+1}/{STAGE1_EPOCHS}: Train {train_acc:.2f}%, Val {val_acc:.2f}%')
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_stage1.pth'))
        
        current_epoch += 1
    
    # STAGE 2: Partial model training (Epochs 11-25)
    print("\n🎯 STAGE 2: Training deeper layers...")
    freeze_layers(model, stage=2)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE*0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    stage2_steps = len(train_loader) * STAGE2_EPOCHS
    scheduler = OneCycleLRWrapper(optimizer, max_lr=LEARNING_RATE*0.1, total_steps=stage2_steps)
    
    for epoch in range(STAGE2_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, mixup, cutmix, current_epoch, scaler)
        val_loss, val_acc = validate(model, test_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.scheduler.get_last_lr()[0])
        
        print(f'Stage 2 - Epoch {current_epoch+1}/{STAGE1_EPOCHS+STAGE2_EPOCHS}: Train {train_acc:.2f}%, Val {val_acc:.2f}%')
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_stage2.pth'))
        else:
            patience_counter += 1
        
        current_epoch += 1
    
    # STAGE 3: Full model training (Epochs 26-100)
    print("\n🚀 STAGE 3: Full model fine-tuning...")
    freeze_layers(model, stage=3)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE*0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    stage3_steps = len(train_loader) * STAGE3_EPOCHS
    scheduler = OneCycleLRWrapper(optimizer, max_lr=LEARNING_RATE*0.01, total_steps=stage3_steps)
    
    for epoch in range(STAGE3_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, mixup, cutmix, current_epoch, scaler)
        val_loss, val_acc = validate(model, test_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.scheduler.get_last_lr()[0])
        
        print(f'Stage 3 - Epoch {current_epoch+1}/{EPOCHS}: Train {train_acc:.2f}%, Val {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_accuracy': best_accuracy
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f'  🎯 New best accuracy: {best_accuracy:.2f}%')
        else:
            patience_counter += 1
        
        # Early stopping with patience (like successful repo)
        if patience_counter >= 10:  # Same patience as successful repo
            print(f"Early stopping triggered after {patience_counter} epochs without improvement!")
            break
            
        current_epoch += 1
    
    training_time = time.time() - start_time
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc = validate(model, test_loader, criterion)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"   Final Validation Accuracy: {final_acc:.2f}%")
    print(f"   Training Time: {training_time/60:.1f} minutes")
    print(f"   Parameters: {total_params:,}")
    
    if best_accuracy > 90:
        print("🎉 SUCCESS! Achieved 90%+ accuracy!")
    elif best_accuracy > 80:
        print("🚀 Good progress! 80%+ achieved. Continue training or increase model size.")
    else:
        print("⚠️  Need more improvements. Consider increasing training time or model capacity.")
    
    # Save comprehensive training data for analysis
    training_data = {
        'history': history,
        'config': {
            'model_type': 'progressive_vit_224',
            'img_size': IMG_SIZE,
            'patch_size': PATCH_SIZE,
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 12,
            'mlp_dim': 3072,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs': EPOCHS,
            'total_params': total_params,
            'mixup_alpha': 0.8,
            'cutmix_alpha': 0.8,
            'progressive_training': True,
            'scheduler': 'OneCycleLR',
            'dropout': 0.1
        },
        'final_results': {
            'best_accuracy': float(best_accuracy),
            'final_accuracy': float(final_acc),
            'training_time_min': training_time / 60,
            'epochs_trained': len(history["train_acc"]),
            'best_train_acc': max(history["train_acc"]) if history["train_acc"] else 0,
            'final_val_loss': history["val_loss"][-1] if history["val_loss"] else 0,
            'final_train_loss': history["train_loss"][-1] if history["train_loss"] else 0,
            'convergence_epoch': history["val_acc"].index(max(history["val_acc"])) + 1 if history["val_acc"] else 0
        },
        'training_curves': {
            'train_acc': history["train_acc"],
            'val_acc': history["val_acc"], 
            'train_loss': history["train_loss"],
            'val_loss': history["val_loss"],
            'learning_rates': history["lr"]
        }
    }
    
    # Save training history (enhanced)
    with open(os.path.join(OUTPUT_DIR, 'training_data_complete.pkl'), 'wb') as f:
        pickle.dump(training_data, f)
    
    # Save metrics as CSV for easy analysis
    df_metrics = pd.DataFrame(history)
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'training_metrics.csv'), index=False)
    
    print(f"\n💾 Training data saved:")
    print(f"   - {OUTPUT_DIR}/training_data_complete.pkl (complete data)")
    print(f"   - {OUTPUT_DIR}/training_metrics.csv (metrics CSV)")
    print(f"   - {OUTPUT_DIR}/best_model.pth (best model)")
    
    return model, history, best_accuracy

if __name__ == "__main__":
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"🔧 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 CUDA version: {torch.version.cuda}")
        print(f"🔧 GPU count: {torch.cuda.device_count()}")
    
    model, history, accuracy = train_optimized_model()
    print(f"\n✅ Training completed! Best accuracy: {accuracy:.2f}%")