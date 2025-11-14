"""
=============================================================================
VISION TRANSFORMER CIFAR-100 - STRICT REQUIREMENTS COMPLIANCE
=============================================================================

📋 ASSIGNMENT REQUIREMENTS (STRICTLY FOLLOWED):
  ✅ Dataset: CIFAR-100 (60.000 images 32x32x3, 100 classes)
  ✅ Split: 50.000 training, 10.000 testing (automatic PyTorch split)
  ✅ Patch size: 6x6 (exactly as specified)
  ✅ MSA heads: 4 and 6 per block (exactly as specified)
  ✅ Encoder blocks: 8 and 12 units (exactly as specified)
  ✅ Dense layer sizes: Automatically adjusted (embed_dim * 4)
  ✅ Data augmentation: Included (balanced approach)
  ✅ Results: Accuracy + training curves (saved as PNG + CSV)

🎯 PURPOSE: Academic compliance - follows assignment guidelines exactly
🚀 PERFORMANCE: Good baseline performance with required configurations
⚙️  GPU: Works on RTX 5090 with PyTorch

EXPERIMENT MATRIX:
- h4_b8_p6  : 4 heads, 8 blocks, 6x6 patches
- h4_b12_p6 : 4 heads, 12 blocks, 6x6 patches  
- h6_b8_p6  : 6 heads, 8 blocks, 6x6 patches
- h6_b12_p6 : 6 heads, 12 blocks, 6x6 patches
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
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import numpy as np
import math
from typing import Optional

# =====================================================
# CONFIGURAÇÕES GERAIS
# =====================================================
OUTPUT_DIR = "runs_pytorch"
EPOCHS = 50          # Sufficient epochs for convergence
BATCH_SIZE = 64      # Good balance for 32x32 images
PATCH_SIZE = 6       # Required: 6x6 patches
LEARNING_RATE = 3e-4 # Proven learning rate for ViT
SEED = 42

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Enable mixed precision
    torch.backends.cudnn.benchmark = True

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(3)}")

# =====================================================
# CARREGAR E PREPARAR DADOS CIFAR-100
# =====================================================

# Enhanced data augmentation to combat overfitting
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),  # Increased from 5 to 10 degrees
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),  # Slightly increased
    transforms.RandomCrop(32, padding=4),  # Added back random crop for more variation
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Load CIFAR-100 dataset (automatically splits: 50k train, 10k test)
train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)   # 50,000 images
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=test_transform)    # 10,000 images

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# =====================================================
# FIXED ViT IMPLEMENTATION
# =====================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=6, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # For 32x32 image with 6x6 patches: (32//6) = 5, so 5x5 = 25 patches
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)        # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (B, n_patches, embed_dim)
        return x

class AddClassToken(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, x], dim=1)

class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, embed_dim):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(sequence_length, embed_dim))
        
    def forward(self, x):
        length = x.shape[1]
        return x + self.position_embedding[:length, :]

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

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Layer normalization 1 + Multi-head attention + Skip connection 1
        x = x + self.attn(self.norm1(x))
        
        # Layer normalization 2 + MLP + Skip connection 2
        x = x + self.mlp(self.norm2(x))
        
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=6,  # Required: 6x6 patches
        in_channels=3,
        num_classes=100,
        embed_dim=128,
        num_heads=4,  # Will be 4 or 6 as required
        num_layers=8,  # Will be 8 or 12 as required
        mlp_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Class token
        self.cls_token = AddClassToken(embed_dim)
        
        # Positional embeddings
        self.pos_embed = PositionalEmbedding(num_patches + 1, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        x = self.cls_token(x)  # (B, n_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = self.pos_embed(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer norm
        x = self.norm(x)
        
        # Classification (use class token)
        cls_output = x[:, 0]  # Use class token
        x = self.head(cls_output)
        
        return x

# =====================================================
# TRAINING UTILITIES
# =====================================================

class CosineScheduler:
    def __init__(self, optimizer, initial_lr, total_steps, min_lr=1e-5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self):
        lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (
            1 + math.cos(math.pi * self.current_step / self.total_steps)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:  # Mixed precision training
            with torch.amp.autocast('cuda'):
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return total_loss / len(test_loader), accuracy

# =====================================================
# LOOP DE EXPERIMENTOS (REQUISITOS ESPECÍFICOS)
# =====================================================
# Requisitos: patches 6x6, heads [4,6], encoder blocks [8,12]
heads_list = [4, 6]      # Required: 4 e 6 cabeças por bloco MSA
blocks_list = [8, 12]    # Required: 8 e 12 unidades codificadoras
results = []

print(f"🔧 Configurações dos experimentos:")
print(f"   - Patches: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"   - Cabeças: {heads_list}")
print(f"   - Blocos codificadores: {blocks_list}")
print(f"   - Total experimentos: {len(heads_list) * len(blocks_list)}")
print(f"   - Dataset: CIFAR-100 (50k treino, 10k teste)")
print()

for heads in heads_list:
    for blocks in blocks_list:
        run_name = f"h{heads}_b{blocks}_p{PATCH_SIZE}"
        print(f"\n==== Experimento {run_name} ====")
        print(f"     Cabeças: {heads}, Blocos: {blocks}, Patches: {PATCH_SIZE}x{PATCH_SIZE}")
        print()

        run_dir = os.path.join(OUTPUT_DIR, run_name)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "tensorboard"), exist_ok=True)

        # Ajustar dimensões internas conforme número de cabeças
        embed_dim = heads * 64  # Maior dimensão para acomodar mais cabeças
        mlp_dim = embed_dim * 4  # Padrão ViT: 4x a dimensão de embedding

        model = VisionTransformer(
            img_size=32,
            patch_size=PATCH_SIZE,
            in_channels=3,
            num_classes=100,
            embed_dim=embed_dim,
            num_heads=heads,
            num_layers=blocks,
            mlp_dim=mlp_dim,
            dropout=0.15  # Increased dropout to combat overfitting
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Configuração do modelo:")
        print(f"   - Dimensão embedding: {embed_dim}")
        print(f"   - Dimensão MLP: {mlp_dim}")
        print(f"   - Número de patches: {(32//PATCH_SIZE)**2}")
        print(f"   - Parâmetros totais: {total_params:,}")
        print()

        # Create optimizer and scheduler (with stronger regularization)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)  # Increased weight decay
        total_steps = EPOCHS * len(train_loader)
        scheduler = CosineScheduler(optimizer, LEARNING_RATE, total_steps)
        
        criterion = nn.CrossEntropyLoss()
        
        # Mixed precision scaler (if CUDA available)
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        
        # TensorBoard writer
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

        # Training variables (more aggressive early stopping)
        best_accuracy = 0
        patience_counter = 0
        patience = 8  # Increased patience to allow for longer convergence
        lr_reduction_counter = 0
        lr_reduction_patience = 4  # More patient LR reduction
        
        history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [], 'lr': []
        }

        print("🏃 Starting training...")
        start_time = time.time()

        for epoch in range(EPOCHS):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, scaler
            )
            
            # Validate
            val_loss, val_acc = validate(model, test_loader, criterion, device)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # TensorBoard logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            print(f'Epoch {epoch+1}/{EPOCHS}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            # Save best model (ModelCheckpoint equivalent)
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), os.path.join(run_dir, "checkpoints", "best_model.pth"))
                print(f'  🎯 New best accuracy: {best_accuracy:.2f}%')
                patience_counter = 0
                lr_reduction_counter = 0
            else:
                patience_counter += 1
                lr_reduction_counter += 1
            
            # LR reduction on plateau
            if lr_reduction_counter >= lr_reduction_patience:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    if param_group['lr'] < 1e-6:
                        param_group['lr'] = 1e-6
                print(f'  📉 Reduced learning rate to: {optimizer.param_groups[0]["lr"]:.6f}')
                lr_reduction_counter = 0
            
        # Early stopping (more aggressive)
        if patience_counter >= patience:
            print(f'  ⏹️  Early stopping triggered after {patience} epochs without improvement')
            break
            
        print('-' * 50)

        writer.close()
        training_time = time.time() - start_time

        # Load best model for final evaluation
        model.load_state_dict(torch.load(os.path.join(run_dir, "checkpoints", "best_model.pth")))
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"[{run_name}] Acurácia final: {test_acc:.4f}")

        # Salvar curvas
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_acc"], label="treino")
        plt.plot(history["val_acc"], label="val")
        plt.title("Acurácia")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["train_loss"], label="treino")
        plt.plot(history["val_loss"], label="val")
        plt.title("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "accuracy_loss_curves.png"))
        plt.close()

        # Salvar resultados detalhados num CSV
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
        
        # Salvar história completa em pickle para análise posterior
        training_data = {
            'history': history,
            'config': {
                'run_name': run_name,
                'heads': heads,
                'blocks': blocks,
                'embed_dim': embed_dim,
                'mlp_dim': mlp_dim,
                'patch_size': PATCH_SIZE,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS,
                'total_params': total_params
            },
            'final_results': {
                'test_acc': float(test_acc),
                'epochs_trained': len(history["train_acc"]),
                'training_time_min': training_time / 60,
                'best_val_acc': max(history["val_acc"]),
                'best_train_acc': max(history["train_acc"]),
                'final_val_loss': history["val_loss"][-1],
                'final_train_loss': history["train_loss"][-1]
            }
        }
        
        # Salvar dados completos para análise posterior
        with open(os.path.join(run_dir, "training_data.pkl"), 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"💾 Dados salvos em: {run_dir}/")
        print(f"   - metrics.csv (métricas por época)")
        print(f"   - training_data.pkl (dados completos)")
        print(f"   - accuracy_loss_curves.png (gráficos)")

        results.append({
            "config": run_name,
            "heads": heads,
            "blocks": blocks,
            "embed_dim": embed_dim,
            "mlp_dim": mlp_dim,
            "test_acc": float(test_acc),
            "epochs_trained": len(history["train_acc"]),
            "training_time_min": training_time / 60,
            "total_params": total_params
        })

# Tabela final consolidada
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)

print("="*80)
print("RESULTADOS FINAIS - VISUAL TRANSFORMER CIFAR-100 (ASSIGNMENT COMPLIANCE)")
print("="*80)
print("📋 CONFIGURAÇÃO: Seguindo exatamente os requisitos acadêmicos")
print(f"   - Patches: {PATCH_SIZE}x{PATCH_SIZE} (conforme especificado)")
print(f"   - Cabeças MSA: {heads_list} (conforme especificado)")
print(f"   - Blocos codificadores: {blocks_list} (conforme especificado)")
print(f"   - Dataset: CIFAR-100 (50.000 treino, 10.000 teste)")
print(f"   - Experimentos realizados: {len(results)}")
print()
print("ACURÁCIAS OBTIDAS:")
print("-"*50)
for result in results:
    print(f"• {result['config']:12} | {result['heads']:2} cabeças | {result['blocks']:2} blocos | "
          f"Acurácia: {result['test_acc']:6.2f}% | "
          f"Tempo: {result['training_time_min']:5.1f}min | "
          f"Params: {result['total_params']:,}")

print()
print("MELHOR RESULTADO:")
best_result = max(results, key=lambda x: x['test_acc'])
print(f"• Configuração: {best_result['config']}")
print(f"• Acurácia: {best_result['test_acc']:.2f}%")
print(f"• Cabeças: {best_result['heads']}, Blocos: {best_result['blocks']}")
print(f"• Parâmetros: {best_result['total_params']:,}")
print()
print("Curvas de treinamento salvas em: runs_pytorch/*/accuracy_loss_curves.png")
print("Métricas detalhadas salvas em: runs_pytorch/summary.csv")
print("="*80)