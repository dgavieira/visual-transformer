"""
Real-time Overfitting Monitor for Vision Transformer Training
============================================================

Monitor training progress and detect overfitting early.
Reads metrics.csv files and alerts when overfitting is detected.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pathlib import Path

def detect_overfitting(train_acc, val_acc, train_loss, val_loss, threshold=10):
    """
    Detect overfitting based on divergence between train and validation metrics
    
    Args:
        train_acc, val_acc, train_loss, val_loss: Lists of metrics
        threshold: Percentage threshold for overfitting detection
    
    Returns:
        dict with overfitting analysis
    """
    if len(train_acc) < 5:  # Need minimum epochs
        return {"overfitting": False, "reason": "Not enough epochs"}
    
    # Check last 5 epochs
    recent_train_acc = np.mean(train_acc[-5:])
    recent_val_acc = np.mean(val_acc[-5:])
    recent_train_loss = np.mean(train_loss[-5:])
    recent_val_loss = np.mean(val_loss[-5:])
    
    # Calculate gaps
    acc_gap = recent_train_acc - recent_val_acc
    loss_ratio = recent_val_loss / recent_train_loss if recent_train_loss > 0 else 1
    
    # Check for validation plateau (no improvement in last 5 epochs)
    val_acc_trend = np.polyfit(range(len(val_acc[-5:])), val_acc[-5:], 1)[0]
    val_loss_trend = np.polyfit(range(len(val_loss[-5:])), val_loss[-5:], 1)[0]
    
    overfitting_indicators = []
    
    if acc_gap > threshold:
        overfitting_indicators.append(f"Large accuracy gap: {acc_gap:.1f}%")
    
    if loss_ratio > 1.5:
        overfitting_indicators.append(f"Validation loss increasing: {loss_ratio:.2f}x training loss")
    
    if val_acc_trend < 0.1 and len(val_acc) > 10:
        overfitting_indicators.append("Validation accuracy plateaued")
    
    if val_loss_trend > 0.01 and len(val_loss) > 10:
        overfitting_indicators.append("Validation loss increasing")
    
    return {
        "overfitting": len(overfitting_indicators) >= 2,
        "indicators": overfitting_indicators,
        "acc_gap": acc_gap,
        "loss_ratio": loss_ratio,
        "val_acc_trend": val_acc_trend,
        "val_loss_trend": val_loss_trend
    }

def monitor_experiment(exp_path):
    """Monitor a single experiment for overfitting"""
    
    metrics_file = exp_path / "metrics.csv"
    if not metrics_file.exists():
        return None
    
    try:
        df = pd.read_csv(metrics_file)
        
        analysis = detect_overfitting(
            df['train_acc'].tolist(),
            df['val_acc'].tolist(), 
            df['train_loss'].tolist(),
            df['val_loss'].tolist()
        )
        
        return {
            "experiment": exp_path.name,
            "epochs": len(df),
            "current_train_acc": df['train_acc'].iloc[-1],
            "current_val_acc": df['val_acc'].iloc[-1],
            "best_val_acc": df['val_acc'].max(),
            "analysis": analysis
        }
        
    except Exception as e:
        return {"experiment": exp_path.name, "error": str(e)}

def plot_overfitting_analysis(exp_path, save_plot=True):
    """Generate overfitting analysis plot"""
    
    metrics_file = exp_path / "metrics.csv"
    if not metrics_file.exists():
        return
    
    df = pd.read_csv(metrics_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Overfitting Analysis - {exp_path.name}', fontsize=14, weight='bold')
    
    epochs = range(1, len(df) + 1)
    
    # Accuracy plot
    axes[0, 0].plot(epochs, df['train_acc'], label='Training', linewidth=2)
    axes[0, 0].plot(epochs, df['val_acc'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Accuracy Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(epochs, df['train_loss'], label='Training', linewidth=2)
    axes[0, 1].plot(epochs, df['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Loss Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Accuracy gap over time
    acc_gap = df['train_acc'] - df['val_acc']
    axes[1, 0].plot(epochs, acc_gap, color='red', linewidth=2)
    axes[1, 0].axhline(y=10, color='orange', linestyle='--', label='Warning threshold')
    axes[1, 0].axhline(y=20, color='red', linestyle='--', label='Critical threshold')
    axes[1, 0].set_title('Train-Val Accuracy Gap')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy Gap (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss ratio over time
    loss_ratio = df['val_loss'] / df['train_loss']
    axes[1, 1].plot(epochs, loss_ratio, color='purple', linewidth=2)
    axes[1, 1].axhline(y=1.5, color='orange', linestyle='--', label='Warning threshold')
    axes[1, 1].axhline(y=2.0, color='red', linestyle='--', label='Critical threshold')
    axes[1, 1].set_title('Val/Train Loss Ratio')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(exp_path / "overfitting_analysis.png", dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Monitor all experiments"""
    
    print("🔍 Monitoring experiments for overfitting...")
    
    runs_dir = Path("runs_pytorch")
    if not runs_dir.exists():
        print("❌ No runs_pytorch directory found")
        return
    
    results = []
    
    for exp_dir in runs_dir.iterdir():
        if exp_dir.is_dir():
            result = monitor_experiment(exp_dir)
            if result:
                results.append(result)
    
    # Display results
    print("\n" + "="*80)
    print("OVERFITTING MONITORING REPORT")
    print("="*80)
    
    for result in results:
        if 'error' in result:
            print(f"\n❌ {result['experiment']}: {result['error']}")
            continue
            
        exp = result['experiment']
        analysis = result['analysis']
        
        print(f"\n📊 {exp}")
        print(f"   Epochs: {result['epochs']}")
        print(f"   Current: Train {result['current_train_acc']:.1f}% | Val {result['current_val_acc']:.1f}%")
        print(f"   Best Val: {result['best_val_acc']:.1f}%")
        
        if analysis['overfitting']:
            print(f"   🚨 OVERFITTING DETECTED!")
            for indicator in analysis['indicators']:
                print(f"      - {indicator}")
        else:
            print(f"   ✅ No significant overfitting detected")
        
        print(f"   Gap: {analysis['acc_gap']:.1f}% | Loss Ratio: {analysis['loss_ratio']:.2f}")
        
        # Generate plot for overfitting cases
        if analysis['overfitting']:
            exp_path = runs_dir / exp
            plot_overfitting_analysis(exp_path)

if __name__ == "__main__":
    main()