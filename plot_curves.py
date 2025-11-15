"""
=============================================================================
SIMPLE TRAINING CURVES PLOTTER
=============================================================================

📊 PURPOSE: Quick utility to plot training curves from saved data
🔍 FEATURES: Load pickle files and generate publication-ready plots
🚀 USAGE: python plot_curves.py

No external dependencies beyond matplotlib, pandas, numpy
=============================================================================
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_experiment_data():
    """Load all available training data"""
    data = {}
    
    # Load assignment compliance experiments
    if os.path.exists("runs_pytorch"):
        for run_dir in Path("runs_pytorch").iterdir():
            if run_dir.is_dir():
                pkl_file = run_dir / "training_data.pkl"
                if pkl_file.exists():
                    with open(pkl_file, 'rb') as f:
                        data[run_dir.name] = pickle.load(f)
    
    # Load high-performance experiment
    hp_file = "runs_optimized_pytorch/training_data_complete.pkl"
    if os.path.exists(hp_file):
        with open(hp_file, 'rb') as f:
            data['high_performance'] = pickle.load(f)
    
    # Load SOTA optimization experiment
    sota_file = "runs_sota_pytorch/training_data_complete.pkl"
    if os.path.exists(sota_file):
        with open(sota_file, 'rb') as f:
            data['sota_optimization'] = pickle.load(f)
    
    # Load SOTA optimization advanced experiment
    sota_opt_file = "runs_sota_pytorch_optimized/training_data_complete.pkl"
    if os.path.exists(sota_opt_file):
        with open(sota_opt_file, 'rb') as f:
            data['sota_optimized'] = pickle.load(f)
    
    return data

def plot_comparison_curves(data, save_path="training_comparison.png"):
    """Plot comparison of all experiments"""
    
    if not data:
        print("❌ No training data found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Vision Transformer Training Analysis - CIFAR-100', fontsize=14, weight='bold')
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(data)))
    
    # Training accuracy
    ax1 = axes[0, 0]
    for i, (name, exp_data) in enumerate(data.items()):
        history = exp_data['history']
        epochs = range(1, len(history['train_acc']) + 1)
        ax1.plot(epochs, history['train_acc'], label=name, color=colors[i], linewidth=2)
    
    ax1.set_title('Training Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax2 = axes[0, 1]
    for i, (name, exp_data) in enumerate(data.items()):
        history = exp_data['history']
        epochs = range(1, len(history['val_acc']) + 1)
        ax2.plot(epochs, history['val_acc'], label=name, color=colors[i], linewidth=2)
    
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Training loss
    ax3 = axes[1, 0]
    for i, (name, exp_data) in enumerate(data.items()):
        history = exp_data['history']
        epochs = range(1, len(history['train_loss']) + 1)
        ax3.plot(epochs, history['train_loss'], label=name, color=colors[i], linewidth=2)
    
    ax3.set_title('Training Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Validation loss
    ax4 = axes[1, 1]
    for i, (name, exp_data) in enumerate(data.items()):
        history = exp_data['history']
        epochs = range(1, len(history['val_loss']) + 1)
        ax4.plot(epochs, history['val_loss'], label=name, color=colors[i], linewidth=2)
    
    ax4.set_title('Validation Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Curves saved: {save_path}")

def plot_individual_curves(data):
    """Plot individual curves for each experiment"""
    
    for name, exp_data in data.items():
        history = exp_data['history']
        config = exp_data.get('config', {})
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Training Curves - {name}', fontsize=14, weight='bold')
        
        epochs = range(1, len(history['train_acc']) + 1)
        
        # Accuracy (including EMA if available)
        axes[0].plot(epochs, history['train_acc'], label='Training', linewidth=2)
        axes[0].plot(epochs, history['val_acc'], label='Validation', linewidth=2)
        
        # Add EMA accuracy if available (SOTA model)
        if 'ema_val_acc' in history and history['ema_val_acc']:
            ema_epochs = range(1, len(history['ema_val_acc']) + 1)
            axes[0].plot(ema_epochs, history['ema_val_acc'], label='Validation (EMA)', 
                        linewidth=2, linestyle='--', alpha=0.8)
        
        axes[0].set_title('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(epochs, history['train_loss'], label='Training', linewidth=2)
        axes[1].plot(epochs, history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        # Add configuration info
        config_text = f"Config: "
        if 'heads' in config:
            config_text += f"Heads={config['heads']}, Blocks={config['blocks']}"
        elif 'num_heads' in config:
            config_text += f"Heads={config['num_heads']}, Layers={config['num_layers']}"
        
        fig.text(0.5, 0.02, config_text, ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"curves_{name}.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Individual curves saved: curves_{name}.png")

def generate_summary_report(data):
    """Generate text summary report"""
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY REPORT")
    print("="*80)
    
    for name, exp_data in data.items():
        config = exp_data.get('config', {})
        final = exp_data.get('final_results', {})
        history = exp_data['history']
        
        print(f"\n📊 {name.upper()}")
        print("-" * 40)
        
        # Configuration
        if 'heads' in config:
            print(f"Configuration: {config['heads']} heads, {config['blocks']} blocks, {config['patch_size']}x{config['patch_size']} patches")
        elif 'num_heads' in config:
            print(f"Configuration: {config['num_heads']} heads, {config['num_layers']} layers, {config['patch_size']}x{config['patch_size']} patches")
        
        print(f"Parameters: {config.get('total_params', 'N/A'):,}")
        
        # Results
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
        best_train_acc = max(history['train_acc']) if history['train_acc'] else 0
        final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
        final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Best Training Accuracy: {best_train_acc:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        print(f"Final Training Accuracy: {final_train_acc:.2f}%")
        
        # Show EMA accuracy if available (SOTA model)
        if 'ema_val_acc' in history and history['ema_val_acc']:
            best_ema_acc = max(history['ema_val_acc'])
            final_ema_acc = history['ema_val_acc'][-1]
            print(f"Best EMA Accuracy: {best_ema_acc:.2f}%")
            print(f"Final EMA Accuracy: {final_ema_acc:.2f}%")
        print(f"Epochs Trained: {len(history['train_acc'])}")
        
        if 'training_time_min' in final:
            print(f"Training Time: {final['training_time_min']:.1f} minutes")

def main():
    """Main function"""
    print("🚀 Loading training data...")
    
    data = load_experiment_data()
    
    if not data:
        print("❌ No training data found!")
        print("   Run the training scripts first to generate data.")
        return
    
    print(f"✅ Loaded {len(data)} experiments")
    
    # Generate plots
    plot_comparison_curves(data)
    plot_individual_curves(data)
    generate_summary_report(data)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()