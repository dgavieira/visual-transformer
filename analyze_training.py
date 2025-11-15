"""
=============================================================================
VISION TRANSFORMER TRAINING ANALYSIS UTILITIES
=============================================================================

📊 PURPOSE: Load and analyze training data from all ViT implementations
🔍 FEATURES: 
  - Load training data from pickle files
  - Generate comprehensive accuracy curves (including EMA)
  - Compare different configurations
  - Statistical analysis of training progress
  - Export publication-ready plots
  - SOTA analysis with EMA tracking

🚀 USAGE:
  python analyze_training.py
  
📁 INPUT FILES:
  - runs_pytorch/*/training_data.pkl (assignment compliance)
  - runs_optimized_pytorch/training_data_complete.pkl (high performance)
  - runs_sota_pytorch/training_data_complete.pkl (SOTA optimization)
  - runs_sota_pytorch_optimized/training_data_complete.pkl (SOTA advanced)
=============================================================================
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Any

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingAnalyzer:
    def __init__(self, data_dirs: List[str] = None):
        if data_dirs is None:
            data_dirs = ["runs_pytorch", "runs_optimized_pytorch", "runs_sota_pytorch", "runs_sota_pytorch_optimized"]
        self.data_dirs = data_dirs
        self.training_data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all training data from pickle files"""
        print("📁 Loading training data...")
        
        # Load assignment compliance data
        if os.path.exists("runs_pytorch"):
            for run_dir in Path("runs_pytorch").iterdir():
                if run_dir.is_dir():
                    pkl_file = run_dir / "training_data.pkl"
                    if pkl_file.exists():
                        with open(pkl_file, 'rb') as f:
                            data = pickle.load(f)
                            self.training_data[run_dir.name] = data
                            print(f"   ✅ Loaded: {run_dir.name}")
        
        # Load high-performance data
        hp_file = "runs_optimized_pytorch/training_data_complete.pkl"
        if os.path.exists(hp_file):
            with open(hp_file, 'rb') as f:
                data = pickle.load(f)
                self.training_data['high_performance'] = data
                print(f"   ✅ Loaded: high_performance")
        
        # Load SOTA optimization data
        sota_file = "runs_sota_pytorch/training_data_complete.pkl"
        if os.path.exists(sota_file):
            with open(sota_file, 'rb') as f:
                data = pickle.load(f)
                self.training_data['sota_optimization'] = data
                print(f"   ✅ Loaded: sota_optimization")
        
        # Load SOTA optimization advanced data
        sota_opt_file = "runs_sota_pytorch_optimized/training_data_complete.pkl"
        if os.path.exists(sota_opt_file):
            with open(sota_opt_file, 'rb') as f:
                data = pickle.load(f)
                self.training_data['sota_optimized'] = data
                print(f"   ✅ Loaded: sota_optimized")
        
        print(f"📊 Total experiments loaded: {len(self.training_data)}")
    
    def plot_accuracy_curves(self, save_path: str = "analysis_accuracy_curves.png"):
        """Generate comprehensive accuracy curves for all experiments"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Vision Transformer Training Analysis - CIFAR-100', fontsize=16, fontweight='bold')
        
        # Plot 1: All training accuracies
        ax1 = axes[0, 0]
        for exp_name, data in self.training_data.items():
            history = data['history']
            epochs = range(1, len(history['train_acc']) + 1)
            ax1.plot(epochs, history['train_acc'], label=f"{exp_name} (train)", alpha=0.8)
        
        ax1.set_title('Training Accuracy Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: All validation accuracies (including EMA)
        ax2 = axes[0, 1]
        for exp_name, data in self.training_data.items():
            history = data['history']
            epochs = range(1, len(history['val_acc']) + 1)
            ax2.plot(epochs, history['val_acc'], label=f"{exp_name} (val)", alpha=0.8)
            
            # Plot EMA accuracy if available (SOTA model)
            if 'ema_val_acc' in history and history['ema_val_acc']:
                ema_epochs = range(1, len(history['ema_val_acc']) + 1)
                ax2.plot(ema_epochs, history['ema_val_acc'], label=f"{exp_name} (EMA)", 
                        alpha=0.8, linestyle='--')
        
        ax2.set_title('Validation Accuracy Curves (+ EMA)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Loss curves
        ax3 = axes[1, 0]
        for exp_name, data in self.training_data.items():
            history = data['history']
            epochs = range(1, len(history['train_loss']) + 1)
            ax3.plot(epochs, history['train_loss'], label=f"{exp_name} (train)", alpha=0.7)
            ax3.plot(epochs, history['val_loss'], label=f"{exp_name} (val)", alpha=0.7, linestyle='--')
        
        ax3.set_title('Loss Curves')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Plot 4: Final accuracy comparison
        ax4 = axes[1, 1]
        exp_names = []
        final_accs = []
        
        for exp_name, data in self.training_data.items():
            if 'final_results' in data:
                exp_names.append(exp_name)
                # Use EMA accuracy if available (SOTA), otherwise regular accuracy
                final_acc = (data['final_results'].get('best_ema_accuracy') or 
                           data['final_results'].get('test_acc') or
                           data['final_results'].get('final_accuracy', 0))
                final_accs.append(final_acc)
        
        bars = ax4.bar(exp_names, final_accs, alpha=0.8)
        ax4.set_title('Final Test Accuracy Comparison')
        ax4.set_ylabel('Accuracy (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Accuracy curves saved: {save_path}")
    
    def generate_summary_table(self, save_path: str = "analysis_summary.csv"):
        """Generate comprehensive summary table"""
        
        summary_data = []
        
        for exp_name, data in self.training_data.items():
            config = data.get('config', {})
            final = data.get('final_results', {})
            history = data.get('history', {})
            
            summary_data.append({
                'Experiment': exp_name,
                'Patch_Size': config.get('patch_size', 'N/A'),
                'Heads': config.get('heads', config.get('num_heads', 'N/A')),
                'Blocks': config.get('blocks', config.get('num_layers', 'N/A')),
                'Embed_Dim': config.get('embed_dim', 'N/A'),
                'Total_Params': config.get('total_params', 'N/A'),
                'Final_Test_Acc': final.get('test_acc', final.get('final_accuracy', 0)),
                'Best_Val_Acc': final.get('best_val_acc', max(history.get('val_acc', [0]))),
                'Best_EMA_Acc': final.get('best_ema_accuracy', 'N/A'),  # SOTA EMA accuracy
                'Best_Train_Acc': final.get('best_train_acc', max(history.get('train_acc', [0]))),
                'Training_Time_Min': final.get('training_time_min', 0),
                'Epochs_Trained': final.get('epochs_trained', len(history.get('train_acc', []))),
                'Final_Train_Loss': final.get('final_train_loss', 0),
                'Final_Val_Loss': final.get('final_val_loss', 0)
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Final_Test_Acc', ascending=False)
        df.to_csv(save_path, index=False)
        
        print(f"\n📊 EXPERIMENT SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        print(f"\n💾 Summary saved: {save_path}")
        
        return df
    
    def plot_learning_rate_analysis(self, save_path: str = "analysis_lr_curves.png"):
        """Plot learning rate schedules if available"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Learning rate curves
        ax1 = axes[0]
        for exp_name, data in self.training_data.items():
            history = data['history']
            if 'lr' in history and history['lr']:
                epochs = range(1, len(history['lr']) + 1)
                ax1.plot(epochs, history['lr'], label=exp_name, alpha=0.8)
        
        ax1.set_title('Learning Rate Schedules')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Learning Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Convergence analysis
        ax2 = axes[1]
        for exp_name, data in self.training_data.items():
            history = data['history']
            if 'val_acc' in history and history['val_acc']:
                # Find convergence point (where validation stops improving significantly)
                val_acc = np.array(history['val_acc'])
                smoothed = np.convolve(val_acc, np.ones(5)/5, mode='valid')
                convergence_epoch = len(smoothed) - np.argmax(smoothed[::-1]) if len(smoothed) > 5 else len(val_acc)
                
                ax2.scatter(convergence_epoch, max(val_acc), 
                           label=f"{exp_name} (epoch {convergence_epoch})", s=100, alpha=0.8)
        
        ax2.set_title('Convergence Analysis')
        ax2.set_xlabel('Convergence Epoch')
        ax2.set_ylabel('Best Validation Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Learning rate analysis saved: {save_path}")
    
    def export_training_data(self, save_path: str = "all_training_data.xlsx"):
        """Export all training data to Excel for detailed analysis"""
        
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_df = self.generate_summary_table("temp_summary.csv")
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual experiment sheets
            for exp_name, data in self.training_data.items():
                history = data['history']
                df_history = pd.DataFrame(history)
                df_history.index.name = 'Epoch'
                df_history.to_excel(writer, sheet_name=exp_name[:30])  # Excel sheet name limit
        
        print(f"📊 Complete training data exported: {save_path}")
        
        # Clean up temp file
        if os.path.exists("temp_summary.csv"):
            os.remove("temp_summary.csv")

def main():
    """Main analysis function"""
    print("🚀 Starting Training Analysis...")
    
    analyzer = TrainingAnalyzer()
    
    if not analyzer.training_data:
        print("❌ No training data found!")
        print("   Make sure you have run the training scripts first.")
        return
    
    # Generate all analyses
    analyzer.plot_accuracy_curves("training_analysis_curves.png")
    analyzer.generate_summary_table("training_analysis_summary.csv") 
    analyzer.plot_learning_rate_analysis("training_analysis_lr.png")
    analyzer.export_training_data("training_analysis_complete.xlsx")
    
    print("\n✅ Analysis complete! Files generated:")
    print("   📊 training_analysis_curves.png - Accuracy and loss curves")
    print("   📊 training_analysis_lr.png - Learning rate analysis")
    print("   📊 training_analysis_summary.csv - Summary table")
    print("   📊 training_analysis_complete.xlsx - Complete data export")

if __name__ == "__main__":
    main()