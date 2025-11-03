# Vision Transformer for CIFAR-100

This repository contains two optimized PyTorch implementations of Vision Transformer for CIFAR-100 classification, plus comprehensive training analytics.

## 📁 Files

### 1. `train_pytorch.py` - Assignment Compliance
- **Purpose**: Strictly follows academic assignment requirements
- **Configuration**: 6x6 patches, [4,6] heads, [8,12] blocks as specified
- **Target**: Meet assignment criteria with good baseline performance
- **Usage**: `python train_pytorch.py`

### 2. `train_high_performance_pytorch.py` - Maximum Performance  
- **Purpose**: Achieve highest possible accuracy using RTX 5090
- **Configuration**: Optimized 4x4 patches, 16 heads, 16 layers, 512 embed_dim
- **Target**: 90%+ validation accuracy through advanced techniques
- **Usage**: `python train_high_performance_pytorch.py`

### 3. Training Analytics Tools

#### `plot_curves.py` - Simple Curve Plotting
- **Purpose**: Quick visualization of training curves
- **Features**: Load saved data and generate comparison plots
- **Usage**: `python plot_curves.py` (after training)

#### `analyze_training.py` - Comprehensive Analysis  
- **Purpose**: Full statistical analysis and reporting
- **Features**: Advanced plots, Excel export, convergence analysis
- **Usage**: `python analyze_training.py` (requires seaborn)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run assignment compliance training
python train_pytorch.py

# Run high-performance training  
python train_high_performance_pytorch.py

# Analyze results
python plot_curves.py
```

## 📊 Training Data Persistence

Both training scripts automatically save:

### Saved Files per Experiment:
- **`training_data.pkl`** - Complete training history and configuration
- **`metrics.csv`** - Epoch-by-epoch metrics for easy analysis
- **`accuracy_loss_curves.png`** - Training curves visualization
- **`best_model.pth`** - Best model checkpoint

### Directory Structure:
```
runs_pytorch/
├── h4_b8_p6/          # 4 heads, 8 blocks experiment
│   ├── training_data.pkl
│   ├── metrics.csv
│   ├── accuracy_loss_curves.png
│   └── best_model.pth
├── h4_b12_p6/         # 4 heads, 12 blocks experiment
├── h6_b8_p6/          # 6 heads, 8 blocks experiment
├── h6_b12_p6/         # 6 heads, 12 blocks experiment
└── summary.csv        # Consolidated results

runs_optimized_pytorch/
├── training_data_complete.pkl
├── training_metrics.csv
├── best_model.pth
└── ...
```

## 📊 Analytics Capabilities

### Training Curve Analysis:
- **Accuracy progression** (training vs validation)
- **Loss convergence** (with log scale)
- **Learning rate schedules**
- **Convergence analysis**

### Data Export Options:
- **CSV files** for spreadsheet analysis
- **Pickle files** for Python analysis
- **Excel files** with multiple sheets
- **PNG plots** for presentations

### Comparison Features:
- **Side-by-side** experiment comparison
- **Statistical summaries** 
- **Performance ranking**
- **Parameter efficiency analysis**

## 📊 Expected Results

- **Assignment version**: 60-75% accuracy (meets requirements)
- **High-performance version**: 85-92% accuracy (state-of-the-art)

## 🔧 Requirements

```bash
pip install torch torchvision pandas matplotlib numpy
# Optional for advanced analysis:
pip install seaborn openpyxl scikit-learn tensorboard
```

Both scripts work with RTX 5090 and automatically use GPU when available.

## 📈 Post-Training Analysis

After training completes, use the analytics tools:

```bash
# Quick visualization (no extra dependencies)
python plot_curves.py

# Full analysis (requires seaborn, openpyxl)
python analyze_training.py
```

This generates publication-ready plots and comprehensive analysis reports for your training sessions.