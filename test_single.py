#!/usr/bin/env python3
"""
Test single ViT configuration to debug issues
"""
import sys
sys.path.append('/home/diegoav-lx/Documentos/ufam/visual-transformer')

# Import from main file
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Simple test with one configuration
def test_single_config():
    # Load small subset of data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")
    x_train = x_train[:2000].astype("float32") / 255.0
    x_test = x_test[:400].astype("float32") / 255.0
    y_train = y_train[:2000].squeeze()
    y_test = y_test[:400].squeeze()

    # Import the functions from train.py
    from train import create_vit

    print("Creating ViT model...")
    model = create_vit(
        input_shape=(32, 32, 3),
        patch_size=6,
        num_classes=100,
        projection_dim=128,  # Small but reasonable size
        num_heads=4,
        num_layers=4,  # Fewer layers for testing
        mlp_dim=256,
        dropout=0.1,
    )

    print(f"Model parameters: {model.count_params():,}")

    # Use simpler optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("Starting training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=64,
        verbose=2
    )

    final_acc = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_acc:.4f}")
    
    if final_acc > 0.02:  # At least better than random
        print("✅ Model is learning!")
        return True
    else:
        print("❌ Model not learning properly")
        return False

if __name__ == "__main__":
    success = test_single_config()
    print("Test completed!")