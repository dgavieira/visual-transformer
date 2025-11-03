#!/usr/bin/env python3
"""
Simple test to verify ViT is working
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Simple ViT test
def create_simple_vit():
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Simple patch extraction
    patches = layers.Conv2D(64, kernel_size=6, strides=6)(inputs)
    patches = layers.Reshape((-1, 64))(patches)
    
    # Add positional embedding
    seq_len = patches.shape[1]
    pos_embed = layers.Embedding(input_dim=100, output_dim=64)
    positions = tf.range(start=0, limit=seq_len, delta=1)
    patches = patches + pos_embed(positions)
    
    # Simple transformer block
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16)(patches, patches)
    x = layers.Add()([patches, attn])
    x = layers.LayerNormalization()(x)
    
    # Classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(100, activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs)
    return model

if __name__ == "__main__":
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")
    x_train = x_train[:1000].astype("float32") / 255.0  # Small subset for testing
    x_test = x_test[:200].astype("float32") / 255.0
    y_train = y_train[:1000].squeeze()
    y_test = y_test[:200].squeeze()
    
    # Create model
    model = create_simple_vit()
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # Test training for a few epochs
    print("Testing training...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=3,
        batch_size=32,
        verbose=1
    )
    
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print("✅ Simple ViT test completed successfully!")