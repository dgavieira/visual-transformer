"""
Fixed Vision Transformer (ViT) for CIFAR-100
Based on proven architectures with proper initialization
"""

import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# =====================================================
# CONFIGURAÇÕES GERAIS
# =====================================================
OUTPUT_DIR = "runs_fixed"
EPOCHS = 20
BATCH_SIZE = 64      # Smaller batch size for better convergence
PATCH_SIZE = 4       # Smaller patches for CIFAR-32
LEARNING_RATE = 3e-4
SEED = 42

tf.random.set_seed(SEED)

# =====================================================
# CARREGAR E PREPARAR DADOS CIFAR-100
# =====================================================
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Simple data augmentation
def augment_data(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, 0.1)
    return x, y

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# =====================================================
# SIMPLIFIED ViT IMPLEMENTATION
# =====================================================

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, projection_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        
    def build(self, input_shape):
        self.projection = layers.Conv2D(
            filters=self.projection_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='valid',
            kernel_initializer='he_normal'
        )
        self.reshape = layers.Reshape((-1, self.projection_dim))
        
    def call(self, x):
        x = self.projection(x)
        x = self.reshape(x)
        return x

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, projection_dim):
        super().__init__()
        self.sequence_length = sequence_length
        self.projection_dim = projection_dim
        
    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.sequence_length, self.projection_dim),
            initializer="random_normal",
        )
        
    def call(self, x):
        length = tf.shape(x)[1]
        return x + self.position_embedding[:length, :]

class AddClassToken(layers.Layer):
    def __init__(self, projection_dim):
        super().__init__()
        self.projection_dim = projection_dim
        
    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.projection_dim),
            initializer="random_normal",
        )
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        return tf.concat([cls_tokens, x], axis=1)

def create_vit_v2(
    input_shape=(32, 32, 3),
    patch_size=4,
    num_classes=100,
    projection_dim=128,
    num_heads=8,
    num_layers=6,
    mlp_dim=256,
    dropout=0.1,
):
    inputs = layers.Input(shape=input_shape)
    
    # Create patches
    patches = PatchEmbedding(patch_size, projection_dim)(inputs)
    
    # Calculate sequence length
    num_patches = (input_shape[0] // patch_size) ** 2
    
    # Add class token
    x = AddClassToken(projection_dim)(patches)
    
    # Add positional embeddings
    x = PositionalEmbedding(num_patches + 1, projection_dim)(x)
    
    # Transformer blocks
    for _ in range(num_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=projection_dim // num_heads,
            dropout=dropout
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, x])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = layers.Dense(mlp_dim, activation="gelu")(x3)
        x3 = layers.Dropout(dropout)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(dropout)(x3)
        
        # Skip connection 2
        x = layers.Add()([x3, x2])
    
    # Final layer norm
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Classification head (use CLS token)
    cls_output = x[:, 0, :]
    
    # Final dense layers
    x = layers.Dense(mlp_dim, activation="gelu")(cls_output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# =====================================================
# TESTING CONFIGURATIONS
# =====================================================

def test_config(config_name, **model_params):
    print(f"\n==== Testing {config_name} ====")
    
    model = create_vit_v2(**model_params)
    print(f"Parameters: {model.count_params():,}")
    
    # Learning rate schedule
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * len(train_ds),
    )
    
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Callbacks
    run_dir = os.path.join(OUTPUT_DIR, config_name)
    os.makedirs(run_dir, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", 
            patience=5, 
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[{config_name}] Final accuracy: {test_acc:.4f}")
    
    return {
        "config": config_name,
        "test_acc": test_acc,
        "params": model.count_params(),
        "epochs": len(history.history["accuracy"])
    }

if __name__ == "__main__":
    results = []
    
    # Test small configuration first
    result = test_config(
        "small_vit",
        patch_size=4,
        projection_dim=64,
        num_heads=4,
        num_layers=4,
        mlp_dim=128
    )
    results.append(result)
    
    # If small works, test medium
    if result["test_acc"] > 0.05:  # Better than 5%
        result = test_config(
            "medium_vit", 
            patch_size=4,
            projection_dim=128,
            num_heads=8,
            num_layers=6,
            mlp_dim=256
        )
        results.append(result)
    
    # Print results
    print("\n===== FINAL RESULTS =====")
    for r in results:
        print(f"{r['config']}: {r['test_acc']:.4f} acc, {r['params']:,} params")