"""
High-Performance Vision Transformer (ViT) for CIFAR-100 - SIMPLIFIED VERSION
Target: 90%+ Validation Accuracy
----------------------------------------------------
Key improvements:
- Stronger data augmentation
- Larger model with proven architecture
- Better optimization
- Label smoothing
- Longer training
"""

import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

# =====================================================
# OPTIMIZED CONFIGURATIONS
# =====================================================
OUTPUT_DIR = "runs_optimized"
EPOCHS = 150         # More epochs for better convergence
BATCH_SIZE = 64      # Smaller batch for memory efficiency
PATCH_SIZE = 4       # Optimal for 32x32 images
LEARNING_RATE = 5e-4 # Slightly lower initial LR
WARMUP_EPOCHS = 10
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# =====================================================
# ADVANCED DATA AUGMENTATION
# =====================================================

def create_strong_augmentation():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15, 0.15),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        # Add some color jittering
        layers.Lambda(lambda x: tf.image.random_hue(x, 0.1)),
        layers.Lambda(lambda x: tf.image.random_saturation(x, 0.7, 1.3)),
    ])

class MixUp(layers.Layer):
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        
    def call(self, inputs, training=None):
        if not training:
            return inputs
            
        x, y = inputs
        batch_size = tf.shape(x)[0]
        
        # Sample lambda from Beta distribution
        alpha = tf.cast(self.alpha, x.dtype)
        lambda_val = tf.random.uniform([], 0.0, alpha, dtype=x.dtype)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        
        # Mix inputs and labels
        x_mixed = lambda_val * x + (1.0 - lambda_val) * tf.gather(x, indices)
        y_mixed = lambda_val * y + (1.0 - lambda_val) * tf.gather(y, indices)
        
        return x_mixed, y_mixed

# =====================================================
# LOAD AND PREPARE DATA
# =====================================================
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Convert to one-hot for label smoothing and mixup
y_train_onehot = tf.one_hot(y_train, 100)
y_test_onehot = tf.one_hot(y_test, 100)

# Create augmentation pipeline
strong_aug = create_strong_augmentation()
mixup = MixUp(alpha=0.2)

def train_preprocess(x, y):
    x = strong_aug(x, training=True)
    return x, y

def mixup_preprocess(x, y):
    return mixup([x, y], training=True)

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train_onehot))
    .shuffle(10000)
    .map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(mixup_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test_onehot))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# =====================================================
# PROVEN ViT ARCHITECTURE
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
            kernel_initializer=keras.initializers.LecunNormal()
        )
        self.reshape = layers.Reshape((-1, self.projection_dim))
        
    def call(self, x):
        x = self.projection(x)
        x = self.reshape(x)
        return x

class AddClassToken(layers.Layer):
    def __init__(self, projection_dim):
        super().__init__()
        self.projection_dim = projection_dim
        
    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.projection_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        return tf.concat([cls_tokens, x], axis=1)

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, projection_dim):
        super().__init__()
        self.sequence_length = sequence_length
        self.projection_dim = projection_dim
        
    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.sequence_length, self.projection_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )
        
    def call(self, x):
        length = tf.shape(x)[1]
        return x + self.position_embedding[:length, :]

def create_optimized_vit(
    input_shape=(32, 32, 3),
    patch_size=4,
    num_classes=100,
    projection_dim=512,  # Larger embedding dimension
    num_heads=16,        # More attention heads
    num_layers=16,       # Deeper network
    mlp_dim=2048,        # 4x projection_dim
    dropout_rate=0.1,
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
    for i in range(num_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout_rate
        )(x1, x1)
        
        # Skip connection 1
        x = layers.Add()([x, attention_output])
        
        # Layer normalization 2
        x2 = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # MLP block
        mlp_output = layers.Dense(mlp_dim, activation="gelu")(x2)
        mlp_output = layers.Dropout(dropout_rate)(mlp_output)
        mlp_output = layers.Dense(projection_dim)(mlp_output)
        mlp_output = layers.Dropout(dropout_rate)(mlp_output)
        
        # Skip connection 2
        x = layers.Add()([x, mlp_output])
    
    # Final layer norm
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Classification head
    cls_output = x[:, 0, :]
    
    # Final MLP head with hidden layer
    x = layers.Dense(mlp_dim // 2, activation="gelu")(cls_output)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(mlp_dim // 4, activation="gelu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# =====================================================
# TRAINING SETUP
# =====================================================

def create_lr_schedule():
    """Warmup + cosine decay with restarts"""
    def lr_fn(epoch):
        if epoch < WARMUP_EPOCHS:
            return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
        else:
            # Cosine decay with occasional restarts
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))
    
    return keras.callbacks.LearningRateScheduler(lr_fn, verbose=1)

def create_callbacks(run_dir):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(run_dir, "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        create_lr_schedule(),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=8,
            min_lr=1e-7,
            verbose=1
        )
    ]

# =====================================================
# MAIN TRAINING
# =====================================================

def train_optimized_model():
    print("🚀 Creating optimized ViT model...")
    
    model = create_optimized_vit(
        projection_dim=512,
        num_heads=16,
        num_layers=16,
        mlp_dim=2048,
        dropout_rate=0.1
    )
    
    print(f"📊 Model parameters: {model.count_params():,}")
    
    # Compile with optimized settings
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=5e-5,
            clipnorm=1.0
        ),
        loss=keras.losses.CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=0.1
        ),
        metrics=["accuracy"]
    )
    
    # Create run directory
    run_dir = os.path.join(OUTPUT_DIR, "optimized_vit_large")
    os.makedirs(run_dir, exist_ok=True)
    
    print("🏃 Starting training...")
    start_time = time.time()
    
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=create_callbacks(run_dir),
        verbose=1,
    )
    
    training_time = time.time() - start_time
    
    # Final evaluation
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Validation Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Training Time: {training_time/60:.1f} minutes")
    print(f"   Parameters: {model.count_params():,}")
    
    if test_acc > 0.90:
        print("🎉 SUCCESS! Achieved 90%+ accuracy!")
    elif test_acc > 0.80:
        print("🚀 Good progress! 80%+ achieved. Continue training or increase model size.")
    else:
        print("⚠️  Need more improvements. Consider increasing training time or model capacity.")
    
    return model, history, test_acc

if __name__ == "__main__":
    model, history, accuracy = train_optimized_model()
    print(f"\n✅ Training completed! Final accuracy: {accuracy*100:.2f}%")