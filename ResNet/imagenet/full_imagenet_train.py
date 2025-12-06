"""
Training script for ResNet with and without Stochastic Depth on Full ImageNet (ILSVRC 2012)

This script implements training of ResNet models on the full ImageNet dataset with options to:
- Train with or without Stochastic Depth regularization
- Use modern training techniques for best accuracy
- Support ResNet-50 and ResNet-101 architectures
- Leverage mixed precision training for efficiency

Expected dataset structure:
    data_dir/
        train/
            n01440764/
                n01440764_10026.JPEG
                ...
            n01443537/
                ...
        val/
            n01440764/
                ILSVRC2012_val_00000293.JPEG
                ...
            n01443537/
                ...

Usage:
    # Train ResNet-50 without Stochastic Depth (baseline)
    python full_imagenet_train.py --data_dir /path/to/imagenet --out ./results_baseline --sd 0
    
    # Train ResNet-50 with Stochastic Depth
    python full_imagenet_train.py --data_dir /path/to/imagenet --out ./results_sd --sd 1 --pL 0.5
    
    # Train ResNet-101 with Stochastic Depth and advanced techniques
    python full_imagenet_train.py --data_dir /path/to/imagenet --depth 101 --sd 1 --pL 0.5 \\
        --mixup 0.2 --label_smoothing 0.1 --epochs 120 --batch 256
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from datasets import load_dataset

# ----------------------------
# Default Hyperparameters (optimized for ImageNet)
# ----------------------------
DEFAULT_IMAGE_SIZE = 224
DEFAULT_NUM_CLASSES = 1000
DEFAULT_BATCH = 256  # for good hardware, can use 512 or 1024
DEFAULT_EPOCHS = 120  # standard for ImageNet with modern techniques
DEFAULT_LR = 0.1  # base learning rate for batch_size=256
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
SHUFFLE_BUFFER = 10000  # for ImageNet can be smaller


# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class ModelConfig:
    """Configuration for ResNet ImageNet training."""
    output_dir: str
    data_dir: Optional[str] = None  # No longer required - using Hugging Face datasets
    image_size: int = DEFAULT_IMAGE_SIZE
    num_classes: int = DEFAULT_NUM_CLASSES
    depth: int = 50  # 50 or 101 for standard ImageNet ResNets
    sd_on: bool = False
    final_survival_prob: float = 0.5
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH
    base_lr: float = DEFAULT_LR
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0  # CutMix augmentation (0 to disable)
    randaugment_n: int = 0  # RandAugment number of transformations (0 to disable)
    randaugment_m: int = 9  # RandAugment magnitude
    label_smoothing: float = 0.1
    use_mixed_precision: bool = True
    use_ema: bool = False  # Exponential Moving Average of model weights
    ema_decay: float = 0.9999  # EMA decay rate
    grad_clip_norm: float = 0.0  # Gradient clipping (0 to disable)
    cache_dataset: bool = False
    seed: int = 42
    auto_scale_lr: bool = True  # Whether to auto-scale LR based on batch size
    
    def __post_init__(self):
        if self.depth not in [50, 101, 152]:
            raise ValueError("Depth must be 50, 101, or 152 for ImageNet ResNet")
        # Scale learning rate with batch size if auto-scaling is enabled
        # Following the linear scaling rule: lr = base_lr * (batch_size / 256)
        if self.auto_scale_lr:
            self.base_lr = 0.1 * (self.batch_size / 256.0)
            print(f"Auto-scaled learning rate: {self.base_lr} (for batch size {self.batch_size})")
        else:
            print(f"Using learning rate: {self.base_lr}")


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


class LrCallback(keras.callbacks.Callback):
    """Callback to log learning rate at each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        lr = float(
            self.model.optimizer.learning_rate(self.model.optimizer.iterations).numpy()
        )
        print(f"LR at epoch {epoch + 1}: {lr:.8f}")


# ----------------------------
# Data loading & augmentation for ImageNet using Hugging Face datasets
# ----------------------------
def make_datasets(config: ModelConfig):
    """Create training and validation datasets for ImageNet using Hugging Face datasets.
    
    Loads ImageNet from Hugging Face Hub (ILSVRC/imagenet-1k) and applies:
    - Training: RandomResizedCrop, RandomFlip, ColorJitter, Normalization
    - Validation: Resize, CenterCrop, Normalization
    """
    seed = config.seed
    image_size = config.image_size
    
    print("Loading ImageNet dataset from Hugging Face Hub...")
    print("Note: First time loading will download ~150GB. Subsequent runs will use cache.")
    
    # Load ImageNet from Hugging Face
    # Login using `huggingface-cli login` to access this dataset
    try:
        dataset = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        print(f"✓ Loaded ImageNet: {len(train_dataset)} train, {len(val_dataset)} validation images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTo use this dataset, you need to:")
        print("1. Accept the terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("2. Login with: huggingface-cli login")
        raise
    
    AUTOTUNE = tf.data.AUTOTUNE
    
    # ImageNet normalization (mean and std from ImageNet statistics)
    imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    
    def prepare_image(example):
        """Convert HF dataset example to TensorFlow tensors."""
        # Get image and label
        image = example["image"]
        label = example["label"]
        
        # Convert PIL image to numpy array
        image = np.array(image.convert('RGB'))
        
        # Convert to TensorFlow tensor
        image = tf.constant(image, dtype=tf.uint8)
        label = tf.constant(label, dtype=tf.int32)
        
        return image, label
    
    def augment_train(image, label):
        """Training augmentation pipeline."""
        # Resize and random crop
        image = tf.image.resize(image, [image_size + 32, image_size + 32])
        image = tf.image.random_crop(image, [image_size, image_size, 3])
        
        # Convert to float and scale to [0,1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Apply RandAugment if enabled
        if config.randaugment_n > 0:
            image = rand_augment_transform(image, config.randaugment_n, config.randaugment_m)
        else:
            # Standard color jittering (brightness, contrast, saturation)
            image = tf.image.random_brightness(image, 0.4)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            image = tf.image.random_saturation(image, 0.8, 1.2)
            image = tf.clip_by_value(image, 0.0, 1.0)
        
        # Normalize with ImageNet statistics
        mean = tf.cast(imagenet_mean, image.dtype)
        std = tf.cast(imagenet_std, image.dtype)
        image = (image - mean) / std
        
        # Convert labels to one-hot for label smoothing and mixup
        label = tf.one_hot(label, config.num_classes)
        
        return image, label
    
    def augment_val(image, label):
        """Validation preprocessing (no augmentation)."""
        # Resize to slightly larger then center crop
        image = tf.image.resize(image, [int(image_size * 1.15), int(image_size * 1.15)])
        image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
        
        # Convert to float and scale to [0,1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Normalize with ImageNet statistics
        mean = tf.cast(imagenet_mean, image.dtype)
        std = tf.cast(imagenet_std, image.dtype)
        image = (image - mean) / std
        
        # Convert labels to one-hot
        label = tf.one_hot(label, config.num_classes)
        
        return image, label
    
    # Convert Hugging Face datasets to TensorFlow datasets
    # Shuffle the dataset once outside generator for efficiency
    train_dataset_shuffled = train_dataset.shuffle(seed=seed)
    
    def generator_train():
        for example in train_dataset_shuffled:
            yield prepare_image(example)
    
    def generator_val():
        for example in val_dataset:
            yield prepare_image(example)
    
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_generator(
        generator_train,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    val_ds = tf.data.Dataset.from_generator(
        generator_val,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    # Apply augmentation and batching
    train_ds = train_ds.map(augment_train, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(augment_val, num_parallel_calls=AUTOTUNE)
    
    # Batch and prefetch
    train_ds = train_ds.batch(config.batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(config.batch_size).prefetch(AUTOTUNE)
    
    # Optional caching (requires lots of RAM for ImageNet)
    if config.cache_dataset:
        print("⚠ Warning: Caching ImageNet requires ~150GB RAM")
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
    
    return train_ds, val_ds


def mixup(batch_x, batch_y, alpha=0.2):
    """MixUp data augmentation.
    
    Mixes pairs of examples and their labels.
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
    """
    if alpha <= 0.0:
        return batch_x, batch_y
    
    lam = np.random.beta(alpha, alpha)
    lam = float(lam)
    batch_size = tf.shape(batch_x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    
    mixed_x = lam * batch_x + (1 - lam) * tf.gather(batch_x, index)
    mixed_y = lam * batch_y + (1 - lam) * tf.gather(batch_y, index)
    
    return mixed_x, mixed_y


def cutmix(batch_x, batch_y, alpha=1.0):
    """CutMix data augmentation.
    
    Cuts and pastes patches between training images.
    Reference: "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)
    """
    if alpha <= 0.0:
        return batch_x, batch_y
    
    batch_size = tf.shape(batch_x)[0]
    image_h = tf.shape(batch_x)[1]
    image_w = tf.shape(batch_x)[2]
    
    # Sample lambda from Beta distribution
    lam = np.random.beta(alpha, alpha)
    
    # Sample random index for mixing
    index = tf.random.shuffle(tf.range(batch_size))
    
    # Sample bounding box
    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(image_h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(image_w, tf.float32) * cut_ratio, tf.int32)
    
    # Uniform random location
    cx = tf.random.uniform([], 0, image_w, dtype=tf.int32)
    cy = tf.random.uniform([], 0, image_h, dtype=tf.int32)
    
    # Bounding box coordinates
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_h)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_w)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_h)
    
    # Create mask
    mask_shape = tf.shape(batch_x)
    mask = tf.ones(mask_shape, dtype=batch_x.dtype)
    
    # Zero out the cut region
    updates = tf.zeros([batch_size, y2 - y1, x2 - x1, 3], dtype=batch_x.dtype)
    mask_slice = mask[:, y1:y2, x1:x2, :]
    mask = tf.tensor_scatter_nd_update(
        mask,
        [[i, y1, x1, 0] for i in range(batch_size)],
        tf.zeros([batch_size], dtype=batch_x.dtype)
    )
    
    # Apply CutMix
    mixed_x = mask * batch_x + (1 - mask) * tf.gather(batch_x, index)
    
    # Adjust lambda based on actual cut area
    actual_lam = 1.0 - (tf.cast((x2 - x1) * (y2 - y1), tf.float32) / 
                        tf.cast(image_h * image_w, tf.float32))
    
    # Mix labels
    mixed_y = actual_lam * batch_y + (1 - actual_lam) * tf.gather(batch_y, index)
    
    return mixed_x, mixed_y


def rand_augment_transform(image, num_layers=2, magnitude=9):
    """Apply RandAugment transformations.
    
    Reference: "RandAugment: Practical automated data augmentation" (Cubuk et al., 2020)
    """
    if num_layers <= 0:
        return image
    
    # Available augmentation operations
    augmentations = [
        lambda img: tf.image.random_brightness(img, magnitude / 30.0),
        lambda img: tf.image.random_contrast(img, 1 - magnitude / 30.0, 1 + magnitude / 30.0),
        lambda img: tf.image.random_saturation(img, 1 - magnitude / 30.0, 1 + magnitude / 30.0),
        lambda img: tf.image.random_hue(img, magnitude / 60.0),
    ]
    
    # Randomly select and apply num_layers augmentations
    for _ in range(num_layers):
        op_idx = tf.random.uniform([], 0, len(augmentations), dtype=tf.int32)
        
        # Apply selected operation
        for idx, aug_fn in enumerate(augmentations):
            image = tf.cond(
                tf.equal(op_idx, idx),
                lambda: aug_fn(image),
                lambda: image
            )
    
    return tf.clip_by_value(image, 0.0, 1.0)


# ----------------------------
# Stochastic Depth Layer
# ----------------------------
class StochasticDepth(layers.Layer):
    """Implements Stochastic Depth (Drop Path) regularization.
    
    Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016)
    """
    def __init__(self, survival_prob=1.0, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob
    
    def call(self, x, residual, training=False):
        """Apply stochastic depth.
        
        Args:
            x: Input (identity path)
            residual: Residual branch output
            training: Whether in training mode
        
        Returns:
            x + residual (with probability survival_prob during training)
            x + survival_prob * residual (during inference)
        """
        if training:
            # Randomly drop entire residual branch
            random_val = tf.random.uniform([], 0, 1)
            return tf.cond(
                random_val < self.survival_prob,
                lambda: x + residual,
                lambda: x
            )
        else:
            # Use expected value at test time
            return x + self.survival_prob * residual
    
    def get_config(self):
        config = super().get_config()
        config["survival_prob"] = self.survival_prob
        return config


# ----------------------------
# ResNet Building Blocks (Bottleneck for ImageNet)
# ----------------------------
def bottleneck_block(x, filters, stride, survival_prob, weight_decay, downsample=False):
    """Bottleneck residual block for ImageNet ResNet.
    
    Uses 1x1 -> 3x3 -> 1x1 convolutions with expansion factor of 4.
    
    Args:
        x: Input tensor
        filters: Number of filters in the 3x3 conv (output will be filters * 4)
        stride: Stride for the 3x3 conv (used for downsampling)
        survival_prob: Probability of keeping the residual branch (for stochastic depth)
        weight_decay: L2 regularization factor
        downsample: Whether this block downsamples (changes dimension)
    
    Returns:
        Output tensor after residual connection
    """
    expansion = 4
    shortcut = x
    
    # Pre-activation: BN -> ReLU before convolution
    x_preact = layers.BatchNormalization()(x)
    x_preact = layers.ReLU()(x_preact)
    
    # 1x1 conv to reduce dimension
    y = layers.Conv2D(
        filters, 1, strides=1, padding='same',
        use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(x_preact)
    
    # 3x3 conv
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(
        filters, 3, strides=stride, padding='same',
        use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(y)
    
    # 1x1 conv to expand dimension
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(
        filters * expansion, 1, strides=1, padding='same',
        use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(y)
    
    # Shortcut connection with projection if needed
    # In full pre-activation, projection also uses pre-activated input
    if downsample or x.shape[-1] != filters * expansion:
        shortcut = layers.Conv2D(
            filters * expansion, 1, strides=stride, padding='same',
            use_bias=False, kernel_initializer='he_normal',
            kernel_regularizer=keras.regularizers.l2(weight_decay)
        )(x_preact)
    
    # Apply stochastic depth or regular residual connection
    if survival_prob < 1.0:
        out = StochasticDepth(survival_prob)(shortcut, y)
    else:
        out = layers.Add()([shortcut, y])
    
    return out


# ----------------------------
# Build ResNet for ImageNet
# ----------------------------
def build_resnet_imagenet(config: ModelConfig):
    """Build ResNet-50, ResNet-101, or ResNet-152 for ImageNet.
    
    Architecture:
    - Input: 224x224x3
    - Conv1: 7x7, 64 filters, stride 2
    - MaxPool: 3x3, stride 2
    - Stage 1: [1x1,64; 3x3,64; 1x1,256] x n1, stride 1
    - Stage 2: [1x1,128; 3x3,128; 1x1,512] x n2, stride 2 for first block
    - Stage 3: [1x1,256; 3x3,256; 1x1,1024] x n3, stride 2 for first block
    - Stage 4: [1x1,512; 3x3,512; 1x1,2048] x n4, stride 2 for first block
    - Global Average Pooling
    - FC 1000
    
    Args:
        config: ModelConfig instance
    
    Returns:
        Keras Model
    """
    # Number of blocks per stage for different ResNet variants
    stage_blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }
    
    if config.depth not in stage_blocks:
        raise ValueError(f"Unsupported depth: {config.depth}. Choose from {list(stage_blocks.keys())}")
    
    blocks_per_stage = stage_blocks[config.depth]
    
    # Calculate total number of blocks for stochastic depth
    total_blocks = sum(blocks_per_stage)
    
    inputs = keras.Input((config.image_size, config.image_size, 3))
    
    # Initial convolution and pooling
    x = layers.Conv2D(
        64, 7, strides=2, padding='same',
        use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(config.weight_decay)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Build stages
    block_idx = 0
    filters_list = [64, 128, 256, 512]
    
    for stage_idx, (num_blocks, filters) in enumerate(zip(blocks_per_stage, filters_list)):
        for block_in_stage in range(num_blocks):
            # First block in stage (except stage 0) downsamples
            stride = 2 if stage_idx > 0 and block_in_stage == 0 else 1
            downsample = (block_in_stage == 0)
            
            # Calculate survival probability for this block
            if config.sd_on:
                # Linear decay: survival_prob = 1 - (block_idx / total_blocks) * (1 - final_survival_prob)
                survival_prob = 1.0 - (block_idx / total_blocks) * (1.0 - config.final_survival_prob)
            else:
                survival_prob = 1.0
            
            x = bottleneck_block(x, filters, stride, survival_prob, config.weight_decay, downsample)
            block_idx += 1
    
    # Final layers
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Output layer (keep in float32 for numerical stability with mixed precision)
    outputs = layers.Dense(
        config.num_classes, activation="softmax",
        kernel_regularizer=keras.regularizers.l2(config.weight_decay),
        dtype="float32"
    )(x)
    
    model = keras.Model(inputs, outputs, name=f"ResNet{config.depth}")
    return model


# ----------------------------
# Learning rate schedule
# ----------------------------
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup and cosine decay.
    
    Reference: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (Goyal et al., 2017)
    """
    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=1e-6):
        super().__init__()
        self.base_lr = tf.cast(base_lr, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        
        # Avoid division by zero
        warmup_denom = tf.maximum(self.warmup_steps, 1.0)
        cosine_denom = tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        
        # Linear warmup
        warmup_lr = (step / warmup_denom) * self.base_lr
        
        # Cosine decay
        cosine_steps = tf.maximum(step - self.warmup_steps, 0.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * cosine_steps / cosine_denom))
        decayed_lr = (self.base_lr - self.min_lr) * cosine_decay + self.min_lr
        
        # Select warmup or decay
        lr = tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)
        return lr
    
    def get_config(self):
        return {
            'base_lr': float(self.base_lr.numpy()),
            'total_steps': float(self.total_steps.numpy()),
            'warmup_steps': float(self.warmup_steps.numpy()),
            'min_lr': float(self.min_lr.numpy())
        }


# ----------------------------
# EMA (Exponential Moving Average) Callback
# ----------------------------
class EMACallback(keras.callbacks.Callback):
    """Maintains exponential moving average of model weights.
    
    EMA improves model generalization by smoothing weight updates.
    Reference: "Mean teachers are better role models" (Tarvainen & Valpola, 2017)
    """
    def __init__(self, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None
    
    def on_train_begin(self, logs=None):
        # Initialize EMA weights
        self.ema_weights = [tf.Variable(w, trainable=False) for w in self.model.weights]
    
    def on_train_batch_end(self, batch, logs=None):
        # Update EMA weights
        for ema_w, model_w in zip(self.ema_weights, self.model.weights):
            ema_w.assign(self.decay * ema_w + (1 - self.decay) * model_w)
    
    def on_epoch_end(self, epoch, logs=None):
        # Optionally swap to EMA weights for validation
        # (Can be done at the end of training)
        pass
    
    def apply_ema_weights(self):
        """Apply EMA weights to the model."""
        for ema_w, model_w in zip(self.ema_weights, self.model.weights):
            model_w.assign(ema_w)
    
    def restore_original_weights(self, original_weights):
        """Restore original weights."""
        for orig_w, model_w in zip(original_weights, self.model.weights):
            model_w.assign(orig_w)


# ----------------------------
# Training function
# ----------------------------
def train_model(config: ModelConfig):
    """Train ResNet on ImageNet with the specified configuration."""
    
    # Enable mixed precision if requested
    if config.use_mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"✓ Enabled mixed precision: {tf.keras.mixed_precision.global_policy()}")
        except Exception as e:
            print(f"⚠ Could not enable mixed precision: {e}")
    
    set_seed(config.seed)
    
    print("\n" + "="*80)
    print("Training Configuration:")
    print("="*80)
    print(f"Model: ResNet-{config.depth}")
    print(f"Stochastic Depth: {'Enabled' if config.sd_on else 'Disabled'}")
    if config.sd_on:
        print(f"Final Survival Probability: {config.final_survival_prob}")
    print(f"Dataset: ImageNet (Hugging Face ILSVRC/imagenet-1k)")
    print(f"Image Size: {config.image_size}x{config.image_size}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Base Learning Rate: {config.base_lr}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"MixUp Alpha: {config.mixup_alpha}")
    print(f"CutMix Alpha: {config.cutmix_alpha}")
    if config.randaugment_n > 0:
        print(f"RandAugment: N={config.randaugment_n}, M={config.randaugment_m}")
    print(f"Label Smoothing: {config.label_smoothing}")
    print(f"Mixed Precision (FP16): {config.use_mixed_precision}")
    print(f"EMA: {config.use_ema}" + (f" (decay={config.ema_decay})" if config.use_ema else ""))
    if config.grad_clip_norm > 0:
        print(f"Gradient Clipping: {config.grad_clip_norm}")
    print("="*80 + "\n")
    
    # Load datasets
    print("Loading datasets...")
    train_ds, val_ds = make_datasets(config)
    
    # Calculate training steps
    # ImageNet has ~1.28M training images
    # Approximate number based on typical ImageNet size
    train_steps_per_epoch = 1281167 // config.batch_size  # exact training images in ImageNet
    val_steps_per_epoch = 50000 // config.batch_size  # exact validation images
    
    total_steps = train_steps_per_epoch * config.epochs
    warmup_steps = train_steps_per_epoch * WARMUP_EPOCHS
    
    print(f"Steps per epoch: {train_steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}\n")
    
    # Create learning rate schedule and optimizer
    lr_schedule = WarmupCosineDecay(
        config.base_lr, 
        total_steps, 
        warmup_steps=warmup_steps, 
        min_lr=1e-6
    )
    
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=config.momentum,
        nesterov=True,
        clipnorm=config.grad_clip_norm if config.grad_clip_norm > 0 else None
    )
    
    # Loss function with label smoothing
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=config.label_smoothing,
        from_logits=False
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    model_path = os.path.join(config.output_dir, "best_model.h5")
    
    # Build or load model
    if os.path.exists(model_path):
        print(f"✓ Found existing model checkpoint: {model_path}")
        print("  Loading model to resume training...\n")
        model = keras.models.load_model(
            model_path,
            custom_objects={
                'StochasticDepth': StochasticDepth,
                'WarmupCosineDecay': WarmupCosineDecay
            }
        )
    else:
        print("Building new model...\n")
        model = build_resnet_imagenet(config)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    # Print model summary
    model.summary()
    
    # Setup callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode='max'
        ),
        keras.callbacks.CSVLogger(
            os.path.join(config.output_dir, 'training_history.csv'),
            append=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.output_dir, 'tensorboard_logs'),
            histogram_freq=0,
            write_graph=False
        ),
        LrCallback(),
        keras.callbacks.TerminateOnNaN(),
    ]
    
    # Add EMA callback if enabled
    ema_callback = None
    if config.use_ema:
        print(f"✓ Using EMA (Exponential Moving Average) with decay={config.ema_decay}\n")
        ema_callback = EMACallback(decay=config.ema_decay)
        callbacks.append(ema_callback)
    
    # Add early stopping for efficiency (optional)
    # callbacks.append(keras.callbacks.EarlyStopping(
    #     monitor='val_accuracy',
    #     patience=10,
    #     restore_best_weights=True
    # ))
    
    # Apply MixUp/CutMix if enabled
    use_augmentation = config.mixup_alpha > 0.0 or config.cutmix_alpha > 0.0
    if use_augmentation:
        if config.mixup_alpha > 0.0 and config.cutmix_alpha > 0.0:
            print(f"✓ Applying MixUp (alpha={config.mixup_alpha}) and CutMix (alpha={config.cutmix_alpha})\n")
        elif config.mixup_alpha > 0.0:
            print(f"✓ Applying MixUp augmentation (alpha={config.mixup_alpha})\n")
        else:
            print(f"✓ Applying CutMix augmentation (alpha={config.cutmix_alpha})\n")
        
        def augmentation_generator(ds):
            for batch_x, batch_y in ds:
                # Randomly choose between MixUp and CutMix if both enabled
                if config.mixup_alpha > 0.0 and config.cutmix_alpha > 0.0:
                    use_mixup = np.random.rand() < 0.5
                    if use_mixup:
                        x, y = tf.numpy_function(
                            func=lambda bx, by: mixup(bx, by, config.mixup_alpha),
                            inp=[batch_x, batch_y],
                            Tout=[tf.float32, tf.float32]
                        )
                    else:
                        x, y = tf.numpy_function(
                            func=lambda bx, by: cutmix(bx, by, config.cutmix_alpha),
                            inp=[batch_x, batch_y],
                            Tout=[tf.float32, tf.float32]
                        )
                elif config.mixup_alpha > 0.0:
                    x, y = tf.numpy_function(
                        func=lambda bx, by: mixup(bx, by, config.mixup_alpha),
                        inp=[batch_x, batch_y],
                        Tout=[tf.float32, tf.float32]
                    )
                else:
                    x, y = tf.numpy_function(
                        func=lambda bx, by: cutmix(bx, by, config.cutmix_alpha),
                        inp=[batch_x, batch_y],
                        Tout=[tf.float32, tf.float32]
                    )
                x.set_shape([None, config.image_size, config.image_size, 3])
                y.set_shape([None, config.num_classes])
                yield x, y
        
        train_for_fit = tf.data.Dataset.from_generator(
            lambda: augmentation_generator(train_ds),
            output_signature=(
                tf.TensorSpec(shape=(None, config.image_size, config.image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, config.num_classes), dtype=tf.float32)
            )
        ).prefetch(tf.data.AUTOTUNE)
    else:
        train_for_fit = train_ds
    
    # Train model
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    history = model.fit(
        train_for_fit,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch
    )
    
    elapsed_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation on Validation Set")
    print("="*80 + "\n")
    
    results = model.evaluate(val_ds, steps=val_steps_per_epoch)
    val_loss, val_acc, val_top5_acc = results
    
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Top-1 Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Validation Top-5 Accuracy: {val_top5_acc:.4f} ({val_top5_acc*100:.2f}%)")
    
    # Save training summary
    save_training_summary(config, elapsed_time, val_loss, val_acc, val_top5_acc)
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print(f"Results saved to: {config.output_dir}")
    print("="*80 + "\n")


def save_training_summary(config: ModelConfig, elapsed_time: float, 
                         val_loss: float, val_acc: float, val_top5_acc: float):
    """Save training summary to CSV."""
    
    summary = {
        'model': f'ResNet-{config.depth}',
        'stochastic_depth': config.sd_on,
        'final_survival_prob': config.final_survival_prob if config.sd_on else None,
        'epochs': config.epochs,
        'batch_size': config.batch_size,
        'base_lr': config.base_lr,
        'weight_decay': config.weight_decay,
        'mixup_alpha': config.mixup_alpha,
        'label_smoothing': config.label_smoothing,
        'mixed_precision': config.use_mixed_precision,
        'training_time_hours': elapsed_time / 3600,
        'val_loss': val_loss,
        'val_top1_accuracy': val_acc,
        'val_top5_accuracy': val_top5_acc
    }
    
    df = pd.DataFrame([summary])
    summary_path = os.path.join(config.output_dir, 'training_summary.csv')
    df.to_csv(summary_path, index=False)
    
    print("\n" + "="*80)
    print("Training Summary:")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


# ----------------------------
# Main CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train ResNet with/without Stochastic Depth on Full ImageNet using Hugging Face datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ResNet-50 baseline (no stochastic depth)
  python full_imagenet_train.py --out ./results_baseline --sd 0
  
  # Train ResNet-50 with stochastic depth
  python full_imagenet_train.py --out ./results_sd --sd 1 --pL 0.5
  
  # Train ResNet-101 with all modern techniques (best accuracy)
  python full_imagenet_train.py --depth 101 --sd 1 --pL 0.5 \\
      --mixup 0.2 --cutmix 1.0 --randaugment_n 2 --label_smoothing 0.1 \\
      --use_ema --grad_clip 1.0 --batch 256 --epochs 120

Note: This script uses Hugging Face datasets to load ImageNet (ILSVRC/imagenet-1k).
      You need to:
      1. Accept terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k
      2. Login with: huggingface-cli login
        """
    )
    
    # Required arguments
    parser.add_argument('--out', '--output_dir', dest='output_dir', type=str, 
                       default='./results_imagenet',
                       help='Output directory for checkpoints and logs (default: ./results_imagenet)')
    
    # Data (optional - kept for backward compatibility)
    parser.add_argument('--data_dir', type=str, default=None,
                       help='[DEPRECATED] Not used - ImageNet is loaded from Hugging Face Hub')
    
    # Model architecture
    parser.add_argument('--depth', type=int, default=50, choices=[50, 101, 152],
                       help='ResNet depth: 50, 101, or 152 (default: 50)')
    
    # Stochastic depth
    parser.add_argument('--sd', type=int, default=0, choices=[0, 1],
                       help='Enable stochastic depth: 0=disabled, 1=enabled (default: 0)')
    parser.add_argument('--pL', type=float, default=0.5,
                       help='Final survival probability for stochastic depth (default: 0.5)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help=f'Number of training epochs (default: {DEFAULT_EPOCHS})')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH,
                       help=f'Batch size (default: {DEFAULT_BATCH})')
    parser.add_argument('--lr', type=float, default=None,
                       help='Base learning rate (default: auto-scaled based on batch size)')
    
    # Regularization and augmentation
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='MixUp alpha parameter (0 to disable, typical: 0.2)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                       help='CutMix alpha parameter (0 to disable, typical: 1.0)')
    parser.add_argument('--randaugment_n', type=int, default=0,
                       help='RandAugment N (number of transformations, 0 to disable, typical: 2)')
    parser.add_argument('--randaugment_m', type=int, default=9,
                       help='RandAugment M (magnitude, typical: 9)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                       help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_WEIGHT_DECAY,
                       help=f'Weight decay (L2 regularization) (default: {DEFAULT_WEIGHT_DECAY})')
    
    # Advanced training techniques
    parser.add_argument('--use_ema', action='store_true',
                       help='Use Exponential Moving Average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                       help='EMA decay rate (default: 0.9999)')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                       help='Gradient clipping norm (0 to disable, typical: 1.0)')
    
    # Performance options
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision training (FP16)')
    parser.add_argument('--cache', action='store_true',
                       help='Cache dataset in memory (requires ~150GB RAM for ImageNet)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ModelConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        depth=args.depth,
        sd_on=bool(args.sd),
        final_survival_prob=args.pL,
        epochs=args.epochs,
        batch_size=args.batch,
        base_lr=args.lr if args.lr is not None else DEFAULT_LR,
        weight_decay=args.weight_decay,
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        randaugment_n=args.randaugment_n,
        randaugment_m=args.randaugment_m,
        label_smoothing=args.label_smoothing,
        use_mixed_precision=not args.no_mixed_precision,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        grad_clip_norm=args.grad_clip,
        cache_dataset=args.cache,
        seed=args.seed,
        auto_scale_lr=(args.lr is None)  # Only auto-scale if user didn't provide explicit LR
    )
    
    # Start training
    train_model(config)
