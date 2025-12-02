import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# ----------------------------
# Defaults & hyperparameters
# ----------------------------
DEFAULT_IMAGE_SIZE = 64
DEFAULT_NUM_CLASSES = 200
DEFAULT_BATCH = 128
DEFAULT_EPOCHS = 120
DEFAULT_LR = 0.1
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4  # implemented as kernel_regularizer
SHUFFLE_BUFFER = 300
# SHUFFLE_BUFFER = 20000
WARMUP_EPOCHS = 5


# ----------------------------
# Config dataclass
# ----------------------------
@dataclass
class ModelConfig:
    data_dir: str
    output_dir: str
    image_size: int = DEFAULT_IMAGE_SIZE
    num_classes: int = DEFAULT_NUM_CLASSES
    depth: int = 56
    sd_on: bool = False
    final_survival_prob: float = 0.5
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH
    base_lr: float = DEFAULT_LR
    momentum: float = DEFAULT_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    mixup_alpha: float = 0.0
    label_smoothing: float = 0.0
    use_mixed_precision: bool = True
    cache_dataset: bool = False
    seed: int = 42

    def __post_init__(self):
        assert (self.depth - 2) % 6 == 0, "Depth must be 20, 32, 44, 56, 110"


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed=42, use_tf=True):
    np.random.seed(seed)
    if use_tf:
        tf.random.set_seed(seed)


class LrCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(
            self.model.optimizer.learning_rate(self.model.optimizer.iterations).numpy()
        )
        print(f"LR at epoch {epoch + 1}: {lr:.8f}")


# ----------------------------
# Data loading & augmentation
# ----------------------------
def make_datasets(config: ModelConfig):
    seed = config.seed
    image_size = config.image_size

    train_dir = os.path.join(config.data_dir, "train")
    val_dir = os.path.join(config.data_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise ValueError("data_dir must contain 'train' and 'val' directories with per-class subfolders.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        batch_size=config.batch_size,
        image_size=(image_size, image_size),
        shuffle=True,
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        batch_size=config.batch_size,
        image_size=(image_size, image_size),
        shuffle=False
    )

    # Prefetch and optional cache
    AUTOTUNE = tf.data.AUTOTUNE

    if config.cache_dataset:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    # Augmentations using Keras preprocessing layers (deterministic per example but different seeds)
    augment_layers = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),  # ~ +/- 6%
        layers.RandomZoom(0.06),
        layers.RandomContrast(0.06),
    ], name="augmentation")

    # Normalization to ImageNet mean/std (commonly used for Tiny-ImageNet)
    imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    def preprocess(x, y, training=False):
        # convert to float and scale to [0,1]
        x = tf.cast(x, tf.float32) / 255.0

        if training:
            x = augment_layers(x)

        # Ensure mean/std have same dtype as x (works with mixed precision)
        mean = tf.cast(imagenet_mean, x.dtype)
        std = tf.cast(imagenet_std, x.dtype)
        x = (x - mean) / std

        # convert labels to one-hot (for label smoothing and mixup)
        y = tf.one_hot(y, config.num_classes)
        return x, y

    train_ds = train_ds.map(lambda x, y: preprocess(x, y, True), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: preprocess(x, y, False), num_parallel_calls=AUTOTUNE)

    # Shuffle, repeat, batch already handled by image_dataset_from_directory
    train_ds = train_ds.shuffle(SHUFFLE_BUFFER).prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


# MixUp augmentation
def mixup(batch_x, batch_y, alpha=0.2):
    if alpha <= 0.0:
        return batch_x, batch_y
    lam = np.random.beta(alpha, alpha)
    lam = float(lam)
    batch_size = tf.shape(batch_x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_x = lam * batch_x + (1 - lam) * tf.gather(batch_x, index)
    mixed_y = lam * batch_y + (1 - lam) * tf.gather(batch_y, index)
    return mixed_x, mixed_y


# ----------------------------
# Stochastic Depth (kept from your original code, minor adapt)
# ----------------------------
class StochasticDepth(layers.Layer):
    def __init__(self, survival_prob=1.0, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob

    def call(self, x, residual, training=False):
        if training:
            # drop entire residual branch
            bern = tf.random.uniform([], 0, 1) < self.survival_prob
            return tf.cond(
                bern,
                lambda: x + residual,
                lambda: x
            )
        else:
            # expected value at test time
            return x + self.survival_prob * residual

    def get_config(self):
        cfg = super().get_config()
        cfg["survival_prob"] = self.survival_prob
        return cfg


# ----------------------------
# Shortcut projection & ResNet block (adapted)
# ----------------------------
def shortcut_projection(x, filters, stride):
    if stride == 1 and x.shape[-1] == filters:
        return x
    x = layers.AveragePooling2D(pool_size=2, strides=stride, padding="valid")(x)
    ch_in = x.shape[-1]
    ch_out = filters
    pad_total = ch_out - ch_in
    pad1 = pad_total // 2
    pad2 = pad_total - pad1
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [pad1, pad2]])
    return x


def resnet_block(x, filters, stride, survival_prob, weight_decay):
    shortcut = x
    y = layers.BatchNormalization()(x)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(weight_decay))(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(weight_decay))(y)
    shortcut = shortcut_projection(shortcut, filters, stride)
    if survival_prob < 1.0:
        out = StochasticDepth(survival_prob)(shortcut, y)
    else:
        out = layers.Add()([shortcut, y])
    return out


# ----------------------------
# Build ResNet (pre-activation CIFAR style)
# ----------------------------
def build_resnet(config: ModelConfig):
    n = (config.depth - 2) // 6
    inputs = keras.Input((config.image_size, config.image_size, 3))
    x = layers.Conv2D(16, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal",
                      kernel_regularizer=keras.regularizers.l2(config.weight_decay))(inputs)
    total_blocks = 3 * n
    block = 0
    filters_list = [16, 32, 64]
    for stage, filters in enumerate(filters_list):
        for i in range(n):
            stride = 2 if stage > 0 and i == 0 else 1
            if config.sd_on:
                block_index = block + 1
                survival_prob = 1 - (block_index / total_blocks) * (1 - config.final_survival_prob)
            else:
                survival_prob = 1.0
            x = resnet_block(x, filters, stride, survival_prob, config.weight_decay)
            block += 1
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(config.num_classes, activation="softmax",
                           kernel_regularizer=keras.regularizers.l2(config.weight_decay),
                           dtype="float32")(x)  # keep logits in float32 for numerical stability
    return keras.Model(inputs, outputs)


# ----------------------------
# Learning rate schedule: linear warmup + cosine decay
# ----------------------------
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps=0, min_lr=1e-6):
        super().__init__()
        # store as floats (Python numbers or tensors are both OK, we'll cast inside __call__)
        self.base_lr = tf.cast(base_lr, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)

    def __call__(self, step):
        # cast step to float32
        step = tf.cast(step, tf.float32)

        # safe denominators to avoid division by zero when warmup_steps == 0 or total_steps == warmup_steps
        warmup_denom = tf.maximum(self.warmup_steps, 1.0)
        cosine_denom = tf.maximum(self.total_steps - self.warmup_steps, 1.0)

        # linear warmup lr (safe even if warmup_steps == 0 because we won't use it in that case)
        warmup_lr = (step / warmup_denom) * self.base_lr

        # cosine decay part (starts at base_lr and decays to min_lr)
        cosine_steps = tf.maximum(step - self.warmup_steps, 0.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(tf.constant(np.pi, dtype=tf.float32) * cosine_steps / cosine_denom))
        decayed_lr = (self.base_lr - self.min_lr) * cosine_decay + self.min_lr

        # if step < warmup_steps -> warmup_lr else decayed_lr
        lr = tf.where(step < self.warmup_steps, warmup_lr, decayed_lr)
        return lr

    def get_config(self):
        return dict(base_lr=float(self.base_lr.numpy()),
                    total_steps=float(self.total_steps.numpy()),
                    warmup_steps=float(self.warmup_steps.numpy()),
                    min_lr=float(self.min_lr.numpy()))


# ----------------------------
# Training loop
# ----------------------------
from tensorflow.keras.models import load_model


def train_model(config: ModelConfig):
    # Mixed precision
    if config.use_mixed_precision:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Enabled mixed precision:", tf.keras.mixed_precision.global_policy())
        except Exception as e:
            print("Could not enable mixed precision:", e)

    set_seed(config.seed)

    train_ds, val_ds = make_datasets(config)

    # Compute steps
    train_count = sum(1 for _ in train_ds.unbatch())
    train_steps = max(1, train_count // config.batch_size)
    total_steps = train_steps * config.epochs
    warmup_steps = train_steps * WARMUP_EPOCHS

    lr_schedule = WarmupCosineDecay(config.base_lr, total_steps, warmup_steps=warmup_steps, min_lr=1e-5)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config.momentum, nesterov=True)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.label_smoothing, from_logits=False)

    os.makedirs(config.output_dir, exist_ok=True)
    model_path = os.path.join(config.output_dir, "best_model.h5")

    # Загружаем модель, если есть
    if os.path.exists(model_path):
        print("Found existing model, loading:", model_path)
        model = load_model(model_path, custom_objects={
            "StochasticDepth": StochasticDepth,
            "WarmupCosineDecay": WarmupCosineDecay
        })
    else:
        print("No existing model found, creating new one.")
        model = build_resnet(config)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.CSVLogger(os.path.join(config.output_dir, "history.csv")),
        keras.callbacks.TensorBoard(log_dir=os.path.join(config.output_dir, "tb_logs")),
        LrCallback(),
    ]

    # MixUp generator
    def mixup_generator(ds):
        for batch_x, batch_y in ds:
            x = batch_x
            y = batch_y
            if config.mixup_alpha > 0.0:
                x, y = tf.numpy_function(
                    func=lambda bx, by: mixup(bx, by, config.mixup_alpha),
                    inp=[x, y],
                    Tout=[tf.float32, tf.float32]
                )
                x.set_shape([None, config.image_size, config.image_size, 3])
                y.set_shape([None, config.num_classes])
            yield x, y

    if config.mixup_alpha > 0.0:
        train_for_fit = tf.data.Dataset.from_generator(
            lambda: mixup_generator(train_ds),
            output_signature=(
                tf.TensorSpec(shape=(None, config.image_size, config.image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, config.num_classes), dtype=tf.float32)
            )
        ).unbatch().batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    else:
        train_for_fit = train_ds

    # Fit
    start = time.time()
    model.fit(
        train_for_fit,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2
    )
    elapsed = time.time() - start

    # Evaluate
    loss_val, acc_val = model.evaluate(val_ds)
    _save_training_summary(config, elapsed, float(loss_val), float(acc_val), lr_schedule=lr_schedule)


def _save_training_summary(config: ModelConfig, elapsed_time, test_loss, test_accuracy, lr_schedule=None):
    # lr_schedule — объект WarmupCosineDecay, можно вычислить lr для шага 0 и финального шага
    if lr_schedule is not None:
        lr_start = float(lr_schedule(0).numpy())
        lr_end = float(lr_schedule(lr_schedule.total_steps).numpy())
    else:
        lr_start = config.base_lr
        lr_end = 0.0  # по умолчанию

    df = pd.DataFrame([{
        "sd_on": config.sd_on,
        "depth": config.depth,
        "final_survival_prob": config.final_survival_prob,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "mixup_alpha": config.mixup_alpha,
        "label_smoothing": config.label_smoothing,
        "use_mixed_precision": config.use_mixed_precision,
        "train_time_s": elapsed_time,
        "test_loss": test_loss,
        "test_acc": test_accuracy,
        "lr_start": lr_start,
        "lr_end": lr_end
    }])
    out = os.path.join(config.output_dir, "summary.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet on Tiny ImageNet with improved pipeline")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root with train/ and val/ subfolders")
    parser.add_argument("--out", "--output_dir", dest="output_dir", type=str, default="./results_tiny_imagenet")
    parser.add_argument("--depth", type=int, default=110, help="ResNet depth (CIFAR variant, 20,32,44,56,110)")
    parser.add_argument("--sd", type=int, default=0, help="Enable stochastic depth (1 to enable)")
    parser.add_argument("--pL", type=float, default=0.5, help="Final survival probability for stochastic depth")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0 to disable)")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--cache", action="store_true", help="Cache datasets in memory (if you have RAM)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = ModelConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        depth=args.depth,
        sd_on=bool(args.sd),
        final_survival_prob=args.pL,
        epochs=args.epochs,
        batch_size=args.batch,
        base_lr=args.lr,
        mixup_alpha=args.mixup,
        label_smoothing=args.label_smoothing,
        use_mixed_precision=not args.no_mixed_precision,
        cache_dataset=args.cache,
        seed=args.seed
    )

    train_model(config)
