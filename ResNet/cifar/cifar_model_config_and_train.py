import argparse
import os
import time
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# Constants
CIFAR100_NUM_CLASSES = 10
# CIFAR100_NUM_CLASSES = 100
CIFAR100_IMAGE_SIZE = 32
AUGMENTED_IMAGE_SIZE = 40
INITIAL_LEARNING_RATE = 0.1
LR_DECAY_FACTOR = 0.1
LR_DECAY_EPOCH_1 = 0.5
LR_DECAY_EPOCH_2 = 0.75
MOMENTUM = 0.9
SHUFFLE_BUFFER_SIZE = 20000


# Config
@dataclass
class ModelConfig:
    """Configuration for ResNet model training."""
    depth: int
    sd_on: bool
    final_survival_prob: float  # pL in the paper
    epochs: int
    batch_size: int
    output_dir: str

    def __post_init__(self):
        assert (self.depth - 2) % 6 == 0, "Depth must be 20, 32, 44, 56, 110"


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    return (x_train, y_train), (x_test, y_test)


def load_cifar100():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    return (x_train, y_train), (x_test, y_test)


def make_dataset(x, y, batch_size, training=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
        ds = ds.map(lambda img, lbl: (augment(img), lbl),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def augment(image):
    image = tf.image.resize_with_crop_or_pad(image, AUGMENTED_IMAGE_SIZE, AUGMENTED_IMAGE_SIZE)
    image = tf.image.random_crop(image, [CIFAR100_IMAGE_SIZE, CIFAR100_IMAGE_SIZE, 3])
    image = tf.image.random_flip_left_right(image)
    return image


# Stochastic Depth
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


# -------------------------------------------------------
# Identity + AvgPool + ZeroPad shortcut (from original CIFAR ResNet paper)
# -------------------------------------------------------
def shortcut_projection(x, filters, stride):
    """Implements the avgpool + zero padding shortcut from the original CIFAR ResNet."""
    if stride == 1 and x.shape[-1] == filters:
        return x
    # Downsample spatially using avgpool
    x = layers.AveragePooling2D(pool_size=2, strides=stride, padding="valid")(x)
    # Zero pad channel dimension
    ch_in = x.shape[-1]
    ch_out = filters
    pad_total = ch_out - ch_in
    pad1 = pad_total // 2
    pad2 = pad_total - pad1
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [pad1, pad2]])
    return x


# -------------------------------------------------------
# Pre-activation ResNet block (He et al., 2016, CIFAR version)
# -------------------------------------------------------
def resnet_block(x, filters, stride, survival_prob):
    shortcut = x
    # Pre-activation
    y = layers.BatchNormalization()(x)
    y = layers.ReLU()(y)
    # First conv
    y = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(y)
    # Second BN + ReLU
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    # Second conv
    y = layers.Conv2D(filters, 3, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(y)
    # Proper shortcut (avgpool+zero pad)
    shortcut = shortcut_projection(shortcut, filters, stride)
    if survival_prob < 1.0:
        out = StochasticDepth(survival_prob)(shortcut, y)
    else:
        out = layers.Add()([shortcut, y])
    return out


# -------------------------------------------------------
# Build ResNet (strict to paper: CIFAR version, pre-activation, correct shortcuts)
# -------------------------------------------------------
def build_resnet(config: ModelConfig):
    n = (config.depth - 2) // 6
    inputs = keras.Input((CIFAR100_IMAGE_SIZE, CIFAR100_IMAGE_SIZE, 3))
    # Initial conv (no batchnorm here)
    x = layers.Conv2D(16, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(inputs)
    total_blocks = 3 * n
    block = 0
    filters_list = [16, 32, 64]
    for stage, filters in enumerate(filters_list):
        for i in range(n):
            stride = 2 if stage > 0 and i == 0 else 1
            # Linear survival prob decay
            if config.sd_on:
                block_index = block + 1
                survival_prob = 1 - (block_index / total_blocks) * (1 - config.final_survival_prob)
            else:
                survival_prob = 1.0
            x = resnet_block(x, filters, stride, survival_prob)
            block += 1
    # Final BN + ReLU
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(CIFAR100_NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inputs, outputs)


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
def _create_optimizer():
    """Create optimizer with paper-specified hyperparameters."""
    return keras.optimizers.SGD(
        learning_rate=INITIAL_LEARNING_RATE,
        momentum=MOMENTUM,
        nesterov=True
    )


def _create_learning_rate_scheduler(total_epochs):
    """Create learning rate scheduler following the paper's schedule."""

    def scheduler(epoch, lr):
        if epoch == int(total_epochs * LR_DECAY_EPOCH_1):
            return lr * LR_DECAY_FACTOR
        if epoch == int(total_epochs * LR_DECAY_EPOCH_2):
            return lr * LR_DECAY_FACTOR
        return lr

    return scheduler


def _create_callbacks(config: ModelConfig):
    """Create training callbacks for logging and checkpointing."""
    os.makedirs(config.output_dir, exist_ok=True)
    return [
        keras.callbacks.LearningRateScheduler(_create_learning_rate_scheduler(config.epochs)),
        keras.callbacks.CSVLogger(os.path.join(config.output_dir, "history.csv")),
        keras.callbacks.ModelCheckpoint(
            os.path.join(config.output_dir, f"best_model_cifar-{CIFAR100_NUM_CLASSES}.h5"),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]


def _save_training_summary(config: ModelConfig, elapsed_time, test_loss, test_accuracy):
    """Save training summary to CSV."""
    df = pd.DataFrame([{
        "sd_on": config.sd_on,
        "depth": config.depth,
        "final_survival_prob": config.final_survival_prob,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "train_time": elapsed_time,
        "test_loss": test_loss,
        "test_acc": test_accuracy
    }])
    df.to_csv(os.path.join(config.output_dir, "summary.csv"), index=False)
    print(df.to_string(index=False))


from tensorflow.keras.models import load_model

def train_model(config: ModelConfig):
    set_seed()
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    # (x_train, y_train), (x_test, y_test) = load_cifar100()
    train_ds = make_dataset(x_train, y_train, config.batch_size, training=True)
    test_ds = make_dataset(x_test, y_test, config.batch_size, training=False)

    model_path = os.path.join(config.output_dir, f"best_model_cifar-{CIFAR100_NUM_CLASSES}.h5")
    os.makedirs(config.output_dir, exist_ok=True)

    # Загрузка существующей модели, если она есть
    if os.path.exists(model_path):
        print("Found existing model, loading:", model_path)
        model = load_model(model_path, custom_objects={"StochasticDepth": StochasticDepth})
    else:
        print("No existing model found, creating a new one.")
        model = build_resnet(config)

    model.compile(
        optimizer=_create_optimizer(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = _create_callbacks(config)

    start = time.time()
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2
    )
    elapsed = time.time() - start

    loss, acc = model.evaluate(test_ds)
    _save_training_summary(config, elapsed, loss, acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd", type=int, default=0)
    parser.add_argument("--depth", type=int, default=56)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--pL", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="results_cifar")
    args = parser.parse_args()

    config = ModelConfig(
        depth=args.depth,
        sd_on=bool(args.sd),
        final_survival_prob=args.pL,
        epochs=args.epochs,
        batch_size=args.batch,
        output_dir=args.out
    )
    train_model(config)
