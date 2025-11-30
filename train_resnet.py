import argparse
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd



# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -------------------------------------------------------
# Load CIFAR-100
# -------------------------------------------------------
def load_cifar100():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    return (x_train, y_train), (x_test, y_test)


# -------------------------------------------------------
# Data Augmentation
# -------------------------------------------------------
def make_dataset(x, y, batch_size, training=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(20000)
        ds = ds.map(lambda img, lbl: (augment(img), lbl),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def augment(image):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image


# -------------------------------------------------------
# Stochastic Depth (Correct version from paper)
# -------------------------------------------------------
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
def build_resnet(depth, sd_on=False, pL=0.5):
    assert (depth - 2) % 6 == 0, "Depth must be 20, 32, 44, 56, 110"
    n = (depth - 2) // 6

    inputs = keras.Input((32, 32, 3))

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
            if sd_on:
                l = block + 1
                survival_prob = 1 - (l / total_blocks) * (1 - pL)
            else:
                survival_prob = 1.0

            x = resnet_block(x, filters, stride, survival_prob)

            block += 1

    # Final BN + ReLU
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(100, activation="softmax")(x)

    return keras.Model(inputs, outputs)


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
def train_model(depth, sd_on, epochs, batch, pL, outdir):
    set_seed()

    (x_train, y_train), (x_test, y_test) = load_cifar100()
    train_ds = make_dataset(x_train, y_train, batch, training=True)
    test_ds = make_dataset(x_test, y_test, batch, training=False)

    model = build_resnet(depth, sd_on, pL)

    # Optimizer EXACTLY as in the paper
    optimizer = keras.optimizers.SGD(
        learning_rate=0.1,
        momentum=0.9,
        nesterov=True
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(outdir, exist_ok=True)

    # Standard LR schedule from the paper
    def scheduler(epoch, lr):
        if epoch == int(epochs * 0.5):
            return lr * 0.1
        if epoch == int(epochs * 0.75):
            return lr * 0.1
        return lr

    callbacks = [
        keras.callbacks.LearningRateScheduler(scheduler),
        keras.callbacks.CSVLogger(os.path.join(outdir, "history.csv")),
        keras.callbacks.ModelCheckpoint(
            os.path.join(outdir, "best.h5"),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    start = time.time()
    model.fit(
        train_ds, validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    elapsed = time.time() - start

    loss, acc = model.evaluate(test_ds)

    df = pd.DataFrame([{
        "sd_on": sd_on,
        "depth": depth,
        "pL": pL,
        "epochs": epochs,
        "batch": batch,
        "train_time": elapsed,
        "test_loss": loss,
        "test_acc": acc
    }])
    df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    print(df.to_string(index=False))


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sd", type=int, default=0)
    p.add_argument("--depth", type=int, default=56)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--pL", type=float, default=0.5)
    p.add_argument("--out", type=str, default="results_cifar100")
    a = p.parse_args()

    train_model(
        depth=a.depth,
        sd_on=bool(a.sd),
        epochs=a.epochs,
        batch=a.batch,
        pL=a.pL,
        outdir=a.out
    )
