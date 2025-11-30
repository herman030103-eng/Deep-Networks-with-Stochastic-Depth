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
# Data Augmentation (standard CIFAR augmentation)
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
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image

# -------------------------------------------------------
# Stochastic Depth Layer
# -------------------------------------------------------
class StochasticDepth(layers.Layer):
    """
    Simple stochastic depth (a.k.a. drop path) layer that either returns
    shortcut + residual with probability `survival_prob` during training,
    or returns shortcut + survival_prob * residual during inference
    (the expected value).
    """

    def __init__(self, survival_prob, **kwargs):
        super().__init__(**kwargs)
        # store as float so it's JSON-serializable in get_config
        self.survival_prob = float(survival_prob)

    def call(self, shortcut, residual, training=None):
        if training:
            r = tf.random.uniform([], 0, 1)
            return tf.cond(
                r < self.survival_prob,
                lambda: shortcut + residual,
                lambda: shortcut
            )
        else:
            return shortcut + self.survival_prob * residual

    def get_config(self):
        """
        Return the config for serialization. Because this layer takes
        an argument in __init__, Keras requires get_config to be implemented
        so models using this layer can be saved/loaded.
        """
        config = super().get_config()
        config.update({
            "survival_prob": self.survival_prob,
        })
        return config

# -------------------------------------------------------
# CIFAR ResNet Block (official version)
# -------------------------------------------------------
def resnet_block(x, filters, stride, survival_prob):
    shortcut = x

    # First conv
    y = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      kernel_initializer="he_normal", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    # Second conv
    y = layers.Conv2D(filters, 3, padding="same",
                      kernel_initializer="he_normal", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    # Match shortcut if shape mismatch
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride,
                                 padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Stochastic Depth
    if survival_prob < 1.0:
        out = StochasticDepth(survival_prob)(shortcut, y)
    else:
        out = layers.Add()([shortcut, y])

    return layers.ReLU()(out)

# -------------------------------------------------------
# Build proper CIFAR-100 ResNet (6n+2)
# -------------------------------------------------------
def build_resnet(depth, sd_on=False, pL=0.5):
    assert (depth - 2) % 6 == 0, "Depth must be 20, 32, 44, 56, 110"
    n = (depth - 2) // 6

    inputs = keras.Input(shape=(32, 32, 3))

    x = layers.Conv2D(16, 3, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    total_blocks = 3 * n
    block_index = 0

    # Stages
    filters_list = [16, 32, 64]

    for stage, filters in enumerate(filters_list):
        for i in range(n):
            stride = 2 if stage > 0 and i == 0 else 1

            if sd_on:
                l = block_index + 1
                survival_prob = 1 - (l / total_blocks) * (1 - pL)
            else:
                survival_prob = 1.0

            x = resnet_block(x, filters, stride, survival_prob)
            block_index += 1

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(100, activation="softmax")(x)

    return keras.Model(inputs, outputs)

# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def train_model(depth, sd_on, epochs, batch, pL, outdir):
    set_seed(42)
    (x_train, y_train), (x_test, y_test) = load_cifar100()

    train_ds = make_dataset(x_train, y_train, batch, training=True)
    test_ds = make_dataset(x_test, y_test, batch, training=False)

    model = build_resnet(depth, sd_on, pL)

    model.compile(
        optimizer=keras.optimizers.SGD(0.001, momentum=0.9, nesterov=True),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(outdir, exist_ok=True)

    callbacks = [
        # keras.callbacks.EarlyStopping(
        #     monitor="val_accuracy", patience=20, restore_best_weights=True
        # ),
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor="val_loss", factor=0.1, patience=10, min_lr=1e-5
        # ),
        keras.callbacks.CSVLogger(os.path.join(outdir, f"history{outdir}.csv")),
        keras.callbacks.ModelCheckpoint(
            os.path.join(outdir, f"{outdir}.h5"),
            monitor="val_accuracy", save_best_only=True
        )
    ]

    start = time.time()
    model.fit(
        train_ds,
        validation_data=test_ds,
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
    df.to_csv(os.path.join(outdir, f"summary_{outdir}.csv"), index=False)

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