# Start
# python train_mnist_stochastic_depth.py --sd 0 --depth 18 --epochs 12 --batch 128 --pL 0.5 --out results_mnist
# python train_mnist_stochastic_depth.py --sd 1 --depth 18 --epochs 12 --batch 128 --pL 0.5 --out results_mnist

import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import os

# -------------------------------------
# Load MNIST
# -------------------------------------
def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

# -------------------------------------
# Correct Stochastic Depth Layer
# -------------------------------------
class StochasticDepth(layers.Layer):
    def __init__(self, survival_prob, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob

    def call(self, shortcut, residual, training=None):
        if training:
            random_val = tf.random.uniform([], 0, 1)
            return tf.cond(
                random_val < self.survival_prob,
                lambda: shortcut + residual,
                lambda: shortcut
            )
        else:
            return shortcut + self.survival_prob * residual

# -------------------------------------
# Residual Block (No Lambda at all!)
# -------------------------------------
def residual_block(x, filters, stride=1, survival_prob=1.0):
    shortcut = x

    # Conv path
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    y = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)

    # Match shortcut if needed
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Stochastic Depth
    if survival_prob < 1.0:
        out = StochasticDepth(survival_prob)(shortcut, y)
    else:
        out = layers.Add()([shortcut, y])

    return layers.ReLU()(out)

# -------------------------------------
# Build MNIST ResNet
# -------------------------------------
def build_resnet_mnist(input_shape, depth, sd_on=False, pL=0.5):
    assert depth % 3 == 0
    blocks_per_stage = depth // 3
    total_blocks = depth

    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    filters = [16, 32, 64]
    block_index = 0

    for stage, f in enumerate(filters):
        for b in range(blocks_per_stage):
            stride = 2 if stage > 0 and b == 0 else 1

            if sd_on:
                l = block_index + 1
                survival_prob = 1 - (l / total_blocks) * (1 - pL)
            else:
                survival_prob = 1.0

            x = residual_block(x, f, stride=stride, survival_prob=survival_prob)
            block_index += 1

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    return keras.Model(inputs, outputs)

# -------------------------------------
# Run Experiment
# -------------------------------------
def run_experiment(sd_on, depth, epochs, batch, pL, out):
    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    model = build_resnet_mnist(x_train.shape[1:], depth, sd_on, pL)
    model.compile(
        optimizer=keras.optimizers.SGD(0.1, momentum=0.9, nesterov=True),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(out, exist_ok=True)

    tag = "SD" if sd_on else "BASE"
    csv_logger = keras.callbacks.CSVLogger(os.path.join(out, f"history_{tag}.csv"))

    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch,
        callbacks=[csv_logger],
        verbose=2
    )
    elapsed = time.time() - start

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    summary_df = pd.DataFrame([{
        "model": tag,
        "sd_on": int(sd_on),
        "depth": depth,
        "pL": pL,
        "epochs": epochs,
        "batch": batch,
        "train_time_sec": elapsed,
        "test_loss": loss,
        "test_acc": acc
    }])

    summary_path = os.path.join(out, "summary.csv")
    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        pd.concat([old, summary_df], ignore_index=True).to_csv(summary_path, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)

    print("✓ Experiment finished:", summary_df.to_string(index=False))


# -------------------------------------
# Main
# -------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd", type=int, default=0)
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--pL", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="results_mnist")
    a = parser.parse_args()

    run_experiment(
        sd_on=bool(a.sd),
        depth=a.depth,
        epochs=a.epochs,
        batch=a.batch,
        pL=a.pL,
        out=a.out
    )
