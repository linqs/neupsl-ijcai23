from collections import namedtuple

import os

import numpy as np
import tensorflow as tf

from baseline_models import MNISTAdditionBaseline
from baseline_models import MNISTAddition2Baseline

N_FOLDS = 10
N_EPOCHS = 500
LEARNING_RATES = [1e-3, 1e-4, 1e-5]
BATCH_SIZES = [16, 32, 64, 128]
PATIENCE=20

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.abspath(os.path.join(THIS_DIR, "../data/mnist-addition"))
RESULTS_PATH = os.path.abspath(os.path.join(THIS_DIR, "../results"))


def load_dataset(path, label_size):
    print(path)
    with open(path, 'r') as data_file:
        data = np.array(eval(data_file.read()))

    x = np.array([example[0] for example in data])
    y = np.array([np.zeros(label_size) for _ in data])
    for i, example in enumerate(data):
        y[i][example[1]] = 1
    return x, y


def test_eval(model, data, labels, output_path=None):
    loss, accuracy = model.evaluate(data, labels)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as output:
        output.write("Loss: {}".format(loss))
        output.write("IMAGESUM -- Categorical Accuracy: {}".format(accuracy))


def split_train_val(x,y, val_percent=0.15):
    indices = np.array(range(len(y)))
    np.random.shuffle(indices)
    split_index = int(len(y) * (1-val_percent))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    return x[train_indices], y[train_indices], x[val_indices], y[val_indices]

def main():
    # MNIST 1

    for training_size in [20, 37, 75, 150, 300, 3000, 25000]:
        for overlap in [0.00, 0.5, 1.0, 2.0]:
            for fold in range(N_FOLDS):
                data_dir = os.path.join(DATA_PATH, "n_digits::{:01d}/fold::{:02d}/train_size::{:05d}/overlap::{:.2f}".format(1, fold, training_size, overlap))
                x, y = load_dataset(os.path.join(data_dir, 'learn/baseline_data.txt'), 19)
                eval_x, eval_y = load_dataset(os.path.join(data_dir, 'eval/baseline_data.txt'), 19)

                for learning_rate in LEARNING_RATES:
                    for batch_size in BATCH_SIZES:
                        model = MNISTAdditionBaseline()
                        optimizer = tf.keras.optimizers.Adam(learning_rate)
                        loss_fun = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                        model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])

                        train_x, train_y, val_x, val_y = split_train_val(x, y)

                        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True, min_delta=0.001)

                        model.fit(train_x, train_y, validation_data = (val_x, val_y),
                                validation_freq=1, callbacks=[callback],
                                batch_size=batch_size, epochs=N_EPOCHS
                                )
                        test_eval(
                            model,
                            eval_x, eval_y,
                            os.path.join(RESULTS_PATH,
                                        "experiment::mnist-addition-{:01d}/model::neural_baseline/train_size::{:05d}/overlap::{:.2f}/fold::{:02d}/lr{}_bs{}_out.txt".format(
                                            1, training_size, overlap, fold, learning_rate, batch_size))
                        )

    # MNIST 2

    for training_size in [10, 18, 37, 75, 150, 1500, 12500]:
        for overlap in [0.00, 0.5, 1.0, 2.0]:
            for fold in range(N_FOLDS):
                data_dir = os.path.join(DATA_PATH, "n_digits::{:01d}/fold::{:02d}/train_size::{:05d}/overlap::{:.2f}".format(2, fold, training_size, overlap))
                x, y = load_dataset(os.path.join(data_dir, 'learn/baseline_data.txt'), 199)
                eval_x, eval_y = load_dataset(os.path.join(data_dir, 'eval/baseline_data.txt'), 199)

                for learning_rate in LEARNING_RATES:
                    for batch_size in BATCH_SIZES:
                        model = MNISTAddition2Baseline()
                        optimizer = tf.keras.optimizers.Adam(learning_rate)
                        loss_fun = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                        model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])

                        train_x, train_y, val_x, val_y = split_train_val(x, y)

                        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True, min_delta=0.001)

                        model.fit(train_x, train_y, validation_data = (val_x, val_y),
                                validation_freq=1, callbacks=[callback],
                                batch_size=batch_size, epochs=N_EPOCHS
                                )
                        test_eval(
                            model,
                            eval_x, eval_y,
                            os.path.join(RESULTS_PATH, "experiment::mnist-addition-{:01d}/model::neural_baseline/train_size::{:05d}/overlap::{:.2f}/fold::{:02d}/lr{}_bs{}_out.txt.txt".format(
                                2, training_size, overlap, fold, learning_rate, batch_size))
                        )


if __name__ == "__main__":
    main()
