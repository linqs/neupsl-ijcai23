from collections import namedtuple

import os

import numpy as np
import tensorflow as tf

from cnn_models import MNISTAdditionBaseline
from cnn_models import MNISTAddition2Baseline

N_FOLDS = 10
N_EPOCHS = 100
LEARNING_RATE = 1e-3

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.abspath(os.path.join(THIS_DIR, "../data/mnist-addition"))
RESULTS_PATH = os.path.abspath(os.path.join(THIS_DIR, "../results"))


def load_dataset(path, label_size):
    print(path)
    with open(path, 'r') as data_file:
        data = np.array(eval(data_file.read()))

    x = np.array([np.array(example[0]) / 255.0 for example in data])
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


def main():
    # MNIST 1
    # Hyperparameter: Search over learning rates for training_size: 3000, overlap: 0.00, fold: 0
    for training_size in [20, 37, 75, 150, 300, 3000, 25000]:
        for overlap in [0.00, 0.5, 1.0, 2.0]:
            for fold in range(N_FOLDS):
                model = MNISTAdditionBaseline()
                optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
                loss_fun = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])

                out_dir = os.path.join(DATA_PATH, "n_digits::{:01d}/fold::{:02d}/train_size::{:05d}/overlap::{:.2f}".format(1, fold, training_size, overlap))
                train_x, train_y = load_dataset(os.path.join(out_dir, 'learn/baseline_data.txt'), 19)
                eval_x, eval_y = load_dataset(os.path.join(out_dir, 'eval/baseline_data.txt'), 19)

                model.fit(train_x, train_y, batch_size=32, epochs=N_EPOCHS)
                test_eval(
                    model,
                    eval_x, eval_y,
                    os.path.join(RESULTS_PATH,
                                 "experiment::mnist-addition-{:01d}/model::neural_baseline/train_size::{:05d}/overlap::{:.2f}/fold::{:02d}/out.txt".format(
                                     1, training_size, overlap, fold))
                )

    # MNIST 2
    # Hyperparameter: Search over learning rates for training_size: 1500, overlap: 0.00, fold: 0
    for training_size in [10, 20, 37, 150, 1500, 12500]:
        for overlap in [0.00, 0.5, 1.0, 2.0]:
            for fold in range(N_FOLDS):
                model = MNISTAddition2Baseline()
                optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
                loss_fun = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])

                out_dir = os.path.join(DATA_PATH, "n_digits::{:01d}/fold::{:02d}/train_size::{:05d}/overlap::{:.2f}".format(2, fold, training_size, overlap))
                train_x, train_y = load_dataset(os.path.join(out_dir, 'learn/baseline_data.txt'), 199)
                eval_x, eval_y = load_dataset(os.path.join(out_dir, 'eval/baseline_data.txt'), 199)

                model.fit(train_x, train_y, batch_size=32, epochs=N_EPOCHS)
                test_eval(
                    model,
                    eval_x, eval_y,
                    os.path.join(RESULTS_PATH, "experiment::mnist-addition-{:01d}/model::neural_baseline/train_size::{:05d}/overlap::{:.2f}/fold::{:02d}/out.txt".format(
                        2, training_size, overlap, fold))
                )


if __name__ == "__main__":
    main()
