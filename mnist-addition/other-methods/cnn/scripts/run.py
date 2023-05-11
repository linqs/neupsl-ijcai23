import importlib
import sys

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy
import tensorflow as tf

from models import MNISTAdditionBaseline
from models import MNISTAddition2Baseline

NUM_SPLITS = 2
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.abspath(os.path.join(THIS_DIR, "../../../data/"))
RESULTS_PATH = os.path.abspath(os.path.join(THIS_DIR, "../results"))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..', '..', 'scripts'))
util = importlib.import_module("util")


def create_cnn_data(raw_data, entity_data_map, label_size):
    features = []
    labels = []
    index = 0
    for entities in raw_data:
        if index % label_size == 0:
            labels.append([])
            features.append([])
            for entity in entities[:-2]:
                features[-1].append([float(value) for value in entity_data_map[entity]['features']])
        labels[-1].append(int(float(entities[-1])))
        index += 1

    features = numpy.array(features).reshape((-1, len(raw_data[0][:-2]), 28, 28, 1)).tolist()

    return features, labels


def load_dataset(path, label_size):
    raw_entity_data_map = util.load_psl_file(os.path.join(path, 'entity-data-map.txt'))
    entity_data_map = {}
    for entity in raw_entity_data_map:
        entity_data_map[entity[0]] = {'features': entity[1:-1], 'label': entity[-1]}

    raw_train_data = util.load_psl_file(os.path.join(path, 'image-sum-truth-train.txt'))
    train_features, train_labels = create_cnn_data(raw_train_data, entity_data_map, label_size)

    raw_test_data = util.load_psl_file(os.path.join(path, 'image-sum-truth-test.txt'))
    test_features, test_labels = create_cnn_data(raw_test_data, entity_data_map, label_size)

    return train_features, train_labels, test_features, test_labels


def test_eval(model, data, labels, output_path=None):
    loss, accuracy = model.evaluate(data, labels)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as output:
        output.write("Loss: {}".format(loss))
        output.write("IMAGESUM -- Categorical Accuracy: {}".format(accuracy))


def main():
    # MNIST 1
    for training_size in [40, 60, 80]:
        for overlap in [0.0, 0.5, 1.0]:
            for split in range(NUM_SPLITS):
                model = MNISTAdditionBaseline()
                optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
                loss_fun = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])

                out_dir = os.path.join(DATA_PATH, "experiment::mnist-1/split::{:01d}/train-size::{:04d}/overlap::{:.2f}".format(split, training_size, overlap))
                train_x, train_y, eval_x, eval_y = load_dataset(out_dir, 19)

                model.fit(train_x, train_y, batch_size=32, epochs=NUM_EPOCHS)
                test_eval(
                    model,
                    eval_x, eval_y,
                    os.path.join(RESULTS_PATH, "experiment::mnist-1/split::{:02d}/train-size::{:04d}/overlap::{:.2f}/out.txt".format(split, training_size, overlap))
                )

    # MNIST 2
    for training_size in [40, 60, 80]:
        for overlap in [0.0, 0.5, 1.0]:
            for split in range(NUM_SPLITS):
                model = MNISTAddition2Baseline()
                optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
                loss_fun = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                model.compile(optimizer=optimizer, loss=loss_fun, metrics=['accuracy'])

                out_dir = os.path.join(DATA_PATH, "experiment::mnist-2/split::{:01d}/train-size::{:04d}/overlap::{:.2f}".format(split, training_size, overlap))
                train_x, train_y, eval_x, eval_y = load_dataset(out_dir, 199)

                model.fit(train_x, train_y, batch_size=32, epochs=NUM_EPOCHS)
                test_eval(
                    model,
                    eval_x, eval_y,
                    os.path.join(RESULTS_PATH, "experiment::mnist-2/split::{:01d}/train-size::{:04d}/overlap::{:.2f}/out.txt".format(split, training_size, overlap))
                )


if __name__ == "__main__":
    main()
