#!/usr/bin/env python3

# Construct the data and neural model for this experiment.
# Before a directory is generated, the existence of a config file for that directory will be checked,
# if it exists generation is skipped.

import datetime
import importlib
import json
import os
import sys

import numpy
import pandas
import tensorflow

from typing import Iterable
from itertools import product

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

DATASET_MNIST_1 = 'mnist-1'
DATASET_MNIST_2 = 'mnist-2'
DATASETS = [DATASET_MNIST_1]

DATASET_CONFIG = {
    DATASET_MNIST_1: {
        "name": DATASET_MNIST_1,
        "class-size": 10,
        "train-sizes": [40, 60, 80],
        "valid-size": 1000,
        "test-size": 1000,
        "num-splits": 1,
        "num-digits": 1,
        "max-sum": 18,
        "overlaps": [0.0, 0.5, 1.0],
    },
    DATASET_MNIST_2: {
        "name": DATASET_MNIST_2,
        "class-size": 10,
        "train-sizes": [40, 60, 80],
        "valid-size": 1000,
        "test-size": 1000,
        "num-splits": 1,
        "num-digits": 2,
        "max-sum": 198,
        "overlaps": [0.0, 0.5, 1.0],
    },
}

CONFIG_FILENAME = "config.json"


def normalize_images(images):
    (numImages, width, height) = images.shape

    # Flatten out the images into a 1d array.
    images = images.reshape(numImages, width * height)

    # Normalize the greyscale intensity to [0,1].
    images = images / 255.0

    # Round so that the output is significantly smaller.
    images = images.round(4)

    return images


def digits_to_number(digits):
    number = 0
    for digit in digits:
        number *= 10
        number += digit
    return number


def digits_to_sum(digits, n_digits):
    return digits_to_number(digits[:n_digits]) + digits_to_number(digits[n_digits:])


def generate_split(config, features, labels, start_index, end_index):
    numpy.random.seed(config['seed'])

    all_indexes = numpy.array(range(len(features)))
    numpy.random.shuffle(all_indexes)

    indexes = all_indexes[start_index:end_index]
    for _ in range(int(len(indexes) * config['overlap'])):
        indexes = numpy.append(indexes, indexes[numpy.random.randint(0, end_index - start_index)])

    numpy.random.shuffle(indexes)
    indexes = indexes[:len(indexes) - (len(indexes) % (2 * config['num-digits']))]
    indexes = numpy.unique(indexes.reshape(-1, 2 * config['num-digits']), axis=0)

    sum_labels = numpy.array([digits_to_sum(digits, config['num-digits']) for digits in labels[indexes]])

    return  indexes, sum_labels


def create_entity_data_map(data, entities_train, entities_valid, entities_test):
    digit_entities_train = numpy.unique(entities_train.reshape(-1))
    digit_entities_valid = numpy.unique(entities_valid.reshape(-1))
    digit_entities_test = numpy.unique(entities_test.reshape(-1))

    entities = numpy.concatenate((digit_entities_train, digit_entities_valid, digit_entities_test))
    features = normalize_images(numpy.concatenate((data[0][0], data[1][0])))[entities]
    labels = numpy.concatenate((data[0][1], data[1][1]))[entities]

    entity_data_map = numpy.concatenate((entities.reshape(-1, 1), features, labels.reshape(-1, 1)), axis=1).tolist()
    return [[int(row[0])] + row[1:] for row in entity_data_map]


def create_image_sum_data(config, entities, sum_labels):
    image_sum_target = []
    image_sum_truth = []
    for index_i in range(len(entities)):
        for index_j in range(config['max-sum'] + 1):
            image_sum_target.append(list(entities[index_i]) + [index_j])
            image_sum_truth.append(list(entities[index_i]) + [index_j] + [1 if index_j == sum_labels[index_i] else 0])

    return image_sum_target, image_sum_truth


def create_neural_data(config, entities):
    neural_data = []
    for index_i in range(len(entities)):
        for index_j in range(config['class-size']):
            neural_data.append(list(entities[index_i]) + [index_j])

    return neural_data


def write_specific_data(config, out_dir, data):
    entities_train, sum_train = generate_split(config, data[0][0], data[0][1], 0, config['train-size'])
    entities_valid, sum_valid = generate_split(config, data[0][0], data[0][1], config['train-size'], config['train-size'] + config['valid-size'])
    entities_test, sum_test = generate_split(config, data[1][0], data[1][1], 0, config['test-size'])

    util.write_psl_file(os.path.join(out_dir, 'image-sum-target-block-train.txt'), entities_train)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-target-block-valid.txt'), entities_valid)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-target-block-test.txt'), entities_test)

    entity_data_map = create_entity_data_map(data, entities_train, entities_valid, entities_test)
    util.write_psl_file(os.path.join(out_dir, 'entity-data-map.txt'), entity_data_map)

    image_sum_target, image_sum_truth = create_image_sum_data(config, entities_train, sum_train)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-target-train.txt'), image_sum_target)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-truth-train.txt'), image_sum_truth)

    image_sum_target, image_sum_truth = create_image_sum_data(config, entities_valid, sum_valid)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-target-valid.txt'), image_sum_target)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-truth-valid.txt'), image_sum_truth)

    image_sum_target, image_sum_truth = create_image_sum_data(config, entities_test, sum_test)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-target-test.txt'), image_sum_target)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-truth-test.txt'), image_sum_truth)

    neural_target_train = create_neural_data(config, numpy.unique(entities_train.reshape(-1)).reshape(-1, 1))
    util.write_psl_file(os.path.join(out_dir, 'neural-target-train.txt'), neural_target_train)

    neural_target_valid = create_neural_data(config, numpy.unique(entities_valid.reshape(-1)).reshape(-1, 1))
    util.write_psl_file(os.path.join(out_dir, 'neural-target-valid.txt'), neural_target_valid)

    neural_target_test = create_neural_data(config, numpy.unique(entities_test.reshape(-1)).reshape(-1, 1))
    util.write_psl_file(os.path.join(out_dir, 'neural-target-test.txt'), neural_target_test)

    util.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def create_sum_data(config):
    number_sum = []
    possible_digits = []
    for index_i in range(config['class-size']):
        for index_j in range(config['class-size']):
            number_sum.append([index_i, index_j, index_i + index_j])
            possible_digits.append([index_i, index_i + index_j])

    return number_sum, possible_digits

def write_shared_data(config, out_dir):
    number_sum, possible_digits = create_sum_data(config)
    util.write_psl_file(os.path.join(out_dir, 'number-sum.txt'), number_sum)
    util.write_psl_file(os.path.join(out_dir, 'possible-digits.txt'), possible_digits)

    util.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def fetch_data(config):
    return tensorflow.keras.datasets.mnist.load_data("mnist.npz")


def main():
    for dataset_id in DATASETS:
        config = DATASET_CONFIG[dataset_id]

        # TODO(Connor): Add data generation for mnist 2.
        if config['name'] == DATASET_MNIST_2:
            print("Mnist 2 data generation is not yet supported.")
            continue

        shared_out_dir = os.path.join(THIS_DIR, "..", "data", dataset_id)
        os.makedirs(shared_out_dir, exist_ok=True)
        if os.path.isfile(os.path.join(shared_out_dir, CONFIG_FILENAME)):
            print("Shared data already exists for %s. Skipping generation." % dataset_id)
        else:
            print("Generating shared data for %s." % dataset_id)
            write_shared_data(config, shared_out_dir)

        for split in range(config['num-splits']):
            config['seed'] = split
            for train_size in config['train-sizes']:
                config['train-size'] = train_size
                for overlap in config['overlaps']:
                    config['overlap'] = overlap

                    out_dir = os.path.join(shared_out_dir, str(split), str(train_size), str(overlap))
                    os.makedirs(out_dir, exist_ok=True)

                    if os.path.isfile(os.path.join(out_dir, CONFIG_FILENAME)):
                        print("Data already exists for %s. Skipping generation." % out_dir)
                        continue

                    print("Generating data for %s." % out_dir)
                    data = fetch_data(config)
                    write_specific_data(config, out_dir, data)


if __name__ == '__main__':
    main()
