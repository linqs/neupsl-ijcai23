#!/usr/bin/env python3

# Construct the data and neural model for this experiment.
# Before a directory is generated, the existence of a config file for that directory will be checked,
# if it exists generation is skipped.

import importlib
import os
import sys

import numpy
import tensorflow

from itertools import product


THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

DATASET_MNIST_1 = 'mnist-1'
DATASET_MNIST_2 = 'mnist-2'
DATASETS = [DATASET_MNIST_1, DATASET_MNIST_2]

DATASET_CONFIG = {
    DATASET_MNIST_1: {
        "name": DATASET_MNIST_1,
        "num-digits": 1,
        "class-size": 10,
        "num-splits": 5,
        "train-sizes": [40, 60, 80],
        "valid-size": 100,
        "test-size": 1000,
        "overlaps": [0.0, 0.5, 1.0],
    },
    DATASET_MNIST_2: {
        "name": DATASET_MNIST_2,
        "num-digits": 2,
        "class-size": 10,
        "num-splits": 5,
        "train-sizes": [40, 60, 80],
        "valid-size": 100,
        "test-size": 1000,
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


def generate_split(config, labels, indexes):
    for _ in range(int(len(indexes) * config['overlap'])):
        indexes = numpy.append(indexes, indexes[numpy.random.randint(0, len(indexes))])

    indexes = indexes[:len(indexes) - (len(indexes) % (2 * config['num-digits']))]
    indexes = numpy.unique(indexes.reshape(-1, 2 * config['num-digits']), axis=0)

    sum_labels = numpy.array([digits_to_sum(digits, config['num-digits']) for digits in labels[indexes]])

    return  indexes, sum_labels


def create_entity_data_map(features, labels, entities):
    features = normalize_images(features)[entities]
    labels = labels[entities].reshape(-1, 1)
    entities = entities.reshape(-1, 1)

    entity_data_map = numpy.concatenate((entities, features, labels), axis=1).tolist()
    return [[int(row[0])] + row[1:] for row in entity_data_map]


def create_image_digit_sum_data(config, sum_entities):
    image_digit_sum_targets = []
    for example_indices in sum_entities:
        for digit in range(config['num-digits']):
            image_digit_sum_targets += [[example_indices[digit], example_indices[config['num-digits'] + digit], digit_sum] for digit_sum in range(config['class-size'] * 2 - 1)]
    image_digit_sum_targets = numpy.unique(image_digit_sum_targets, axis=0).tolist()

    return image_digit_sum_targets


def create_image_sum_data(config, sum_entities, sum_labels):
    image_sum_target = []
    image_sum_truth = []
    for index_i in range(len(sum_entities)):
        for index_j in range(config['max-sum'] + 1):
            image_sum_target.append(list(sum_entities[index_i]) + [index_j])
            image_sum_truth.append(list(sum_entities[index_i]) + [index_j] + [1 if index_j == sum_labels[index_i] else 0])
    image_sum_target = numpy.unique(image_sum_target, axis=0).tolist()
    image_sum_truth = numpy.unique(image_sum_truth, axis=0).tolist()

    return image_sum_target, image_sum_truth


def create_image_data(config, entities):
    image_target = []
    for index_i in range(len(entities)):
        for index_j in range(config['class-size']):
            image_target.append(list(entities[index_i]) + [index_j])

    return image_target


def write_specific_data(config, out_dir, features, labels):
    total_image_entities = numpy.array([], dtype=numpy.int32)

    numpy.random.seed(config['seed'])

    all_indexes = numpy.array(range(len(features)))
    numpy.random.shuffle(all_indexes)

    partition_indexes = {
        'train': all_indexes[0: config['train-size']],
        'valid': all_indexes[config['train-size']: config['train-size'] + config['valid-size']],
        'test': all_indexes[config['train-size'] + config['valid-size']: config['train-size'] + config['valid-size'] + config['test-size']]
    }

    for partition in ['train', 'valid', 'test']:
        image_sum_entities, image_sum_labels = generate_split(config, labels, partition_indexes[partition])
        image_sum_target, image_sum_truth = create_image_sum_data(config, image_sum_entities, image_sum_labels)

        if config["num-digits"] != 1:
            image_digit_sum_targets = create_image_digit_sum_data(config, image_sum_entities)
            util.write_psl_file(os.path.join(out_dir, f'image-digit-sum-target-{partition}.txt'), image_digit_sum_targets)

        image_entities = numpy.unique(image_sum_entities.reshape(-1)).reshape(-1, 1)
        image_target = create_image_data(config, image_entities)

        total_image_entities = numpy.append(total_image_entities, image_entities)

        util.write_psl_file(os.path.join(out_dir, f'image-sum-block-{partition}.txt'), image_sum_entities)
        util.write_psl_file(os.path.join(out_dir, f'image-sum-target-{partition}.txt'), image_sum_target)
        util.write_psl_file(os.path.join(out_dir, f'image-sum-truth-{partition}.txt'), image_sum_truth)
        util.write_psl_file(os.path.join(out_dir, f'image-target-{partition}.txt'), image_target)
        util.write_psl_file(os.path.join(out_dir, f'image-digit-labels-{partition}.txt'), list(zip(partition_indexes[partition], labels[partition_indexes[partition]])))

    entity_data_map = create_entity_data_map(features, labels, total_image_entities)
    util.write_psl_file(os.path.join(out_dir, 'entity-data-map.txt'), entity_data_map)

    util.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def create_sum_data_add1(config):
    number_sum = []
    possible_digits = []
    for index_i in range(config['class-size']):
        for index_j in range(config['class-size']):
            number_sum.append([index_i, index_j, index_i + index_j])
            possible_digits.append([index_i, index_i + index_j])

    return number_sum, possible_digits


def create_sum_data_add2(config):
    # Possible tens place digits.
    digits_sums = product(range(10), repeat=4)
    possible_tens_digits_dict = {}
    for digits_sum in digits_sums:
        if digits_sum[0] in possible_tens_digits_dict:
            possible_tens_digits_dict[digits_sum[0]].add(
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3])
        else:
            possible_tens_digits_dict[digits_sum[0]] = {
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3]}

    possible_tens_digits = []
    for key in possible_tens_digits_dict:
        for value in possible_tens_digits_dict[key]:
            possible_tens_digits.append([key, value])

    # Possible ones place digits.
    digits_sums = product(range(10), repeat=4)
    possible_ones_digits_dict = {}
    for digits_sum in digits_sums:
        if digits_sum[1] in possible_ones_digits_dict:
            possible_ones_digits_dict[digits_sum[1]].add(
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3])
        else:
            possible_ones_digits_dict[digits_sum[1]] = {
                10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3]}

    possible_ones_digits = []
    for key in possible_ones_digits_dict:
        for value in possible_ones_digits_dict[key]:
            possible_ones_digits.append([key, value])

    # Placed number sum.
    placed_number_sums = []
    digit_sums = product(range(19), repeat=2)
    for digit_sum in digit_sums:
        placed_number_sums += [[digit_sum[0], digit_sum[1], 10 * digit_sum[0] + digit_sum[1]]]

    # Possible sums.
    possible_ones_sums = []
    possible_tens_sums = []
    digit_sums = product(range(19), repeat=2)
    for digit_sum in digit_sums:
        possible_ones_sums += [[digit_sum[1], 10 * digit_sum[0] + digit_sum[1]]]
        possible_tens_sums += [[digit_sum[0], 10 * digit_sum[0] + digit_sum[1]]]

    return possible_tens_digits, possible_ones_digits, placed_number_sums, possible_ones_sums, possible_tens_sums


def write_shared_data(config, out_dir):
    number_sum, possible_digits = create_sum_data_add1(config)
    util.write_psl_file(os.path.join(out_dir, 'number-sum.txt'), number_sum)
    util.write_psl_file(os.path.join(out_dir, 'possible-digits.txt'), possible_digits)

    if config['num-digits'] == 2:
        possible_tens_digits, possible_ones_digits, placed_number_sums, possible_ones_sums, possible_tens_sums = create_sum_data_add2(config)
        util.write_psl_file(os.path.join(out_dir, 'possible-tens-digits.txt'), possible_tens_digits)
        util.write_psl_file(os.path.join(out_dir, 'possible-ones-digits.txt'), possible_ones_digits)
        util.write_psl_file(os.path.join(out_dir, 'placed-number-sums.txt'), placed_number_sums)
        util.write_psl_file(os.path.join(out_dir, 'possible-ones-sums.txt'), possible_ones_sums)
        util.write_psl_file(os.path.join(out_dir, 'possible-tens-sums.txt'), possible_tens_sums)

    util.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def fetch_data():
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data("mnist.npz")
    return numpy.concatenate((x_train, x_test)), numpy.concatenate((y_train, y_test))


def main():
    for dataset_id in DATASETS:
        config = DATASET_CONFIG[dataset_id]
        config['max-number'] = 10 ** config['num-digits'] - 1
        config['max-sum'] = 2 * config['max-number']

        shared_out_dir = os.path.join(THIS_DIR, "..", "data", "experiment::" + dataset_id)
        os.makedirs(shared_out_dir, exist_ok=True)
        if os.path.isfile(os.path.join(shared_out_dir, CONFIG_FILENAME)):
            print("Shared data already exists for %s. Skipping generation." % dataset_id)
        else:
            print("Generating shared data for %s." % dataset_id)
            write_shared_data(config, shared_out_dir)

        for split in range(config['num-splits']):
            for train_size in config['train-sizes']:
                config['train-size'] = train_size
                config['seed'] = 10 * (10 * train_size + split) + config['num-digits']
                print("Using seed %d." % config['seed'])
                for overlap in config['overlaps']:
                    config['overlap'] = overlap
                    out_dir = os.path.join(shared_out_dir, "split::%01d" % split, "train-size::%04d" % train_size, "overlap::%.2f" % overlap)
                    os.makedirs(out_dir, exist_ok=True)

                    if os.path.isfile(os.path.join(out_dir, CONFIG_FILENAME)):
                        print("Data already exists for %s. Skipping generation." % out_dir)
                        continue

                    print("Generating data for %s." % out_dir)
                    features, labels = fetch_data()
                    write_specific_data(config, out_dir, features, labels)


if __name__ == '__main__':
    main()
