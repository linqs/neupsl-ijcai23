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
DATASETS = [DATASET_MNIST_1, DATASET_MNIST_2]

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

SIGNIFICANT_DIGITS = 4
CONFIG_FILENAME = "config.json"

def normalize_images(images):
    (numImages, width, height) = images.shape

    # Flatten out the images into a 1d array.
    images = images.reshape(numImages, width * height)

    # Normalize the greyscale intensity to [0,1].
    images = images / 255.0

    # Round so that the output is significantly smaller.
    images = images.round(SIGNIFICANT_DIGITS)

    return images


def addition_to_json(features, labels):
    """
    Convert to JSON, for easy comparisons with other systems.

    Format is [EXAMPLE, ...]
    EXAMPLE :- [ARGS, expected_result]
    ARGS :- [MULTI_DIGIT_NUMBER, ...]
    MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
    """
    data = [(features[i].tolist(), labels[i].tolist()) for i in range(len(features))]
    return json.dumps(data)


def digits_to_number(digits: Iterable[int]) -> int:
    number = 0
    for d in digits:
        number *= 10
        number += d
    return number


def digits_to_sum(digits: Iterable[int], n_digits: int) -> int:
    return digits_to_number(digits[: n_digits]) + digits_to_number(digits[n_digits:])


def write_data(config, out_dir, data):
    numpy.random.seed(config['seed'])

    train_partition_indices = numpy.array(range(len(data[0][0])))
    numpy.random.shuffle(train_partition_indices)

    test_partition_indices = numpy.array(range(len(data[1][0])))
    numpy.random.shuffle(test_partition_indices)

    train_indexes = train_partition_indices[:config['train-size']]
    valid_indexes = train_partition_indices[config['train-size']:config['train-size'] + config['valid-size']]
    test_indexes = test_partition_indices[:config['test-size']]

    for _ in range(int(config['train-size'] * config['overlap'])):
        train_indexes = numpy.append(train_indexes, train_indexes[numpy.random.randint(0, config['train-size'])])

    for _ in range(int(config['valid-size'] * config['overlap'])):
        valid_indexes = numpy.append(valid_indexes, valid_indexes[numpy.random.randint(0, config['valid-size'])])

    for _ in range(int(config['test-size'] * config['overlap'])):
        test_indexes = numpy.append(test_indexes, test_indexes[numpy.random.randint(0, config['test-size'])])

    numpy.random.shuffle(train_indexes)
    numpy.random.shuffle(valid_indexes)
    numpy.random.shuffle(test_indexes)

    train_indexes = train_indexes[:len(train_indexes) - (len(train_indexes) % (2 * config['num-digits']))]
    valid_indexes = valid_indexes[:len(valid_indexes) - (len(valid_indexes) % (2 * config['num-digits']))]
    test_indexes = test_indexes[:len(test_indexes) - (len(test_indexes) % (2 * config['num-digits']))]

    train_indexes = train_indexes.reshape(-1, 2 * config['num-digits'])
    valid_indexes = valid_indexes.reshape(-1, 2 * config['num-digits'])
    test_indexes = test_indexes.reshape(-1, 2 * config['num-digits'])

    return

    # Create sum data and labels.
    addition_train_indices = np.array([np.array([train_digit_indices[i + j] for j in range(2 * n_digits)])
                                       for i in range(0, len(train_digit_indices), (2 * n_digits))])
    addition_train_indices = np.unique(addition_train_indices, axis=0)
    addition_validation_indices = np.array([np.array([validation_digit_indices[i + j] for j in range(2 * n_digits)])
                                            for i in range(0, len(validation_digit_indices), (2 * n_digits))])
    addition_validation_indices = np.unique(addition_validation_indices, axis=0)
    addition_test_indices = np.array([np.array([test_indices[i + j] for j in range(2 * n_digits)])
                                      for i in range(0, len(test_indices), (2 * n_digits))])
    addition_test_indices = np.unique(addition_test_indices, axis=0)

    addition_x_train = np.array([np.array([digit_features[i] for i in example]) for example in addition_train_indices])
    addition_y_train = np.array([digits_to_sum(np.array([digit_labels[i] for i in example]), n_digits) for example in addition_train_indices])
    addition_x_validation = np.array([np.array([digit_features[i] for i in example]) for example in addition_validation_indices])
    addition_y_validation = np.array([digits_to_sum(np.array([digit_labels[i] for i in example]), n_digits) for example in addition_validation_indices])
    addition_x_test = np.array([np.array([digit_features_test[i] for i in example]) for example in addition_test_indices])
    addition_y_test = np.array([digits_to_sum(np.array([digit_labels_test[i] for i in example]), n_digits) for example in addition_test_indices])

    number_sums = []
    digits = product(range(10), repeat=1)
    digit_pairs = product(digits, repeat=2)
    for digit_pair in digit_pairs:
        number_sums += [list(digit_pair[0]) + list(digit_pair[1])
                        + [digits_to_sum(list(digit_pair[0]) + list(digit_pair[1]), 1)]]

    possible_digits = []
    digits = product(range(10), repeat=1)
    digit_pairs = product(digits, repeat=2)
    for digit_pair in digit_pairs:
        possible_digits += [[digit_pair[0][0], digit_pair[0][0] + digit_pair[1][0]]]

    # Predicates used for multi digit addition.
    if n_digits == 2:
        # Number sums 2:
        number_sums_2 = []
        digits = product(range(10), repeat=1)
        digit_pairs = product(digits, repeat=4)
        for digit_pair in digit_pairs:
            number_sums_2 += [list(digit_pair[0]) + list(digit_pair[1]) + list(digit_pair[2]) + list(digit_pair[3])
                              + [digits_to_sum(list(digit_pair[0]) + list(digit_pair[1]) + list(digit_pair[2]) + list(digit_pair[3]), 2)]]

        # Possible tens place digits.
        digits_sums = product(range(10), repeat=4)
        possible_tens_digits_dict = {}
        for digits_sum in digits_sums:
            if digits_sum[0] in possible_tens_digits_dict:
                possible_tens_digits_dict[digits_sum[0]].add(10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3])
            else:
                possible_tens_digits_dict[digits_sum[0]] = {10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3]}

        possible_tens_digits = []
        for key in possible_tens_digits_dict:
            for value in possible_tens_digits_dict[key]:
                possible_tens_digits.append([key, value])

        # Possible ones place digits.
        digits_sums = product(range(10), repeat=4)
        possible_ones_digits_dict = {}
        for digits_sum in digits_sums:
            if digits_sum[1] in possible_ones_digits_dict:
                possible_ones_digits_dict[digits_sum[1]].add(10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3])
            else:
                possible_ones_digits_dict[digits_sum[1]] = {10 * digits_sum[0] + digits_sum[1] + 10 * digits_sum[2] + digits_sum[3]}

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

        learn_image_digit_sum_targets = []
        for example_indices in addition_train_indices:
            learn_image_digit_sum_targets += [[example_indices[0], example_indices[2], k] for k in range(19)]
            learn_image_digit_sum_targets += [[example_indices[1], example_indices[3], k] for k in range(19)]
        learn_image_digit_sum_targets = np.unique(learn_image_digit_sum_targets, axis=0).tolist()

        eval_image_digit_sum_targets = []
        for example_indices in addition_validation_indices:
            eval_image_digit_sum_targets += [[example_indices[0], example_indices[2], k] for k in range(19)]
            eval_image_digit_sum_targets += [[example_indices[1], example_indices[3], k] for k in range(19)]
        eval_image_digit_sum_targets = np.unique(eval_image_digit_sum_targets, axis=0).tolist()

        test_image_digit_sum_targets = []
        for example_indices in addition_test_indices:
            test_image_digit_sum_targets += [[example_indices[0], example_indices[2], k] for k in range(19)]
            test_image_digit_sum_targets += [[example_indices[1], example_indices[3], k] for k in range(19)]
        test_image_digit_sum_targets = np.unique(test_image_digit_sum_targets, axis=0).tolist()

    # Write train and test features and labels.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    write_file(os.path.join(out_dir, '../../../numbersum_obs.txt'), number_sums)
    write_file(os.path.join(out_dir, '../../../possibledigits_obs.txt'), possible_digits)

    if n_digits == 2:
        write_file(os.path.join(out_dir, '../../../numbersum2_obs.txt'), number_sums_2)
        write_file(os.path.join(out_dir, '../../../placednumbersum_obs.txt'), placed_number_sums)
        write_file(os.path.join(out_dir, '../../../possibleonessums_obs.txt'), possible_ones_sums)
        write_file(os.path.join(out_dir, '../../../possibleonesdigits_obs.txt'), possible_ones_digits)
        write_file(os.path.join(out_dir, '../../../possibletenssums_obs.txt'), possible_tens_sums)
        write_file(os.path.join(out_dir, '../../../possibletensdigits_obs.txt'), possible_tens_digits)

    learn_out_dir = os.path.join(out_dir, 'learn')
    if not os.path.exists(learn_out_dir):
        os.makedirs(learn_out_dir)

    eval_out_dir = os.path.join(out_dir, 'eval')
    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    test_out_dir = os.path.join(out_dir, 'test')
    if not os.path.exists(test_out_dir):
        os.makedirs(test_out_dir)

    with open(os.path.join(learn_out_dir, 'baseline_data.txt'), 'w') as file:
        file.write(addition_to_json(addition_x_train, addition_y_train))

    with open(os.path.join(eval_out_dir, 'baseline_data.txt'), 'w') as file:
        file.write(addition_to_json(addition_x_validation, addition_y_validation))

    with open(os.path.join(test_out_dir, 'baseline_data.txt'), 'w') as file:
        file.write(addition_to_json(addition_x_test, addition_y_test))

    learn_image_sum_target_block = [addition_train_indices[i].tolist() for i in range(len(addition_train_indices))]
    learn_image_sum_targets = [example_indices.tolist() + [k] for example_indices in addition_train_indices for k in labels]
    learn_image_sum_truth = [addition_train_indices[i].tolist() + [addition_y_train[i]] for i in range(len(addition_y_train))]
    learn_predicted_number_targets = [[unique_train_digit_indices[i]] + [k] for i in range((2 * n_digits) * train_size) for k in range(10)]
    learn_predicted_number_truth = [[unique_train_digit_indices[i]] + [digit_labels[unique_train_digit_indices[i]]] for i in range((2 * n_digits) * train_size)]
    learn_neuralclassifier_features = [[unique_train_digit_indices[i]] + normalize_images(np.array([digit_features[unique_train_digit_indices[i]]]))[0].tolist() for i in range((2 * n_digits) * train_size)]
    write_file(os.path.join(learn_out_dir, 'neuralclassifier_features.txt'), learn_neuralclassifier_features)
    write_file(os.path.join(learn_out_dir, 'neuralclassifier_labels.txt'), [[i] for i in range(10)])
    write_file(os.path.join(learn_out_dir, 'imagesumtargetblock_obs.txt'), learn_image_sum_target_block)
    write_file(os.path.join(learn_out_dir, 'imagesum_targets.txt'), learn_image_sum_targets)
    write_file(os.path.join(learn_out_dir, 'imagesum_truth.txt'), learn_image_sum_truth)
    write_file(os.path.join(learn_out_dir, 'predicted_number_targets.txt'), learn_predicted_number_targets)
    write_file(os.path.join(learn_out_dir, 'predicted_number_truth.txt'), learn_predicted_number_truth)

    if n_digits == 2:
        write_file(os.path.join(learn_out_dir, 'imagedigitsum_targets.txt'), learn_image_digit_sum_targets)

    eval_image_sum_target_block = [addition_validation_indices[i].tolist() for i in range(len(addition_validation_indices))]
    eval_image_sum_targets = [example_indices.tolist() + [k] for example_indices in addition_validation_indices for k in labels]
    eval_image_sum_truth = [addition_validation_indices[i].tolist() + [addition_y_validation[i]] for i in range(len(addition_y_validation))]
    eval_predicted_number_targets = [[unique_validation_digit_indices[i]] + [k] for i in range((2 * n_digits) * validation_size) for k in range(10)]
    eval_predicted_number_truth = [[unique_validation_digit_indices[i]] + [digit_labels[unique_validation_digit_indices[i]]] for i in range((2 * n_digits) * validation_size)]
    eval_neuralclassifier_features = [[unique_validation_digit_indices[i]] + normalize_images(np.array([digit_features[unique_validation_digit_indices[i]]]))[0].tolist() for i in range((2 * n_digits) * validation_size)]
    write_file(os.path.join(eval_out_dir, 'neuralclassifier_features.txt'), eval_neuralclassifier_features)
    write_file(os.path.join(eval_out_dir, 'neuralclassifier_labels.txt'), [[i] for i in range(10)])
    write_file(os.path.join(eval_out_dir, 'imagesumtargetblock_obs.txt'), eval_image_sum_target_block)
    write_file(os.path.join(eval_out_dir, 'imagesum_targets.txt'), eval_image_sum_targets)
    write_file(os.path.join(eval_out_dir, 'imagesum_truth.txt'), eval_image_sum_truth)
    write_file(os.path.join(eval_out_dir, 'predicted_number_targets.txt'), eval_predicted_number_targets)
    write_file(os.path.join(eval_out_dir, 'predicted_number_truth.txt'), eval_predicted_number_truth)

    if n_digits == 2:
        write_file(os.path.join(eval_out_dir, 'imagedigitsum_targets.txt'), eval_image_digit_sum_targets)

    test_image_sum_target_block = [addition_test_indices[i].tolist() for i in range(len(addition_test_indices))]
    test_image_sum_targets = [example_indices.tolist() + [k] for example_indices in addition_test_indices for k in labels]
    test_image_sum_truth = [addition_test_indices[i].tolist() + [addition_y_test[i]] for i in range(len(addition_y_test))]
    test_predicted_number_targets = [[unique_test_indices[i]] + [k] for i in range(len(unique_test_indices)) for k in range(10)]
    test_predicted_number_truth = [[unique_test_indices[i]] + [digit_labels_test[unique_test_indices[i]]] for i in range(len(unique_test_indices))]
    test_neuralclassifier_features = [[unique_test_indices[i]] + normalize_images(np.array([digit_features_test[unique_test_indices[i]]]))[0].tolist() for i in range(len(unique_test_indices))]
    write_file(os.path.join(test_out_dir, 'neuralclassifier_features.txt'), test_neuralclassifier_features)
    write_file(os.path.join(test_out_dir, 'neuralclassifier_labels.txt'), [[i] for i in range(10)])
    write_file(os.path.join(test_out_dir, 'imagesumtargetblock_obs.txt'), test_image_sum_target_block)
    write_file(os.path.join(test_out_dir, 'imagesum_targets.txt'), test_image_sum_targets)
    write_file(os.path.join(test_out_dir, 'imagesum_truth.txt'), test_image_sum_truth)
    write_file(os.path.join(test_out_dir, 'predicted_number_targets.txt'), test_predicted_number_targets)
    write_file(os.path.join(test_out_dir, 'predicted_number_truth.txt'), test_predicted_number_truth)

    if n_digits == 2:
        write_file(os.path.join(test_out_dir, 'imagedigitsum_targets.txt'), test_image_digit_sum_targets)


def fetch_data(config):
    return tensorflow.keras.datasets.mnist.load_data("mnist.npz")

def main():
    for dataset_id in DATASETS:
        config = DATASET_CONFIG[dataset_id]
        for split in range(config['num-splits']):
            config['seed'] = split
            for train_size in config['train-sizes']:
                config['train-size'] = train_size
                for overlap in config['overlaps']:
                    config['overlap'] = overlap

                    out_dir = os.path.join(THIS_DIR, "..", "data", dataset_id, str(split), str(train_size), str(overlap))
                    if os.path.isfile(os.path.join(out_dir, CONFIG_FILENAME)):
                        print("Data already exists for %s. Skipping generation." % out_dir)
                        continue

                    data = fetch_data(config)
                    write_data(config, out_dir, data)


if __name__ == '__main__':
    main()
