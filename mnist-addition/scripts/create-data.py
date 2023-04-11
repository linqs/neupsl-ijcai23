#!/usr/bin/env python3

# Construct the data and neural model for this experiment.
# Before a directory is generated, the existence of a config file for that directory will be checked,
# if it exists generation is skipped.

import datetime
import json
import os
import random

import numpy as np
import pandas as pd
import pslpython.neupsl
import tensorflow as tf

from typing import Iterable
from itertools import product

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

# In this path, include the format string for the subpath.
# The subpath itself may have more subs, but only one will occur for each child.
DATA_PATH = os.path.abspath(os.path.join(THIS_DIR, "../data/mnist-addition"))

EVAL_DIRNAME = 'eval'
LEARN_DIRNAME = 'learn'

UNTRAINED_MODEL_H5_FILE_NAME = 'neuralclassifier_model_untrained.h5'
UNTRAINED_MODEL_TF_FILE_NAME = 'neuralclassifier_model_untrained_tf'

# This is also the order of the labels in the output layer.
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

SIGNIFICANT_DIGITS = 4

EPOCHS = 0
LEARNING_RATES = [1e-2, 1e-3, 1e-4]

N_FOLDS = 10
TRAINING_SIZES = [40, 60, 75, 80, 150, 300, 600, 6000, 50000]
FULL_TRAINING_SET_SIZE = 60000
VALIDATION_SET_SIZE = 1000
TEST_SET_SIZE = 1000
OVERLAP_RESAMPLE_PROPORTIONS = [0.0, 0.5, 1.0, 2.0]

N_DIGITS = [1, 2]

# MNIST images are 28 x 28 = 784.
MNIST_DIMENSION = 28
INPUT_SIZE = MNIST_DIMENSION * MNIST_DIMENSION

# We will keep cycling through the seeds if there are not enough.
# None means we make a new seed (which may be determined by the previous seed).
SEEDS = [None]


def normalize_images(images):
    (numImages, width, height) = images.shape

    # Flatten out the images into a 1d array.
    images = images.reshape(numImages, width * height)

    # Normalize the greyscale intensity to [0,1].
    images = images / 255.0

    # Round so that the output is significantly smaller.
    images = images.round(SIGNIFICANT_DIGITS)

    return images


def load_features(path):
    dataframe = pd.read_csv(path, header=None, sep='\t', index_col=0)
    dataframe.iloc[0, :] = dataframe.iloc[0, :].astype('int')
    return dataframe.values


def load_truth(path):
    dataframe = pd.read_csv(path, header=None, sep='\t', index_col=0)
    dataframe.iloc[0, :] = dataframe.iloc[0, :].astype('int')
    truth_data = dataframe.values

    y = np.array([np.zeros(10) for _ in truth_data])
    for i, label in enumerate(truth_data):
        y[i][label] = 1
    return y


def write_file(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")


def evaluate(model, images, labels, verbose=1):
    loss, accuracy = model.evaluate(images, labels, verbose=verbose)
    return loss, accuracy


def train(model_wrapper, raw_train_features, raw_train_labels, epochs=EPOCHS):
    assert(len(raw_train_features) == len(raw_train_labels))

    indexes = list(range(len(raw_train_features)))

    for epoch in range(epochs):
        random.shuffle(indexes)

        labels = np.array([raw_train_labels[index] for index in indexes])
        images = np.array([raw_train_features[index] for index in indexes])

        model_wrapper.model.fit(images, labels)

        loss, accuracy = evaluate(model_wrapper.model, images, labels, verbose=0)
        print("Epoch %d - Loss: %f, Accuracy: %f" % (epoch + 1, loss, accuracy))

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


def write_data(out_dir: str, n_digits, overlap_resample_proportion, labels, train_size, validation_size, seed):
    # Load MNIST dataset.
    (digit_features, digit_labels), (digit_features_test, digit_labels_test) = tf.keras.datasets.mnist.load_data("mnist.npz")

    # Sample training and validation digits.
    indices = np.array(range(len(digit_features)))
    unique_test_indices = np.array(range(len(digit_features_test)))
    if seed != -1:
        print("Shuffling")
        np.random.seed(seed)
        np.random.shuffle(indices)

    unique_train_digit_indices = indices[: (2 * n_digits) * train_size]
    train_digit_indices = indices[: (2 * n_digits) * train_size]
    if seed != -1:
        for _ in range((2 * n_digits) * int(overlap_resample_proportion * train_size)):
            sample_index = np.random.randint(0, (2 * n_digits) * train_size)
            train_digit_indices = np.append(train_digit_indices, train_digit_indices[sample_index])
        np.random.shuffle(train_digit_indices)

    unique_validation_digit_indices = indices[(2 * n_digits) * train_size: (2 * n_digits) * train_size + (2 * n_digits) * validation_size]
    validation_digit_indices = indices[(2 * n_digits) * train_size: (2 * n_digits) * train_size + (2 * n_digits) * validation_size]
    if seed != -1:
        for _ in range((2 * n_digits) * int(overlap_resample_proportion * validation_size)):
            sample_index = np.random.randint(0, (2 * n_digits) * validation_size)
            validation_digit_indices = np.append(validation_digit_indices, validation_digit_indices[sample_index])
        np.random.shuffle(validation_digit_indices)

    test_indices = np.array(range(len(digit_features_test)))
    np.random.shuffle(test_indices)
    test_indices = test_indices[:TEST_SET_SIZE]
    if overlap_resample_proportion != 0.0:
        for _ in range((2 * n_digits) * int(overlap_resample_proportion * (TEST_SET_SIZE // (2 * n_digits)))):
            sample_index = np.random.randint(0, TEST_SET_SIZE)
            test_indices = np.append(test_indices, test_indices[sample_index])
        np.random.shuffle(test_indices)

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


def create_addition_dataset_and_models(n_digits, overlap_resample_proportion, labels, train_size, validation_size, fold):
    """Generate dataset for MNIST addition: [img, img, ..., img] + [img, img, ..., img].
    :param n_digits: Number of digits representing each number.
    :param overlap_resample_proportion: The proportion of data that is resampled to create overlaps.
    :param labels: The possible sum labels.
    :param train_size: Size of training dataset
    :param validation_size: Size of validation dataset
    :param fold: The fold identifier. Used as the seed for RNG generating split
    """
    out_dir = os.path.join(DATA_PATH, "n_digits::{:01d}/fold::{:02d}/train_size::{:05d}/overlap::{:.2f}".format(n_digits, fold, train_size, overlap_resample_proportion))
    config_path = os.path.join(out_dir, 'config.json')
    if os.path.isfile(config_path):
        print("Found existing config file, skipping generation. " + config_path)
        return

    write_data(out_dir, n_digits, overlap_resample_proportion, labels, train_size, validation_size, fold)

    config = {
        'architecture': {
            'reference': [
                'https://github.com/ML-KULeuven/deepproblog/blob/master/src/deepproblog/examples/MNIST/network.py#L44',
                'https://arxiv.org/pdf/1907.08194.pdf#page=30',
            ]
        },
        'network': {},
        'validationSamples': int(validation_size),
        'numTrainPairs': int(train_size),
        'fold': int(fold),
        'timestamp': str(datetime.datetime.now()),
    }

    for i, learning_rate in enumerate(LEARNING_RATES):
        create_model(learning_rate, n_digits, overlap_resample_proportion, train_size, fold)

    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)


def main():
    for n_digits in N_DIGITS:
        train_sizes = np.array(TRAINING_SIZES) // (2 * n_digits)
        validation_size = VALIDATION_SET_SIZE // (2 * n_digits)
        max_number = sum([9 * (10 ** i) for i in range(n_digits)])
        labels = list(range(2 * max_number + 1))
        for train_size in train_sizes:
            for overlap_resample_proportion in OVERLAP_RESAMPLE_PROPORTIONS:
                for fold in range(-1, N_FOLDS):
                    create_addition_dataset_and_models(n_digits, overlap_resample_proportion, labels, train_size, validation_size, fold)


if __name__ == '__main__':
    main()
