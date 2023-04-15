#!/usr/bin/env python3

# Construct the data and neural model for this experiment.
# Before a directory is generated, the existence of a config file for that directory will be checked,
# if it exists generation is skipped.

import importlib
import os
import sys

import numpy
import tensorflow


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
        "num-splits": 2,
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


def create_image_sum_data(config, sum_entities, sum_labels):
    image_sum_target = []
    image_sum_truth = []
    for index_i in range(len(sum_entities)):
        for index_j in range(config['max-sum'] + 1):
            image_sum_target.append(list(sum_entities[index_i]) + [index_j])
            image_sum_truth.append(list(sum_entities[index_i]) + [index_j] + [1 if index_j == sum_labels[index_i] else 0])

    return image_sum_target, image_sum_truth


def create_image_data(config, entities):
    image_target = []
    for index_i in range(len(entities)):
        for index_j in range(config['class-size']):
            image_target.append(list(entities[index_i]) + [index_j])

    return image_target


def write_specific_data(config, out_dir, features, labels):
    total_image_entities = numpy.array([], dtype=numpy.int32)
    indexes = {
        'train': [0, config['train-size']],
        'valid': [config['train-size'], config['train-size'] + config['valid-size']],
        'test': [config['train-size'] + config['valid-size'], config['train-size'] + config['valid-size'] + config['test-size']]
    }

    for partition in ['train', 'valid', 'test']:
        image_sum_entities, image_sum_labels = generate_split(config, features, labels, indexes[partition][0], indexes[partition][1])
        image_sum_target, image_sum_truth = create_image_sum_data(config, image_sum_entities, image_sum_labels)

        image_entities = numpy.unique(image_sum_entities.reshape(-1)).reshape(-1, 1)
        image_target = create_image_data(config, image_entities)

        total_image_entities = numpy.append(total_image_entities, image_entities)

        util.write_psl_file(os.path.join(out_dir, f'image-sum-block-{partition}.txt'), image_sum_entities)
        util.write_psl_file(os.path.join(out_dir, f'image-sum-target-{partition}.txt'), image_sum_target)
        util.write_psl_file(os.path.join(out_dir, f'image-sum-truth-{partition}.txt'), image_sum_truth)
        util.write_psl_file(os.path.join(out_dir, f'image-target-{partition}.txt'), image_target)

    entity_data_map = create_entity_data_map(features, labels, total_image_entities)
    util.write_psl_file(os.path.join(out_dir, 'entity-data-map.txt'), entity_data_map)

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


def fetch_data():
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data("mnist.npz")

    # TODO(Connor): Currently concatenating the train and test partitions, and random sample.
    return numpy.concatenate((x_train, x_test)), numpy.concatenate((y_train, y_test))


def main():
    for dataset_id in DATASETS:
        config = DATASET_CONFIG[dataset_id]

        # TODO(Connor): Add data generation for mnist 2.
        if config['name'] == DATASET_MNIST_2:
            print("Mnist 2 data generation is not yet supported.")
            continue

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
