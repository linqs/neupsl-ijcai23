#!/usr/bin/env python3

import datetime
import importlib
import json
import os
import random
import sys
import time

import numpy
import tensorflow


THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..', '..', 'data'))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..', '..', 'scripts'))
util = importlib.import_module("util")

DATASET_MNIST_4X4 = 'mnist-4x4'
DATASETS = [DATASET_MNIST_4X4]

ENTITY_ARGUMENT_INDEXES = [0, 1, 2]

HYPERPARAMETERS = {
    'learning-rate': [1.0e-2, 1.0e-3],
    'epochs': [1000],
    'batch-size': [32],
    'loss': ['binary_crossentropy'],
    'metrics': [[tensorflow.keras.metrics.CategoricalAccuracy(name='acc')]],
}

DEFAULT_PARAMETERS = {
    DATASET_MNIST_4X4: {
        'learning-rate': 1.0e-4,
        'epochs': 1000,
        'batch-size': 32,
        'loss': 'binary_crossentropy',
        'metrics': [tensorflow.keras.metrics.CategoricalAccuracy(name='acc')],
    }
}

VERBOSE = 1
NUM_RANDOM_SEEDS = 1

SAVED_NETWORKS_DIRECTORY = 'saved-networks'
CONFIG_FILENAME = 'config.json'

RUN_HYPERPARAMETER_SEARCH = False


def build_network(data, config, hyperparameters):
    layers = [
        tensorflow.keras.layers.Input(shape=config['input-size']),
        tensorflow.keras.layers.Reshape((config['digit-dimension'], config['digit-dimension'], 1), input_shape=(config['input-size'],)),
        tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'),
        tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tensorflow.keras.layers.Flatten(),
        tensorflow.keras.layers.Dense(units=256, activation='relu'),
        tensorflow.keras.layers.Dense(units=256, activation='relu'),
        tensorflow.keras.layers.Dense(units=128, activation='relu'),
        tensorflow.keras.layers.Dense(config['output-size'], activation='softmax'),
    ]

    model = tensorflow.keras.Sequential(layers = layers)

    model.compile(
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate = hyperparameters['learning-rate']),
        loss = hyperparameters['loss'],
        metrics = hyperparameters['metrics'],
    )

    return model


def fit_model(model, data, config, hyperparameters):
    x_train = data['train']['features']
    y_train = data['train']['labels']
    x_test = data['test']['features']
    y_test = data['test']['labels']
    x_valid = data['valid']['features']
    y_valid = data['valid']['labels']

    early_stop = []
    early_stop.append(tensorflow.keras.callbacks.EarlyStopping(monitor="val_acc", patience=250, restore_best_weights=True))

    train_history = model.fit(
        x=x_train,
        y=y_train,
        epochs=hyperparameters['epochs'],
        batch_size=hyperparameters['batch-size'],
        validation_data=(x_valid, y_valid),
        verbose=VERBOSE,
        callbacks=early_stop,
    )

    test_start_time = time.time()
    _, test_acc = model.evaluate(x_test, y_test, verbose=VERBOSE)
    test_end_time = time.time()
    _, valid_acc = model.evaluate(x_valid, y_valid, verbose=VERBOSE)
    print("Test Accuracy: %f Valid Accuracy: %f" % (test_acc, valid_acc))

    results = {
        'trained': {
            'inference-test-time': (test_end_time - test_start_time),
            'test-accuracy': test_acc,
            'valid-accuracy': valid_acc,
            'train-history': train_history.history,
        },
    }

    return results, model


def build_model(data, config, out_dir, hyperparameters, tuning_hyperparameters):
    os.makedirs(out_dir, exist_ok=True)

    config_path = os.path.join(out_dir, CONFIG_FILENAME)
    if os.path.isfile(config_path):
        print("Existing config file found (%s), skipping..." % (config_path,))
        return

    max_validation_accuracy = 0
    max_results = None
    max_seed = None
    max_model = None

    for index in range(NUM_RANDOM_SEEDS):
        seed = random.randrange(2 ** 64)
        tensorflow.random.set_seed(seed)
        print("Starting: %s -- Current Run: %d -- Seed: %d -- Max Validation Accuracy: %f" % (config['dataset'], index, seed, max_validation_accuracy))

        model = build_network(data, config, hyperparameters)
        results, model = fit_model(model, data, config, hyperparameters)

        if results['trained']['valid-accuracy'] > max_validation_accuracy:
            max_validation_accuracy = results['trained']['valid-accuracy']
            max_results = results
            max_seed = seed
            max_model = model

    if tuning_hyperparameters:
        return max_results

    for partition in ['train', 'test', 'valid']:
        deep_predictions_data = []
        deep_probability_data = []

        deep_predictions = max_model.predict(data[partition]['features'], verbose=VERBOSE)
        for entity, predictions in zip(data[partition]['features'], deep_predictions):
            max_index = numpy.argmax(predictions)
            for index in range(len(predictions)):
                deep_probability_data.append([entity, index] + [predictions[index]])
                deep_predictions_data.append([entity, index, 1.0 if index == max_index else 0.0])

        util.write_psl_file(os.path.join(out_dir, "deep-predictions-%s.txt" % partition), deep_predictions_data)
        util.write_psl_file(os.path.join(out_dir, "deep-probabilities-%s.txt" % partition), deep_probability_data)

    max_model.save(os.path.join(out_dir, "pre-trained-tf"), save_format='tf')

    write_config = hyperparameters.copy()
    write_config['metrics'] = [str(metric) for metric in hyperparameters['metrics']]
    config = {
        'timestamp': str(datetime.datetime.now()),
        'seed': max_seed,
        'generator': os.path.basename(os.path.realpath(__file__)),
        'network': {
            'number-random-seeds': NUM_RANDOM_SEEDS,
            'settings': write_config,
            'results': max_results,
        },
    }

    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)


def load_data(dataset_dir):
    raw_data = util.load_json_file(os.path.join(dataset_dir, "neural-data.json"))
    setting = util.load_json_file(os.path.join(dataset_dir, "config.json"))
    config = {
        'input-size': len(raw_data['train']['digit_features'][0][len(ENTITY_ARGUMENT_INDEXES):] * (len(setting['labels']) ** 2)),
        'output-size': 2,
        'digit-dimension': 28 * len(setting['labels']),
        'puzzle-dimension': len(setting['labels']),
    }

    data = {}
    for partition in raw_data:
        data[partition] = {'features': [], 'labels': []}

        puzzles = {}
        for puzzle in raw_data[partition]['puzzle_truths']:
            puzzles[puzzle[0]] = util.one_hot_encoding(puzzle[1], 2)

        digits = {}
        for visual_digit in raw_data[partition]['digit_features']:
            digits[tuple(visual_digit[:len(ENTITY_ARGUMENT_INDEXES)])] = visual_digit[len(ENTITY_ARGUMENT_INDEXES):]

        count = 0
        for digit in raw_data[partition]['digit_truths']:
            if count % len(setting['labels']) ** 2 == 0:
                data[partition]['features'].append([])
                data[partition]['labels'].append(puzzles[digit[0]])
            data[partition]['features'][-1] += digits[tuple(digit[:len(ENTITY_ARGUMENT_INDEXES)])]
            count += 1

    return data, config


def main():
    hyperparameters = util.enumerate_hyperparameters(HYPERPARAMETERS)

    for dataset in DATASETS:
        dataset_dir = os.path.join(DATA_DIR, 'experiment::' + dataset)
        for split_id in sorted(os.listdir(dataset_dir)):
            split_dir = os.path.join(dataset_dir, split_id)
            for train_size in sorted(os.listdir(split_dir)):
                train_dir = os.path.join(split_dir, train_size)
                for overlap in sorted(os.listdir(train_dir)):
                    overlap_dir = os.path.join(train_dir, overlap)

                    max_valid_accuracy = 0.0
                    hyperparameter_setting = DEFAULT_PARAMETERS[dataset]
                    out_dir = os.path.join(THIS_DIR, '..', 'results', 'experiment::' + dataset, split_id, train_size, overlap)

                    data, config = load_data(overlap_dir)

                    config['dataset'] = dataset
                    config['seed'] = int(split_id.split('::')[1])

                    random.seed(config['seed'])

                    if RUN_HYPERPARAMETER_SEARCH and int(split_id.split('::')[1]) == 0:
                        for index in range(len(hyperparameters)):
                            hyperparameters_string = ''
                            for key in sorted(hyperparameters[index].keys()):
                                hyperparameters_string = hyperparameters_string + key + ':' + str(hyperparameters[index][key]) + ' -- '
                            print("\n%d \ %d -- %s" % (index, len(hyperparameters), hyperparameters_string[:-3]))

                            results = build_model(data, config, out_dir, hyperparameters[index], True)
                            if results is None:
                                break
                            if results["trained"]["valid-accuracy"] > max_valid_accuracy:
                                max_valid_accuracy = results["trained"]["valid-accuracy"]
                                hyperparameter_setting = hyperparameters[index]

                    build_model(data, config, out_dir, hyperparameter_setting, False)


if __name__ == '__main__':
    main()