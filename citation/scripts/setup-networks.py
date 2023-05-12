#!/usr/bin/env python3

"""
Create the networks for the citation experiment.
"""
import datetime
import importlib
import json
import os
import random
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy
import tensorflow

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data'))


sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

DATASET_CITESEER = 'citeseer'
DATASET_CORA = 'cora'
DATASETS = [DATASET_CITESEER, DATASET_CORA]

MODEL_SIMPLE = 'simple'
MODEL_SMOOTHED = 'smoothed'
MODELS = [MODEL_SIMPLE, MODEL_SMOOTHED]

HYPERPARAMETERS = {
    'hidden-units': [0, 16, 32, 64],
    'learning-rate': [1.0e-0, 1.5e-0],
    'weight-regularizer': [5.0e-5, 1.0e-6],
    'epochs': [250],
    'batch': [1024],
    'loss': [tensorflow.keras.losses.KLDivergence()],
    'metrics': [[tensorflow.keras.metrics.CategoricalAccuracy(name='acc')]],
}

DEFAULT_PARAMETERS = {
    (DATASET_CITESEER, MODEL_SIMPLE): {
        'hidden-units': 0,
        'learning-rate': 1.0e-0,
        'weight-regularizer': 1.0e-6,
        'epochs': 250,
        'batch': 1024,
        'loss': tensorflow.keras.losses.KLDivergence(),
        'metrics': [tensorflow.keras.metrics.CategoricalAccuracy(name='acc')],
    },
    (DATASET_CITESEER, MODEL_SMOOTHED): {
        'hidden-units': 0,
        'learning-rate': 1.5e-0,
        'weight-regularizer': 1.0e-6,
        'epochs': 250,
        'batch': 1024,
        'loss': tensorflow.keras.losses.KLDivergence(),
        'metrics': [tensorflow.keras.metrics.CategoricalAccuracy(name='acc')],
    },
    (DATASET_CORA, MODEL_SIMPLE): {
        'hidden-units': 0,
        'learning-rate': 1.5e-0,
        'weight-regularizer': 5.0e-5,
        'epochs': 250,
        'batch': 1024,
        'loss': tensorflow.keras.losses.KLDivergence(),
        'metrics': [tensorflow.keras.metrics.CategoricalAccuracy(name='acc')],
    },
    (DATASET_CORA, MODEL_SMOOTHED): {
        'hidden-units': 0,
        'learning-rate': 1.5e-0,
        'weight-regularizer': 5.0e-7,
        'epochs': 250,
        'batch': 1024,
        'loss': tensorflow.keras.losses.KLDivergence(),
        'metrics': [tensorflow.keras.metrics.CategoricalAccuracy(name='acc')],
    }
}

VERBOSE = 0
NUM_RANDOM_SEEDS = 10

SAVED_NETWORKS_DIRECTORY = 'saved-networks'
CONFIG_FILENAME = 'config.json'

RUN_HYPERPARAMETER_SEARCH = False


def build_network(config, hyperparameters):
    if hyperparameters['hidden-units'] != 0:
        layers = [
            tensorflow.keras.layers.Input(shape=config['input-size']),
            tensorflow.keras.layers.Dense(hyperparameters['hidden-units'],
                                          kernel_regularizer=tensorflow.keras.regularizers.l2(hyperparameters['weight-regularizer']),
                                          bias_regularizer=tensorflow.keras.regularizers.l2(hyperparameters['weight-regularizer']),
                                          activation='relu'),
            tensorflow.keras.layers.Dense(config['output-size'],
                                          kernel_regularizer=tensorflow.keras.regularizers.l2(hyperparameters['weight-regularizer']),
                                          bias_regularizer=tensorflow.keras.regularizers.l2(hyperparameters['weight-regularizer']),
                                          activation='softmax'),
        ]
    else:
        layers = [
            tensorflow.keras.layers.Input(shape=config['input-size']),
            tensorflow.keras.layers.Dense(config['output-size'],
                                          kernel_regularizer=tensorflow.keras.regularizers.l2(hyperparameters['weight-regularizer']),
                                          bias_regularizer=tensorflow.keras.regularizers.l2(hyperparameters['weight-regularizer']),
                                          activation="softmax"),
        ]

    model = tensorflow.keras.Sequential(layers=layers)

    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=hyperparameters['learning-rate']),
        loss=hyperparameters['loss'],
        metrics=hyperparameters['metrics'],
    )

    return model


def fit_model(model, data, config, hyperparameters):
    x_train = data['train']['features']
    y_train = data['train']['labels']
    x_test = data['test']['features']
    y_test = data['test']['labels']
    x_valid = data['valid']['features']
    y_valid = data['valid']['labels']

    early_stop = [tensorflow.keras.callbacks.EarlyStopping(monitor="val_acc", patience=150, restore_best_weights=True)]

    train_history = model.fit(
        x_train,
        y_train,
        epochs=hyperparameters['epochs'],
        validation_data=(x_valid, y_valid),
        verbose=VERBOSE,
        callbacks=early_stop
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
        print("Existing config file found (%s), skipping generation." % (config_path,))
        return

    max_validation_accuracy = 0
    max_results = None
    max_seed = None
    max_model = None

    for index in range(NUM_RANDOM_SEEDS):
        seed = random.randrange(2 ** 64)
        tensorflow.random.set_seed(seed)
        print("Starting: %s -- Current Run: %d -- Seed: %d -- Max Validation Accuracy: %f" % (config['dataset'], index, seed, max_validation_accuracy))

        model = build_network(config, hyperparameters)
        results, model = fit_model(model, data, config, hyperparameters)

        if results['trained']['valid-accuracy'] > max_validation_accuracy:
            max_validation_accuracy = results['trained']['valid-accuracy']
            max_results = results
            max_seed = seed
            max_model = model

    if tuning_hyperparameters:
        return max_results

    for partition in ['train', 'test', 'valid', 'latent']:
        deep_predictions_data = []
        deep_probaility_data = []

        deep_predictions = max_model.predict(data[partition]['features'], verbose=VERBOSE)
        for entity, predictions in zip(data[partition]['entity-ids'], deep_predictions):
            max_index = numpy.argmax(predictions)
            for index in range(len(predictions)):
                deep_probaility_data.append([entity, index] + [predictions[index]])
                deep_predictions_data.append([entity, index, 1.0 if index == max_index else 0.0])

        util.write_psl_file(os.path.join(out_dir, "deep-predictions-%s.txt" % partition), deep_predictions_data)
        util.write_psl_file(os.path.join(out_dir, "deep-probabilities-%s.txt" % partition), deep_probaility_data)

    max_model.save(os.path.join(out_dir, "pre-trained-tf"), save_format='tf')

    write_config = hyperparameters.copy()
    write_config['loss'] = str(hyperparameters['loss'])
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
    data = util.load_json_file(os.path.join(dataset_dir, "deep-data.json"))
    config = {'input-size': len(data['train']['features'][0]),
              'output-size': util.load_json_file(os.path.join(dataset_dir, "config.json"))['class-size']}

    for partition in data:
        data[partition]['labels'] = [util.one_hot_encoding(label, config['output-size']) for label in data[partition]['labels']]

    return data, config


def main():
    hyperparameters = util.enumerate_hyperparameters(HYPERPARAMETERS)

    for dataset in DATASETS:
        dataset_dir = os.path.join(DATA_DIR, 'experiment::' + dataset)

        for split_id in sorted(os.listdir(dataset_dir)):
            split_dir = os.path.join(dataset_dir, split_id)
            if os.path.isfile(split_dir):
                continue

            for model_id in sorted(os.listdir(split_dir)):
                model_dir = os.path.join(split_dir, model_id)
                if os.path.isfile(model_dir):
                    continue

                max_valid_accuracy = 0.0
                hyperparameter_setting = DEFAULT_PARAMETERS[(dataset, model_id.split('::')[1])]
                out_dir = os.path.join(model_dir, SAVED_NETWORKS_DIRECTORY)

                data, config = load_data(model_dir)

                config['model'] = model_id.split('::')[1]
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
