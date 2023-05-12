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

import models

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', '..', '..', 'data'))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..', '..', 'scripts'))
util = importlib.import_module("util")

DATASET_CITESEER = 'citeseer'
DATASET_CORA = 'cora'
DATASETS = [DATASET_CITESEER, DATASET_CORA]

HYPERPARAMETERS = {
    'hidden-units': [[16], [32], [64]],
    'learning-rate': [1.0e-2, 1.0e-3],
    'dropout-rate': [0.5],
    'epochs': [1000],
    'batch-size': [1024],
    'regularizer-weight': [1.0e-3, 5.0e-4],
    'aggregation-type': ["sum"],
    'combination-type': ["none"],
    'metrics': [[tensorflow.keras.metrics.SparseCategoricalAccuracy(name='acc')]],
}

DEFAULT_PARAMETERS = {
    DATASET_CITESEER: {
        'hidden-units': [64],
        'learning-rate': 1.0e-3,
        'dropout-rate': 0.5,
        'epochs': 1000,
        'batch-size': 1024,
        'regularizer-weight': 1.0e-3,
        'aggregation-type': "sum",
        'combination-type': "none",
        'metrics': [tensorflow.keras.metrics.SparseCategoricalAccuracy(name='acc')],
    },
    DATASET_CORA: {
        'hidden-units': [64],
        'learning-rate': 1.0e-3,
        'dropout-rate': 0.5,
        'epochs': 1000,
        'batch-size': 1024,
        'regularizer-weight': 1.0e-3,
        'aggregation-type': "sum",
        'combination-type': "none",
        'metrics': [tensorflow.keras.metrics.SparseCategoricalAccuracy(name='acc')],
    }
}

VERBOSE = 0
NUM_RANDOM_SEEDS = 10

SAVED_NETWORKS_DIRECTORY = 'saved-networks'
CONFIG_FILENAME = 'config.json'

RUN_HYPERPARAMETER_SEARCH = False


def create_graph_info(data):
    edges = tensorflow.constant(data['edges'], dtype=tensorflow.dtypes.int32)
    features = tensorflow.constant(data['features'], dtype=tensorflow.dtypes.float32)

    # Precompute GCN edge weights.
    node_degrees = {node: 0 for node in data['nodes']}
    for edge in edges.numpy():
        node_degrees[edge[0]] += 1

    edge_weights = numpy.zeros(shape=edges.numpy().shape[0])
    for i, edge in enumerate(edges.numpy()):
        edge_weights[i] = 1 / numpy.sqrt(node_degrees[edge[0]] * node_degrees[edge[1]])

    graph_info = (features, edges.numpy().T, edge_weights)
    return graph_info


def build_network(data, config, hyperparameters):
    graph_info = create_graph_info(data)

    model = models.GNNNodeClassifier(
        graph_info=graph_info,
        class_size=config['output-size'],
        hidden_units=hyperparameters['hidden-units'],
        dropout_rate=hyperparameters['dropout-rate'],
        aggregation_type=hyperparameters['aggregation-type'],
        combination_type=hyperparameters['combination-type'],
        regularizer_weight=hyperparameters['regularizer-weight'],
    )

    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=hyperparameters['learning-rate']),
        loss=model.one_hot_kl_divergence,
        metrics=hyperparameters['metrics']
    )

    return model


def fit_model(model, data, config, hyperparameters):
    x_train = data['train']['entity-ids']
    y_train = data['train']['labels']
    x_test = data['test']['entity-ids']
    y_test = data['test']['labels']
    x_valid = data['valid']['entity-ids']
    y_valid = data['valid']['labels']

    early_stop = []
    early_stop.append(tensorflow.keras.callbacks.EarlyStopping(monitor="val_acc", patience=200, restore_best_weights=True))

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

    for partition in ['train', 'test', 'valid', 'latent']:
        deep_predictions_data = []
        deep_probability_data = []

        deep_predictions = max_model.predict(data[partition]['entity-ids'], verbose=VERBOSE)
        for entity, predictions in zip(data[partition]['entity-ids'], deep_predictions):
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
    data = util.load_json_file(os.path.join(dataset_dir, "deep-data.json"))
    config = {'input-size': len(data['train']['features'][0]),
              'output-size': util.load_json_file(os.path.join(dataset_dir, "config.json"))['class-size']}

    data['edges'] = []
    data['features'] = []
    data['nodes'] = []
    for example in sorted(util.load_psl_file(os.path.join(dataset_dir, "entity-data-map.txt")), key=lambda x: int(x[0])):
        data['edges'].append([int(example[0]), int(example[0])])
        data['features'].append([float(features) for features in example[1:-1]])
        data['nodes'].append(int(example[0]))

    for edge in util.load_psl_file(os.path.join(dataset_dir, "edges.txt")):
        data['edges'].append([int(edge[0]), int(edge[1])])
        data['edges'].append([int(edge[1]), int(edge[0])])

    return data, config


def main():
    hyperparameters = util.enumerate_hyperparameters(HYPERPARAMETERS)

    for dataset in DATASETS:
        dataset_dir = os.path.join(DATA_DIR, 'experiment::' + dataset)

        for split_id in sorted(os.listdir(dataset_dir)):
            split_dir = os.path.join(dataset_dir, split_id, 'method::simple')
            if os.path.isfile(split_dir):
                continue

            max_valid_accuracy = 0.0
            hyperparameter_setting = DEFAULT_PARAMETERS[dataset]
            out_dir = os.path.join(THIS_DIR, '..', 'results', 'experiment::' + dataset, split_id)

            data, config = load_data(split_dir)

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