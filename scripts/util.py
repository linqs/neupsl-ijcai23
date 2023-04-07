import json

import numpy


def write_json_file(path, data, indent=4):
    with open(path, "w") as file:
        json.dump(data, file, indent=indent)


def load_json_file(path):
    with open(path, "r") as file:
        return json.load(file)


def load_fake_json_file(path, size=None):
    with open(path, "r") as file:
        data = []
        current_line = 0
        for line in file:
            if size is not None and current_line > size - 1:
                break
            data.append(json.loads(line))
            current_line += 1

    return data


def write_psl_file(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")


def load_psl_file(path, dtype=str):
    data = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '':
                continue

            data.append(list(map(dtype, line.split("\t"))))

    return data


def enumerate_hyperparameters(hyperparameters_dict, current_hyperparameters={}):
    for key in sorted(hyperparameters_dict):
        hyperparameters = []
        for value in hyperparameters_dict[key]:
            next_hyperparameters = current_hyperparameters.copy()
            next_hyperparameters[key] = value

            remaining_hyperparameters = hyperparameters_dict.copy()
            remaining_hyperparameters.pop(key)

            if remaining_hyperparameters:
                hyperparameters = hyperparameters + enumerate_hyperparameters(remaining_hyperparameters, next_hyperparameters)
            else:
                hyperparameters.append(next_hyperparameters)
        return hyperparameters


def calculate_metrics(y_pred, y_truth, metrics):
    results = {}
    for metric in metrics:
        if metric == 'categorical_accuracy':
            results['categorical_accuracy'] = _categorical_accuracy(y_pred, y_truth)
        else:
            raise ValueError('Unknown metric: {}'.format(metric))
    return results


def _categorical_accuracy(y_pred, y_truth):
    correct = 0
    for i in range(len(y_truth)):
        if numpy.argmax(y_pred[i]) == numpy.argmax(y_truth[i]):
            correct += 1
    return correct / len(y_truth)
