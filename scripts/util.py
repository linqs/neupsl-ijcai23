import json


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
        hyperparameters_list = []
        for value in hyperparameters_dict[key]:
            next_hyperparameters = current_hyperparameters.copy()
            next_hyperparameters[key] = value

            remaining_hyperparameters = hyperparameters_dict.copy()
            remaining_hyperparameters.pop(key)

            if remaining_hyperparameters:
                hyperparameters_list = hyperparameters_list + enumerate_hyperparameters(remaining_hyperparameters, next_hyperparameters)
            else:
                hyperparameters_list.append(next_hyperparameters)
        return hyperparameters_list


def one_hot_encoding(label, num_labels):
    encoding = [0] * num_labels
    encoding[label] = 1
    return encoding
