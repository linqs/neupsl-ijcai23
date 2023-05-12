#!/usr/bin/env python3

# Parse out the results.

import os
import re
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.abspath(os.path.join(THIS_DIR, '../data'))

DIR_FILENAME = 'saved-networks'
LOG_FILENAME = 'config.json'


def print_results(results):
    for experiment, result in sorted(results.items()):
        print("Experiment: %s" % (experiment,))
        print(' '.join(result['header']))
        for row in result['rows']:
            print(' '.join([str(value) for value in row]))


def get_log_paths(path):
    log_paths = []
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if DIR_FILENAME in dir:
                log_paths.append(os.path.join(root, dir))

    return sorted(log_paths)


def parse_log(log_path):
    results = []
    with open(log_path, 'r') as file:
        for line in file:
            match = re.search(r'"test-accuracy": (\d+\.\d+)', line)
            if match is not None:
                results.append(float(match.group(1)))

    return results


def main():
    results = {}
    for experiment in sorted(os.listdir(RESULTS_DIR)):
        results[experiment] = {'header': [], 'rows': []}
        log_paths = get_log_paths(os.path.join(RESULTS_DIR, experiment))
        for log_path in log_paths:
            parts = os.path.dirname(log_path.split(experiment + "/")[1]).split("/")
            if len(results[experiment]['rows']) == 0:
                results[experiment]['header'] = [row.split("::")[0] for row in parts]
                results[experiment]['header'].append('Categorical Accuracy')
            results[experiment]['rows'].append([row.split("::")[1] for row in parts])

            for log_result in parse_log(os.path.join(log_path, LOG_FILENAME)):
                results[experiment]['rows'][-1].append(log_result)
    print_results(results)


def _load_args(args):
    executable = args.pop(0)
    if len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 %s" % (executable,), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    _load_args(sys.argv)
    main()
