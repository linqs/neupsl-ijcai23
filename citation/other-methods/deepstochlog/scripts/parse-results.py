#!/usr/bin/env python3

# Parse out the results.

import os
import re
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.abspath(os.path.join(THIS_DIR, '../results'))

LOG_FILENAME = 'out.txt'


def print_results(results):
    for experiment, result in sorted(results.items()):
        print("Experiment: %s" % (experiment,))
        print(' '.join(result['header']))
        for row in result['rows']:
            print(' '.join([str(value) for value in row]))


def get_log_paths(path):
    log_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if LOG_FILENAME == file.split("/")[-1]:
                log_paths.append(os.path.join(root,file))

    return sorted(log_paths)


def parse_log(log_path):
    results = []
    previous_line = ""
    next_line = False

    with open(log_path, 'r') as file:
        for line in file:
            if line.strip() == "":
                continue

            line = ' '.join(line.split())

            if "Training" in line:
                match = re.search(r'(\d+) (\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+)', previous_line)
                results.append(float(match.group(5)))

            if next_line:
                match = re.search(r'(\d+\.\d+) (\d+\.\d+) (\d+\.\d+)', line)
                results.append(float(match.group(1)))
                results.append(float(match.group(3)))
                next_line = False

            if "Test acc P(correct) time" == line:
                next_line = True

            previous_line = line

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
                results[experiment]['header'].extend(['Training-Test-Accuracy', 'Test-Accuracy', 'Inference-Time'])
            results[experiment]['rows'].append([row.split("::")[1] for row in parts])

            for log_result in parse_log(log_path):
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
