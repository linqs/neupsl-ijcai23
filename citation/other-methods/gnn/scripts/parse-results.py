#!/usr/bin/env python3
import importlib
import os
import re
import sys

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..', '..', 'scripts'))
results_parser = importlib.import_module('results-parser')
util = importlib.import_module("util")

RESULTS_DIR = os.path.join(THIS_DIR, '..', 'results')
LOG_FILENAME = 'config.json'
ADDITIONAL_HEADERS = ['Categorical-Accuracy', 'Inference-Runtime', 'Learning-Runtime']

class GNNResultsParser(results_parser.AbstractResultsParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_log_path(self, log_path):
        results = []
        inference_runtime = -1
        learning_runtime = -1

        with open(log_path, 'r') as file:
            for line in file:
                match = re.search(r'"test-accuracy": (\d+\.\d+)', line)
                if match is not None:
                    results.append(float(match.group(1)))

                match = re.search(r'"inference-test-time": (\d+\.\d+)', line)
                if match is not None:
                    inference_runtime = float(match.group(1))

        results.append(inference_runtime)
        results.append(learning_runtime)
        return results


def main():
    gnn_results_parser = GNNResultsParser(results_dir=RESULTS_DIR, log_filename=LOG_FILENAME, additional_headers=ADDITIONAL_HEADERS)
    results = gnn_results_parser.parse_results()
    gnn_results_parser.print_results(results)

    util.write_json_file('results.json', results)

if __name__ == '__main__':
    main()
