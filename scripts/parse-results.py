#!/usr/bin/env python3
import importlib
import os
import re

results_parser = importlib.import_module("results-parser")

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(THIS_DIR, '..', 'results')
util = importlib.import_module("util")

LOG_FILENAME = 'out.txt'
ADDITIONAL_HEADERS = []
SPECIALIZED_HEADERS = {
    'citation': ['Categorical-Accuracy'],
    'mnist-addition': ['Categorical-Accuracy'],
    'vspc': ['Puzzle-Accuracy', 'Puzzle-AUROC', 'Digit-Categorical-Accuracy']
}

class NeuPSLResultsParser(results_parser.AbstractResultsParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_log_path(self, log_path):
        results = []
        with open(log_path, 'r') as file:
            for line in file:
                if 'Evaluation results' in line:
                    match = re.search(r': ([\d\.]+)', line)
                    results.append(float(match.group(1)))

        return results


def main():
    neupsl_results_parser = NeuPSLResultsParser(results_dir=RESULTS_DIR, log_filename=LOG_FILENAME, additional_headers=ADDITIONAL_HEADERS, specialized_headers=SPECIALIZED_HEADERS)
    results = neupsl_results_parser.parse_results()
    neupsl_results_parser.print_results(results)

    util.write_json_file('results.json', results)

if __name__ == '__main__':
    main()
