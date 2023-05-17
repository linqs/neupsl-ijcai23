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
LOG_FILENAME = 'out.txt'
ADDITIONAL_HEADERS = ['Categorical-Accuracy']

class CNNResultsParser(results_parser.AbstractResultsParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_log_path(self, log_path):
        results = []
        with open(log_path, 'r') as file:
            for line in file:
                if 'Categorical Accuracy:' in line:
                    match = re.search(r'^Loss: (\d+\.\d+)IMAGESUM -- Categorical Accuracy: (\d+\.\d+)', line)
                    results.append(float(match.group(2)))

        return results


def main():
    cnn_results_parser = CNNResultsParser(results_dir=RESULTS_DIR, log_filename=LOG_FILENAME, additional_headers=ADDITIONAL_HEADERS)
    results = cnn_results_parser.parse_results()
    cnn_results_parser.print_results(results)

    util.write_json_file('results.json', results)

if __name__ == '__main__':
    main()