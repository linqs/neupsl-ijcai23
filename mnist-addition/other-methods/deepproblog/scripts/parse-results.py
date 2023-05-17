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
ADDITIONAL_HEADERS = ['Categorical-Accuracy']

class DPLResultsParser(results_parser.AbstractResultsParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_log_path(self, log_path):
        results = []
        with open(log_path, 'r') as file:
            for line in file:
                match = re.search(r'"accuracy": (\d+\.\d+)', line)
                if match is not None:
                    results.append(float(match.group(1)))

        return results


def main():
    dpl_results_parser = DPLResultsParser(results_dir=RESULTS_DIR, log_filename=LOG_FILENAME, additional_headers=ADDITIONAL_HEADERS)
    results = dpl_results_parser.parse_results()
    dpl_results_parser.print_results(results)

    util.write_json_file('results.json', results)

if __name__ == '__main__':
    main()
