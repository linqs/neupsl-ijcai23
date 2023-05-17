#!/usr/bin/env python3
import importlib
import os
import re
import sys

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', '..', '..', 'scripts'))
results_parser = importlib.import_module('results-parser')
util = importlib.import_module("util")

RESULTS_DIR = os.path.join(THIS_DIR, '..', '..', '..', 'data')
LOG_FILENAME = 'config.json'
ADDITIONAL_HEADERS = ['Categorical-Accuracy']

class PSLResultsParser(results_parser.AbstractResultsParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_log_path(self, log_path):
        if 'saved-networks' not in log_path:
            return None

        results = []
        with open(log_path, 'r') as file:
            for line in file:

                if '"test-accuracy":' in line:
                    match = re.search(r'"test-accuracy": (\d+\.\d+)', line)
                    results.append(float(match.group(1)))

        return results


def main():
    psl_results_parser = PSLResultsParser(results_dir=RESULTS_DIR, log_filename=LOG_FILENAME, additional_headers=ADDITIONAL_HEADERS)
    results = psl_results_parser.parse_results()
    psl_results_parser.print_results(results)

    util.write_json_file('results.json', results)

if __name__ == '__main__':
    main()