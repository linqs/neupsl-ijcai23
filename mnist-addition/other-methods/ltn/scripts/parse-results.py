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
ADDITIONAL_HEADERS = ['Categorical-Accuracy', 'Inference-Runtime', 'Learning-Runtime']

class LTNResultsParser(results_parser.AbstractResultsParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_log_path(self, log_path):
        results = [0.0]
        with open(log_path, 'r') as file:
            for line in file:
                match = re.search(r'test_accuracy: (\d+\.\d+)', line)
                if match is not None:
                    results[-1] = float(match.group(1))

        train_time, test_time = self._load_timing_info(log_path)
        results.append(test_time)
        results.append(train_time)
        return results

    def _load_timing_info(self, log_path):
        train_time = -1
        test_time = -1

        timing_info_path = os.path.join(os.path.dirname(log_path), 'out.err')

        with open(timing_info_path, 'r') as file:
            for line in file:
                match = re.search(r'Train Time.*(\d+\.\d+)', line)
                if match is not None:
                    train_time = float(match.group(1))

                match = re.search(r'Test Time.*(\d+\.\d+)', line)
                if match is not None:
                    test_time = float(match.group(1))

        return train_time, test_time


def main():
    ltn_results_parser = LTNResultsParser(results_dir=RESULTS_DIR, log_filename=LOG_FILENAME, additional_headers=ADDITIONAL_HEADERS)
    results = ltn_results_parser.parse_results()
    ltn_results_parser.print_results(results)

    util.write_json_file('results.json', results)

if __name__ == '__main__':
    main()