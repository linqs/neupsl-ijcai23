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

class DPLResultsParser(results_parser.AbstractResultsParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_log_path(self, log_path):
        results = []

        learning_time = -1
        inference_start = -1
        inference_end = -1

        with open(log_path, 'r') as file:
            for line in file:
                match = re.search(r'"accuracy": (\d+\.\d+)', line)
                if match is not None:
                    results.append(float(match.group(1)))

                match = re.search(r'"learning_total_time": (\d+\.\d+)', line)
                if match is not None:
                    learning_time = float(match.group(1))

                match = re.search(r'"learning_time_end": (\d+\.\d+)', line)
                if match is not None:
                    inference_start = float(match.group(1))

                match = re.search(r'"program_time_end": (\d+\.\d+)', line)
                if match is not None:
                    inference_end = float(match.group(1))

        results.append(inference_end - inference_start)
        results.append(learning_time)
        return results


def main():
    dpl_results_parser = DPLResultsParser(results_dir=RESULTS_DIR, log_filename=LOG_FILENAME, additional_headers=ADDITIONAL_HEADERS)
    results = dpl_results_parser.parse_results()
    dpl_results_parser.print_results(results)

    util.write_json_file('results.json', results)

if __name__ == '__main__':
    main()
