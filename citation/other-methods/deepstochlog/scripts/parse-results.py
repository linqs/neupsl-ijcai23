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
ADDITIONAL_HEADERS = ['Training-Test-Accuracy', 'Test-Accuracy', 'Inference-Runtime', 'Learning-Runtime']

class DSLResultsParser(results_parser.AbstractResultsParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_log_path(self, log_path):
        results = []
        previous_line = ""
        next_line = False
        learning_runtime = -1

        with open(log_path, 'r') as file:
            for line in file:
                if line.strip() == "":
                    continue

                line = ' '.join(line.strip().split())

                if "Training" in line:
                    match = re.search(r'(\d+) (\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+)', previous_line)
                    results.append(float(match.group(5)))
                    learning_runtime = float(match.group(7))

                if next_line:
                    match = re.search(r'(\d+\.\d+) (\d+\.\d+) (\d+\.\d+)', line)
                    results.append(float(match.group(1)))
                    results.append(float(match.group(3)))
                    next_line = False

                if "Test acc P(correct) time" == line:
                    next_line = True

                previous_line = line

        results.append(learning_runtime)

        return results


def main():
    dsl_results_parser = DSLResultsParser(results_dir=RESULTS_DIR, log_filename=LOG_FILENAME, additional_headers=ADDITIONAL_HEADERS)
    results = dsl_results_parser.parse_results()
    dsl_results_parser.print_results(results)

    util.write_json_file('results.json', results)

if __name__ == '__main__':
    main()
