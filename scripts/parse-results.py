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
    'citation': ['Categorical-Accuracy', 'Inference-Runtime', 'Learning-Runtime'],
    'mnist-addition': ['Categorical-Accuracy', 'Inference-Runtime', 'Learning-Runtime'],
    'vspc': ['Puzzle-Accuracy', 'Puzzle-AUROC', 'Digit-Categorical-Accuracy', 'Inference-Runtime', 'Learning-Runtime'],
}

class NeuPSLResultsParser(results_parser.AbstractResultsParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_log_path(self, log_path):
        results = []

        inference_start = -1
        inference_end = -1
        learning_start = -1
        learning_end = -1
        capture_inference_time = False

        with open(log_path, 'r') as file:
            for line in file:
                if 'Evaluation results' in line:
                    match = re.search(r': ([\d\.]+)', line)
                    results.append(float(match.group(1)))

                if 'Found value true for option runtime.inference' in line:
                    capture_inference_time = True

                if 'Weight Learning Start' in line:
                    match = re.search(r'^(\d+).*Weight Learning Start\.', line)
                    learning_start = int(match.group(1))

                if 'Final Weight Learning' in line:
                    match = re.search(r'^(\d+).*Final Weight Learning.*', line)
                    learning_end = int(match.group(1))

                if capture_inference_time:
                    match = re.search(r'^(\d+).*Beginning inference\.', line)
                    if match is not None:
                        inference_start = int(match.group(1))

                    match = re.search(r'^(\d+).*Inference complete\.', line)
                    if match is not None:
                        inference_end = int(match.group(1))

        results.append((inference_end - inference_start) / 1000)
        results.append((learning_end - learning_start) / 1000)
        return results


def main():
    neupsl_results_parser = NeuPSLResultsParser(results_dir=RESULTS_DIR, log_filename=LOG_FILENAME, additional_headers=ADDITIONAL_HEADERS, specialized_headers=SPECIALIZED_HEADERS)
    results = neupsl_results_parser.parse_results()
    neupsl_results_parser.print_results(results)

    util.write_json_file('results.json', results)

if __name__ == '__main__':
    main()
