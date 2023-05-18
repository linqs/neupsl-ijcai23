#!/usr/bin/env python3

import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

CNN_RESULTS = os.path.join(THIS_DIR, '..', 'other-methods', 'cnn', 'scripts', 'results.json')
DSL_RESULTS = os.path.join(THIS_DIR, '..', 'other-methods', 'deepstochlog', 'scripts', 'results.json')
GNN_RESULTS = os.path.join(THIS_DIR, '..', 'other-methods', 'gnn', 'scripts', 'results.json')
PSL_RESULTS = os.path.join(THIS_DIR, '..', 'other-methods', 'psl', 'scripts', 'results.json')
NEUPSL_RESULTS = os.path.join(THIS_DIR, '..', '..', 'scripts', 'results.json')


def parse_raw_results(rows, header, settings_start_index, settings_end_index, results_indexes, ignore_rows_with_entries=None, ignore_rows_with_entry_indexies=None):
    results = {}
    for row in rows:
        if len(row) < len(header):
            continue

        if ignore_rows_with_entries is not None:
            skip = False
            for ignore_entry, ignore_entry_index in zip(ignore_rows_with_entries, ignore_rows_with_entry_indexies):
                if row[ignore_entry_index] == ignore_entry:
                    skip = True
            if skip:
                continue

        if settings_start_index is None or settings_end_index is None:
            if 'total' not in results:
                results['total'] = {'results': [[] for _ in range(len(results_indexes))], 'mean': 0.0, 'std': 0.0}
            for index in range(len(results_indexes)):
                results['total']['results'][index].append(row[results_indexes[index]])
        else:
            if tuple(row[settings_start_index:settings_end_index]) not in results:
                results[tuple(row[settings_start_index:settings_end_index])] = {'results': [[] for _ in range(len(results_indexes))], 'mean': 0.0, 'std': 0.0}
            for index in range(len(results_indexes)):
                results[tuple(row[settings_start_index:settings_end_index])]['results'][index].append(row[results_indexes[index]])

    for key in results:
        results[key]['mean'] = numpy.mean(results[key]['results'], axis=1)
        results[key]['std'] = numpy.std(results[key]['results'], axis=1)

    return results


def print_mean_std(results, name):
    print(name)
    for key in results:
        print("Categorical Accuracy: %.2f \u00B1 %.2f Inference Runtime: %.2f \u00B1 %.2f Learning Runtime: %.2f \u00B1 %.2f" % (100 * results[key]['mean'][0], 100 * results[key]['std'][0], results[key]['mean'][1], results[key]['std'][1], results[key]['mean'][2], results[key]['std'][2]))


def main():
    raw_cnn_results = util.load_json_file(CNN_RESULTS)
    raw_dsl_results = util.load_json_file(DSL_RESULTS)
    raw_gnn_results = util.load_json_file(GNN_RESULTS)
    raw_psl_results = util.load_json_file(PSL_RESULTS)
    raw_neupsl_results = util.load_json_file(NEUPSL_RESULTS)

    cnn_citeseer_simple_results = parse_raw_results(raw_cnn_results['experiment::citeseer']['rows'], raw_cnn_results['experiment::citeseer']['header'], None, None, [3, 4, 5], ignore_rows_with_entries=['smoothed'], ignore_rows_with_entry_indexies=[1])
    cnn_citeseer_smoothed_results = parse_raw_results(raw_cnn_results['experiment::citeseer']['rows'], raw_cnn_results['experiment::citeseer']['header'], None, None, [3, 4, 5], ignore_rows_with_entries=['simple'], ignore_rows_with_entry_indexies=[1])
    dsl_citeseer_results = parse_raw_results(raw_dsl_results['experiment::citeseer']['rows'], raw_dsl_results['experiment::citeseer']['header'], None, None, [2, 3, 4])
    gnn_citeseer_results = parse_raw_results(raw_gnn_results['experiment::citeseer']['rows'], raw_gnn_results['experiment::citeseer']['header'], None, None, [1, 2, 3])
    psl_citeseer_simple_results = parse_raw_results(raw_psl_results['experiment::citeseer']['rows'], raw_psl_results['experiment::citeseer']['header'], None, None, [2, 3, 4], ignore_rows_with_entries=['smoothed'], ignore_rows_with_entry_indexies=[1])
    psl_citeseer_smoothed_results = parse_raw_results(raw_psl_results['experiment::citeseer']['rows'], raw_psl_results['experiment::citeseer']['header'], None, None, [2, 3, 4], ignore_rows_with_entries=['simple'], ignore_rows_with_entry_indexies=[1])
    neupsl_citeseer_simple_results = parse_raw_results(raw_neupsl_results['citation']['rows'], raw_neupsl_results['citation']['header'], None, None, [4, 5, 6], ignore_rows_with_entries=['cora', 'smoothed'], ignore_rows_with_entry_indexies=[1, 3])
    neupsl_citeseer_smoothed_results = parse_raw_results(raw_neupsl_results['citation']['rows'], raw_neupsl_results['citation']['header'], None, None, [4, 5, 6], ignore_rows_with_entries=['cora', 'simple'], ignore_rows_with_entry_indexies=[1, 3])

    print_mean_std(cnn_citeseer_simple_results, "CNN Citeseer Simple Results:")
    print_mean_std(cnn_citeseer_smoothed_results, "CNN Citeseer Smoothed Results:")
    print_mean_std(dsl_citeseer_results, "DSL Citeseer Results:")
    print_mean_std(gnn_citeseer_results, "GNN Citeseer Results:")
    print_mean_std(psl_citeseer_simple_results, "PSL Citeseer Simple Results:")
    print_mean_std(psl_citeseer_smoothed_results, "PSL Citeseer Smoothed Results:")
    print_mean_std(neupsl_citeseer_simple_results, "NeuPSL Citeseer Simple Results:")
    print_mean_std(neupsl_citeseer_smoothed_results, "NeuPSL Citeseer Smoothed Results:")

    cnn_cora_simple_results = parse_raw_results(raw_cnn_results['experiment::cora']['rows'], raw_cnn_results['experiment::cora']['header'], None, None, [3, 4, 5], ignore_rows_with_entries=['smoothed'], ignore_rows_with_entry_indexies=[1])
    cnn_cora_smoothed_results = parse_raw_results(raw_cnn_results['experiment::cora']['rows'], raw_cnn_results['experiment::cora']['header'], None, None, [3, 4, 5], ignore_rows_with_entries=['simple'], ignore_rows_with_entry_indexies=[1])
    dsl_cora_results = parse_raw_results(raw_dsl_results['experiment::cora']['rows'], raw_dsl_results['experiment::cora']['header'], None, None, [2, 3, 4])
    gnn_cora_results = parse_raw_results(raw_gnn_results['experiment::cora']['rows'], raw_gnn_results['experiment::cora']['header'], None, None, [1, 2, 3])
    psl_cora_simple_results = parse_raw_results(raw_psl_results['experiment::cora']['rows'], raw_psl_results['experiment::cora']['header'], None, None, [2, 3, 4], ignore_rows_with_entries=['smoothed'], ignore_rows_with_entry_indexies=[1])
    psl_cora_smoothed_results = parse_raw_results(raw_psl_results['experiment::cora']['rows'], raw_psl_results['experiment::cora']['header'], None, None, [2, 3, 4], ignore_rows_with_entries=['simple'], ignore_rows_with_entry_indexies=[1])
    neupsl_cora_simple_results = parse_raw_results(raw_neupsl_results['citation']['rows'], raw_neupsl_results['citation']['header'], None, None, [4, 5, 6], ignore_rows_with_entries=['citeseer', 'smoothed'], ignore_rows_with_entry_indexies=[1, 3])
    neupsl_cora_smoothed_results = parse_raw_results(raw_neupsl_results['citation']['rows'], raw_neupsl_results['citation']['header'], None, None, [4, 5, 6], ignore_rows_with_entries=['citeseer', 'simple'], ignore_rows_with_entry_indexies=[1, 3])

    print_mean_std(cnn_cora_simple_results, "CNN Cora Simple Results:")
    print_mean_std(cnn_cora_smoothed_results, "CNN Cora Smoothed Results:")
    print_mean_std(dsl_cora_results, "DSL Cora Results:")
    print_mean_std(gnn_cora_results, "GNN Cora Results:")
    print_mean_std(psl_cora_simple_results, "PSL Cora Simple Results:")
    print_mean_std(psl_cora_smoothed_results, "PSL Cora Smoothed Results:")
    print_mean_std(neupsl_cora_simple_results, "NeuPSL Cora Simple Results:")
    print_mean_std(neupsl_cora_smoothed_results, "NeuPSL Cora Smoothed Results:")


if __name__ == '__main__':
    main()