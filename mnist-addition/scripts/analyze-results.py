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
DPL_RESULTS = os.path.join(THIS_DIR, '..', 'other-methods', 'deepproblog', 'scripts', 'results.json')
LTN_RESULTS = os.path.join(THIS_DIR, '..', 'other-methods', 'ltn', 'scripts', 'results.json')
NEUPSL_RESULTS = os.path.join(THIS_DIR, '..', '..', 'scripts', 'results.json')

Y_LIM = 0.0
BUFFER = 0.20
TICK_OFFSET = 1.0e-3
BAR_WIDTH = 0.5
GROUP_SIZE = 4 * BAR_WIDTH + BUFFER

LEGEND_KEYS = ['CNN', 'LTN', 'DeepProbLog','NeuPSL']
Y_LABEL = 'Categorical Accuracy'

CNN_COLOR = 'forestgreen'
DPL_COLOR = 'tomato'
LTN_COLOR = 'mediumorchid'
NEUPSL_COLOR = 'mediumslateblue'


def plot_results(cnn_results, dpl_results, ltn_results, neupsl_results, minor_xtick_labels, major_xtick_labels, title):
    fig, ax = plt.subplots(figsize=(12, 2.5))

    index = 0
    for key in neupsl_results:
        for method_results in [cnn_results, dpl_results, ltn_results]:
            if key not in method_results:
                method_results[key] = {'mean': [0.0], 'std': [0.0]}
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (0 + index // 3), cnn_results[key]['mean'][0], BAR_WIDTH, yerr=cnn_results[key]['std'][0], capsize=2, color=CNN_COLOR, edgecolor='black')
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (1 + index // 3), ltn_results[key]['mean'][0], BAR_WIDTH, yerr=ltn_results[key]['std'][0], capsize=2, color=LTN_COLOR, edgecolor='black')
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (2 + index // 3), dpl_results[key]['mean'][0], BAR_WIDTH, yerr=dpl_results[key]['std'][0], capsize=2, color=DPL_COLOR, edgecolor='black')
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (3 + index // 3), neupsl_results[key]['mean'][0], BAR_WIDTH, yerr=neupsl_results[key]['std'][0], capsize=2, color=NEUPSL_COLOR, edgecolor='black')
        index += 1

    ax.legend(LEGEND_KEYS)

    ax.set_xticks([index * GROUP_SIZE + BAR_WIDTH * (1.5 + index // 3) + TICK_OFFSET for index in range(len(neupsl_results))], minor=True)
    ax.set_xticklabels(minor_xtick_labels, minor=True)

    ax.set_xticks([GROUP_SIZE + 1.5 * BAR_WIDTH, 4 * GROUP_SIZE + 2.5 * BAR_WIDTH, 7 * GROUP_SIZE + 3.5 * BAR_WIDTH], minor=False)
    ax.set_xticklabels(major_xtick_labels, minor=False)
    ax.tick_params(axis='x', which='major', pad=20, size=0)
    ax.set_ylim(Y_LIM)

    ax.set_ylabel(Y_LABEL)
    ax.set_title(title)

    plt.axvline(x = 3 * GROUP_SIZE + 0 * BAR_WIDTH - BUFFER / 2, color='black', linewidth=1)
    plt.axvline(x = 6 * GROUP_SIZE + 1 * BAR_WIDTH - BUFFER / 2, color='black', linewidth=1)

    fig.tight_layout()
    plt.show()


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
        print("Train Size: %d Overlap: %.2f Categorical Accuracy: %.2f \u00B1 %.2f Inference Runtime: %.2f \u00B1 %.2f Learning Runtime: %.2f \u00B1 %.2f" % (int(key[0]), float(key[1]), 100 * results[key]['mean'][0], 100 * results[key]['std'][0], results[key]['mean'][1], results[key]['std'][1], results[key]['mean'][2], results[key]['std'][2]))


def fix_ltn(ltn_results):
    for experiment in ltn_results:
        for index in range(len(ltn_results[experiment]['rows'])):
            ltn_results[experiment]['rows'][index][2] = "%04d" % (2 * int(ltn_results[experiment]['rows'][index][0].split('-')[1]) * int(ltn_results[experiment]['rows'][index][2]))
            ltn_results[experiment]['rows'][index][3] = "%.2f" % float(ltn_results[experiment]['rows'][index][3])
    return ltn_results

def main():
    cnn_mnist_1_results = {}
    dpl_mnist_1_results = {}
    ltn_mnist_1_results = {}
    neupsl_mnist_1_results = {}

    if os.path.isfile(CNN_RESULTS):
        raw_cnn_results = util.load_json_file(CNN_RESULTS)
        cnn_mnist_1_results = parse_raw_results(raw_cnn_results['experiment::mnist-1']['rows'], raw_cnn_results['experiment::mnist-1']['header'], 1, 3, [3, 4, 5])
        print_mean_std(cnn_mnist_1_results, "CNN MNIST-1 Results:")

    if os.path.isfile(DPL_RESULTS):
        raw_dpl_results = util.load_json_file(DPL_RESULTS)
        dpl_mnist_1_results = parse_raw_results(raw_dpl_results['experiment::mnist-1']['rows'], raw_dpl_results['experiment::mnist-1']['header'], 1, 3, [3, 4, 5])
        print_mean_std(dpl_mnist_1_results, "DPL MNIST-1 Results:")

    if os.path.isfile(LTN_RESULTS):
        raw_ltn_results = util.load_json_file(LTN_RESULTS)
        raw_ltn_results = fix_ltn(raw_ltn_results)
        ltn_mnist_1_results = parse_raw_results(raw_ltn_results['method::ltn']['rows'], raw_ltn_results['method::ltn']['header'], 2, 4, [4, 5, 6], ignore_rows_with_entries=['mnist-2'], ignore_rows_with_entry_indexies=[0])
        print_mean_std(ltn_mnist_1_results, "LTN MNIST-1 Results:")

    if os.path.isfile(NEUPSL_RESULTS):
        raw_neupsl_results = util.load_json_file(NEUPSL_RESULTS)
        neupsl_mnist_1_results = parse_raw_results(raw_neupsl_results['mnist-addition']['rows'], raw_neupsl_results['mnist-addition']['header'], 3, 5, [5, 6, 7], ignore_rows_with_entries=['mnist-2'], ignore_rows_with_entry_indexies=[1])
        print_mean_std(neupsl_mnist_1_results, "NeuPSL MNIST-1 Results:")

    minor_xtick_labels = ["20", "30", "40", "30", "45", "60", "40", "60", "80"]
    major_xtick_labels = ["Number of Puzzles \n" + r"$\mathbf{40 \, Unique MNIST \, Images}$",
                          "Number of Puzzles \n" + r"$\mathbf{60 \, Unique MNIST \, Images}$",
                          "Number of Puzzles \n" + r"$\mathbf{80 \, Unique MNIST \, Images}$"]
    plot_results(cnn_mnist_1_results, dpl_mnist_1_results, ltn_mnist_1_results, neupsl_mnist_1_results, minor_xtick_labels, major_xtick_labels, "MNIST Addition 1")

    cnn_mnist_2_results = {}
    dpl_mnist_2_results = {}
    ltn_mnist_2_results = {}
    neupsl_mnist_2_results = {}

    if os.path.isfile(CNN_RESULTS):
        raw_cnn_results = util.load_json_file(CNN_RESULTS)
        cnn_mnist_2_results = parse_raw_results(raw_cnn_results['experiment::mnist-2']['rows'], raw_cnn_results['experiment::mnist-2']['header'], 1, 3, [3, 4, 5])
        print_mean_std(cnn_mnist_2_results, "CNN MNIST-2 Results:")

    if os.path.isfile(DPL_RESULTS):
        raw_dpl_results = util.load_json_file(DPL_RESULTS)
        dpl_mnist_2_results = parse_raw_results(raw_dpl_results['experiment::mnist-2']['rows'], raw_dpl_results['experiment::mnist-2']['header'], 1, 3, [3, 4, 5])
        print_mean_std(dpl_mnist_2_results, "DPL MNIST-2 Results:")

    if os.path.isfile(LTN_RESULTS):
        raw_ltn_results = util.load_json_file(LTN_RESULTS)
        raw_ltn_results = fix_ltn(raw_ltn_results)
        ltn_mnist_2_results = parse_raw_results(raw_ltn_results['method::ltn']['rows'], raw_ltn_results['method::ltn']['header'], 2, 4, [4, 5, 6], ignore_rows_with_entries=['mnist-1'], ignore_rows_with_entry_indexies=[0])
        print_mean_std(ltn_mnist_2_results, "LTN MNIST-2 Results:")

    if os.path.isfile(NEUPSL_RESULTS):
        raw_neupsl_results = util.load_json_file(NEUPSL_RESULTS)
        neupsl_mnist_2_results = parse_raw_results(raw_neupsl_results['mnist-addition']['rows'], raw_neupsl_results['mnist-addition']['header'], 3, 5, [5, 6, 7], ignore_rows_with_entries=['mnist-1'], ignore_rows_with_entry_indexies=[1])
        print_mean_std(neupsl_mnist_2_results, "NeuPSL MNIST-2 Results:")

    minor_xtick_labels = ["10", "15", "20", "15", "22", "30", "20", "30", "40"]
    major_xtick_labels = ["Number of Puzzles \n" + r"$\mathbf{40 \, Unique MNIST \, Images}$",
                          "Number of Puzzles \n" + r"$\mathbf{60 \, Unique MNIST \, Images}$",
                          "Number of Puzzles \n" + r"$\mathbf{80 \, Unique MNIST \, Images}$"]
    plot_results(cnn_mnist_2_results, dpl_mnist_2_results, ltn_mnist_2_results, neupsl_mnist_2_results, minor_xtick_labels, major_xtick_labels, "MNIST Addition 2")


if __name__ == '__main__':
    main()