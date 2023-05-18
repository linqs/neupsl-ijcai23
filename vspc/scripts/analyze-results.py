#!/usr/bin/env python3

import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

CNN_DIGIT_RESULTS = os.path.join(THIS_DIR, '..', 'other-methods', 'cnn-digit', 'scripts', 'results.json')
CNN_VISUAL_RESULTS = os.path.join(THIS_DIR, '..', 'other-methods', 'cnn-visual', 'scripts', 'results.json')
NEUPSL_RESULTS = os.path.join(THIS_DIR, '..', '..', 'scripts', 'results.json')

Y_LIM = 0.45
BUFFER = 0.25
TICK_OFFSET = 1.0e-3
BAR_WIDTH = 0.5
GROUP_SIZE = 3 * BAR_WIDTH + BUFFER

LEGEND_KEYS = ['CNN-Visual', 'CNN-Digit', 'NeuPSL']
Y_LABEL = 'Categorical Accuracy'

CNN_DIGIT_COLOR = 'tomato'
CNN_VISUAL_COLOR = 'forestgreen'
NEUPSL_COLOR = 'mediumslateblue'


def plot_results(cnn_digit_results, cnn_visual_results, neupsl_results, minor_xtick_labels, major_xtick_labels, title):
    fig, ax = plt.subplots(figsize=(12, 2.5))

    index = 0
    for key in neupsl_results:
        for method_results in [cnn_digit_results, cnn_visual_results]:
            if key not in method_results:
                method_results[key] = {'mean': [0.0], 'std': [0.0]}
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (0 + index // 3), cnn_visual_results[key]['mean'][0], BAR_WIDTH, yerr=cnn_visual_results[key]['std'][0], capsize=2, color=CNN_VISUAL_COLOR, edgecolor='black')
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (1 + index // 3), cnn_digit_results[key]['mean'][0], BAR_WIDTH, yerr=cnn_digit_results[key]['std'][0], capsize=2, color=CNN_DIGIT_COLOR, edgecolor='black')
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (2 + index // 3), neupsl_results[key]['mean'][0], BAR_WIDTH, yerr=neupsl_results[key]['std'][0], capsize=2, color=NEUPSL_COLOR, edgecolor='black')
        index += 1

    ax.legend(LEGEND_KEYS)

    ax.set_xticks([index * GROUP_SIZE + BAR_WIDTH * (1 + index // 3) + TICK_OFFSET for index in range(len(neupsl_results))], minor=True)
    ax.set_xticklabels(minor_xtick_labels, minor=True)

    ax.set_xticks([GROUP_SIZE + BAR_WIDTH, 4 * GROUP_SIZE + 2 * BAR_WIDTH, 7 * GROUP_SIZE + 3 * BAR_WIDTH], minor=False)
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


def main():
    cnn_digit_results = {}
    cnn_visual_results = {}
    neupsl_results = {}

    if os.path.isfile(CNN_DIGIT_RESULTS):
        raw_cnn_digit_results = util.load_json_file(CNN_DIGIT_RESULTS)
        cnn_digit_results = parse_raw_results(raw_cnn_digit_results['experiment::mnist-4x4']['rows'], raw_cnn_digit_results['experiment::mnist-4x4']['header'], 1, 3, [3, 4, 5])
        print_mean_std(cnn_digit_results, "CNN-Digit Results:")

    if os.path.isfile(CNN_VISUAL_RESULTS):
        raw_cnn_visual_results = util.load_json_file(CNN_VISUAL_RESULTS)
        cnn_visual_results = parse_raw_results(raw_cnn_visual_results['experiment::mnist-4x4']['rows'], raw_cnn_visual_results['experiment::mnist-4x4']['header'], 1, 3, [3, 4, 5])
        print_mean_std(cnn_visual_results, "CNN-Visual Results:")

    if os.path.isfile(NEUPSL_RESULTS):
        raw_neupsl_results = util.load_json_file(NEUPSL_RESULTS)
        neupsl_results = parse_raw_results(raw_neupsl_results['vspc']['rows'], raw_neupsl_results['vspc']['header'], 3, 5, [5, 8, 9])
        print_mean_std(neupsl_results, "NeuPSL Results:")

    minor_xtick_labels = ["4", "8", "16", "8", "16", "32", "16", "32", "64",]
    major_xtick_labels = ["Number of Puzzles \n" + r"$\mathbf{64 \, Unique MNIST \, Images}$",
                          "Number of Puzzles \n" + r"$\mathbf{128 \, Unique MNIST \, Images}$",
                          "Number of Puzzles \n" + r"$\mathbf{256 \, Unique MNIST \, Images}$"]
    plot_results(cnn_digit_results, cnn_visual_results, neupsl_results, minor_xtick_labels, major_xtick_labels, "Visual-Sudoku-Classification")


if __name__ == '__main__':
    main()