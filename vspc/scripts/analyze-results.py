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

BUFFER = 0.25
TICK_OFFSET = 1.0e-3
BAR_WIDTH = 0.5
GROUP_SIZE = 3 * BAR_WIDTH + BUFFER

NEUPSL_COLOR = 'mediumslateblue'
BASELINE_DIGIT_COLOR = 'tomato'
BASELINE_VISUAL_COLOR = 'forestgreen'


def plot_results(cnn_digit_results, cnn_visual_results, neupsl_results):
    fig, ax = plt.subplots(figsize=(12, 3))

    index = 0
    for key in neupsl_results:
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (0 + index // 3), cnn_visual_results[key]['mean'], BAR_WIDTH, yerr=cnn_visual_results[key]['std'], capsize=2, color=BASELINE_VISUAL_COLOR, edgecolor='black')
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (1 + index // 3), cnn_digit_results[key]['mean'], BAR_WIDTH, yerr=cnn_digit_results[key]['std'], capsize=2, color=BASELINE_DIGIT_COLOR, edgecolor='black')
        ax.bar(index * GROUP_SIZE + BAR_WIDTH * (2 + index // 3), neupsl_results[key]['mean'], BAR_WIDTH, yerr=neupsl_results[key]['std'], capsize=2, color=NEUPSL_COLOR, edgecolor='black')
        index += 1

    ax.legend(['CNN-Visual', 'CNN-Digit', 'NeuPSL'])

    ax.set_xticks([index * GROUP_SIZE + BAR_WIDTH * (1 + index // 3) + TICK_OFFSET for index in range(len(neupsl_results))], minor=True)
    ax.set_xticklabels(["4", "8", "16", "8", "16", "32", "16", "32", "64",], minor=True)

    ax.set_xticks([GROUP_SIZE + BAR_WIDTH, 4 * GROUP_SIZE + 2 * BAR_WIDTH, 7 * GROUP_SIZE + 3 * BAR_WIDTH], minor=False)
    ax.set_xticklabels(["Number of Puzzles \n" + r"$\mathbf{64 \, Unique MNIST \, Images}$",
                        "Number of Puzzles \n" + r"$\mathbf{128 \, Unique MNIST \, Images}$",
                        "Number of Puzzles \n" + r"$\mathbf{256 \, Unique MNIST \, Images}$"], minor=False)
    ax.tick_params(axis='x', which='major', pad=20, size=0)
    ax.set_ylim(0.45)

    ax.set_ylabel("Accuracy")
    ax.set_title("Visual-Sudoku-Classification")

    plt.axvline(x = 3 * GROUP_SIZE + 0 * BAR_WIDTH - BUFFER / 2, color='black', linewidth=1)
    plt.axvline(x = 6 * GROUP_SIZE + 1 * BAR_WIDTH - BUFFER / 2, color='black', linewidth=1)

    fig.tight_layout()
    plt.show()


def parse_raw_results(rows, header, settings_start_index, settings_end_index, results_index):
    results = {}
    for row in rows:
        if len(row) < len(header):
            continue
        if tuple(row[settings_start_index:settings_end_index]) not in results:
            results[tuple(row[settings_start_index:settings_end_index])] = {'results': [], 'mean': 0.0, 'std': 0.0}
        results[tuple(row[settings_start_index:settings_end_index])]['results'].append(row[results_index])

    for key in results:
        results[key]['mean'] = numpy.mean(results[key]['results'])
        results[key]['std'] = numpy.std(results[key]['results'])

    return results


def print_mean_std(results, name):
    print(name)
    for key in results:
        print("Train Size: %d Overlap: %.2f Categorical Accuracy: %.2f \u00B1 %.2f" % (int(key[0]), float(key[1]), 100 * results[key]['mean'], 100 * results[key]['std']))


def main():
    raw_cnn_digit_results = util.load_json_file(CNN_DIGIT_RESULTS)
    raw_cnn_visual_results = util.load_json_file(CNN_VISUAL_RESULTS)
    raw_neupsl_results = util.load_json_file(NEUPSL_RESULTS)

    cnn_digit_results = parse_raw_results(raw_cnn_digit_results['experiment::mnist-4x4']['rows'], raw_cnn_digit_results['experiment::mnist-4x4']['header'], 1, 3, 3)
    cnn_visual_results = parse_raw_results(raw_cnn_visual_results['experiment::mnist-4x4']['rows'], raw_cnn_visual_results['experiment::mnist-4x4']['header'], 1, 3, 3)
    neupsl_results = parse_raw_results(raw_neupsl_results['vspc']['rows'], raw_neupsl_results['vspc']['header'], 3, 5, 5)

    print_mean_std(cnn_digit_results, "CNN-Digit Results:")
    print_mean_std(cnn_visual_results, "CNN-Visual Results:")
    print_mean_std(neupsl_results, "NeuPSL Results:")

    plot_results(cnn_digit_results, cnn_visual_results, neupsl_results)


if __name__ == '__main__':
    main()