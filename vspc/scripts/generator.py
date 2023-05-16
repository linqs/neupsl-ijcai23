#!/usr/bin/env python3

# A setup script similar to setup.py,
# except this script will not rely on existing puzzles.
# Instead, fully new puzzles will be generated.

import argparse
import copy
import datetime
import importlib
import json
import math
import os
import random
import sys

import numpy
import tensorflow

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data', '{}'))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

SUBPATH_FORMAT = os.path.join('experiment::mnist-{:01d}x{:01d}', 'split::{:02d}', 'positive-train-size::{:04d}', 'overlap::{:1.2f}')
CONFIG_PATH = os.path.join(DATA_DIR, 'config.json')

DIGIT_MAP_PATH = os.path.join(DATA_DIR, 'digit-map.txt')

ENTITY_DATA_MAP_PATH = os.path.join(DATA_DIR, 'entity-data-map')
DIGIT_TARGETS_PATH = os.path.join(DATA_DIR, 'digit-targets')
DIGIT_TRUTH_PATH = os.path.join(DATA_DIR, 'digit-truth')
DIGIT_MAPPED_TRUTH_PATH = os.path.join(DATA_DIR, 'digit-mapped-truth')

FIRST_PUZZLE_OBSERVATION_PATH = os.path.join(DATA_DIR, 'first-puzzle-observation.txt')

ROW_COLUMN_TARGETS_PATH = os.path.join(DATA_DIR, 'row-column-targets.txt')
PUZZLE_TARGETS_PATH = os.path.join(DATA_DIR, 'puzzle-targets.txt')
PUZZLE_TRUTH_PATH = os.path.join(DATA_DIR, 'puzzle-truth.txt')

LABELS = list(range(0, 10))
BINARY_LABELS = [0, 1]
BINARY_LABEL_POSITIVE = [0, 1]
BINARY_LABEL_NEGATIVE = [1, 0]

# MNIST images are 28 x 28 = 784.
MNIST_DIMENSION = 28

PUZZLE_CORRUPTION_REPLACE_CHANCE = 0.50
PUZZLE_CORRUPTION_REPLACE_MAX = 10
PUZZLE_CORRUPTION_SWAP_CHANCE = 0.50
PUZZLE_CORRUPTION_SWAP_MAX = 10

SIGNIFICANT_DIGITS = 4
NORMAL_PUZZLE_NOTE = 'normal'


class DigitChooser(object):
    # digits: {label: [image, ...], ...}
    def __init__(self, digits):
        self.digits = digits
        self.nextIndexes = {label: 0 for label in digits}

    # Takes the next image for a digit,
    def takeDigit(self, label):
        assert(self.nextIndexes[label] < len(self.digits[label]))

        image = self.digits[label][self.nextIndexes[label]]
        self.nextIndexes[label] += 1
        return image

    # Get a digit randomly from anywhere in the sequence.
    def getDigit(self, label):
        return random.choice(self.digits[label])


def create_digit_chooser(labels, num_train_puzzles, num_test_puzzles, overlap):
    digit_images = loadMNIST()

    unique_digit_count = (num_train_puzzles + num_test_puzzles) * len(labels) * 2

    unique_digits = {label: digit_images[label][0:unique_digit_count] for label in labels}
    digits = {label: digits for (label, digits) in unique_digits.items()}

    for label in labels:
        digits[label].extend(random.choices(unique_digits[label], k = (int(unique_digit_count * overlap))))
        random.shuffle(digits[label])

    return DigitChooser(digits)


def normalizeMNISTImages(images):
    (numImages, width, height) = images.shape

    # Flatten out the images into a 1d array.
    images = images.reshape(numImages, width * height)

    # Normalize the greyscale intensity to [0,1].
    images = images / 255.0

    # Round so that the output is significantly smaller.
    images = images.round(SIGNIFICANT_DIGITS)

    return images


# Returns: {digit: [image, ...], ...}
def loadMNIST(shuffle = True):
    mnist = tensorflow.keras.datasets.mnist
    (trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

    trainImages = normalizeMNISTImages(trainImages)
    testImages = normalizeMNISTImages(testImages)

    mnistLabels = LABELS

    # {digit: [image, ...], ...}
    digits = {label: [] for label in mnistLabels}
    for i in range(len(trainImages)):
        digits[int(trainLabels[i])].append(trainImages[i])

    for i in range(len(testImages)):
        digits[int(testLabels[i])].append(testImages[i])

    if (shuffle):
        for label in mnistLabels:
            random.shuffle(digits[label])
            random.shuffle(digits[label])

    return digits


def generatePuzzle(digitChooser, labels):
    puzzleImages = [[None] * len(labels) for i in range(len(labels))]
    puzzleLabels = [[None] * len(labels) for i in range(len(labels))]

    # Keep track of the possible options for each location.
    # [row][col][label].
    # Remove options as we add to the puzzle.
    options = [[list(labels) for j in range(len(labels))] for i in range(len(labels))]

    blockSize = int(math.sqrt(len(labels)))

    for row in range(len(labels)):
        for col in range(len(labels)):
            if (len(options[row][col]) == 0):
                # Failed to create a puzzle, try again.
                return None, None

            label = random.choice(options[row][col])
            options[row][col].clear()

            puzzleLabels[row][col] = label

            blockRow = row // blockSize
            blockCol = col // blockSize

            # Remove the chosen digit from row/col/grid options.
            for i in range(len(labels)):
                if label in options[i][col]:
                    options[i][col].remove(label)

                if label in options[row][i]:
                    options[row][i].remove(label)

                for j in range(len(labels)):
                    if (i // blockSize == blockRow and j // blockSize == blockCol):
                        if label in options[i][j]:
                            options[i][j].remove(label)

    # Once we have a complete puzzle, choose the digits.
    for row in range(len(labels)):
        for col in range(len(labels)):
            puzzleImages[row][col] = digitChooser.takeDigit(puzzleLabels[row][col])

    return puzzleImages, puzzleLabels


# Return true of the digit labels create a correct puzzle.
# Note that this does not check for values in |digitLabels| that are not in |labels|.
def checkPuzzle(labels, digitLabels):
    # {row/col: {value, ...}, ...}
    seenInRow = {}
    seenInCol = {}

    # {blockRowId: {blockColId: {value, ...}, ...}, ...}
    seenInBlock = {}

    size = len(labels)
    blockSize = int(math.sqrt(len(labels)))

    # Pre-load the seen data structures.
    for i in range(size):
        seenInRow[i] = set()
        seenInCol[i] = set()

    for blockRowID in range(blockSize):
        seenInBlock[blockRowID] = {}

        for blockColID in range(blockSize):
            seenInBlock[blockRowID][blockColID] = set()

    # Load the seen data structures.
    for row in range(size):
        for col in range(size):
            digit = digitLabels[row][col]

            seenInRow[row].add(digit)
            seenInCol[col].add(digit)
            seenInBlock[row // blockSize][col // blockSize].add(digit)

    # Check for valid rows/cols.
    for i in range(size):
        if (len(seenInRow[i]) != size):
            return False

        if (len(seenInCol[i]) != size):
            return False

    # Check for valid grids.
    for blockRowID in range(blockSize):
        for blockColID in range(blockSize):
            if (len(seenInBlock[blockRowID][blockColID]) != size):
                return False

    return True


def generatePuzzles(digitChooser, labels, numPositivePuzzles):
    # [puzzleIndex][row][col]
    allPuzzleImages = []
    allDigitLabels = []

    # [puzzleIndex]
    allPuzzleLabels = []
    allPuzzleNotes = []

    count = 0

    while (count < numPositivePuzzles):
        puzzleImages, digitLabels = generatePuzzle(digitChooser, labels)
        if (puzzleImages is None):
            continue

        allPuzzleImages.append(puzzleImages)
        allDigitLabels.append(digitLabels)
        allPuzzleLabels.append(BINARY_LABEL_POSITIVE)
        allPuzzleNotes.append([NORMAL_PUZZLE_NOTE])

        corruptDigitLabels = None
        while (corruptDigitLabels is None or checkPuzzle(labels, corruptDigitLabels)):
            corruptImages, corruptDigitLabels, corruptNote = corruptPuzzle(digitChooser, labels, puzzleImages, digitLabels)

        allPuzzleImages.append(corruptImages)
        allDigitLabels.append(corruptDigitLabels)
        allPuzzleLabels.append(BINARY_LABEL_NEGATIVE)
        allPuzzleNotes.append([corruptNote])

        count += 1

    return numpy.stack(allPuzzleImages), numpy.stack(allPuzzleLabels), numpy.stack(allDigitLabels), numpy.stack(allPuzzleNotes)


def get_puzzle_digits(puzzles, puzzle_labels, puzzle_digits, labels, start_index, only_positive_puzzles):
    digit_targets = []
    digit_truths = []
    digit_features = []

    row_column_targets = []
    puzzle_targets = []
    puzzle_truths = []

    for index in range(len(puzzles)):
        if only_positive_puzzles and puzzle_labels[index].tolist() != BINARY_LABEL_POSITIVE:
            continue

        puzzleId = start_index + index

        puzzle_targets.append((puzzleId, ))
        puzzle_truths.append((puzzleId, int(puzzle_labels[index].tolist() != BINARY_LABEL_POSITIVE)))

        for row in range(len(puzzles[index])):
            for digit in labels:
                row_column_targets.append((puzzleId, row, digit))

            for col in range(len(puzzles[index][row])):
                digit_features.append([puzzleId, row, col] + puzzles[index][row][col].tolist())

                for digit in labels:
                    digit_targets.append([puzzleId, row, col, digit])
                    digit_truths.append([puzzleId, row, col, digit, int(digit == puzzle_digits[index][row][col])])

    return digit_targets, digit_truths, digit_features, row_column_targets, puzzle_targets, puzzle_truths


def randCell(dimension, skipLocations = set()):
    row = None
    col = None

    while (row is None or (row, col) in skipLocations):
        row = random.randrange(0, dimension)
        col = random.randrange(0, dimension)

    return row, col


# Corrupt by swapping cells from the same puzzle.
def corruptPuzzleBySwap(digitChooser, labels, corruptImages, corruptLabels):
    count = 0
    seenLocations = set()
    maxSwaps = min(PUZZLE_CORRUPTION_SWAP_MAX, len(labels) ** 2 // 2)

    while ((count < maxSwaps) and (count == 0 or random.random() < PUZZLE_CORRUPTION_SWAP_CHANCE)):
        count += 1

        row1, col1 = randCell(len(labels), seenLocations)
        seenLocations.add((row1, col1))

        row2, col2 = randCell(len(labels), seenLocations)
        seenLocations.add((row2, col2))

        corruptImages[row1][col1], corruptImages[row2][col2] = corruptImages[row2][col2], corruptImages[row1][col1]
        corruptLabels[row1][col1], corruptLabels[row2][col2] = corruptLabels[row2][col2], corruptLabels[row1][col1]

    return corruptImages, corruptLabels, "swap(%d)" % (count)


# Corrupt by replacing single cells at a time.
def corruptPuzzleByReplacement(digitChooser, labels, corruptImages, corruptLabels):
    count = 0
    seenLocations = set()
    maxReplacements = min(PUZZLE_CORRUPTION_REPLACE_MAX, len(labels) ** 2)

    while ((count < maxReplacements) and (count == 0 or random.random() < PUZZLE_CORRUPTION_REPLACE_CHANCE)):
        count += 1

        corruptRow, corruptCol = randCell(len(labels), seenLocations)
        seenLocations.add((corruptRow, corruptCol))

        oldDigit = corruptLabels[corruptRow][corruptCol]
        newDigit = oldDigit
        while (oldDigit == newDigit):
            newDigit = random.choice(labels)

        corruptImages[corruptRow][corruptCol] = digitChooser.getDigit(newDigit)
        corruptLabels[corruptRow][corruptCol] = newDigit

    return corruptImages, corruptLabels, "replace(%d)" % (count)


def corruptPuzzle(digitChooser, labels, originalImages, originalLabels):
    corruptImages = copy.deepcopy(originalImages)
    corruptLabels = copy.deepcopy(originalLabels)

    if (random.randrange(2) == 0):
        return corruptPuzzleByReplacement(digitChooser, labels, corruptImages, corruptLabels)
    else:
        return corruptPuzzleBySwap(digitChooser, labels, corruptImages, corruptLabels)


def testPuzzle(model, puzzles, labels):
    loss, accuracy, auc = model.evaluate(puzzles, labels)
    prob = model.predict(puzzles)

    return (loss, accuracy, auc, prob)


def write_data(subpath, labels, train_puzzles, train_puzzle_labels, train_puzzle_digits, train_puzzle_notes, test_puzzles, test_puzzle_labels, test_puzzle_digits, test_puzzle_notes):
    # Create a mapping of the labels in the first row of the first positive puzzle target to the actual labels.
    # In the model, we will pin these first row values to the labels (since we are just trying to differentiate digits, not classify them).
    label_mapping = {}

    for index_i in range(len(train_puzzles)):
        if train_puzzle_labels[index_i].tolist() != BINARY_LABEL_POSITIVE:
            continue
        for index_j in range(len(labels)):
            label_mapping[int(train_puzzle_digits[index_i][0][index_j])] = int(labels[index_j])
        break
    util.write_psl_file(DIGIT_MAP_PATH.format(subpath), [(key, value) for (key, value) in label_mapping.items()])

    partition_data = {
        'train': [train_puzzles, train_puzzle_labels, train_puzzle_digits, labels, 0, True],
        'test': [test_puzzles, test_puzzle_labels, test_puzzle_digits, labels, len(train_puzzles), False]
    }

    for partition in partition_data:
        suffix = "-" + partition + ".txt"

        puzzles, puzzle_labels, puzzle_digits, labels, offset, only_positive_puzzles = partition_data[partition]
        digit_targets, digit_truths, digit_features, row_column_targets, puzzle_targets, puzzle_truths = get_puzzle_digits(puzzles, puzzle_labels, puzzle_digits, labels, offset, only_positive_puzzles)

        util.write_psl_file(DIGIT_TARGETS_PATH.format(subpath) + suffix, digit_targets)
        util.write_psl_file(DIGIT_TRUTH_PATH.format(subpath) + suffix, digit_truths)
        util.write_psl_file(ENTITY_DATA_MAP_PATH.format(subpath) + suffix, digit_features)

        for digit_truth in digit_truths:
            digit_truth[3] = label_mapping[digit_truth[3]]
        util.write_psl_file(DIGIT_MAPPED_TRUTH_PATH.format(subpath) + suffix, digit_truths)

        if partition == 'train':
            first_puzzle_digit_truths = []
            for col in range(len(labels)):
                for labelIndex in range(len(labels)):
                    first_puzzle_digit_truths.append((digit_targets[0][0], 0, col, labelIndex, int(col == labelIndex)))

            util.write_psl_file(FIRST_PUZZLE_OBSERVATION_PATH.format(subpath), first_puzzle_digit_truths)
        elif partition == 'test':
            util.write_psl_file(ROW_COLUMN_TARGETS_PATH.format(subpath), row_column_targets)
            util.write_psl_file(PUZZLE_TARGETS_PATH.format(subpath), puzzle_targets)
            util.write_psl_file(PUZZLE_TRUTH_PATH.format(subpath), puzzle_truths)


def build_dataset(digit_chooser, labels, split, num_positive_train_puzzles, num_positive_test_puzzles, overlap, seed):
    subpath = SUBPATH_FORMAT.format(len(labels), len(labels), split, num_positive_train_puzzles, overlap)
    num_positive_train_puzzles_with_overlap = int(num_positive_train_puzzles * (1 + overlap))
    num_positive_test_puzzles_with_overlap = int(num_positive_test_puzzles * (1 + overlap))

    config_path = CONFIG_PATH.format(subpath)
    if os.path.isfile(config_path):
        print("Found existing config file, skipping generation. " + config_path)
        return
    print("Generating data defined in: " + config_path)

    train_puzzles, train_puzzle_labels, train_puzzle_digits, train_puzzle_notes = generatePuzzles(digit_chooser, labels, num_positive_train_puzzles_with_overlap)
    test_puzzles, test_puzzle_labels, test_puzzle_digits, test_puzzle_notes = generatePuzzles(digit_chooser, labels, num_positive_test_puzzles_with_overlap)

    os.makedirs(DATA_DIR.format(subpath), exist_ok = True)

    write_data(subpath, labels, train_puzzles, train_puzzle_labels, train_puzzle_digits, train_puzzle_notes, test_puzzles, test_puzzle_labels, test_puzzle_digits, test_puzzle_notes)

    config = {
        'labels': labels,
        'num_train_puzzles': num_positive_train_puzzles_with_overlap * 2,
        'num_test_puzzles': num_positive_test_puzzles_with_overlap * 2,
        'num_positive_train_puzzles': num_positive_train_puzzles_with_overlap,
        'num_positive_test_puzzles': num_positive_test_puzzles_with_overlap,
        'seed': seed,
        'timestamp': str(datetime.datetime.now()),
        'generator': os.path.basename(os.path.realpath(__file__)),
    }

    with open(config_path, 'w') as file:
        json.dump(config, file, indent = 4)


def _load_args():
    parser = argparse.ArgumentParser(description = 'Generate custom sudoku puzzle data.')

    parser.add_argument('--dimension', dest = 'dimension',
                        action = 'store', type = int, default = len(LABELS),
                        choices = [4, 9],
                        help = 'Size of the square puzzle (must have an integer square root).')

    parser.add_argument('--num-positive-train-puzzles', dest = 'numPositiveTrainPuzzles',
                        action = 'store', type = int, default = 100,
                        help = 'The number of positive train puzzles to generate per split (the same number of negative puzzles will also be generated).')

    parser.add_argument('--num-positive-test-puzzles', dest = 'numPositiveTestPuzzles',
                        action = 'store', type = int, default = 100,
                        help = 'The number of positive test puzzles to generate per split (the same number of negative puzzles will also be generated).')

    parser.add_argument('--overlap', dest = 'overlap',
                        action = 'store', type = float, default = 0.0,
                        help = 'The amount of digit images that come from resampling existing digit images.')

    parser.add_argument('--seed', dest = 'seed',
                        action = 'store', type = int, default = None,
                        help = 'Random seed.')

    parser.add_argument('--splits', dest = 'splits',
                        action = 'store', type = int, default = 1,
                        help = 'The number of splits to generate.')

    arguments = parser.parse_args()

    if arguments.numPositiveTrainPuzzles < 1:
        print("Number of positive train puzzles must be >= 1, got: %d." % (arguments.numTrainPuzzles,), file = sys.stderr)
        sys.exit(2)

    if arguments.numPositiveTestPuzzles < 1:
        print("Number of positive test puzzles must be >= 1, got: %d." % (arguments.numTestPuzzles,), file = sys.stderr)
        sys.exit(2)

    if arguments.splits < 1:
        print("Number of splits must be >= 1, got: %d." % (arguments.splits), file = sys.stderr)
        sys.exit(2)

    if int(math.sqrt(arguments.dimension)) ** 2 != arguments.dimension:
        print("Puzzle dimension must have an integer square root, got: %f." % (arguments.dimension), file = sys.stderr)
        sys.exit(2)

    return arguments


def main(arguments):
    seed = arguments.seed
    if seed is None:
        seed = random.randrange(2 ** 64)
    seed_random = random.Random(seed)

    labels = list(LABELS[0:arguments.dimension])

    for split in range(arguments.splits):
        split_seed = seed_random.randrange(2 ** 64)
        random.seed(split_seed)
        tensorflow.random.set_seed(split_seed)

        digits = create_digit_chooser(labels, arguments.numPositiveTrainPuzzles, arguments.numPositiveTestPuzzles, arguments.overlap)
        build_dataset(digits, labels, split, arguments.numPositiveTrainPuzzles, arguments.numPositiveTestPuzzles, arguments.overlap, split_seed)

if __name__ == '__main__':
    main(_load_args())
