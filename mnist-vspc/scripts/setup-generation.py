#!/usr/bin/env python3

# A setup script similar to setup.py,
# except this script will not rely on existing puzzles.
# Instead, fully new puzzles will be generated.

import argparse
import copy
import datetime
import json
import math
import os
import random
import shutil
import sys

import numpy
import pslpython.neupsl
import tensorflow

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

# In this path, include the format string for the subpath.
# The subpath itself may have more subs, but only one will occur for each child.
DATA_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data', '{}'))
SUBPATH_FORMAT = os.path.join('visual-sudoku-{}', 'puzzleDim::{:02d}', 'numPositivePuzzles::{:05d}', 'trainPercent::{:02d}', 'overlapPercent::{:02d}', 'neuralLearningRate::{:0.6f}', 'split::{:02d}', '{:s}')

CONFIG_PATH = os.path.join(DATA_DIR, 'config.json')

PUZZLE_IDS_PATH = os.path.join(DATA_DIR, 'puzzle.txt')
PUZZLE_POSITIVE_IDS_PATH = os.path.join(DATA_DIR, 'puzzle_positive.txt')
PUZZLE_FIRST_ID_PATH = os.path.join(DATA_DIR, 'first_puzzle.txt')
PUZZLE_POSITIVE_FIRST_ID_PATH = os.path.join(DATA_DIR, 'first_positive_puzzle.txt')
BLOCKS_PATH = os.path.join(DATA_DIR, 'block.txt')

DIGIT_LABELS_PATH = os.path.join(DATA_DIR, 'digit_labels.txt')
DIGIT_MAP_PATH = os.path.join(DATA_DIR, 'digit_map.txt')
DIGIT_PINNED_TRUTH_PATH = os.path.join(DATA_DIR, 'pinned_digit_mapped_truth.txt')

# Digits from all examples.
DIGIT_FEATURES_PATH = os.path.join(DATA_DIR, 'digit_features.txt')
DIGIT_TARGETS_PATH = os.path.join(DATA_DIR, 'digit_targets.txt')
DIGIT_TRUTH_PATH = os.path.join(DATA_DIR, 'digit_truth.txt')
DIGIT_MAPPED_TRUTH_PATH = os.path.join(DATA_DIR, 'digit_mapped_truth.txt')

# Only digits from positive examples.
DIGIT_POSITIVE_FEATURES_PATH = os.path.join(DATA_DIR, 'digit_positive_features.txt')
DIGIT_POSITIVE_TARGETS_PATH = os.path.join(DATA_DIR, 'digit_positive_targets.txt')
DIGIT_POSITIVE_TRUTH_PATH = os.path.join(DATA_DIR, 'digit_positive_truth.txt')
DIGIT_POSITIVE_MAPPED_TRUTH_PATH = os.path.join(DATA_DIR, 'digit_positive_mapped_truth.txt')

UNTRAINED_DIGIT_MODEL_H5_PATH = os.path.join(DATA_DIR, 'digit_model_untrained.h5')
TRAINED_DIGIT_MODEL_H5_PATH = os.path.join(DATA_DIR, 'digit_model_trained.h5')
UNTRAINED_DIGIT_MODEL_TF_PATH = os.path.join(DATA_DIR, 'digit_model_untrained_tf')
TRAINED_DIGIT_MODEL_TF_PATH = os.path.join(DATA_DIR, 'digit_model_trained_tf')

ROW_VIOLATIONS_PATH = os.path.join(DATA_DIR, 'row_col_violation_targets.txt')
VIOLATIONS_TARGETS_PATH = os.path.join(DATA_DIR, 'violation_targets.txt')
VIOLATIONS_TRUTH_PATH = os.path.join(DATA_DIR, 'violation_truth.txt')

ROW_VIOLATIONS_POSITIVE_PATH = os.path.join(DATA_DIR, 'row_col_violation_positive_targets.txt')
VIOLATIONS_POSITIVE_TARGETS_PATH = os.path.join(DATA_DIR, 'violation_positive_targets.txt')
VIOLATIONS_POSITIVE_TRUTH_PATH = os.path.join(DATA_DIR, 'violation_positive_truth.txt')

PUZZLE_MODEL_DIR = os.path.join(DATA_DIR, 'puzzle_model')

PUZZLE_TRAIN_FEATURES_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_train_features.txt')
PUZZLE_TRAIN_LABELS_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_train_labels.txt')
PUZZLE_TRAIN_NOTES_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_train_notes.txt')
PUZZLE_TRAIN_DIGITS_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_train_digits.txt')
PUZZLE_TEST_FEATURES_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_test_features.txt')
PUZZLE_TEST_LABELS_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_test_labels.txt')
PUZZLE_TEST_NOTES_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_test_notes.txt')
PUZZLE_TEST_DIGITS_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_test_digits.txt')

UNTRAINED_PUZZLE_VISUAL_MODEL_TF_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_visual_model_untrained_tf')
TRAINED_PUZZLE_VISUAL_MODEL_TF_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_visual_model_trained_tf')
UNTRAINED_PUZZLE_DIGIT_MODEL_TF_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_digit_model_untrained_tf')
TRAINED_PUZZLE_DIGIT_MODEL_TF_PATH = os.path.join(PUZZLE_MODEL_DIR, 'puzzle_digit_model_trained_tf')

LABELS = list(range(1, 10))
BINARY_LABELS = [0, 1]
BINARY_LABEL_POSITIVE = [0, 1]
BINARY_LABEL_NEGATIVE = [1, 0]

# MNIST images are 28 x 28 = 784.
MNIST_DIMENSION = 28

NEURAL_EPOCHS = 100
NEURAL_LOSS = 'KLDivergence'
NEURAL_METRICS = ['categorical_accuracy']
DEFAULT_NEURAL_LEARNING_RATE = 1.0e-3

# The chances to continue the respective corruption.
PUZZLE_CORRUPTION_REPLACE_CHANCE = 0.50
PUZZLE_CORRUPTION_REPLACE_MAX = 10
PUZZLE_CORRUPTION_SWAP_CHANCE = 0.50
PUZZLE_CORRUPTION_SWAP_MAX = 10

DEFAULT_TRAIN_PERCENT = 0.5
DEFAULT_OVERLAP_PERCENT = 0.0
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

# Create two digit choosers (learn/eval) that select from separate pools of possibly overlapping digits.
def createDigitChoosers(labels, numPositivePuzzles, overlapPercent):
    digitImages = loadMNIST()

    # We know exactly how many of each digit we will need right away:
    # count * 2 (learn/eval) * 2 (positive/negative) * |labels|
    digitCount = numPositivePuzzles * 2 * 2 * len(labels)

    # Incorporating the overlap tells us how many unique digits we need.
    uniqueDigitCount = int(digitCount * (1.0 - overlapPercent))

    if (uniqueDigitCount > len(digitImages[0])):
        print("Number of required unique digits (%d) exceeds number of available digits (%d)." % (digitCount, len(digitImages[0])),
              file = sys.stderr)
        sys.exit(3)

    # Sample the necessary unique digits.
    uniqueDigits = {label: digitImages[label][0:uniqueDigitCount] for label in labels}

    # Evenly split between learn/eval.
    splitUniqueDigitCount = uniqueDigitCount // 2
    uniqueLearnDigits = {label: digits[:splitUniqueDigitCount] for (label, digits) in uniqueDigits.items()}
    uniqueEvalDigits = {label: digits[splitUniqueDigitCount:] for (label, digits) in uniqueDigits.items()}

    # Sample the duplicate images.
    learnDigits = {label: digits for (label, digits) in uniqueLearnDigits.items()}
    evalDigits = {label: digits for (label, digits) in uniqueEvalDigits.items()}

    for label in labels:
        learnDigits[label].extend(random.choices(uniqueLearnDigits[label], k = (digitCount - uniqueDigitCount)))
        evalDigits[label].extend(random.choices(uniqueEvalDigits[label], k = (digitCount - uniqueDigitCount)))

        random.shuffle(learnDigits[label])
        random.shuffle(evalDigits[label])

    return DigitChooser(learnDigits), DigitChooser(evalDigits)

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

    mnistLabels = [0] + LABELS

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

def onehot(labels, label):
    encoding = [0] * len(labels)
    encoding[labels.index(label)] = 1
    return encoding

def getDigits(labels, puzzleImages, puzzleLabels):
    digitImages = []
    digitLabels = []

    for i in range(len(puzzleImages)):
        for row in range(len(puzzleImages[i])):
            for col in range(len(puzzleImages[i][row])):
                digitImages.append(puzzleImages[i][row][col])
                digitLabels.append(onehot(labels, puzzleLabels[i][row][col]))

    return numpy.stack(digitImages), numpy.stack(digitLabels)

def getPuzzleDigits(labels, onlyPositiveExamples,
                    trainPuzzles, trainPuzzleLabels, trainPuzzleDigits,
                    testPuzzles, testPuzzleLabels, testPuzzleDigits):
    digitTargets = []
    digitTruths = []
    digitFeatures = []

    rowViolations = []
    violationTargets = []
    violationTruths = []

    for i in range(len(trainPuzzles)):
        if (onlyPositiveExamples and trainPuzzleLabels[i].tolist() != BINARY_LABEL_POSITIVE):
            continue

        puzzleId = i

        violationTargets.append((puzzleId, ))
        violationTruths.append((puzzleId, int(trainPuzzleLabels[i].tolist() != BINARY_LABEL_POSITIVE)))

        for row in range(len(trainPuzzles[i])):
            for digit in labels:
                rowViolations.append((puzzleId, row, digit))

            for col in range(len(trainPuzzles[i][row])):
                digitFeatures.append([puzzleId, row, col] + trainPuzzles[i][row][col].tolist())

                for digit in labels:
                    digitTargets.append([puzzleId, row, col, digit])
                    digitTruths.append([puzzleId, row, col, digit, int(digit == trainPuzzleDigits[i][row][col])])

    for i in range(len(testPuzzles)):
        if (onlyPositiveExamples and testPuzzleLabels[i].tolist() != BINARY_LABEL_POSITIVE):
            continue

        puzzleId = len(trainPuzzles) + i

        violationTargets.append((puzzleId, ))
        violationTruths.append((puzzleId, int(testPuzzleLabels[i].tolist() != BINARY_LABEL_POSITIVE)))

        for row in range(len(testPuzzles[i])):
            for digit in labels:
                rowViolations.append((puzzleId, row, digit))

            for col in range(len(testPuzzles[i][row])):
                digitFeatures.append([puzzleId, row, col] + testPuzzles[i][row][col].tolist())

                for digit in labels:
                    digitTargets.append([puzzleId, row, col, digit])
                    digitTruths.append([puzzleId, row, col, digit, int(digit == testPuzzleDigits[i][row][col])])

    return digitTargets, digitTruths, digitFeatures, rowViolations, violationTargets, violationTruths

def getPuzzleFeatures(labels, puzzles, puzzleDigits):
    digitImageSize = len(puzzles[0][0][0])

    features = []
    digits = []

    for i in range(len(puzzles)):
        features.append(puzzles[i].reshape((len(labels) ** 2 * digitImageSize, )))
        digits.append(puzzleDigits[i].reshape((len(labels) ** 2, )))

    return numpy.stack(features), numpy.stack(digits)

def randCell(dimension, skipLocations = set()):
    row = None
    col = None

    while (row is None or (row, col) in skipLocations):
        row = random.randrange(0, dimension)
        col = random.randrange(0, dimension)

    return row, col

# Corrupt by swaping cells from the same puzzle.
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

def buildDigitNetwork(inputSize, labels, neuralLearningRate):
    layers = [
        tensorflow.keras.layers.Input(shape=inputSize, name='input'),
        tensorflow.keras.layers.Reshape((MNIST_DIMENSION, MNIST_DIMENSION, 1), input_shape=(inputSize,), name='01-reshape'),
        tensorflow.keras.layers.Conv2D(filters=6, kernel_size=5, data_format='channels_last', name='03-conv2d_6_5'),
        tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2), data_format='channels_last', name='04-mp_2_2'),
        tensorflow.keras.layers.Activation('relu', name='05-relu'),
        tensorflow.keras.layers.Conv2D(filters=16, kernel_size=5, data_format='channels_last', name='06-conv2d_16_5'),
        tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2), data_format='channels_last', name='07-mp_2_2'),
        tensorflow.keras.layers.Activation('relu', name='08-relu'),
        tensorflow.keras.layers.Flatten(name='09-flatten'),
        tensorflow.keras.layers.Dense(120, activation='relu', name='10-dense_120'),
        tensorflow.keras.layers.Dense(84, activation='relu', name='11-dense_84'),
        tensorflow.keras.layers.Dense(len(labels), activation='softmax', name='output'),
    ]

    model = tensorflow.keras.Sequential(layers = layers, name = 'digitNetwork')

    model.compile(
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate = neuralLearningRate),
        loss = NEURAL_LOSS,
        metrics = NEURAL_METRICS
    )

    wrapper = pslpython.neupsl.NeuPSLWrapper(model, inputSize, len(labels))
    wrapper.model.summary()

    return wrapper

def buildPuzzleVisualNetwork(digitInputSize, labels, neuralLearningRate):
    binaryLabels = BINARY_LABEL_POSITIVE
    inputSize = digitInputSize * (len(labels) ** 2)
    visualPuzzleDim = MNIST_DIMENSION * len(labels)

    layers = [
        tensorflow.keras.layers.Input(shape=inputSize, name='input'),
        tensorflow.keras.layers.Reshape((visualPuzzleDim, visualPuzzleDim, 1), input_shape=(inputSize,), name='01-reshape'),
        tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', name='02-conv_16'),
        tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2), name='03-maxpool_2'),
        tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', name='04-conv_16'),
        tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2), name='05-maxpool_2'),
        tensorflow.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', name='06-conv_16'),
        tensorflow.keras.layers.MaxPooling2D(pool_size=(2,2), name='07-maxpool_2'),
        tensorflow.keras.layers.Flatten(name='08-flatten'),
        tensorflow.keras.layers.Dense(units=256, activation='relu', name='09-dense_256'),
        tensorflow.keras.layers.Dense(units=256, activation='relu', name='10-dense_256'),
        tensorflow.keras.layers.Dense(units=128, activation='relu', name='11-dense_128'),
        tensorflow.keras.layers.Dense(len(binaryLabels), activation='softmax', name='output'),
    ]

    model = tensorflow.keras.Sequential(layers = layers, name = 'puzzleVisualNetwork')

    model.compile(
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate = neuralLearningRate),
        loss = 'binary_crossentropy',
        metrics = ['categorical_accuracy', tensorflow.keras.metrics.AUC()],
    )

    model.summary()

    return model

def buildPuzzleDigitNetwork(labels, neuralLearningRate):
    binaryLabels = BINARY_LABEL_POSITIVE
    inputSize = len(labels) ** 2

    layers = [
        tensorflow.keras.layers.Input(shape=inputSize, name='input'),
        tensorflow.keras.layers.Dense(512, activation='relu', name='01-dense_512'),
        tensorflow.keras.layers.Dense(512, activation='relu', name='02-dense_512'),
        tensorflow.keras.layers.Dense(256, activation='relu', name='03-dense_256'),
        tensorflow.keras.layers.Dense(len(binaryLabels), activation='softmax', name='output'),
    ]

    model = tensorflow.keras.Sequential(layers = layers, name = 'puzzleDigitNetwork')

    model.compile(
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate = neuralLearningRate),
        loss = 'binary_crossentropy',
        metrics = ['categorical_accuracy', tensorflow.keras.metrics.AUC()],
    )

    model.summary()

    return model

def testDigit(model, images, labels):
    loss, accuracy = model.evaluate(images, labels)
    prob = model.predict(images)

    return (loss, accuracy, prob)

def writeFile(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")

def writeData(
    subpath, labels,
    trainPuzzles, trainPuzzleLabels, trainPuzzleDigits, trainPuzzleNotes,
    testPuzzles, testPuzzleLabels, testPuzzleDigits, testPuzzleNotes,
    labelMapping = None):
    # Sudoku block/grid information.
    blockSize = int(math.sqrt(len(labels)))
    blocks = []

    for row in range(9):
        for col in range(9):
            blockRow = row // blockSize
            blockCol = col // blockSize
            blockIndex = blockRow * blockSize + blockCol

            blocks.append([row, col, blockIndex])

    writeFile(BLOCKS_PATH.format(subpath), blocks)

    # Digit information.

    writeFile(DIGIT_LABELS_PATH.format(subpath), [[label] for label in labels])

    # All examples.
    digitTargets, digitTruths, digitFeatures, rowViolations, violationTargets, violationTruths = getPuzzleDigits(labels, False,
                                                                                                                 trainPuzzles, trainPuzzleLabels, trainPuzzleDigits, testPuzzles, testPuzzleLabels, testPuzzleDigits)

    allDigitTruths = digitTruths

    writeFile(DIGIT_TARGETS_PATH.format(subpath), digitTargets)
    writeFile(DIGIT_TRUTH_PATH.format(subpath), digitTruths)
    writeFile(DIGIT_FEATURES_PATH.format(subpath), digitFeatures)

    writeFile(ROW_VIOLATIONS_PATH.format(subpath), rowViolations)
    writeFile(VIOLATIONS_TARGETS_PATH.format(subpath), violationTargets)
    writeFile(VIOLATIONS_TRUTH_PATH.format(subpath), violationTruths)

    # Only positive examples.
    digitTargets, digitTruths, digitFeatures, rowViolations, violationTargets, violationTruths = getPuzzleDigits(labels, True,
                                                                                                                 trainPuzzles, trainPuzzleLabels, trainPuzzleDigits, testPuzzles, testPuzzleLabels, testPuzzleDigits)

    positiveDigitTruths = digitTruths

    writeFile(DIGIT_POSITIVE_TARGETS_PATH.format(subpath), digitTargets)
    writeFile(DIGIT_POSITIVE_TRUTH_PATH.format(subpath), digitTruths)
    writeFile(DIGIT_POSITIVE_FEATURES_PATH.format(subpath), digitFeatures)

    writeFile(ROW_VIOLATIONS_POSITIVE_PATH.format(subpath), rowViolations)
    writeFile(VIOLATIONS_POSITIVE_TARGETS_PATH.format(subpath), violationTargets)
    writeFile(VIOLATIONS_POSITIVE_TRUTH_PATH.format(subpath), violationTruths)

    # Puzzle information.

    puzzleTrainFeatures, puzzleTrainDigits = getPuzzleFeatures(labels, trainPuzzles, trainPuzzleDigits)
    puzzleTestFeatures, puzzleTestDigits = getPuzzleFeatures(labels, testPuzzles, testPuzzleDigits)

    positivePuzzleTrainDigits = puzzleTrainDigits

    writeFile(PUZZLE_IDS_PATH.format(subpath), [[i] for i in range(len(trainPuzzles) + len(testPuzzles))])
    writeFile(PUZZLE_POSITIVE_IDS_PATH.format(subpath), [[digitTarget[0]] for digitTarget in digitTargets])
    writeFile(PUZZLE_FIRST_ID_PATH.format(subpath), [[0]])
    writeFile(PUZZLE_POSITIVE_FIRST_ID_PATH.format(subpath), [[digitTargets[0][0]]])

    writeFile(PUZZLE_TRAIN_FEATURES_PATH.format(subpath), puzzleTrainFeatures)
    writeFile(PUZZLE_TRAIN_LABELS_PATH.format(subpath), trainPuzzleLabels)
    writeFile(PUZZLE_TRAIN_NOTES_PATH.format(subpath), trainPuzzleNotes)
    writeFile(PUZZLE_TRAIN_DIGITS_PATH.format(subpath), puzzleTrainDigits)

    writeFile(PUZZLE_TEST_FEATURES_PATH.format(subpath), puzzleTestFeatures)
    writeFile(PUZZLE_TEST_LABELS_PATH.format(subpath), testPuzzleLabels)
    writeFile(PUZZLE_TEST_NOTES_PATH.format(subpath), testPuzzleNotes)
    writeFile(PUZZLE_TEST_DIGITS_PATH.format(subpath), puzzleTestDigits)

    # Create a mapping of the labels in the first row of the first positive puzzle target to the actual labels.
    # In the model, we will pin these first row values to the labels (since we are just trying to differentiate digits, not classify them).

    newMapping = (labelMapping is None)

    if (newMapping):
        labelMapping = {}
        for i in range(len(labels)):
            labelMapping[int(positivePuzzleTrainDigits[0][i])] = int(labels[i])

    for digitTruth in allDigitTruths:
        digitTruth[3] = labelMapping[digitTruth[3]]

    for digitTruth in positiveDigitTruths:
        digitTruth[3] = labelMapping[digitTruth[3]]

    writeFile(DIGIT_MAP_PATH.format(subpath), [(key, value) for (key, value) in labelMapping.items()])
    writeFile(DIGIT_MAPPED_TRUTH_PATH.format(subpath), allDigitTruths)
    writeFile(DIGIT_POSITIVE_MAPPED_TRUTH_PATH.format(subpath), positiveDigitTruths)

    # Write truth data for the first pinned digits (but only when the mapping is first created).

    if (newMapping):
        pinnedTruthDigits = []

        for col in range(len(labels)):
            for labelIndex in range(len(labels)):
                pinnedTruthDigits.append((0, 0, col, labels[labelIndex], int(col == labelIndex)))

        writeFile(DIGIT_PINNED_TRUTH_PATH.format(subpath), pinnedTruthDigits)

    return labelMapping

def buildPuzzleVisualModel(labels, subpath, neuralLearningRate,
                           trainPuzzles, trainPuzzleLabels, trainPuzzleDigits,
                           testPuzzles, testPuzzleLabels, testPuzzleDigits):
    puzzleTrainFeatures, puzzleTrainDigits = getPuzzleFeatures(labels, trainPuzzles, trainPuzzleDigits)
    puzzleTestFeatures, puzzleTestDigits = getPuzzleFeatures(labels, testPuzzles, testPuzzleDigits)

    puzzleModel = buildPuzzleVisualNetwork(len(trainPuzzles[0][0][0]), labels, neuralLearningRate)

    untrainedLoss, untrainedAccuracy, untrainedAUC, _ = testPuzzle(puzzleModel, puzzleTestFeatures, testPuzzleLabels)
    print("Untrained Puzzle Visual Model -- Loss: %f, Accuracy: %f, AUROC: %f" % (untrainedLoss, untrainedAccuracy, untrainedAUC))
    puzzleModel.save(UNTRAINED_PUZZLE_VISUAL_MODEL_TF_PATH.format(subpath), save_format = 'tf', include_optimizer = True)

    trainHistory = puzzleModel.fit(puzzleTrainFeatures, trainPuzzleLabels, epochs = NEURAL_EPOCHS)

    trainedLoss, trainedAccuracy, trainedAUC, _ = testPuzzle(puzzleModel, puzzleTestFeatures, testPuzzleLabels)
    print("Trained Puzzle Visual Model -- Loss: %f, Accuracy: %f, AUROC: %f" % (trainedLoss, trainedAccuracy, trainedAUC))
    puzzleModel.save(TRAINED_PUZZLE_VISUAL_MODEL_TF_PATH.format(subpath), save_format = 'tf', include_optimizer = True)

    return {
        'untrained': {
            'neuralLearningRate': neuralLearningRate,
            'loss': untrainedLoss,
            'accuracy': untrainedAccuracy,
            'AUROC': untrainedAUC,
        },
        'pretrained': {
            'epochs': NEURAL_EPOCHS,
            'neuralLearningRate': neuralLearningRate,
            'loss': trainedLoss,
            'accuracy': trainedAccuracy,
            'AUROC': trainedAUC,
            'trainHistory': trainHistory.history,
        },
    }

def buildPuzzleDigitModel(labels, subpath, neuralLearningRate,
                          trainPuzzles, trainPuzzleLabels, trainPuzzleDigits,
                          testPuzzles, testPuzzleLabels, testPuzzleDigits):
    puzzleTrainFeatures, puzzleTrainDigits = getPuzzleFeatures(labels, trainPuzzles, trainPuzzleDigits)
    puzzleTestFeatures, puzzleTestDigits = getPuzzleFeatures(labels, testPuzzles, testPuzzleDigits)

    puzzleModel = buildPuzzleDigitNetwork(labels, neuralLearningRate)

    untrainedLoss, untrainedAccuracy, untrainedAUC, _ = testPuzzle(puzzleModel, puzzleTestDigits, testPuzzleLabels)
    print("Untrained Puzzle Model -- Loss: %f, Accuracy: %f, AUROC: %f" % (untrainedLoss, untrainedAccuracy, untrainedAUC))
    puzzleModel.save(UNTRAINED_PUZZLE_DIGIT_MODEL_TF_PATH.format(subpath), save_format = 'tf', include_optimizer = True)

    trainHistory = puzzleModel.fit(puzzleTrainDigits, trainPuzzleLabels, epochs = NEURAL_EPOCHS)

    trainedLoss, trainedAccuracy, trainedAUC, _ = testPuzzle(puzzleModel, puzzleTestDigits, testPuzzleLabels)
    print("Trained Puzzle Model -- Loss: %f, Accuracy: %f, AUROC: %f" % (trainedLoss, trainedAccuracy, trainedAUC))
    puzzleModel.save(TRAINED_PUZZLE_DIGIT_MODEL_TF_PATH.format(subpath), save_format = 'tf', include_optimizer = True)

    return {
        'untrained': {
            'neuralLearningRate': neuralLearningRate,
            'loss': untrainedLoss,
            'accuracy': untrainedAccuracy,
            'AUROC': untrainedAUC,
        },
        'pretrained': {
            'epochs': NEURAL_EPOCHS,
            'neuralLearningRate': neuralLearningRate,
            'loss': trainedLoss,
            'accuracy': trainedAccuracy,
            'AUROC': trainedAUC,
            'trainHistory': trainHistory.history,
        },
    }

def buildDigitModel(labels, subpath, neuralLearningRate,
                    trainPuzzles, trainPuzzleDigits,
                    testPuzzles, testPuzzleDigits):
    trainDigitImages, trainDigitLabels = getDigits(labels, trainPuzzles, trainPuzzleDigits)
    testDigitImages, testDigitLabels = getDigits(labels, testPuzzles, testPuzzleDigits)

    modelWrapper = buildDigitNetwork(len(trainDigitImages[0]), labels, neuralLearningRate)
    modelWrapper.save(UNTRAINED_DIGIT_MODEL_H5_PATH.format(subpath), UNTRAINED_DIGIT_MODEL_TF_PATH.format(subpath))

    untrainedLoss, untrainedAccuracy, _ = testDigit(modelWrapper.model, testDigitImages, testDigitLabels)
    print("Untrained Model -- Loss: %f, Accuracy: %f" % (untrainedLoss, untrainedAccuracy))

    modelWrapper = buildDigitNetwork(len(trainDigitImages[0]), labels, neuralLearningRate)
    trainHistory = modelWrapper.model.fit(trainDigitImages, trainDigitLabels, epochs = NEURAL_EPOCHS)

    modelWrapper.save(TRAINED_DIGIT_MODEL_H5_PATH.format(subpath), TRAINED_DIGIT_MODEL_TF_PATH.format(subpath))

    trainedLoss, trainedAccuracy, testPredictions = testDigit(modelWrapper.model, testDigitImages, testDigitLabels)
    print("Trained Model -- Loss: %f, Accuracy: %f" % (trainedLoss, trainedAccuracy))

    pretrainedModel = tensorflow.keras.models.load_model(TRAINED_DIGIT_MODEL_H5_PATH.format(subpath))

    pretrainedLoss, pretrainedAccuracy, _ = testDigit(pretrainedModel, testDigitImages, testDigitLabels)
    print("Pretrained Model -- Loss: %f, Accuracy: %f" % (pretrainedLoss, pretrainedAccuracy))

    if (not math.isclose(trainedAccuracy, pretrainedAccuracy)):
        print("ERROR: Trained and pretrained accuracy do not match.", file = sys.stderr)
        sys.exit(2)

    return {
        'untrained': {
            'neuralLearningRate': neuralLearningRate,
            'loss': untrainedLoss,
            'accuracy': untrainedAccuracy,
        },
        'pretrained': {
            'epochs': NEURAL_EPOCHS,
            'neuralLearningRate': neuralLearningRate,
            'loss': trainedLoss,
            'accuracy': trainedAccuracy,
            'trainHistory': trainHistory.history,
        },
    }

def buildDataset(suffix, labels, split, digitChooser, numPositivePuzzles, trainPercent, overlapPercent, neuralLearningRate, seed, foldType,
                 force = False, labelMapping = None):
    subpath = SUBPATH_FORMAT.format(suffix, len(labels), numPositivePuzzles, int(trainPercent * 100), int(overlapPercent * 100), neuralLearningRate, split, foldType)

    configPath = CONFIG_PATH.format(subpath)
    if (os.path.isfile(configPath)):
        if (not force):
            print("Found existing config file, skipping generation. " + configPath)
            return

        print("Found existing config file, but forcing over it. " + configPath)
        shutil.rmtree(DATA_DIR.format(subpath))
    print("Generating data defined in: " + configPath)

    numTrainPuzzles = int(numPositivePuzzles * trainPercent)
    numTestPuzzles = numPositivePuzzles - numTrainPuzzles

    trainPuzzles, trainPuzzleLabels, trainPuzzleDigits, trainPuzzleNotes = generatePuzzles(digitChooser, labels, numTrainPuzzles)
    testPuzzles, testPuzzleLabels, testPuzzleDigits, testPuzzleNotes = generatePuzzles(digitChooser, labels, numTestPuzzles)

    os.makedirs(DATA_DIR.format(subpath), exist_ok = True)
    os.makedirs(PUZZLE_MODEL_DIR.format(subpath), exist_ok = True)

    # digitModelConfigInfo = buildDigitModel(labels, subpath, neuralLearningRate, trainPuzzles, trainPuzzleDigits, testPuzzles, testPuzzleDigits)
    puzzleVisualModelConfigInfo = buildPuzzleVisualModel(labels, subpath, neuralLearningRate, trainPuzzles, trainPuzzleLabels, trainPuzzleDigits, testPuzzles, testPuzzleLabels, testPuzzleDigits)
    puzzleDigitModelConfigInfo = buildPuzzleDigitModel(labels, subpath, neuralLearningRate, trainPuzzles, trainPuzzleLabels, trainPuzzleDigits, testPuzzles, testPuzzleLabels, testPuzzleDigits)

    labelMapping = writeData(subpath, labels,
                             trainPuzzles, trainPuzzleLabels, trainPuzzleDigits, trainPuzzleNotes,
                             testPuzzles, testPuzzleLabels, testPuzzleDigits, testPuzzleNotes,
                             labelMapping = labelMapping)

    config = {
        'labels': labels,
        # 'digitModel': digitModelConfigInfo,
        'puzzleVisualModel': puzzleVisualModelConfigInfo,
        'puzzleDigitModel': puzzleDigitModelConfigInfo,
        'numPositivePuzzles': numPositivePuzzles,
        'numTrainPuzzles': numTrainPuzzles,
        'numTestPuzzles': numTestPuzzles,
        'trainPercent': trainPercent,
        'labelMapping': labelMapping,
        'seed': seed,
        'timestamp': str(datetime.datetime.now()),
        'generator': os.path.basename(os.path.realpath(__file__)),
    }

    with open(configPath, 'w') as file:
        json.dump(config, file, indent = 4)

    return labelMapping

def _load_args():
    parser = argparse.ArgumentParser(description = 'Generate custom sudoku puzzle data.')

    parser.add_argument('--dimension', dest = 'dimension',
                        action = 'store', type = int, default = len(LABELS),
                        choices = [4, 9],
                        help = 'Size of the square puzzle (must have an integer square root).')

    parser.add_argument('--force', dest = 'force',
                        action = 'store_true', default = False,
                        help = 'Ignore existing data directories and write over them.')

    parser.add_argument('--neural-learning-rate', dest = 'neuralLearningRate',
                        action = 'store', type = float, default = DEFAULT_NEURAL_LEARNING_RATE,
                        help = 'The learning rate for the neural network.')

    parser.add_argument('--num-puzzles', dest = 'count',
                        action = 'store', type = int, default = 100,
                        help = 'The number of correct puzzles to generate per split (the same number of negative puzzles will also be generated).')

    parser.add_argument('--overlap-percent', dest = 'overlapPercent',
                        action = 'store', type = float, default = DEFAULT_OVERLAP_PERCENT,
                        help = 'The percentage of digit images that come from resampling existing digit images.')

    parser.add_argument('--seed', dest = 'seed',
                        action = 'store', type = int, default = None,
                        help = 'Random seed.')

    parser.add_argument('--splits', dest = 'splits',
                        action = 'store', type = int, default = 1,
                        help = 'The number of splits to generate.')

    parser.add_argument('--suffix', dest = 'suffix',
                        action = 'store', type = str, default = 'test',
                        help = 'The suffix to use when outputting the data (visual-sudoku-<suffix>).')

    parser.add_argument('--train-percent', dest = 'trainPercent',
                        action = 'store', type = float, default = DEFAULT_TRAIN_PERCENT,
                        help = 'The percentage of data used for training (rest is for test).')

    arguments = parser.parse_args()

    if (arguments.count < 1):
        print("Number of puzzles must be >= 1, got: %d." % (arguments.count), file = sys.stderr)
        sys.exit(2)

    if (arguments.splits < 1):
        print("Number of splits must be >= 1, got: %d." % (arguments.splits), file = sys.stderr)
        sys.exit(2)

    if (arguments.trainPercent < 0.0 or arguments.trainPercent > 1.0):
        print("Train percent must be in [0.0, 1.0], got: %f." % (arguments.trainPercent), file = sys.stderr)
        sys.exit(2)

    if (arguments.overlapPercent < 0.0 or arguments.overlapPercent > 1.0):
        print("Overlap percent must be in [0.0, 1.0], got: %f." % (arguments.overlapPercent), file = sys.stderr)
        sys.exit(2)

    blockSize = int(math.sqrt(arguments.dimension))
    if (blockSize ** 2 != arguments.dimension):
        print("Puzzle dimension must have an integer square root, got: %f." % (arguments.dimension), file = sys.stderr)
        sys.exit(2)

    return arguments

def main(arguments):
    # Make an RNG for generating seeds.
    seed = arguments.seed
    if (seed is None):
        seed = random.randrange(2 ** 64)
    seedRandom = random.Random(seed)

    labels = list(LABELS[0:arguments.dimension])

    for split in range(arguments.splits):
        splitSeed = seedRandom.randrange(2 ** 64)
        random.seed(splitSeed)
        tensorflow.random.set_seed(splitSeed)

        learnDigits, evalDigits = createDigitChoosers(labels, arguments.count, arguments.overlapPercent)

        labelMapping = buildDataset(arguments.suffix, labels, split, learnDigits,
                                    arguments.count, arguments.trainPercent, arguments.overlapPercent, arguments.neuralLearningRate,
                                    splitSeed, 'learn',
                                    arguments.force, labelMapping = None)

        buildDataset(arguments.suffix, labels, split, evalDigits,
                     arguments.count, arguments.trainPercent, arguments.overlapPercent, arguments.neuralLearningRate,
                     splitSeed, 'eval',
                     arguments.force, labelMapping = labelMapping)

if (__name__ == '__main__'):
    main(_load_args())
