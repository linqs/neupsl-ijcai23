#!/usr/bin/env python3

# Run baselines after the data has already been created.

import glob
import json
import os
import re
import sys
import time

import numpy
import tensorflow

EXPERIMENT = 'visual-sudoku-experiment'

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(THIS_DIR, '..', 'results')
DATA_DIR = os.path.join(THIS_DIR, '..', '..', '..', 'data', EXPERIMENT)

OUT_FILENAME = 'out-baseline.json'
OUT_TRAIN_PREDICTIONS_FILENAME = 'baseline-train-predictions.txt'
OUT_TRAIN_LABELS_FILENAME = 'baseline-train-labels.txt'
OUT_TEST_PREDICTIONS_FILENAME = 'baseline-test-predictions.txt'
OUT_TEST_LABELS_FILENAME = 'baseline-test-labels.txt'
OUT_KEYS = [
    'experiment', 'method', 'puzzleDim', 'numPositivePuzzles', 'trainPercent', 'overlapPercent',
    'neuralLearningRate', 'split',
]

SPLITS = ['learn', 'eval']

BASELINE_NAME_DIGIT = 'baseline-digit'
BASELINE_NAME_VISUAL = 'baseline-visual'

MODEL_REL_DIR = {
    BASELINE_NAME_DIGIT: os.path.join('learn', 'puzzle_model', 'puzzle_digit_model_untrained_tf'),
    BASELINE_NAME_VISUAL: os.path.join('learn', 'puzzle_model', 'puzzle_visual_model_untrained_tf'),
}

# {method: [(train, test), ...]
MODEL_LABELS_REL_PATHS = {
    BASELINE_NAME_DIGIT: [
        [os.path.join(split, 'puzzle_model', 'puzzle_test_labels.txt') for split in SPLITS],
        [os.path.join(split, 'puzzle_model', 'puzzle_train_labels.txt') for split in SPLITS],
    ],
    BASELINE_NAME_VISUAL: [
        [os.path.join(split, 'puzzle_model', 'puzzle_test_labels.txt') for split in SPLITS],
        [os.path.join(split, 'puzzle_model', 'puzzle_train_labels.txt') for split in SPLITS],
    ],
}

# {method: [(train, test), ...]
MODEL_FEATURES_REL_PATHS = {
    BASELINE_NAME_DIGIT: [
        [os.path.join(split, 'puzzle_model', 'puzzle_test_digits.txt') for split in SPLITS],
        [os.path.join(split, 'puzzle_model', 'puzzle_train_digits.txt') for split in SPLITS],
    ],
    BASELINE_NAME_VISUAL: [
        [os.path.join(split, 'puzzle_model', 'puzzle_test_features.txt') for split in SPLITS],
        [os.path.join(split, 'puzzle_model', 'puzzle_train_features.txt') for split in SPLITS],
    ],
}

MODEL_FEATURES_TYPE = {
    BASELINE_NAME_DIGIT: int,
    BASELINE_NAME_VISUAL: float,
}

CONFIG_FILENAME = 'config.json'

EPOCHS = 100

def writeFile(path, data, dtype = str):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(dtype(item)) for item in row]) + "\n")

def loadData(method, dataDir):
    trainFeatures = []
    trainLabels = []
    testFeatures = []
    testLabels = []

    for (trainRelpath, testRelpath) in MODEL_LABELS_REL_PATHS[method]:
        trainPath = os.path.join(dataDir, trainRelpath)
        trainLabels += numpy.loadtxt(trainPath, delimiter = "\t", dtype = int).tolist()

        testPath = os.path.join(dataDir, testRelpath)
        testLabels += numpy.loadtxt(testPath, delimiter = "\t", dtype = int).tolist()

    for (trainRelpath, testRelpath) in MODEL_FEATURES_REL_PATHS[method]:
        trainPath = os.path.join(dataDir, trainRelpath)
        trainFeatures += numpy.loadtxt(trainPath, delimiter = "\t", dtype = MODEL_FEATURES_TYPE[method]).tolist()

        testPath = os.path.join(dataDir, testRelpath)
        testFeatures += numpy.loadtxt(testPath, delimiter = "\t", dtype = MODEL_FEATURES_TYPE[method]).tolist()

    return numpy.stack(trainFeatures), numpy.stack(trainLabels), numpy.stack(testFeatures), numpy.stack(testLabels)

def runBaseline(dataDir, method):
    attributes = {
        'experiment': EXPERIMENT,
        'method': method,
    }

    for (key, value) in re.findall(r'([\w\-\.]+)::([\w\-\.]+)', dataDir):
        attributes[key] = value

    assert('neuralLearningRate' in attributes)

    outDir = os.path.join(RESULTS_DIR, *["%s::%s" % (key, attributes[key]) for key in OUT_KEYS])
    outPath = os.path.join(outDir, OUT_FILENAME)

    if (os.path.isfile(outPath)):
        print("Found existing baseline output, skipping run: " + outPath)
        return
    print("Running baseline defined in: " + outPath)

    totalStartTime = int(time.time() * 1000)

    trainFeatures, trainLabels, testFeatures, testLabels = loadData(method, dataDir)

    modelDir = os.path.join(dataDir, MODEL_REL_DIR[method])
    model = tensorflow.keras.models.load_model(modelDir)

    model.summary()
    model.compile(
        optimizer = tensorflow.keras.optimizers.Adam(learning_rate = float(attributes['neuralLearningRate'])),
        loss = 'binary_crossentropy',
        metrics = ['categorical_accuracy', tensorflow.keras.metrics.AUC()],
    )

    startTime = int(time.time() * 1000)

    trainHistory = model.fit(trainFeatures, trainLabels, epochs = EPOCHS)

    endTime = int(time.time() * 1000)
    trainTime = endTime - startTime
    startTime = endTime

    loss, accuracy, auc = model.evaluate(testFeatures, testLabels)

    endTime = int(time.time() * 1000)
    testTime = endTime - startTime
    totalTime = endTime - totalStartTime

    trainPredictions = model.predict(trainFeatures)
    testPredictions = model.predict(testFeatures)

    print("%s Results -- Loss: %f, Accuracy: %f, AUROC: %f" % (method, loss, accuracy, auc))

    results = {
        'method': method,
        'epochs': EPOCHS,
        'neuralLearningRate': float(attributes['neuralLearningRate']),
        'loss': loss,
        'accuracy': accuracy,
        'AUROC': auc,
        'runtime': totalTime,
        'trainTime': trainTime,
        'testTime': testTime,
        'trainHistory': trainHistory.history,
    }

    os.makedirs(outDir, exist_ok = True)

    writeFile(os.path.join(outDir, OUT_TRAIN_PREDICTIONS_FILENAME), trainPredictions, float)
    writeFile(os.path.join(outDir, OUT_TRAIN_LABELS_FILENAME), trainLabels, int)

    writeFile(os.path.join(outDir, OUT_TEST_PREDICTIONS_FILENAME), testPredictions, float)
    writeFile(os.path.join(outDir, OUT_TEST_LABELS_FILENAME), testLabels, int)

    with open(outPath, 'w') as file:
        json.dump(results, file, indent = 4)

def main():
    # Search for config files that indicate complete data dits.
    for configPath in glob.glob("%s/**/eval/%s" % (DATA_DIR, CONFIG_FILENAME), recursive = True):
        dataDir = os.path.join(os.path.dirname(configPath), '..')

        for baseline in MODEL_REL_DIR:
            runBaseline(dataDir, baseline)

def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s" % (executable), file = sys.stderr)
        sys.exit(1)

if (__name__ == '__main__'):
    _load_args(sys.argv)
    main()
