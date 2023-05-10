#!/usr/bin/env python3

# Parse out the results.

import os
import re
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.abspath(os.path.join(THIS_DIR, '../results'))

EVAL_LOG_FILENAME = 'out.txt'

OPERATION_OVERALL = 'OVERALL'

OVERALL_HEADER = [
    'Digit Accuracy',
    'Sum Accuracy',
    'Memory Used',
    'Learn Runtime',
]


def parseLog(logPath):
    results = {
        'evaluation': None,
    }

    with open(logPath, 'r') as file:
        for line in file:
            line = line.strip()
            if line == '':
                continue

            match = re.search(r'^Loss: (\d+\.\d+)IMAGESUM -- Categorical Accuracy: (\d+\.\d+)', line)

            if match is not None:
                results['evaluation'] = float(match.group(1))

    return results


def parseDir(resultDir):
    evalLogPath = os.path.join(resultDir, EVAL_LOG_FILENAME)
    if (not os.path.isfile(evalLogPath)):
        return None

    try:
        results = parseLog(evalLogPath)
    except Exception as ex:
        print("Failed to parse log: " + evalLogPath, file=sys.stderr)
        return None

    return results


# Descend into the results directory to look for results.
# Before the actual run directories will be an unknown number of directories specifying data/hyper parameters.
# [(hyper/data params, results), ...]
def fetchResults(runs=[], baseDir=RESULTS_DIR, directoryParams={}):
    logPath = os.path.join(baseDir, EVAL_LOG_FILENAME)
    if os.path.isfile(logPath):
        # We found a run directory.
        if directoryParams["model"] != "neural_baseline":
            return runs

        results = parseDir(baseDir)
        if results is not None:
            runs.append((directoryParams.copy(), results))
        return runs

    # Not a run directory, descend.
    for dirent in sorted(os.listdir(baseDir)):
        direntPath = os.path.join(baseDir, dirent)
        if (os.path.isfile(direntPath)):
            continue

        paramParts = dirent.split('::')
        if (len(paramParts) != 2):
            raise ValueError("Bad parameter path (%s): '%s'." % (dirent, direntPath))

        directoryParams[paramParts[0]] = paramParts[1]

        fetchResults(runs, direntPath, directoryParams)

    return runs


def outputOverallStats(runs):
    header = OVERALL_HEADER.copy()
    runParamKeys = None

    rows = []

    for (runParams, results) in runs:
        # Make sure the header is filled in.
        if (runParamKeys is None):
            runParamKeys = list(sorted(runParams.keys()))
            header += runParamKeys

        row = [results['evaluation']]

        for runParamKey in runParamKeys:
            row.append(runParams[runParamKey])

        rows.append(row)

    print('\t'.join(header))
    for row in rows:
        print('\t'.join(map(str, row)))


def main(operation):
    runs = fetchResults()
    if (len(runs) == 0):
        return

    if (operation == OPERATION_OVERALL):
        outputOverallStats(runs)
    else:
        print("Unknown operation: '%s'." % (operation), file=sys.stderr)
        sys.exit(3)


def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <%s>" % (executable, OPERATION_OVERALL), file=sys.stderr)
        sys.exit(1)

    operation = args.pop(0).upper().strip()
    if (operation not in [OPERATION_OVERALL]):
        print("Unknown operation: '%s'." % (operation), file=sys.stderr)
        sys.exit(2)

    return operation


if (__name__ == '__main__'):
    main(_load_args(sys.argv))
