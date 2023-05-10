#!/usr/bin/env python3

# Parse out the results.

import glob
import os
import re
import sys

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(THIS_DIR, '..', 'results')

LOG_FILENAME = 'results.csv'
ERR_FILENAME = 'out.err'
TIME_START_FILENAME = 'time-start.txt'
TIME_END_FILENAME = 'time-end.txt'

HEADER = [
    # Identifiers
    'experiment',
    'problem',
    'method',
    'size',
    'overlap',
    'split',
    'epochs',
    # Results
    'runtime',
    'trainTime',
    'testTime',
    'trainLoss',
    'trainAccuracy',
    'testLoss',
    'testAccuracy',
]

def parseLog(logPath):
    results = {}

    # Fetch the run identifiers off of the path.
    for (key, value) in re.findall(r'([\w\-\.]+)::([\w\-\.]+)', logPath):
        results[key] = value

    firstLine = True
    with open(logPath, 'r') as file:
        for line in file:
            if (firstLine):
                firstLine = False
                continue

            line = line.strip()
            if (line == ''):
                continue

            parts = line.split(',')

            # Just keep overriding until the last one stays.
            results['epochs'] = int(parts[0]) + 1
            results['trainLoss'] = float(parts[1])
            results['trainAccuracy'] = float(parts[2])
            results['testLoss'] = float(parts[3])
            results['testAccuracy'] = float(parts[4])

    return results

def parseTime(logPath):
    startPath = os.path.join(os.path.dirname(logPath), TIME_START_FILENAME)
    endPath = os.path.join(os.path.dirname(logPath), TIME_END_FILENAME)
    errPath = os.path.join(os.path.dirname(logPath), ERR_FILENAME)

    if (not os.path.isfile(startPath) or not os.path.isfile(endPath) or not os.path.isfile(errPath)):
        return None, None, None

    with open(startPath, 'r') as file:
        startTime = int(file.read())

    with open(endPath, 'r') as file:
        endTime = int(file.read())

    trainTime = None
    testTime = None

    with open(errPath, 'r') as file:
        for line in file:
            line = line.strip()
            if (line == ''):
                continue

            match = re.search(r'Train Time -  (\d+.\d+)', line)
            if (match is not None):
                trainTime = float(match.group(1))

            match = re.search(r'Test Time -  (\d+.\d+)', line)
            if (match is not None):
                testTime = float(match.group(1))

    return endTime - startTime, trainTime, testTime

# [{key, value, ...}, ...]
def fetchResults():
    runs = []

    for logPath in glob.glob("%s/**/%s" % (RESULTS_DIR, LOG_FILENAME), recursive = True):
        run = parseLog(logPath)
        if (run is None):
            continue

        runtime, trainTime, testTime = parseTime(logPath)
        if (runtime is None):
            continue

        run['runtime'] = runtime
        run['trainTime'] = trainTime
        run['testTime'] = testTime

        runs.append(run)

    return runs

def main():
    runs = fetchResults()
    if (len(runs) == 0):
        return

    rows = []
    for run in runs:
        rows.append([run.get(key, '') for key in HEADER])

    print("\t".join(HEADER))
    for row in rows:
        print("\t".join(map(str, row)))

def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s" % (executable), file = sys.stderr)
        sys.exit(1)

if (__name__ == '__main__'):
    _load_args(sys.argv)
    main()
