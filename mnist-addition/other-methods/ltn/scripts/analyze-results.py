#!/usr/bin/env python3

'''
Analyze the results.
The input to this script should be the output from parse-results.py, ex:
```
./scripts/parse-results.py > results.txt
./scripts/analyze-results.py BASE results.txt
```
'''

import math
import os
import sqlite3
import sys

# Get all results with an actual value (e.g. ignore incomplete runs).
BASE_QUERY = '''
    SELECT *
    FROM Stats
    WHERE testAccuracy IS NOT NULL
    ORDER BY
        experiment,
        problem,
        method,
        size,
        overlap,
        split
'''

# Aggregate over splits.
AGGREGATE_QUERY = '''
    SELECT
        S.experiment,
        S.problem,
        S.method,
        S.size,
        S.overlap,
        COUNT(S.split) AS numSplits,
        AVG(S.testAccuracy) AS testAccuracy_mean,
        STDEV(S.testAccuracy) AS testAccuracy_std,
        AVG(S.runtime) AS runtime_mean,
        STDEV(S.runtime) AS runtime_std,
        AVG(S.trainTime) AS trainTime_mean,
        STDEV(S.trainTime) AS trainTime_std,
        AVG(S.testTime) AS testTime_mean,
        STDEV(S.testTime) AS testTime_std
    FROM Stats S
    GROUP BY
        S.experiment,
        S.problem,
        S.method,
        S.size,
        S.overlap
    ORDER BY
        S.experiment,
        S.problem,
        S.method,
        S.size,
        S.overlap
'''

# Aggregate over splits, but only keep the top 10.
TOP_AGGREGATE_QUERY = '''
    SELECT
        S.experiment,
        S.problem,
        S.method,
        S.size,
        S.overlap,
        COUNT(S.split) AS numSplits,
        AVG(S.testAccuracy) AS testAccuracy_mean,
        STDEV(S.testAccuracy) AS testAccuracy_std,
        AVG(S.runtime) AS runtime_mean,
        STDEV(S.runtime) AS runtime_std,
        AVG(S.trainTime) AS trainTime_mean,
        STDEV(S.trainTime) AS trainTime_std,
        AVG(S.testTime) AS testTime_mean,
        STDEV(S.testTime) AS testTime_std
    FROM
        (
            SELECT
                ROW_NUMBER() OVER SplitWindow AS splitRank,
                *
            FROM Stats
            WINDOW SplitWindow AS (
                PARTITION BY
                    experiment,
                    problem,
                    method,
                    size,
                    overlap
                ORDER BY testAccuracy DESC
            )
        ) S
    WHERE S.splitRank <= 10
    GROUP BY
        S.experiment,
        S.problem,
        S.method,
        S.size,
        S.overlap
    ORDER BY
        S.experiment,
        S.problem,
        S.method,
        S.size,
        S.overlap
'''

BOOL_COLUMNS = {
}

INT_COLUMNS = {
    'size',
    'split',
    'epochs',
    'runtime',
}

FLOAT_COLUMNS = {
    'overlap',
    'trainTime',
    'testTime',
    'trainLoss',
    'trainAccuracy',
    'testLoss',
    'testAccuracy',
}

# {key: (query, description), ...}
RUN_MODES = {
    'BASE': (
        BASE_QUERY,
        'Just get the results with no additional processing.',
    ),
    'AGGREGATE': (
        AGGREGATE_QUERY,
        'Aggregate over all splits.',
    ),
    'TOP_AGGREGATE': (
        TOP_AGGREGATE_QUERY,
        'Aggregate over the best 10 splits.',
    ),
}

# ([header, ...], [[value, ...], ...])
def fetchResults(path):
    rows = []
    header = None

    with open(path, 'r') as file:
        for line in file:
            line = line.strip("\n ")
            if (line == ''):
                continue

            row = line.split("\t")

            # Get the header first.
            if (header is None):
                header = row
                continue

            assert(len(header) == len(row))

            for i in range(len(row)):
                if (row[i] == ''):
                    row[i] = None
                elif (header[i] in BOOL_COLUMNS):
                    row[i] = (row[i].upper() == 'TRUE')
                elif (header[i] in INT_COLUMNS):
                    row[i] = int(row[i])
                elif (header[i] in FLOAT_COLUMNS):
                    row[i] = float(row[i])

            rows.append(row)

    return header, rows

# Standard deviation UDF for sqlite3.
# Taken from: https://www.alexforencich.com/wiki/en/scripts/python/stdev
class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k-2))

def main(mode, resultsPath):
    columns, data = fetchResults(resultsPath)
    if (len(data) == 0):
        return

    quotedColumns = ["'%s'" % column for column in columns]

    columnDefs = []
    for i in range(len(columns)):
        column = columns[i]
        quotedColumn = quotedColumns[i]

        if (column in BOOL_COLUMNS):
            columnDefs.append("%s INTEGER" % (quotedColumn))
        elif (column in INT_COLUMNS):
            columnDefs.append("%s INTEGER" % (quotedColumn))
        elif (column in FLOAT_COLUMNS):
            columnDefs.append("%s FLOAT" % (quotedColumn))
        else:
            columnDefs.append("%s TEXT" % (quotedColumn))

    connection = sqlite3.connect(":memory:")
    connection.create_aggregate("STDEV", 1, StdevFunc)

    connection.execute("CREATE TABLE Stats(%s)" % (', '.join(columnDefs)))

    connection.executemany("INSERT INTO Stats(%s) VALUES (%s)" % (', '.join(columns), ', '.join(['?'] * len(columns))), data)

    query = RUN_MODES[mode][0]
    rows = connection.execute(query)

    print("\t".join([column[0] for column in rows.description]))
    for row in rows:
        print("\t".join(map(str, row)))

    connection.close()

def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 2 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 %s <results path> <mode>" % (executable), file = sys.stderr)
        print("modes:", file = sys.stderr)
        for (key, (query, description)) in RUN_MODES.items():
            print("    %s - %s" % (key, description), file = sys.stderr)
        sys.exit(1)

    resultsPath = args.pop(0)
    if (not os.path.isfile(resultsPath)):
        raise ValueError("Can't find the specified results path: " + resultsPath)

    mode = args.pop(0).upper()
    if (mode not in RUN_MODES):
        raise ValueError("Unknown mode: '%s'." % (mode))

    return mode, resultsPath

if (__name__ == '__main__'):
    main(*_load_args(sys.argv))
