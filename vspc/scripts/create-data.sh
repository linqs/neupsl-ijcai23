#!/bin/bash

# Create all the data necessary for experiments.
# This involves repeatedly calling the generator.py script.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
readonly SETUP_SCRIPT="${THIS_DIR}/generator.py"

readonly DIMENSIONS='4'
readonly NUM_POSITIVE_TRAIN_PUZZLES='002 004 008'
readonly NUM_POSITIVE_TEST_PUZZLES='050'
readonly NUM_POSITIVE_VALID_PUZZLES='025'
readonly OVERLAP='0.00 1.00 3.00'
readonly NUM_SPLITS='10'

function main() {
	set -e
	trap exit SIGINT

    for dimension in ${DIMENSIONS} ; do
      for num_positive_train_puzzles in ${NUM_POSITIVE_TRAIN_PUZZLES} ; do
        for num_positive_test_puzzles in ${NUM_POSITIVE_TEST_PUZZLES} ; do
          for num_positive_valid_puzzles in ${NUM_POSITIVE_VALID_PUZZLES} ; do
            for overlap in ${OVERLAP} ; do
              "${SETUP_SCRIPT}" \
                  --dimension "${dimension}" \
                  --num-positive-train-puzzles "${num_positive_train_puzzles}" \
                  --num-positive-test-puzzles "${num_positive_test_puzzles}" \
                  --num-positive-valid-puzzles "${num_positive_valid_puzzles}" \
                  --overlap "${overlap}" \
                  --splits ${NUM_SPLITS}
            done
          done
        done
      done
    done
}

main "$@"
