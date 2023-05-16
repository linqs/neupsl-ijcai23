#!/bin/bash

# Create all the data necessary for experiments.
# This involves repeatedly calling the generator.py script.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
readonly SETUP_SCRIPT="${THIS_DIR}/generator.py"

readonly SUFFIX=experiment
readonly NUM_SPLITS='10'

readonly DIMENSIONS='4'
readonly NUM_PUZZLES='002 004 008 016 032 064'
readonly NEURAL_LEARNING_RATES='1e-3'
readonly OVERLAP_PERCENTS='0.00 0.50 0.75'
readonly TRAIN_PERCENTS='0.50'

function main() {
	set -e
	trap exit SIGINT

    for dimension in ${DIMENSIONS} ; do
        for numPuzzles in ${NUM_PUZZLES} ; do
            for neuralLearningRate in ${NEURAL_LEARNING_RATES} ; do
                for overlapPercent in ${OVERLAP_PERCENTS} ; do
                    for trainPercent in ${TRAIN_PERCENTS} ; do
                        "${SETUP_SCRIPT}" \
                            --dimension "${dimension}" \
                            --neural-learning-rate "${neuralLearningRate}" \
                            --num-puzzles "${numPuzzles}" \
                            --overlap-percent "${overlapPercent}" \
                            --splits ${NUM_SPLITS} \
                            --suffix "${SUFFIX}" \
                            --train-percent "${trainPercent}"
                    done
                done
            done
        done
    done
}

main "$@"
