#!/usr/bin/bash

# Run LTN MNIST experiments as seen in their paper.
# This means we will run 15 times and take the top 10.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_OUT_DIR="${THIS_DIR}/../results"

readonly BASE_REPO_DIR="${THIS_DIR}/../logictensornetworks-neurips22/examples/mnist"
readonly MNIST_1_BASELINE_SCRIPT='single_digits_addition_baseline.py'
readonly MNIST_1_LTN_SCRIPT='single_digits_addition_ltn.py'
readonly MNIST_2_BASELINE_SCRIPT='multi_digits_addition_baseline.py'
readonly MNIST_2_LTN_SCRIPT='multi_digits_addition_ltn.py'

readonly SPLITS=2
readonly EPOCHS=150
readonly TEST_SIZE=1000

readonly OVERLAPS='0.0 0.5 1.0'

readonly MNIST_1_SIZES='20 30 40'
readonly MNIST_2_SIZES='10 15 20'

function run_ltn() {
    local outDir=$1
    local options=$2
    local script=$3

    mkdir -p "${outDir}"

    local outPath="${outDir}/out.txt"
    local errPath="${outDir}/out.err"
    local timeStartPath="${outDir}/time-start.txt"
    local timeEndPath="${outDir}/time-end.txt"
    local resultsPath="${outDir}/results.csv"

    if [[ -e "${timeEndPath}" ]]; then
        echo "Output file already exists, skipping: ${timeEndPath}"
        return 0
    fi

    pushd . > /dev/null
        cd "${BASE_REPO_DIR}"

        date +%s > "${timeStartPath}"

        python3 "${script}" ${options} --csv-path "${resultsPath}" > "${outPath}" 2> "${errPath}"

        date +%s > "${timeEndPath}"
    popd > /dev/null
}

function run_mnist() {
    local mnistType=$1
    local method=$2
    local sizes=$3
    local script=$4

    for split in $(seq 0 ${SPLITS}) ; do
        for size in ${sizes} ; do
            for overlap in ${OVERLAPS} ; do
                local outDir="${BASE_OUT_DIR}/method::${method}/experiment::mnist-${mnistType}/split::${split}/train-size::${size}/overlap::${overlap}/"
                local options="--epochs ${EPOCHS} --n-examples-test ${TEST_SIZE} --n-examples-train ${size} --overlap ${overlap}"

                echo "Running ${outDir}."
                run_ltn "${outDir}" "${options}" "${script}"
            done
        done
    done
}

function main() {
    if [[ $# -ne 0 ]]; then
        echo "USAGE: $0"
        exit 1
    fi

    trap exit SIGINT

    run_mnist 1 'baseline' "${MNIST_1_SIZES}" "${MNIST_1_BASELINE_SCRIPT}"
    run_mnist 1 'ltn' "${MNIST_1_SIZES}" "${MNIST_1_LTN_SCRIPT}"

    run_mnist 2 'baseline' "${MNIST_2_SIZES}" "${MNIST_2_BASELINE_SCRIPT}"
    run_mnist 2 'ltn' "${MNIST_2_SIZES}" "${MNIST_2_LTN_SCRIPT}"
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
