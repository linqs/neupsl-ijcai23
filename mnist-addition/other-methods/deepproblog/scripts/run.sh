#!/bin/bash

# Run the full experiment.

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly SCRIPTS_DIR="${THIS_DIR}"
readonly RESULTS_DIR="${THIS_DIR}/../results"
readonly DATA_DIR="${THIS_DIR}/../../../data"

readonly BUILD_SCRIPT="${SCRIPTS_DIR}/setup.sh"
readonly IN_DOCKER_RUN_SCRIPT="/home/${USER}/scripts/run-in-docker.sh"

readonly IMAGE_NAME='deepproblog-mnist'

function main() {
    if [[ $# -ne 0 ]]; then
        echo "USAGE: $0"
        exit 1
    fi

    trap exit SIGINT
    mkdir -p "${RESULTS_DIR}"

    "${BUILD_SCRIPT}"

    docker run --rm -it -v "${SCRIPTS_DIR}:/home/${USER}/scripts" -v "${RESULTS_DIR}:/home/${USER}/results" -v "${DATA_DIR}:/home/${USER}/psl_data/" "${IMAGE_NAME}" "${IN_DOCKER_RUN_SCRIPT}"
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
