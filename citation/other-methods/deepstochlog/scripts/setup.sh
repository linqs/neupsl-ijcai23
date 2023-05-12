#!/bin/bash

# Setup the experiments (docker image).

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."

readonly IMAGE_NAME='deepstochlog-citation'

function main() {
    if [[ $# -ne 0 ]]; then
        echo "USAGE: $0"
        exit 1
    fi

    trap exit SIGINT

    pushd . > /dev/null
    cd "${BASE_DIR}"
        docker build -t ${IMAGE_NAME} --build-arg USER=$USER --build-arg UID=$(id -u) --build-arg GID=$(id -g) .
    popd > /dev/null
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
