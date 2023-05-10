#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly BASE_DIR="${THIS_DIR}/.."

readonly REPO='https://github.com/linqs/logictensornetworks-neurips22.git'

function main() {
    if [[ $# -ne 0 ]]; then
        echo "USAGE: $0"
        exit 1
    fi

    set -e
    trap exit SIGINT

    cd "${BASE_DIR}"
    git clone "${REPO}"
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
