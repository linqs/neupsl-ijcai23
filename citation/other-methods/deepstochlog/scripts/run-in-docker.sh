#!/bin/bash

readonly ROOT_DIR="/home/$(whoami)"
readonly RESULTS_DIR="${ROOT_DIR}/results"
readonly DATA_DIR="${ROOT_DIR}/data"

readonly OUT_FILENAME="out.txt"
readonly ERR_FILENAME="out.err"

function main() {
    if [[ $# -ne 0 ]]; then
        echo "USAGE: $0"
        exit 1
    fi

    trap exit SIGINT

    cp /home/$(whoami)/scripts/run_citation.py /home/$(whoami)/deepstochlog/run_citation.py
    cd deepstochlog

    for options_path in $(find "${DATA_DIR}" -name entity-data-map.txt | sort) ; do
      if [[ ${options_path} == *"method::smoothed"* ]]; then
        continue
      fi

      local param_path=$(dirname "${options_path}" | sed "s#^.*data/\(.*\)/method.*\$#\1#")
      echo "Running ${param_path}"

      if [ -f ${RESULTS_DIR}/${param_path}/${OUT_FILENAME} ]; then
        echo "Results found for ${RESULTS_DIR}/${param_path}. Skipping..."
        continue
      fi

      python3 run_citation.py ${param_path} > ${RESULTS_DIR}/${param_path}/${OUT_FILENAME} 2> ${RESULTS_DIR}/${param_path}/${ERR_FILENAME}
    done
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
