#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
readonly RESULTS_DIR="${THIS_DIR}/../results"

function run_psl() {
    local results_dir=$1
    local cli_dir=$2

    mkdir -p "${results_dir}"

    local log_out_path="${results_dir}/out.txt"
    local log_err_path="${results_dir}/out.err"
    local time_path="${results_dir}/time.txt"

    if [[ -e "${log_out_path}" ]]; then
        echo "Output file already exists, skipping: ${outPath}"
        return 0
    fi

    pushd . > /dev/null
        cd "${cli_dir}"

        # Run PSL.
        time ./run.sh > "${log_out_path}" 2> "${log_err_path}"

        # Copy any artifacts into the output directory.
        mv inferred-predicates "${results_dir}/"
        mv *.json "${results_dir}/"
    popd > /dev/null
}

function run() {
    local experiment=$1

    if [[ ! -d "${THIS_DIR}/../${experiment}" ]]; then
        echo "Experiment does not exist: ${experiment}"
        exit 1
    fi

    local cli_dir="${THIS_DIR}/../${experiment}/cli"
    local data_dir="${THIS_DIR}/../${experiment}/data"
    local neupsl_models_dir="${cli_dir}/neupsl-models"

    rm ${cli_dir}/*.jar 2> /dev/null
    rm ${cli_dir}/*.json 2> /dev/null
    rm ${cli_dir}/inferred-predicates 2> /dev/null

    for model in $(ls ${neupsl_models_dir}); do
        if [[ ! ${model} == *.json ]]; then
            continue
        fi

        local model_name=$(echo ${model} | sed "s/.json//")

        local name_identifier="entity-data-map.txt"
        if [[ ${experiment} == "vspc" ]]; then
          name_identifier="entity-data-map-test.txt"
        fi

        for options_path in $(find "${data_dir}/${model_name}" -name ${name_identifier} | sort) ; do
            cp ${neupsl_models_dir}/${model} ${cli_dir}/${experiment}.json

            local file_identifier=${name_identifier}
            if [[ ${experiment} == "vspc" ]]; then
              if [[ ${options_path} == *"/learn/entity-data-map"* || ${options_path} == *"/test/entity-data-map-train"* ]]; then
                echo "Skipping '${options_path}'."
                continue
              fi
              options_path=$(dirname $(dirname "${options_path}"))/${file_identifier}
              file_identifier="eval/entity-data-map-test.txt"
            fi

            local original_param_path=$(grep ${file_identifier} "${cli_dir}/${experiment}.json" | sed "s#^.*data/${model_name}/\(.*\)/${file_identifier}.*\$#\1#")
            local current_param_path=$(dirname "${options_path}" | sed "s#^.*data/${model_name}/##")

            # Change the .data files to use the current settings.
            sed -i'' -e "s#${original_param_path}#${current_param_path}#" "${cli_dir}/${experiment}.json"
            local results_dir="${RESULTS_DIR}/${experiment}/method::neupsl/${model_name}/${current_param_path}"

            echo "Running '${results_dir}'."
            run_psl "${results_dir}" "${cli_dir}"
        done
    done

    rm ${cli_dir}/*.jar 2> /dev/null
    rm ${cli_dir}/*.json 2> /dev/null
    rm ${cli_dir}/inferred-predicates 2> /dev/null
}

function main() {
    if [[ $# -ne 1 ]]; then
        echo "USAGE: $0 <setting>"
        exit 1
    fi

    trap exit SIGINT

    run $1
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"
