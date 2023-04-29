#!/usr/bin/env python3

import json
import os

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
CLI_DIR = os.path.join(THIS_DIR, "../cli")
RESULTS_BASE_DIR = os.path.join(THIS_DIR, "../results")
PERFORMANCE_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "performance")

DATASETS = ["citeseer-extended"]
MODEL_TYPES = ["smoothed", "simple"]
SPLITS = ["0", "1", "2", "3", "4"]

STANDARD_EXPERIMENT_OPTIONS = {
    "runtime.log.level": "TRACE",
    "gradientdescent.scalestepsize": "false",
    "weightlearning.inference": "DistributedDualBCDInference",
    "runtime.inference.method": "DistributedDualBCDInference",
    "gradientdescent.numsteps": "1000",
    "gradientdescent.runfulliterations": "false",
    "duallcqp.computeperiod": "10",
    "duallcqp.maxiterations": "10000",
}

STANDARD_DATASET_OPTIONS = {
    "citeseer-extended": {
        "duallcqp.primaldualthreshold": "1.0"
    },
    "citeseer": {
        "duallcqp.primaldualthreshold": "0.1"
    },
    "cora": {
        "duallcqp.primaldualthreshold": "0.1"
    }
}

DATAPATH_NAME = {
    "citeseer-extended": "citeseer"
}

INFERENCE_OPTION_RANGES = {
    "duallcqp.regularizationparameter": ["1.0e-2"]
}

FIRST_ORDER_WL_METHODS = ["BinaryCrossEntropy", "Energy"]

FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES = {
    "gradientdescent.stepsize": ["1.0e-3", "1.0e-5"],
    "gradientdescent.negativelogregularization": ["1.0e-1"],
    "gradientdescent.negativeentropyregularization": ["0.0"]
}

FIRST_ORDER_WL_METHODS_OPTION_RANGES = {
    "Energy": {
        "runtime.learn.method": ["Energy"]
    },
    "MeanSquaredError": {
        "runtime.learn.method": ["MeanSquaredError"],
        "minimizer.objectivedifferencetolerance": ["0.1"],
        "minimizer.proxruleweight": ["1.0e-1", "1.0e-2"],
        "minimizer.numinternaliterations": ["250"]
    },
    "BinaryCrossEntropy": {
        "runtime.learn.method": ["BinaryCrossEntropy"],
        "minimizer.objectivedifferencetolerance": ["0.1"],
        "minimizer.proxruleweight": ["1.0e-1", "1.0e-2"],
        "minimizer.numinternaliterations": ["250"]
    }
}

NEURAL_NETWORK_OPTIONS = {
    "learning-rate": ["1.0e-2", "1.0e-4"],
    "dropout": ["0.0", "0.1"]
}


def enumerate_hyperparameters(hyperparameters_dict: dict, current_hyperparameters={}):
    for key in sorted(hyperparameters_dict):
        hyperparameters = []
        for value in hyperparameters_dict[key]:
            next_hyperparameters = current_hyperparameters.copy()
            next_hyperparameters[key] = value

            remaining_hyperparameters = hyperparameters_dict.copy()
            remaining_hyperparameters.pop(key)

            if remaining_hyperparameters:
                hyperparameters = hyperparameters + enumerate_hyperparameters(remaining_hyperparameters, next_hyperparameters)
            else:
                hyperparameters.append(next_hyperparameters)
        return hyperparameters


def set_data_path(dataset_json, split, dataset_name, neural_model_type):
    dataset_json["predicates"]["Neural/2"]["options"]["entity-data-map-path"] = \
        "../data/experiment::{}/split::{}/entity-data-map.txt".format(dataset_name, split)
    dataset_json["predicates"]["Neural/2"]["options"]["load-path"] = \
        "../data/experiment::{}/split::{}/saved-networks/{}/pre-trained-tf".format(dataset_name, split, neural_model_type)
    dataset_json["predicates"]["Neural/2"]["options"]["save-path"] = \
        "../data/experiment::{}/split::{}/saved-networks/{}/nesy-trained-tf".format(dataset_name, split, neural_model_type)
    dataset_json["predicates"]["Neural/2"]["targets"]["learn"] = \
        ["../data/experiment::{}/split::{}/category-target-train.txt".format(dataset_name, split),
         "../data/experiment::{}/split::{}/category-target-test.txt".format(dataset_name, split),
         "../data/experiment::{}/split::{}/category-target-valid.txt".format(dataset_name, split),
         "../data/experiment::{}/split::{}/category-target-latent.txt".format(dataset_name, split)]
    dataset_json["predicates"]["Neural/2"]["targets"]["infer"] = \
        ["../data/experiment::{}/split::{}/category-target-test.txt".format(dataset_name, split),
         "../data/experiment::{}/split::{}/category-target-latent.txt".format(dataset_name, split)]

    dataset_json["predicates"]["Link/2"]["observations"]["learn"] = \
        ["../data/experiment::{}/split::{}/edges.txt".format(dataset_name, split)]
    dataset_json["predicates"]["Link/2"]["observations"]["infer"] = \
        ["../data/experiment::{}/split::{}/edges.txt".format(dataset_name, split)]

    dataset_json["predicates"]["HasCat/2"]["observations"]["learn"] = \
        ["../data/experiment::{}/split::{}/category-truth-obs-train.txt".format(dataset_name, split)]
    dataset_json["predicates"]["HasCat/2"]["observations"]["infer"] = \
        ["../data/experiment::{}/split::{}/category-truth-train.txt".format(dataset_name, split),
         "../data/experiment::{}/split::{}/category-truth-valid.txt".format(dataset_name, split)]
    dataset_json["predicates"]["HasCat/2"]["targets"]["learn"] = \
        ["../data/experiment::{}/split::{}/category-target-unobs-train.txt".format(dataset_name, split),
         "../data/experiment::{}/split::{}/category-target-test.txt".format(dataset_name, split),
         "../data/experiment::{}/split::{}/category-target-valid.txt".format(dataset_name, split),
         "../data/experiment::{}/split::{}/category-target-latent.txt".format(dataset_name, split)]
    dataset_json["predicates"]["HasCat/2"]["targets"]["infer"] = \
        ["../data/experiment::{}/split::{}/category-target-test.txt".format(dataset_name, split),
         "../data/experiment::{}/split::{}/category-target-latent.txt".format(dataset_name, split)]
    dataset_json["predicates"]["HasCat/2"]["truth"]["learn"] = \
        ["../data/experiment::{}/split::{}/category-truth-unobs-train.txt".format(dataset_name, split)]
    dataset_json["predicates"]["HasCat/2"]["truth"]["infer"] = \
        ["../data/experiment::{}/split::{}/category-truth-test.txt".format(dataset_name, split)]


def run_first_order_wl_methods(dataset_name, neural_model_type):
    base_out_dir = os.path.join(PERFORMANCE_RESULTS_DIR, dataset_name, "first_order_wl_methods", neural_model_type)
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(CLI_DIR, "neupsl-models/experiment::{}.json".format(dataset_name))

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]
    original_neural_options = dataset_json["predicates"]["Neural/2"]["options"]

    standard_experiment_option_ranges = {**INFERENCE_OPTION_RANGES,
                                         **FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES}

    for method in FIRST_ORDER_WL_METHODS:
        for split in SPLITS:
            split_out_dir = os.path.join(base_out_dir, "{}/split::{}".format(method, split))
            os.makedirs(split_out_dir, exist_ok=True)

            # Iterate over every combination options values.
            method_options_dict = {**standard_experiment_option_ranges,
                                   **FIRST_ORDER_WL_METHODS_OPTION_RANGES[method]}
            for options in enumerate_hyperparameters(method_options_dict):
                for learning_rate in NEURAL_NETWORK_OPTIONS["learning-rate"]:
                    for dropout in NEURAL_NETWORK_OPTIONS["dropout"]:
                        experiment_out_dir = split_out_dir
                        for key, value in sorted(options.items()):
                            experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))
                        experiment_out_dir = os.path.join(experiment_out_dir, "learning-rate::{}".format(learning_rate))
                        experiment_out_dir = os.path.join(experiment_out_dir, "dropout::{}".format(dropout))

                        os.makedirs(experiment_out_dir, exist_ok=True)

                        if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                            print("Skipping experiment: {}.".format(experiment_out_dir))
                            continue

                        dataset_json.update({"options":{**original_options,
                                                        **STANDARD_DATASET_OPTIONS[dataset_name],
                                                        **STANDARD_EXPERIMENT_OPTIONS,
                                                        **options,
                                                        "runtime.learn.output.model.path": "./citation_learned.psl"}})

                        dataset_json["predicates"]["Neural/2"]["options"]["learning-rate"] = learning_rate
                        dataset_json["predicates"]["Neural/2"]["options"]["dropout"] = dropout

                        # Set the data path.
                        set_data_path(dataset_json, split, DATAPATH_NAME[dataset_name], neural_model_type)

                        # Write the options the json file.
                        with open(os.path.join(CLI_DIR, "citation.json"), "w") as file:
                            json.dump(dataset_json, file, indent=4)

                        # Run the experiment.
                        print("Running experiment: {}.".format(experiment_out_dir))
                        exit_code = os.system("cd {} && ./run.sh {} > out.txt 2> out.err".format(CLI_DIR, experiment_out_dir))

                        if exit_code != 0:
                            print("Experiment failed: {}.".format(experiment_out_dir))
                            exit()

                        # Save the output and json file.
                        os.system("mv {} {}".format(os.path.join(CLI_DIR, "out.txt"), experiment_out_dir))
                        os.system("mv {} {}".format(os.path.join(CLI_DIR, "out.err"), experiment_out_dir))
                        os.system("cp {} {}".format(os.path.join(CLI_DIR, "citation.json"), experiment_out_dir))
                        os.system("cp {} {}".format(os.path.join(CLI_DIR, "citation_learned.psl"), experiment_out_dir))
                        os.system("cp -r {} {}".format(os.path.join(CLI_DIR, "inferred-predicates"), experiment_out_dir))


def main():
    for dataset in DATASETS:
        for neural_model_type in MODEL_TYPES:
            run_first_order_wl_methods(dataset, neural_model_type)


if __name__ == '__main__':
    main()
