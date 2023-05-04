#!/usr/bin/env python3

import json
import os

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
MNIST_CLI_DIR = os.path.join(THIS_DIR, "../cli")
RESULTS_BASE_DIR = os.path.join(THIS_DIR, "../results")
PERFORMANCE_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "performance")

SPLITS = ["0", "1", "2", "3", "4"]
TRAIN_SIZES = ["0040", "0060", "0080"]
OVERLAPS = ["0.00", "0.50", "1.00"]

STANDARD_EXPERIMENT_OPTIONS = {
    "inference.normalize": "false",
    "runtime.log.level": "TRACE",
    "gradientdescent.scalestepsize": "false",
    "weightlearning.inference": "DistributedDualBCDInference",
    "runtime.inference.method": "DistributedDualBCDInference",
    "gradientdescent.numsteps": "5000",
    "gradientdescent.runfulliterations": "false",
    "duallcqp.computeperiod": "50",
    "duallcqp.maxiterations": "500",
}

STANDARD_DATASET_OPTIONS = {
    "mnist-addition": {
        "duallcqp.primaldualthreshold": "0.01"
    }
}

INFERENCE_OPTION_RANGES = {
    "duallcqp.regularizationparameter": ["1.0e-2"]
}

FIRST_ORDER_WL_METHODS = ["Energy", "BinaryCrossEntropy"]

FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES = {
    "gradientdescent.stepsize": ["1.0e-2", "1.0e-4"],
    "gradientdescent.negativelogregularization": ["1.0e-1", "1.0e-3"],
    "gradientdescent.negativeentropyregularization": ["0.0"]
}

FIRST_ORDER_WL_METHODS_OPTION_RANGES = {
    "Energy": {
        "runtime.learn.method": ["Energy"],
        "gradientdescent.runfulliterations": ["true"],
    },
    "MeanSquaredError": {
        "runtime.learn.method": ["MeanSquaredError"],
        "minimizer.objectivedifferencetolerance": ["0.1", "0.01"],
        "minimizer.proxruleweight": ["1.0e-1", "1.0e-2"],
        "minimizer.numinternaliterations": ["500"]
    },
    "BinaryCrossEntropy": {
        "runtime.learn.method": ["BinaryCrossEntropy"],
        "minimizer.objectivedifferencetolerance": ["0.1", "0.01"],
        "minimizer.proxruleweight": ["1.0e-1", "1.0e-2"],
        "minimizer.numinternaliterations": ["500"]
    }
}

NEURAL_NETWORK_OPTIONS = {
    "dropout": ["0.0", "0.1"]
}

BEST_HYPERPARAMETERS = {
    "Energy": {
        "0040": {
            "0.00": {
                "runtime.learn.method": "Energy",
                "gradientdescent.runfulliterations": "true",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "Energy",
                "gradientdescent.runfulliterations": "true",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "Energy",
                "gradientdescent.runfulliterations": "true",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0060": {
            "0.00": {
                "runtime.learn.method": "Energy",
                "gradientdescent.runfulliterations": "true",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "Energy",
                "gradientdescent.runfulliterations": "true",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "Energy",
                "gradientdescent.runfulliterations": "true",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0080": {
            "0.00": {
                "runtime.learn.method": "Energy",
                "gradientdescent.runfulliterations": "true",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "Energy",
                "gradientdescent.runfulliterations": "true",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "Energy",
                "gradientdescent.runfulliterations": "true",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        }
    },
    "MeanSquaredError": {
        "0040": {
            "0.00": {
                "runtime.learn.method": "MeanSquaredError",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "MeanSquaredError",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "MeanSquaredError",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-2",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0060": {
            "0.00": {
                "runtime.learn.method": "MeanSquaredError",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "MeanSquaredError",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "MeanSquaredError",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0080": {
            "0.00": {
                "runtime.learn.method": "MeanSquaredError",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "MeanSquaredError",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "MeanSquaredError",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        }
    },
    "BinaryCrossEntropy": {
        "0040": {
            "0.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0060": {
            "0.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        },
        "0080": {
            "0.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "0.50": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            },
            "1.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.objectivedifferencetolerance": "0.01",
                "minimizer.proxruleweight": "1.0e-1",
                "minimizer.numinternaliterations": "500",
                "duallcqp.regularizationparameter": "1.0e-2",
                "gradientdescent.stepsize": "1.0e-3",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0"
            }
        }
    }
}

BEST_NEURAL_NETWORK_HYPERPARAMETERS = {
    "Energy": {
        "0040": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0060": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0080": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        }
    },
    "MeanSquaredError": {
        "0040": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0060": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0080": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        }
    },
    "BinaryCrossEntropy": {
        "0040": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0060": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        },
        "0080": {
            "0.00": {
                "dropout": "0.1"
            },
            "0.50": {
                "dropout": "0.1"
            },
            "1.00": {
                "dropout": "0.1"
            }
        }
    }
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


def set_data_path(dataset_json, split, train_size, overlap):
    dataset_json["predicates"]["NeuralClassifier/2"]["options"]["entity-data-map-path"] = \
        "../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/entity-data-map.txt".format(split, train_size, overlap)
    dataset_json["predicates"]["NeuralClassifier/2"]["options"]["save-path"] = \
        "../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/saved-networks/nesy-trained-tf".format(split, train_size, overlap)
    dataset_json["predicates"]["NeuralClassifier/2"]["targets"]["learn"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-target-train.txt".format(split, train_size, overlap)]
    dataset_json["predicates"]["NeuralClassifier/2"]["targets"]["infer"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-target-test.txt".format(split, train_size, overlap)]

    dataset_json["predicates"]["ImageSum/5"]["targets"]["learn"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-sum-target-train.txt".format(split, train_size, overlap)]
    dataset_json["predicates"]["ImageSum/5"]["targets"]["infer"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-sum-target-test.txt".format(split, train_size, overlap)]
    dataset_json["predicates"]["ImageSum/5"]["truth"]["learn"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-sum-truth-train.txt".format(split, train_size, overlap)]
    dataset_json["predicates"]["ImageSum/5"]["truth"]["infer"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-sum-truth-test.txt".format(split, train_size, overlap)]

    dataset_json["predicates"]["ImageDigitSum/3"]["targets"]["learn"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-digit-sum-target-train.txt".format(split, train_size, overlap)]
    dataset_json["predicates"]["ImageDigitSum/3"]["targets"]["infer"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-digit-sum-target-test.txt".format(split, train_size, overlap)]

    dataset_json["predicates"]["ImageSumBlock/4"]["observations"]["learn"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-sum-block-train.txt".format(split, train_size, overlap)]
    dataset_json["predicates"]["ImageSumBlock/4"]["observations"]["infer"] = \
        ["../data/experiment::mnist-2/split::{}/train-size::{}/overlap::{}/image-sum-block-test.txt".format(split, train_size, overlap)]


def run_first_order_wl_methods_hyperparamter_search():
    base_out_dir = os.path.join(PERFORMANCE_RESULTS_DIR, "mnist-add2", "first_order_wl_methods")
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(MNIST_CLI_DIR, "neupsl-models/experiment::mnist-add2.json")

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    standard_experiment_option_ranges = {**INFERENCE_OPTION_RANGES,
                                         **FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES}

    for method in FIRST_ORDER_WL_METHODS:
        method_out_dir = os.path.join(base_out_dir, method)
        os.makedirs(method_out_dir, exist_ok=True)

        for train_size in ["0080"]:
            for overlap in ["1.00"]:
                for split in ["0"]:
                    split_out_dir = os.path.join(method_out_dir, "split::{}/train-size::{}/overlap::{}".format(split, train_size, overlap))
                    os.makedirs(split_out_dir, exist_ok=True)

                    # Iterate over every combination options values.
                    method_options_dict = {**standard_experiment_option_ranges,
                                           **FIRST_ORDER_WL_METHODS_OPTION_RANGES[method]}
                    for options in enumerate_hyperparameters(method_options_dict):
                        for dropout in NEURAL_NETWORK_OPTIONS["dropout"]:
                            experiment_out_dir = split_out_dir
                            for key, value in sorted(options.items()):
                                experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))
                            experiment_out_dir = os.path.join(experiment_out_dir, "dropout::{}".format(dropout))

                            os.makedirs(experiment_out_dir, exist_ok=True)

                            if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                                print("Skipping experiment: {}.".format(experiment_out_dir))
                                continue

                            dataset_json.update({"options":{**original_options,
                                                            **STANDARD_EXPERIMENT_OPTIONS,
                                                            **STANDARD_DATASET_OPTIONS["mnist-addition"],
                                                            **options,
                                                            "runtime.learn.output.model.path": "./mnist-addition_learned.psl"}})

                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["learning-rate"] = options["gradientdescent.stepsize"]
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["dropout"] = dropout

                            # Set the data path.
                            set_data_path(dataset_json, split, train_size, overlap)

                            # Write the options the json file.
                            with open(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), "w") as file:
                                json.dump(dataset_json, file, indent=4)

                            # Run the experiment.
                            print("Running experiment: {}.".format(experiment_out_dir))
                            exit_code = os.system("cd {} && ./run.sh {} > out.txt 2> out.err".format(MNIST_CLI_DIR, experiment_out_dir))

                            if exit_code != 0:
                                print("Experiment failed: {}.".format(experiment_out_dir))
                                exit()

                            # Save the output and json file.
                            os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.txt"), experiment_out_dir))
                            os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.err"), experiment_out_dir))
                            os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), experiment_out_dir))
                            os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition_learned.psl"), experiment_out_dir))
                            os.system("cp -r {} {}".format(os.path.join(MNIST_CLI_DIR, "inferred-predicates"), experiment_out_dir))

def run_first_order_wl_methods():
    base_out_dir = os.path.join(PERFORMANCE_RESULTS_DIR, "mnist-add2", "first_order_wl_methods")
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(MNIST_CLI_DIR, "neupsl-models/experiment::mnist-add2.json")

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    for method in FIRST_ORDER_WL_METHODS:
        method_out_dir = os.path.join(base_out_dir, method)
        os.makedirs(method_out_dir, exist_ok=True)

        for train_size in ["0080"]:
            for overlap in ["1.00"]:
                options = BEST_HYPERPARAMETERS[method][train_size][overlap]
                dropout = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["dropout"]

                for split in SPLITS:
                    split_out_dir = os.path.join(method_out_dir, "split::{}/train-size::{}/overlap::{}".format(split, train_size, overlap))
                    os.makedirs(split_out_dir, exist_ok=True)

                    experiment_out_dir = split_out_dir
                    for key, value in sorted(options.items()):
                        experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))
                    experiment_out_dir = os.path.join(experiment_out_dir, "dropout::{}".format(dropout))

                    os.makedirs(experiment_out_dir, exist_ok=True)

                    if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                        print("Skipping experiment: {}.".format(experiment_out_dir))
                        continue

                    dataset_json.update({"options":{**original_options,
                                                    **STANDARD_EXPERIMENT_OPTIONS,
                                                    **STANDARD_DATASET_OPTIONS["mnist-addition"],
                                                    **options,
                                                    "runtime.learn.output.model.path": "./mnist-addition_learned.psl"}})

                    dataset_json["predicates"]["NeuralClassifier/2"]["options"]["learning-rate"] = options["gradientdescent.stepsize"]
                    dataset_json["predicates"]["NeuralClassifier/2"]["options"]["dropout"] = dropout

                    # Set the data path.
                    set_data_path(dataset_json, split, train_size, overlap)

                    # Write the options the json file.
                    with open(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), "w") as file:
                        json.dump(dataset_json, file, indent=4)

                    # Run the experiment.
                    print("Running experiment: {}.".format(experiment_out_dir))
                    exit_code = os.system("cd {} && ./run.sh {} > out.txt 2> out.err".format(MNIST_CLI_DIR, experiment_out_dir))

                    if exit_code != 0:
                        print("Experiment failed: {}.".format(experiment_out_dir))
                        exit()

                    # Save the output and json file.
                    os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.txt"), experiment_out_dir))
                    os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.err"), experiment_out_dir))
                    os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), experiment_out_dir))
                    os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition_learned.psl"), experiment_out_dir))
                    os.system("cp -r {} {}".format(os.path.join(MNIST_CLI_DIR, "inferred-predicates"), experiment_out_dir))

def main():
    run_first_order_wl_methods_hyperparamter_search()
    run_first_order_wl_methods()

if __name__ == '__main__':
    main()
