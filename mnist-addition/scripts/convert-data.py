"""
Converts IJCAI 2023 data format required for NeuPSL in the psl-fork to the
format required for current NeuPSL implementation in PSL main.
"""
import importlib
import os
import shutil
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

CONVERTED_DATA_DIR = 'converted-data'
TOTAL_TRAIN_SIZE = 60000
MAX_IMAGE_SUM_MNIST_1 = 18


def update_psl_data_mnist_1(data_dir, out_dir):
    eval_image_targets = util.load_psl_file(os.path.join(data_dir, 'eval', 'predicted_number_targets.txt'))
    eval_image_sum_targets = util.load_psl_file(os.path.join(data_dir, 'eval', 'imagesum_targets.txt'))
    eval_image_sum_truth = util.load_psl_file(os.path.join(data_dir, 'eval', 'imagesum_truth.txt'))
    eval_image_sum_block = util.load_psl_file(os.path.join(data_dir, 'eval', 'imagesumtargetblock_obs.txt'))
    learn_image_targets = util.load_psl_file(os.path.join(data_dir, 'learn', 'predicted_number_targets.txt'))
    learn_image_sum_targets = util.load_psl_file(os.path.join(data_dir, 'learn', 'imagesum_targets.txt'))
    learn_image_sum_truth = util.load_psl_file(os.path.join(data_dir, 'learn', 'imagesum_truth.txt'))
    learn_image_sum_block = util.load_psl_file(os.path.join(data_dir, 'learn', 'imagesumtargetblock_obs.txt'))

    new_eval_image_sum_truth = []
    for truth in eval_image_sum_truth:
        for index in range(MAX_IMAGE_SUM_MNIST_1 + 1):
            new_eval_image_sum_truth.append([truth[0], truth[1], index, 0 if index != int(truth[2]) else 1])

    new_learn_image_sum_truth = []
    for truth in learn_image_sum_truth:
        for index in range(MAX_IMAGE_SUM_MNIST_1 + 1):
            new_learn_image_sum_truth.append([truth[0], truth[1], index, 0 if index != int(truth[2]) else 1])

    util.write_psl_file(os.path.join(out_dir, 'image-target-test.txt'), eval_image_targets)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-target-test.txt'), eval_image_sum_targets)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-truth-test.txt'), new_eval_image_sum_truth)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-block-test.txt'), eval_image_sum_block)
    util.write_psl_file(os.path.join(out_dir, 'image-target-train.txt'), learn_image_targets)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-target-train.txt'), learn_image_sum_targets)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-truth-train.txt'), new_learn_image_sum_truth)
    util.write_psl_file(os.path.join(out_dir, 'image-sum-block-train.txt'), learn_image_sum_block)

    config  = {'converted-data': True}
    util.write_json_file(os.path.join(out_dir, 'config.json'), config)


def create_entity_data_map(data_dir):
    eval_features = util.load_psl_file(os.path.join(data_dir, 'eval', 'neuralclassifier_features.txt'))
    eval_labels = util.load_psl_file(os.path.join(data_dir, 'eval', 'predicted_number_truth.txt'))
    learn_features = util.load_psl_file(os.path.join(data_dir, 'learn', 'neuralclassifier_features.txt'))
    learn_labels = util.load_psl_file(os.path.join(data_dir, 'learn', 'predicted_number_truth.txt'))

    entity_data_map = []
    for features, label in zip(eval_features, eval_labels):
        entity_data_map.append(features + [label[1]])

    for features, label in zip(learn_features, learn_labels):
        entity_data_map.append(features + [label[1]])

    return entity_data_map


def convert_mnist_1(old_data_dir, data_dir):
    experiment = 'experiment::mnist-1'
    os.makedirs(os.path.join(data_dir, CONVERTED_DATA_DIR, experiment), exist_ok=True)

    for split in os.listdir(os.path.join(old_data_dir, 'n_digits::1')):
        if not os.path.isdir(os.path.join(old_data_dir, 'n_digits::1', split)):
            if split == 'numbersum_obs.txt':
                shutil.copy(os.path.join(old_data_dir, 'n_digits::1', split), os.path.join(data_dir, CONVERTED_DATA_DIR, experiment, 'number-sum.txt'))
            elif split == 'possibledigits_obs.txt':
                shutil.copy(os.path.join(old_data_dir, 'n_digits::1', split), os.path.join(data_dir, CONVERTED_DATA_DIR, experiment, 'possible-digits.txt'))
            continue

        out_split_dir = os.path.join(data_dir, CONVERTED_DATA_DIR, experiment, 'split::' + split.split('::')[1])
        for train_size in os.listdir(os.path.join(old_data_dir, 'n_digits::1', split)):
            out_train_dir = os.path.join(out_split_dir, 'train_size::' + train_size.split('::')[1])
            for overlap in os.listdir(os.path.join(old_data_dir, 'n_digits::1', split, train_size)):
                out_overlap_dir = os.path.join(out_train_dir, 'overlap::' + overlap.split('::')[1])

                os.makedirs(out_overlap_dir, exist_ok=True)

                update_psl_data_mnist_1(os.path.join(old_data_dir, 'n_digits::1', split, train_size, overlap), out_overlap_dir)
                entity_data_map = create_entity_data_map(os.path.join(old_data_dir, 'n_digits::1', split, train_size, overlap))
                util.write_psl_file(os.path.join(out_overlap_dir, 'entity-data-map.txt'), entity_data_map)


def main(old_data_dir):
    data_dir = os.path.join(THIS_DIR, '..', 'data')

    for experiment in os.listdir(old_data_dir):
        match experiment:
            case 'n_digits::1':
                convert_mnist_1(old_data_dir, data_dir)
            case _:
                print("Conversion for experiment '%s' not implemented." % (experiment,))

    return 0


def _load_args(args):
    executable = args.pop(0)
    if len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 %s /path/to/old/data" % (executable,), file = sys.stderr)
        sys.exit(1)

    return args.pop(0)


if __name__ == '__main__':
    main(_load_args(sys.argv))