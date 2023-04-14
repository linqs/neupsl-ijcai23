"""
Converts IJCAI 2023 data format required for NeuPSL in the psl-fork to the
format required for current NeuPSL implementation in PSL main.
"""
import importlib
import os
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

CONVERTED_DATA_DIR = 'converted-data'


def create_entity_data_map(data_dir):
    eval_features = util.load_psl_file(os.path.join(data_dir, 'eval', 'neuralclassifier_features.txt'))
    eval_labels = util.load_psl_file(os.path.join(data_dir, 'eval', 'predicted_number_truth.txt'))
    learn_features = util.load_psl_file(os.path.join(data_dir, 'learn', 'neuralclassifier_features.txt'))
    learn_labels = util.load_psl_file(os.path.join(data_dir, 'learn', 'predicted_number_truth.txt'))

    entities = set()
    entity_data_map = []
    for features, label in zip(eval_features, eval_labels):
        entity_data_map.append(features + [label[1]])
        if features[0] in entities:
            print("Duplicate entity '%s' in eval data." % (features[0],))
        entities.add(features[0])

    for features, label in zip(learn_features, learn_labels):
        entity_data_map.append(features + [label[1]])
        if features[0] in entities:
            print("Duplicate entity '%s' in learn data." % (features[0],))
        entities.add(features[0])

    return entity_data_map


def main(input_dir):
    data_dir = os.path.join(THIS_DIR, '..', 'data', input_dir)
    if not os.path.exists(data_dir):
        print("ERROR: Data directory '%s' does not exist." % (data_dir,), file = sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.join(data_dir, CONVERTED_DATA_DIR), exist_ok=True)

    entity_data_map = create_entity_data_map(data_dir)
    util.write_psl_file(os.path.join(data_dir, CONVERTED_DATA_DIR, 'entity-data-map.txt'), entity_data_map)

    return 0


def _load_args(args):
    executable = args.pop(0)
    if len(args) != 1 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 %s /path/within/data/directory" % (executable,), file = sys.stderr)
        sys.exit(1)

    return args.pop(0)


if __name__ == '__main__':
    main(_load_args(sys.argv))