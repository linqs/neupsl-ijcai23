#!/usr/bin/env python3

"""
Create the data for the citation experiments.
This script requires the deep graph library, dgl, (which in-turn requires torch),
which will not be listed as a general dependency for this project,
since prepared data will already be directly provided.
"""
import importlib
import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import dgl
import numpy
import torch

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..', 'scripts'))
util = importlib.import_module("util")

DATASET_CITESEER = 'citeseer'
DATASET_CORA = 'cora'
DATASETS = [DATASET_CITESEER, DATASET_CORA]

METHOD_SIMPLE = 'simple'
METHOD_SMOOTHED = 'smoothed'
METHODS = [METHOD_SIMPLE, METHOD_SMOOTHED]

DATASET_CONFIG = {
    DATASET_CITESEER: {
        "name": DATASET_CITESEER,
        "class-size": 6,
        "train-size": 165,
        "valid-size": 165,
        "test-size": 1000,
        "num-splits": 5,
        "num-sgc-layers": 2,
    },
    DATASET_CORA: {
        "name": DATASET_CORA,
        "class-size": 7,
        "train-size": 135,
        "valid-size": 135,
        "test-size": 1000,
        "num-splits": 5,
        "num-sgc-layers": 2,
    },
}

CONFIG_FILENAME = "config.json"


def _generate_partitions(graph, device, class_size, train_count, test_count, valid_count):
    """
    Generate train, test, and valid partition masks. Guarantee at least one node per class for each partition.
    """
    found_sample = False
    while not found_sample:
        graph.ndata["train-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["test-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["valid-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        graph.ndata["latent-mask"] = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)

        permutation = torch.randperm(graph.num_nodes(), device=device)

        graph.ndata["train-mask"][permutation[:train_count]] = True
        graph.ndata["test-mask"][permutation[train_count:train_count + test_count]] = True
        graph.ndata["valid-mask"][permutation[train_count + test_count:train_count + test_count + valid_count]] = True
        graph.ndata["latent-mask"][permutation[train_count + test_count + valid_count:]] = True

        for mask_name in ["train-mask", "test-mask", "valid-mask"]:
            found_sample = found_sample or len(torch.unique(graph.ndata["label"][graph.ndata[mask_name]])) == class_size

    return graph


def fetch_data(config):
    random.seed(config['seed'])
    numpy.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    dgl.seed(config['seed'])

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    if config['name'] == DATASET_CITESEER:
        graph = dgl.data.CiteseerGraphDataset()[0]
    elif config['name'] == DATASET_CORA:
        graph = dgl.data.CoraGraphDataset()[0]
    else:
        raise ValueError("Unknown dataset: '%s'." % (config['name'],))

    graph = dgl.add_self_loop(dgl.remove_self_loop(graph)).to(device)

    graph = _generate_partitions(graph, device, config['class-size'], config['train-size'],
                                 config['test-size'], config['valid-size'])

    # GCN baseline.
    symmetric_graph = dgl.to_simple(dgl.add_reverse_edges(graph).to('cpu')).to(device)

    sgconv_layer = dgl.nn.pytorch.conv.SGConv(symmetric_graph.ndata['feat'].shape[1],
                                              symmetric_graph.ndata['feat'].shape[1],
                                              k=config['num-sgc-layers']).to(device)
    torch.nn.init.eye_(sgconv_layer.fc.weight)
    smoothed_features = sgconv_layer(symmetric_graph, symmetric_graph.ndata['feat']).detach().cpu().numpy()

    graph = dgl.remove_self_loop(graph)
    data = {}
    for (partition, indexes) in (
            [('train', graph.nodes()[graph.ndata["train-mask"]]),
             ('test', graph.nodes()[graph.ndata["test-mask"]]),
             ('valid', graph.nodes()[graph.ndata["valid-mask"]]),
             ('latent', graph.nodes()[graph.ndata["latent-mask"]])]):
        data[partition] = {
            'entity-ids': [int(index) for index in indexes],
            'labels': [int(graph.ndata['label'][index]) for index in indexes],
            'features-simple': [graph.ndata['feat'][index].tolist() for index in indexes],
            'features-smoothed': [smoothed_features[index].tolist() for index in indexes],
        }

    edges = graph.cpu().edges()
    edges = numpy.transpose(numpy.array([edges[0].numpy(), edges[1].numpy()]))

    return data, edges, graph


def write_data(config, out_dir, graph, data, edges):
    entity_data_map = []
    deep_data = {}

    for key in data:
        category_targets = []
        category_truth = []
        deep_data[key] = {}
        for entity_index in range(len(data[key]['entity-ids'])):
            deep_data[key]['entity-ids'] = data[key]['entity-ids']
            deep_data[key]['labels'] = data[key]['labels']
            entity = data[key]['entity-ids'][entity_index]

            if config['method'] == METHOD_SIMPLE:
                deep_data[key]['features'] = data[key]['features-simple']
                entity_data_map.append([entity] + data[key]['features-simple'][entity_index] + [data[key]['labels'][entity_index]])
            else:
                deep_data[key]['features'] = data[key]['features-smoothed']
                entity_data_map.append([entity] + data[key]['features-smoothed'][entity_index] + [data[key]['labels'][entity_index]])

            for label_index in range(config['class-size']):
                label = "0" if label_index != data[key]['labels'][entity_index] else "1"

                category_targets.append([entity, str(label_index)])
                category_truth.append([entity, str(label_index), label])

        util.write_psl_file(os.path.join(out_dir, "category-target-%s.txt" % key), category_targets)
        util.write_psl_file(os.path.join(out_dir, "category-truth-%s.txt" % key), category_truth)

    util.write_psl_file(os.path.join(out_dir, "edges.txt"), edges)

    util.write_psl_file(os.path.join(out_dir, "entity-data-map.txt"), entity_data_map)
    util.write_json_file(os.path.join(out_dir, "deep-data.json"), deep_data, indent=None)

    dgl.save_graphs(os.path.join(out_dir, 'dgl-graph.bin'), [graph])

    util.write_json_file(os.path.join(out_dir, CONFIG_FILENAME), config)


def main():
    for dataset_id in DATASETS:
        config = DATASET_CONFIG[dataset_id]
        for split in range(config['num-splits']):
            config['seed'] = split
            split_dir = os.path.join(THIS_DIR, '..', 'data', 'experiment::' + dataset_id, 'split::' + str(split))

            data, edges, graph = fetch_data(config)
            for method in METHODS:
                config['method'] = method
                out_dir = os.path.join(split_dir, 'method::' + method)
                os.makedirs(out_dir, exist_ok=True)

                if os.path.isfile(os.path.join(out_dir, CONFIG_FILENAME)):
                    print("Data already exists for %s. Skipping generation." % out_dir)
                    continue

                write_data(config, out_dir, graph, data, edges)

if __name__ == '__main__':
    main()
