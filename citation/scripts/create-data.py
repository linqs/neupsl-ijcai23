#!/usr/bin/env python3

"""
Create the data for the citation experiments.
This script requires the deep graph library, dgl, (which in-turn requires torch),
which will not be listed as a general dependency for this project,
since prepared data will already be directly provided.
"""

import json
import os
import random

import dgl
from dgl.nn.pytorch import GraphConv
import numpy as np
import pandas as pd
import torch

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(THIS_DIR, '..', 'data')

DATASET_CITESEER = 'citeseer'
DATASET_CORA = 'cora'

DATASETS = [DATASET_CITESEER, DATASET_CORA]

LABELS = {
    DATASET_CITESEER: list(range(6)),
    DATASET_CORA: list(range(7)),
}

TRAIN_COUNT = {
    DATASET_CITESEER: 165,
    DATASET_CORA: 135,
}

VAL_COUNT = {
    DATASET_CITESEER: 165,
    DATASET_CORA: 135,
}

TEST_COUNT = {
    DATASET_CITESEER: 1000,
    DATASET_CORA: 1000,
}

SPLITS = 5
SGC_NUM_LAYERS = 2


def write_json(results, path):
    with open(path, "w") as outfile:
        json.dump(results, outfile, indent=4)


def evaluate(model, graph, mask, device, n_layers):
    model.eval()

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    dataloader = dgl.dataloading.DataLoader(
        graph, torch.arange(0, graph.num_nodes(), dtype=torch.int64, device=device)[mask], sampler,
        batch_size=16 * 1024, shuffle=False, drop_last=False, device=device)

    with torch.no_grad():
        correct_avg = 0.0
        batch_count = 0
        for input_nodes, output_nodes, mfgs in dataloader:
            inputs = mfgs[0].srcdata['feat']
            predictions = model(mfgs, inputs)
            labels = graph.ndata["label"][output_nodes]
            _, indices = torch.max(predictions, dim=1)
            correct_avg += torch.sum(indices == labels).item() / indices.shape[0]
            batch_count += 1
        correct_avg = correct_avg / batch_count
        return correct_avg


def load_directed_citation_network(dataset_id):
    data_dir = os.path.join(DATA_DIR, dataset_id)

    citations = pd.read_csv(os.path.join(data_dir, "{}.cites".format(DIRECTED_ROOT_NAME_MAP[dataset_id])),
                            sep="\t", header=None, dtype=str)
    features = pd.read_csv(os.path.join(data_dir, "{}.content".format(DIRECTED_ROOT_NAME_MAP[dataset_id])),
                           sep="\t", index_col=0, header=None, dtype=str)
    features.index = features.index.astype(str)

    label_column_name = features.columns[-1]
    labels = features.loc[:, label_column_name]
    features = features.drop(label_column_name, axis=1)

    index_to_int_id_map = labels.reset_index().drop(label_column_name, axis=1).reset_index().set_index(0)
    unique_labels = labels.unique()
    labels_to_int_id_map = dict(zip(unique_labels, np.arange(unique_labels.shape[0])))

    labels.index = index_to_int_id_map.loc[labels.index, "index"].values
    labels = labels.map(labels_to_int_id_map)
    features.index = index_to_int_id_map.loc[features.index, "index"].values
    features = features.astype(int)

    citations.loc[:, 0] = citations.loc[:, 0].map(index_to_int_id_map.to_dict()["index"])
    citations.loc[:, 1] = citations.loc[:, 1].map(index_to_int_id_map.to_dict()["index"])
    citations = citations.dropna(axis=0).astype(int)

    graph = dgl.graph(data=(torch.tensor(citations.loc[:, 0].values), torch.tensor(citations.loc[:, 1].values)))
    graph.ndata["feat"] = torch.tensor(features.values)
    graph.ndata["label"] = torch.tensor(labels.values)

    return graph


def fetch_data(dataset_id, split):
    random.seed(split)
    np.random.seed(split)
    torch.manual_seed(split)
    dgl.seed(split)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    graph = None
    if dataset_id == DATASET_CITESEER:
        graph = dgl.data.CiteseerGraphDataset()[0]
    elif dataset_id == DATASET_CORA:
        graph = dgl.data.CoraGraphDataset()[0]
    elif (dataset_id == DATASET_CITESEER_DIRECTED) or (dataset_id == DATASET_CORA_DIRECTED):
        graph = load_directed_citation_network(dataset_id)
    else:
        raise ValueError("Unknown dataset: '%s'." % (dataset_id,))
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    graph = graph.to(device)

    # Create train, validation, test masks.
    labeled_sample = False
    while not labeled_sample:
        train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        val_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)
        test_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=device)

        permutation = torch.randperm(graph.num_nodes(), device=device)

        train_mask[permutation[:TRAIN_COUNT[dataset_id]]] = True
        val_mask[permutation[TRAIN_COUNT[dataset_id]: TRAIN_COUNT[dataset_id] + VAL_COUNT[dataset_id]]] = True
        test_mask[permutation[TRAIN_COUNT[dataset_id] + VAL_COUNT[dataset_id]:
                              TRAIN_COUNT[dataset_id] + VAL_COUNT[dataset_id] + TEST_COUNT[dataset_id]]] = True

        labeled_sample = (
                (len(LABELS[dataset_id]) == (len(torch.unique(graph.ndata["label"][train_mask.to(torch.bool)]))))
                and (len(LABELS[dataset_id]) == (len(torch.unique(graph.ndata["label"][val_mask.to(torch.bool)]))))
        )

    graph.ndata["train_mask"] = train_mask.to(torch.bool)
    graph.ndata["val_mask"] = val_mask.to(torch.bool)
    graph.ndata["test_mask"] = test_mask.to(torch.bool)
    graph.ndata["latent_mask"] = ~(train_mask.to(torch.bool) | val_mask.to(torch.bool) | test_mask.to(torch.bool))

    # GCN baseline.
    symmetric_graph = dgl.add_reverse_edges(graph)
    symmetric_graph = symmetric_graph.to('cpu')
    symmetric_graph = dgl.to_simple(symmetric_graph)
    symmetric_graph = symmetric_graph.to(device)

    sgconv_layer = dgl.nn.pytorch.conv.SGConv(symmetric_graph.ndata['feat'].shape[1],
                                              symmetric_graph.ndata['feat'].shape[1],
                                              k=SGC_NUM_LAYERS).to(device)
    torch.nn.init.eye_(sgconv_layer.fc.weight)
    smoothed_features = sgconv_layer(symmetric_graph, symmetric_graph.ndata['feat']).detach().cpu().numpy()

    graph = dgl.remove_self_loop(graph)
    data = {}
    for (split_id, indexes) in (
            [('train', graph.nodes()[train_mask]),
             ('valid', graph.nodes()[val_mask]),
             ('test', graph.nodes()[test_mask]),
             ('latent', graph.nodes()[graph.ndata["latent_mask"]])]):
        data[split_id] = {
            'ids': [int(index) for index in indexes],
            'labels': [int(graph.ndata['label'][index]) for index in indexes],
            'features': [graph.ndata['feat'][index].tolist() for index in indexes],
            'smoothed_features': [smoothed_features[index].tolist() for index in indexes],
        }

    all_edges = graph.cpu().edges()
    all_edges = np.transpose(np.array([all_edges[0].numpy(), all_edges[1].numpy()]))

    return data, all_edges, graph.cpu()


def write_file(path, data):
    with open(path, 'w') as file:
        for row in data:
            file.write('\t'.join([str(item) for item in row]) + "\n")


def write_data(dataset_id, graph, papers, train_edges, test_edges, split):
    out_dir = os.path.join(DATA_DIR, dataset_id, str(split))

    if os.path.isdir(out_dir):
        print("Target dir (%s) already exists, skipping generation." % out_dir)
        return

    os.makedirs(out_dir, exist_ok=True)

    all_id_features = []
    all_id_features_gnn = []
    all_id_smoothed_features = []

    for split_id in ['train', 'test', 'valid', 'latent']:
        write_file(os.path.join(out_dir, "ids_%s.txt" % (split_id,)),
                   [[paper_id] for paper_id in papers[split_id]['ids']])
        write_file(os.path.join(out_dir, "labels_%s.txt" % (split_id,)),
                   [[label] for label in papers[split_id]['labels']])

        write_file(os.path.join(out_dir, "features_%s.txt" % (split_id,)),
                   papers[split_id]['features'])
        write_file(
            os.path.join(out_dir, "smoothed_features_%s.txt" % (split_id,)),
            papers[split_id]['smoothed_features'])

        id_features = [
            [papers[split_id]['ids'][i]] + papers[split_id]['features'][i] for i
            in range(len(papers[split_id]['ids']))]
        all_id_features += id_features
        all_id_features_gnn += [[papers[split_id]['ids'][i], papers[split_id]['ids'][i]] for i in range(len(papers[split_id]['ids']))]

        id_smoothed_features = [
            [papers[split_id]['ids'][i]] +
            papers[split_id]['smoothed_features'][
                i]
            for i in range(len(papers[split_id]['ids']))]
        all_id_smoothed_features += id_smoothed_features

        write_file(os.path.join(out_dir, "id_features_%s.txt" % (split_id,)),
                   id_features)
        write_file(
            os.path.join(out_dir, "id_smoothed_features_%s.txt" % (split_id,)),
            id_smoothed_features)

        write_file(os.path.join(out_dir, "id_labels_%s.txt" % (split_id,)),
                   [[papers[split_id]['ids'][i], papers[split_id]['labels'][i]]
                    for
                    i in range(len(papers[split_id]['ids']))])
        write_file(os.path.join(out_dir, "targets_%s.txt" % (split_id,)),
                   [[paper_id, label] for paper_id in papers[split_id]['ids']
                    for
                    label in LABELS[dataset_id]])

        # Fully specify labels.
        full_labels = []
        for i in range(len(papers[split_id]['ids'])):
            paper_id = papers[split_id]['ids'][i]
            true_label = papers[split_id]['labels'][i]

            for label in LABELS[dataset_id]:
                full_labels.append([paper_id, label, int(true_label == label)])

        write_file(os.path.join(out_dir, "id_labels_full_%s.txt" % (split_id,)),
                   full_labels)

    write_file(os.path.join(out_dir, "id_features_gnn.txt"), all_id_features_gnn)
    write_file(os.path.join(out_dir, "id_features_all.txt"), all_id_features)
    write_file(os.path.join(out_dir, "id_smoothed_features_all.txt"),
               all_id_smoothed_features)

    write_file(os.path.join(out_dir, 'labels.txt'),
               [[label] for label in LABELS[dataset_id]])
    write_file(os.path.join(out_dir, 'train_edges.txt'), train_edges)
    write_file(os.path.join(out_dir, 'test_edges.txt'), test_edges)

    dgl.save_graphs(os.path.join(out_dir, 'dgl_graph.bin'), [graph])


def main():
    for dataset_id in DATASETS:
        for split in range(SPLITS):

            out_dir = os.path.join(DATA_DIR, dataset_id, str(split))
            if os.path.isdir(out_dir):
                print("Target dir (%s) already exists, skipping generation." % out_dir)
                continue

            data, edges, graph = fetch_data(dataset_id, split)
            write_data(dataset_id, graph, data, edges, edges, split)

if __name__ == '__main__':
    main()
