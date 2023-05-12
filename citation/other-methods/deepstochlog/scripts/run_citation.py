import os
import pwd
import sys

import dgl
import torch
from torch.optim import Adam
from time import time

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join("/home", pwd.getpwuid(os.getuid())[0], "deepstochlog"))

DATA_PATH = os.path.join(THIS_DIR, "/home", pwd.getpwuid(os.getuid())[0], "data")

from deepstochlog.network import Network, NetworkStore
from examples.citeseer.citeseer_utils import Classifier, RuleWeights, AccuracyCalculator, create_model_accuracy_calculator
from deepstochlog.utils import set_fixed_seed
from deepstochlog.context import ContextualizedTerm, Context
from deepstochlog.dataset import ContextualizedTermDataset
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.trainer import DeepStochLogTrainer, print_logger
from deepstochlog.term import Term, List


class Dataset(ContextualizedTermDataset):
    def __init__(self, dataset_name, split: str, ids, labels, documents):
        self.dataset_name = dataset_name
        self.ids = ids
        self.labels = labels
        self.is_test = True if split in ("test", "valid") else False
        self.documents = documents
        self.dataset = []

        context = {Term(str(i)): d for i, d in enumerate(self.documents)}

        if dataset_name == "citeseer":
            n_classes = 6
        else:
            n_classes = 7

        for i in range(n_classes):
            context[Term("class" + str(i))] = torch.tensor([i])
        context = Context(context)
        self.queries_for_model = []
        for did in self.ids:
            label = Term("class" + str(self.labels[did]))
            query = ContextualizedTerm(
                context=context,
                term=Term("s", label, List(did)))
            self.dataset.append(query)
            if self.is_test or (dataset_name == "citeseer"):
                query_model = Term("s", Term("_"), List(did))
            else:
                query_model = query.term
            self.queries_for_model.append(query_model)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if type(item) is slice:
            return (self[i] for i in range(*item.indices(len(self))))
        return self.dataset[item]


def run(
    dataset_name="citeseer",
    epochs=100,
    lr=0.01,
    log_freq=50,
    logger=print_logger,
    seed=None,
    verbose=False,
):
    set_fixed_seed(seed)

    # Load train, validation, and test data.
    g_list, _ = dgl.load_graphs(os.path.join(DATA_PATH, '{}/{}/method::simple/dgl-graph.bin'.format("experiment::" + dataset_name, "split::" + str(seed))))
    g = g_list[0]

    # get node feature.
    documents = g.ndata['feat'].to(torch.float)

    # get labels
    labels = g.ndata['label'].numpy()

    # get train, validation, test split.
    train_ids = torch.arange(0, g.num_nodes(), dtype=torch.int64)[g.ndata["train-mask"]].numpy()
    valid_ids = torch.arange(0, g.num_nodes(), dtype=torch.int64)[g.ndata["val-mask"]].numpy()
    test_ids = torch.arange(0, g.num_nodes(), dtype=torch.int64)[g.ndata["test-mask"]].numpy()

    citations = []
    for eid in range(g.num_edges()):
        a, b = g.find_edges(eid)
        a, b = a.numpy().tolist()[0], b.numpy().tolist()[0],
        citations.append("cite(%d, %d)." % (a, b))
    citations = "\n".join(citations)

    train_dataset = Dataset(dataset_name=dataset_name, split="train", documents=documents, labels=labels, ids=train_ids)
    valid_dataset = Dataset(dataset_name=dataset_name, split="valid", documents=documents, labels=labels, ids=valid_ids)
    test_dataset = Dataset(dataset_name=dataset_name, split="test", documents=documents, labels=labels, ids=test_ids)

    queries_for_model = train_dataset.queries_for_model + valid_dataset.queries_for_model + test_dataset.queries_for_model

    # Load the MNIST model, and Adam optimiser
    classifier = Classifier(input_size=len(train_dataset.documents[0]))

    if dataset_name == "citeseer":
        n_classes = 6
    else:
        n_classes = 7

    rule_weights = RuleWeights(num_rules=2, num_classes=n_classes)
    classifier_network = Network(
        "classifier",
        classifier,
        index_list=[Term("class"+str(i)) for i in range(n_classes)],
    )
    rule_weight = Network(
        "rule_weight",
        rule_weights,
        index_list=[Term(str("neural")), Term(str("cite"))],
    )
    networks = NetworkStore(classifier_network, rule_weight)

    device = torch.device("cpu")

    proving_start = time()
    model = DeepStochLogModel.from_file(
        file_location=os.path.join(
            THIS_DIR, "examples/{}/with_rule_weights/{}_ruleweights.pl".format(dataset_name, dataset_name)
        ),
        query=queries_for_model,
        networks=networks,
        device=device,
        prolog_facts=citations,
        normalization=DeepStochLogModel.FULL_NORM
    )
    optimizer = Adam(model.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()
    proving_time = time() - proving_start

    if verbose:
        logger.print("\nProving the program took {:.2f} seconds".format(proving_time))

    # Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))

    # Create test functions
    calculate_model_accuracy = AccuracyCalculator(model=model,
                                                  valid=valid_dataset,
                                                  test=test_dataset,
                                                  start_time=time(),
                                                  after_epoch=0)

    # Train the DeepStochLog model
    trainer = DeepStochLogTrainer(
        log_freq=log_freq,
        accuracy_tester=(calculate_model_accuracy.header, calculate_model_accuracy),
        logger=logger,
        print_time=verbose,
    )
    trainer.train(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        epochs=epochs,
    )

    # Inference
    inference_header, accuracy_calculator = create_model_accuracy_calculator(model, test_dataset, time())
    logger.print(inference_header)
    logger.print(accuracy_calculator())

    return None


def _load_args(args):
    executable = args.pop(0)
    if (len(args) != 2 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args})):
        print("USAGE: python3 {} <'citeseer' or 'cora'> <split>".format(executable), file=sys.stderr)
        sys.exit(1)
    return args[0], int(args[1])


if __name__ == "__main__":
    dataset_name, seed = _load_args(sys.argv)

    if dataset_name == "citeseer":
        epochs = 1000
    else:
        epochs = 200

    run(
        dataset_name=dataset_name,
        epochs=epochs,
        seed=seed,
        log_freq=1,
        verbose=True,
    )
