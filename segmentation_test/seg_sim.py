def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated
    except ImportError:
        pass


tensorflow_shutup()


import os
import math
import random
import sys

import server

sys.path.append('/home/ram264/Federated/Utils/')
from data_loader import demo_train_val_test
from seg_models import build_unet
import csv
from time import time
import flwr as fl
import tensorflow as tf
import numpy as np
import argparse
from typing import List, Tuple, cast, Union, Dict, Optional, Callable
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

parser = argparse.ArgumentParser(
    description='Run Federated Experiment with inputs: # of clients ( --clients ) , Aggregation strategy ( --strat ), Data Distribution (--dist) ')
parser.add_argument('--clients', type=int)
parser.add_argument('--strat', type=str)
parser.add_argument('--dist',  type=str)

argparse = parser.parse_args()

NUM_CLIENTS = argparse.clients

strats = {
        'Fed_avg': fl.server.strategy.FedAvg,
        'Fed_avgM': fl.server.strategy.FedAvgM,
        'QFed_avg': fl.server.strategy.QFedAvg,
        'FT_Fed_avg': fl.server.strategy.FaultTolerantFedAvg,
        }

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, train, val) -> None:
        super().__init__()
        self.model = model
        self.train = train
        self.val = val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        hist = self.model.fit(self.train, epochs=2, verbose=2)
        print(f'hist loss: {hist.history["loss"]}')
        return self.model.get_weights(), len(self.train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.val, verbose=2)
        return loss, len(self.val), {"accuracy": acc}


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "dropout": True,
        "learning_rate": 0.00085,
        "batch_size": 32,
        "current_round": server_round,
        "local_epochs": 5,
    }
    return config


def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #print(f'~~~~~~~~~~~~~~~~~~~Client:{cid}~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    train, val, _ = demo_train_val_test()
    #print(f'x:{len(x_train)}  y:{len(y_train)}')
    # Create and return client
    return FlwrClient(model, train, val)


class SaveModelStrategy(strats[argparse.strat]):

    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):

        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"weights/round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

    def evaluate_config(self,rnd: int):
        """Return evaluation configuration dict for each round.
        Perform five local evaluation steps on each client (i.e., use five
        batches) during rounds one to three, then increase to ten local
        evaluation steps.
        """
        val_steps = 5 if rnd < 4 else 10
        return {"val_steps": val_steps}

    def aggregate_evaluate(self,
                           rnd,
                           results,
                           failures
                           ):
        if not results:
            return None
            # Weigh accuracy of each client by number of examples used
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        #preds, labels = results[0].metrics["preds"], results.metrics["labels"]
        # Aggregate and print custom metric
        accuracy_aggregated = sum(accuracies) / sum(examples)
        print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)


def main() -> None:

    model = build_unet()

    strat = SaveModelStrategy(evaluate_fn=server.get_eval_fn(model),
                              fraction_fit=0.2,
                              min_available_clients=NUM_CLIENTS,
                              on_fit_config_fn=fit_config
                              )

    # headers = ['accuracy', 'loss', 'AUC', 'F1(weighted)']
    # csv_file = f'/home/ram264/Federated_Learning/flower/mnist/results/{argparse.dist}/{argparse.strat}_{argparse.clients}_metrics.csv'
    # try:
    #     with open(csv_file, 'w') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=headers)
    #         writer.writeheader()
    # except FileNotFoundError:
    #     pass

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=20),
        client_resources={"num_cpus": 4},
        strategy=strat
    )


if __name__ == "__main__":
    main()
