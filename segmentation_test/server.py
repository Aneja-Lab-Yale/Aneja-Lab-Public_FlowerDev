"""
U-NET TEST SERVER
"""

import flwr as fl
from typing import Dict, Optional, Tuple
from seg_models import build_unet
from data_loader import demo_train_val_test
import argparse

parser = argparse.ArgumentParser(
    description='Run Federated Experiment with inputs: # of clients ( --clients ) , Aggregation strategy ( --strat ), Data Distribution (--dist) ')
parser.add_argument('--clients', type=int)
parser.add_argument('--strat', type=str)
parser.add_argument('--dist',  type=str)

argparse = parser.parse_args()

NUM_CLIENTS = argparse.clients

weights_path = r'/home/ram264/TF/classification/chexnet/simulation_weights/'
file_type = r'\*h5'

# strats = {
#         'FedProx': FedProx,
#         'Fed_avg': fl.server.strategy.FedAvg,
#         'Fed_avgM': fl.server.strategy.FedAvgM,
#         'QFed_avg': fl.server.strategy.QFedAvg,
#         'FT_Fed_avg': fl.server.strategy.FaultTolerantFedAvg,
#         }

"""
Adapted from https://github.com/adap/flower/blob/main/examples/advanced_tensorflow/client.py
By Daniel J. Beutel and Taner Topal
"""


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = build_unet()
    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,

    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        grpc_max_message_length=1024 * 1024 * 10,
        strategy=strategy,
    )


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself

    _, _, test, _ = demo_train_val_test()

    # The `evaluate` function will be called after every round
    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(test)

        metrics = {"accuracy": float(accuracy), 'loss': float(loss)}

        return loss, metrics

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
