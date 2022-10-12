"""
U-NET TEST CLIENT
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from pathlib import Path
import flwr as fl
from seg_models import build_unet
from data_loader import demo_train_val_test

parser = argparse.ArgumentParser(
    description='Run Federated client, Arg: Data Distribution (--dist) ')
parser.add_argument('--dist',  type=str)


"""
Adapted from https://github.com/adap/flower/blob/main/examples/advanced_tensorflow/client.py
By Daniel J. Beutel and Taner Topal
"""

# Input central server IP
ip = "18.219.159.248"


class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, train, val):
        self.model = model
        self.train, self.val = train, val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Train on the local data and return the updated model weights and results

        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train[:2500],  # TODO
            self.y_train[:2500],  # TODO
            config["batch_size"],
            config["local_epochs"],
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)  # TODO x_train
        results = {
            "loss": float(history.history["loss"][0]),
            "accuracy": float(history.history["accuracy"][0]),
            "val_loss": float(history.history["val_loss"][0]),
            "val_accuracy": float(history.history["val_accuracy"][0]),
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # TODO use config["val_steps"] ?
        loss, acc = self.model.evaluate(self.x_train[2501:], self.y_train[2501:], verbose=2)  # TODO x_train/y_train
        return loss, len(self.val), {"accuracy": acc}

def client_fn(cid: str) -> fl.client.Client:
    # Load model
    train, val, _, _ = demo_train_val_test()
    model = build_unet()
    model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics='accuracy')
    client = FlwrClient(model, train, val)

    print(f'~~~~~~~~~~~~~~~~~~~Client:{cid}~~~~~~~~~~~~~~~~~~~~~~~~~~')

    print(f'x:{len(x_train)}  y:{len(y_train)}')  # TODO x_train/y_train
    # Create and return client
    return client


if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address=ip,
        client=client_fn(0),
        grpc_max_message_length=1024*1024*10
    )
