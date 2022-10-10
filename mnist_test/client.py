import flwr as fl
from 2d_cnn import CNN_2D
import tensorflow as tf
import tensorflow_datasets as tfds

model = CNN_2D((28,28,1))
model.compile("adam","sparse_categorical_crossentropy", metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = tfds.load('mnist_test')
x_val, y_val = x_test[:2000], y_test[:2000]

class MnistClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=2, verbose=2)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_val, y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MnistClient())