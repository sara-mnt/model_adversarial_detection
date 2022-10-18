import tempfile
from typing import List
import requests
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.layers import Dense, Flatten
from keras.models import Sequential

from config.app_config import ORCHESTRATOR, MODELS_FOLDER


# y_train = np.expand_dims(y_train, axis=-1)  # (60000, 28, 28, 1)
# y_test = np.expand_dims(y_test, axis=-1)


class NeuralNetwork:

    def create_neural_network(self):
        pass

    def train(self,
              X_test: np.array,
              X_train: np.array,
              y_test: np.array,
              y_train: np.array,
              loss: str = "categorical_crossentropy",
              optimizer: str = "adam",
              metrics: List = None,
              model_name: str = "model",
              save: bool = True):
        pass


class BaseNeuralNetwork(NeuralNetwork):

    def __init__(self):
        self.model = Sequential()

    def __call__(self, *args, **kwargs):
        self.create_neural_network()

    def create_neural_network(self):
        self.model.add(Flatten(input_shape=(28, 28, 1)))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(20, activation='sigmoid'))
        self.model.add(Dense(10, activation='softmax'))

    def train(self,
              X_test: np.array,
              X_train: np.array,
              y_test: np.array,
              y_train: np.array,
              loss: str = "categorical_crossentropy",
              optimizer: str = "adam",
              metrics: List = None,
              model_name: str = "model",
              save: bool = True, requests=None):

        if not metrics:
            metrics = ["acc"]

        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics,
                           )

        self.model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

        if save:
            self.save_model(model_name=model_name)

    def save_model(self, model_name:str):
        directory_path = ORCHESTRATOR + "files" + MODELS_FOLDER
        file_writer_path = directory_path + "/" + model_name + ".h5"
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            self.model.save(tmp.name)
            tmp.seek(0)
            data = open(tmp.name, "rb").read()
            res = requests.post(file_writer_path, data=data)
            # TODO LOGGER




if __name__ == '__main__':

    nn = BaseNeuralNetwork()
    nn()

    predictions = nn.model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)

    print(predictions)

    # Load the model from the file
    model = models.load_model('mnist.h5')
    # Use the loaded model to make predictions
    predictions = model.predict(X_test)

    print(predictions)
