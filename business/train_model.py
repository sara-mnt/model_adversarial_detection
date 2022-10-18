import os
import pickle
from typing import List

import numpy
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
from numpy.array_api import astype

from utils.targets_utils import get_inputs_targets_test_train

# y_train = np.expand_dims(y_train, axis=-1)  # (60000, 28, 28, 1)
# y_test = np.expand_dims(y_test, axis=-1)

def get_training_data_from_dataset(dataset_folder:str):
    dataset_path = "/home/sara/loko/datasets/mnist"
    for file in os.listdir(dataset_path):
        f = np.load(file)
        data_name = list(f.keys())[0]
        print(data_name)


    #(X_train, y_train), (X_test, y_test)

    #(X_train, y_train), (X_test, y_test) = get_inputs_targets_test_train()

    #X_train = np.expand_dims(X_train, axis=-1)  # (60000, 28, 28, 1)
    #X_test = np.expand_dims(X_test, axis=-1)

    return

get_training_data_from_dataset(dataset_folder="")
class NeuralNetork:

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


class BaseNeuralNetork(NeuralNetork):

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
              save: bool = True):

        if not metrics:
            metrics = ["acc"]

        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=metrics,
                           )

        model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

        if save:
            self.model.save(f'{model_name}.h5')


if __name__ == '__main__':
    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)

    print(predictions)

    # Load the model from the file
    model = models.load_model('mnist.h5')
    # Use the loaded model to make predictions
    predictions = model.predict(X_test)

    print(predictions)
