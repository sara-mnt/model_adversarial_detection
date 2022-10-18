import numpy as np
from keras.datasets import mnist

from art.attacks.evasion import CarliniL0Method, DPatch, CarliniL2Method
from art.estimators.classification import KerasClassifier
from keras import models

from utils.targets_utils import get_inputs_targets_test_train

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def estimate_model_affidability(model_path: str, X_train, X_test, y_train, y_test):
    model = models.load_model(model_path)
    estimator = KerasClassifier(model=model)
    carlini_wagner_attack = CarliniL2Method(classifier=estimator, confidence=0.5,
                                            batch_size=40, max_iter=2)

    (X_train, y_train), (X_test, y_test) = get_inputs_targets_test_train(X_train, X_test, y_train, y_test)

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    (X_train, y_train) = (X_train[:3000], y_train[:3000])

    y_train = np.argmax(y_train, axis=2)
    x_adversarial = carlini_wagner_attack.generate(x=X_train, y=y_train)

    print(carlini_wagner_attack.confidence)

    print(x_adversarial.shape)

if __name__ == '__main__':
    modelpath = 'mnist.h5'

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    estimate_model_affidability(model_path=modelpath,
                                X_train=X_train,
                                X_test=X_test,
                                y_train=y_train,
                                y_test=y_test)
# predictions = np.argmax(predictions, axis=1)
# model.compile(loss='categorical_crossentropy',
# optimizer='adam',
#  metrics=['acc'],
# )
# print(model.compiled_metrics)
# metrics = model.compute_metrics(x=x_adversarial, y=y_train, y_pred=predictions, sample_weight=np.ones(shape=(len(y_train),)))
# print(model.compiled_metrics)
# print(metrics)
