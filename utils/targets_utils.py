import numpy as np

from keras.datasets import mnist


def get_hot_encoded_targets(y_train: np.ndarray, y_test: np.ndarray):
    # Convert y_train into one-hot format
    y_train_hot_encoded = list()
    for i in range(len(y_train)):
        y_train_hot_encoded.append(to_categorical(y_train[i], num_classes=10))
    y_train = np.array(y_train_hot_encoded)

    # Convert y_test into one-hot format
    y_test_hot_encoded = list()
    for i in range(len(y_test)):
        y_test_hot_encoded.append(to_categorical(y_test[i], num_classes=10))
    y_test = np.array(y_test_hot_encoded)

    return y_train, y_test


def get_inputs_targets_test_train(X_train, X_test, y_train, y_test, hot_encoded: bool = True):
    if hot_encoded:
        y_train, y_test = get_hot_encoded_targets(y_train, y_test)

    (X_train, y_train), (X_test, y_test) = (X_train[:12000], y_train[:12000]), (X_test[:2000], y_test[:2000])

    X_train, x_test = X_train.astype(np.float32), X_test.astype(np.float32)

    return (X_train, y_train), (X_test, y_test)
