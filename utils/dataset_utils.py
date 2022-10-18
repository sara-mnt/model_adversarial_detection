import tempfile
import numpy as np
from config.app_config import ORCHESTRATOR, DATASETS_FOLDER
from utils.targets_utils import get_inputs_targets_test_train
import requests

# y_train = np.expand_dims(y_train, axis=-1)  # (60000, 28, 28, 1)
# y_test = np.expand_dims(y_test, axis=-1)


def get_training_data_from_dataset(dataset_folder:str):
    directory_path = ORCHESTRATOR + "files" + DATASETS_FOLDER
    datasets_folder_path = directory_path + "/" + dataset_folder
    res = requests.get(datasets_folder_path)
    datasets = res.json().get("items")
    data_dict= dict()
    for dataset in datasets:
        f = ORCHESTRATOR + "files" + "/" + dataset.get("path")
        r = requests.get(f)
        with tempfile.NamedTemporaryFile() as t:
            t.write(r.content)
            t.seek(0)
            data = np.load(t)
            data_name = list(data.keys())[0]
            data_dict.update({data_name: data.get(data_name)})

    X_test = data_dict.get("X_test")
    X_train = data_dict.get("X_train")
    y_test = data_dict.get("y_test")
    y_train = data_dict.get("y_train")

    (X_train, y_train), (X_test, y_test) = get_inputs_targets_test_train(X_train, X_test, y_train, y_test)

    #X_train = np.expand_dims(X_train, axis=-1)  # (60000, 28, 28, 1)
    #X_test = np.expand_dims(X_test, axis=-1)

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    get_training_data_from_dataset(dataset_folder="mnist")