import os
import tempfile

import numpy as np

from utils.file_utils import extract_from_gzip, save_dataset
from utils.logger_utils import stream_logger
from utils.targets_utils import get_inputs_targets_test_train

logger = stream_logger(__name__)

def read_dataset(file, args):
    #file = file[0]
    internal_folder = args.get("internal_folder")
    data_name = args.get("data_name")
    with tempfile.NamedTemporaryFile(mode="wb", suffix="." + file.name.split(".")[-1]) as tt:
        tt.write(file.read())
        tt.seek(0)
        with open(tt.name, "rb") as f:
            data = extract_from_gzip(f)
            temp_npz = tempfile.NamedTemporaryFile(suffix=".npz")
            np.savez(temp_npz, **{data_name: data})
            file_name = file.name.split("/")[-1]
            file_name = file_name.split(".")[0] + ".npz"
            save_dataset(file=temp_npz, file_name=file_name, internal_folder=internal_folder)

    logger.debug(f'ARGS: {args}')
    #logger.debug(f'JSON: {file[0].name}')


def get_training_data_from_dataset(dataset_folder: str):
    dataset_path = "/home/sara/loko/datasets/mnist"
    data_dict = dict()
    for file in os.listdir(dataset_path):
        fpath = dataset_path + "/" + file
        f = np.load(fpath)
        data_name = list(f.keys())[0]
        print(data_name)
        data_dict.update({data_name: f.get(data_name)})

    X_test = data_dict.get("X_test")
    X_train = data_dict.get("X_train")
    y_test = data_dict.get("y_test")
    y_train = data_dict.get("y_train")

    (X_train, y_train), (X_test, y_test) = get_inputs_targets_test_train(X_train, X_test, y_train, y_test)

    X_train = np.expand_dims(X_train, axis=-1)  # (60000, 28, 28, 1)
    X_test = np.expand_dims(X_test, axis=-1)

    return (X_train, y_train), (X_test, y_test)


with open("/home/sara/Scaricati/train-images-idx3-ubyte.gz", "rb") as f:
    print(f.name)
    read_dataset(file=f, args={"data_name":"X_train", "internal_folder":"mnist"})