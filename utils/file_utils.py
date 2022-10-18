import gzip
import json
import os

import numpy as np

from config.app_config import ORCHESTRATOR, DATASETS_FOLDER, MODELS_FOLDER
import requests

from utils.loko_extensions import extract_value_args


## se voglio leggere direttamente i file senza passare dal file manager

## oppure repo e frontend


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(bytestream):
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    print(data.shape)
    #assert data.shape[3] == 1
    #data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    #data = data.astype(np.float32)
    #data = np.multiply(data, 1.0 / 255.0)
    return data


def extract_from_gzip(f):
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic == 2049:
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels
        elif magic == 2051:
            print('Extracting', f.name)
            return extract_images(bytestream)
        else:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                            (magic, f.name))


class FileStorage:
    pass


def save_dataset(file:FileStorage, internal_folder:str, file_name:str=None,) -> str:
    directory_path = ORCHESTRATOR + "files" + DATASETS_FOLDER
    #file_name = file.filename.split(".")[0] + ".npz"
    if not file_name:
        file_name = file.filename
    file_writer_path = directory_path + "/" + internal_folder + "/" + file_name
    data = open(file.name, "rb").read()
    #data = file.stream.read()
    #np.savez(file_writer_path, data)
    res = requests.post(file_writer_path, data=data)
    return "ok"

def save_trained_model(file:FileStorage, internal_folder:str, file_name:str=None,) -> str:
    directory_path = ORCHESTRATOR + "files" + MODELS_FOLDER
    #file_name = file.filename.split(".")[0] + ".npz"
    if not file_name:
        file_name = file.filename
    file_writer_path = directory_path + "/" + internal_folder + "/" + file_name
    data = open(file.name, "rb").read()
    #data = file.stream.read()
    #np.savez(file_writer_path, data)
    res = requests.post(file_writer_path, data=data)
    return "ok"


if __name__ == '__main__':
    data = np.load("/home/sara/loko/datasets/mnist/t10k-labels-idx1-ubyte.npz", allow_pickle=True)
    print(data["y_test"])
    print(list(data.keys()))
    mnist_folder = "/home/sara/loko/datasets/mnist/"
    #for nzp_file in os.listdir(mnist_folder)

