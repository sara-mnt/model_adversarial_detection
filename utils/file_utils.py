import gzip
import json
import os

import numpy as np

from config.app_config import ORCHESTRATOR, DATASETS_FOLDER
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

def save_dataset_file(dataset_file_name, data, internal_folder:str=None):
    os.listdir(DATASETS_FOLDER)
    if not internal_folder:
        internal_folder = "dataset_" + str(len(os.listdir(DATASETS_FOLDER)))
    fpath = DATASETS_FOLDER + "/" + internal_folder + "/" + dataset_file_name
    file_writer_path = ORCHESTRATOR + "files" + fpath
    data = json.dumps(data)
    res = requests.post(file_writer_path, data=data)

if __name__ == '__main__':
    data = np.load("/home/sara/loko/datasets/mnist/t10k-labels-idx1-ubyte.npz", allow_pickle=True)
    print(data["y_test"])
    print(list(data.keys()))
    mnist_folder = "/home/sara/loko/datasets/mnist/"
    #for nzp_file in os.listdir(mnist_folder)

