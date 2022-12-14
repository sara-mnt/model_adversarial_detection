import gzip
import json
import os
import tempfile
from typing import List, Dict
import matplotlib.pyplot as plt
import werkzeug
import numpy as np

from ds4biz_flask.model.ds4bizflask import DS4BizFlask
from flask_compress import Compress
from flask_cors import CORS
from keras.datasets import mnist
import requests
from loko_extensions.business.decorators import extract_value_args

from werkzeug.datastructures import FileStorage

from business.carlini_wagner_adversarial import estimate_model_affidability

import tensorflow as tf

from config.app_config import ORCHESTRATOR, DATASETS_FOLDER
from utils.file_utils import extract_from_gzip, save_dataset_file

tf.compat.v1.disable_eager_execution()

app = DS4BizFlask("prova", version="0.0.1")


CORS(app, expose_headers="Content-Disposition")
Compress(app)

app.url_map.strict_slashes = False
root = "/services"
app.root = "/"


def estimate_model(model_saved:str, data_file:str=None) -> str:
    if not data_file:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    estimate_model_affidability(model_saved, X_train, y_train, X_test, y_test)

    return "ok"


def get_models_filenames() -> List[str]:
    fpath = "files/data/models"
    file_writer_path = ORCHESTRATOR + fpath
    res = requests.get(file_writer_path)
    if res.status_code == 200:
        items = res.json().get("items")
        models = [item.get("name") for item in items]
        return models


def get_datasets() -> List[str]:
    fpath = "files/data/datasets"
    file_writer_path = ORCHESTRATOR + fpath
    res = requests.get(file_writer_path)
    if res.status_code == 200:
        items = res.json().get("items")
        print(items)
        datasets = [item.get("name") for item in items]
        return datasets


def save_dataset(file:FileStorage,file_name:str=None, internal_folder:str=None) -> str:
    if not internal_folder:
        len_datasets_dir = len(get_datasets())
        internal_folder = "dataset_" + str(len_datasets_dir)
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

from werkzeug.local import Local
l = Local()

# these are proxies
request = l('request')
user = l('user')

from werkzeug.local import LocalStack
_response_local = LocalStack()

# this is a proxy
response = _response_local()

@extract_value_args(_request=response, file=True)
def read_dataset(file:FileStorage, args:Dict[str, str]) -> str:
    internal_folder = args.get("internal_folder")
    data_name = args.get("data_name")
    with tempfile.NamedTemporaryFile(mode="wb", suffix="." + file.filename.split(".")[-1]) as tt:
        tt.write(file.stream.read())
        tt.seek(0)
        with open(tt.name, "rb") as f:
            data = extract_from_gzip(f)
            temp_npz = tempfile.NamedTemporaryFile(suffix=".npz")
            np.savez(temp_npz,**{data_name: data})
            file_name = file.filename.split(".")[0] + ".npz"
            save_dataset(file=temp_npz, file_name=file_name, internal_folder=internal_folder)

    return "ok"

def load_dataset():
    data = np.load("/home/sara/loko/datasets/mnist/t10k-images-idx3-ubyte.npz")
    print(data)

def merge_datasets(datasets):
    np.savez()
    return datasets

def train_model(model_name:str, dataset_folder:str) -> str:
    print(model_name)
    fpath = f"files/data/datasets/{dataset_folder}"
    file_writer_path = ORCHESTRATOR + fpath
    res = requests.get(file_writer_path)
    if res.status_code != 200:
        raise Exception("Chiamata non andata a buon fine")
    items = res.json().get("items")
    print(items)
    datasets = [item.get("name") for item in items]
    merged_datasets = merge_datasets(datasets)
    return "ok"




app.add_service(root + "/model/estimate", estimate_model, method="POST")
app.add_service(root + "/models", get_models_filenames, method="GET")
app.add_service(root + "/dataset/read", read_dataset, method="POST")
app.add_service(root + "/dataset", save_dataset, method="POST")
app.add_service(root + "/datasets", get_datasets, method="GET")
app.add_service(root + "/model/train/<model_name>/<dataset_folder>", train_model, method="GET")

app.setup()

SERVICES_HOST = "localhost"
SERVICES_PORT = 8080
app.run(SERVICES_HOST, SERVICES_PORT, debug=False)

