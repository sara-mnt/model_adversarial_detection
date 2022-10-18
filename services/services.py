import tempfile
from typing import List

import numpy as np

from business.carlini_wagner_adversarial import estimate_model_affidability
from business.train_model import BaseNeuralNetwork
from config.app_config import ORCHESTRATOR, MODELS_FOLDER
from utils.dataset_utils import get_training_data_from_dataset
from utils.file_utils import extract_from_gzip, save_dataset
from utils.logger_utils import stream_logger
import sanic
import traceback
from sanic import Sanic, Blueprint
from sanic.exceptions import NotFound
from sanic_openapi import swagger_blueprint
from sanic_openapi.openapi2 import doc
from loko_extensions.business.decorators import extract_value_args

from tensorflow import keras
from keras import models

import requests

import tensorflow as tf

logger = stream_logger(__name__)
tf.compat.v1.disable_eager_execution()


def get_app(name):
    app = Sanic(name)
    swagger_blueprint.url_prefix = "/api"
    app.blueprint(swagger_blueprint)
    return app


name = "first_project"
app = get_app(name)
bp = Blueprint("default", url_prefix=f"/")
app.config["API_TITLE"] = name


@bp.get('/models')
@doc.consumes()
def get_models_filenames(request):
    # get models for async select

    fpath = "files/data/models"
    file_writer_path = ORCHESTRATOR + fpath
    res = requests.get(file_writer_path)
    if res.status_code == 200:
        items = res.json().get("items")
        models = [item.get("name") for item in items]
        return sanic.json(models)
    else:
        return sanic.json([])


@bp.get("/datasets")
def get_datasets(request):
    # get datasets for async select

    fpath = "files/data/datasets"
    file_writer_path = ORCHESTRATOR + fpath
    res = requests.get(file_writer_path)
    if res.status_code == 200:
        items = res.json().get("items")
        print(items)
        datasets = [item.get("name") for item in items]
        return sanic.json(datasets)
    else:
        return sanic.json([])


# @bp.post('/myfirstservice')
# @doc.consumes(doc.JsonBody({"value": dict, "args": {"data_name": str}}), location="body")
# @extract_value_args()

@bp.post('/model/train')
@doc.consumes(doc.JsonBody({"value": dict, "args": {"internal_folder": str, "model": str}}), location="body")
@extract_value_args()
def train_model(value, args):
    model_name = args.get("model")
    internal_folder = args.get("internal_folder")
    (X_train, y_train), (X_test, y_test) = \
        get_training_data_from_dataset(dataset_folder=internal_folder)

    # TODO MODEL FACTORY
    if model_name == "base":
        neural_net = BaseNeuralNetwork()
        neural_net()
        neural_net.train(X_train=X_train,
                         X_test=X_test,
                         y_train=y_train,
                         y_test=y_test,
                         model_name=model_name)

    return sanic.json(dict(msg="Train del modello eseguito con successo"))


@bp.post('/model/detect')
@doc.consumes(doc.JsonBody({"value": dict, "args": {"model": str, "internal_folder": str, "detection_method": str}}),
              location="body")
@extract_value_args()
def detect_model(value, args):
    logger.debug(f'ARGS: {args}')
    logger.debug(f'JSON: {value}')
    model_name = args.get("model")
    internal_folder = args.get("internal_folder")
    detection_method = args.get("detection_method")
    (X_train, y_train), (X_test, y_test) = get_training_data_from_dataset(dataset_folder=internal_folder)
    model_url = ORCHESTRATOR + "files" + MODELS_FOLDER + "/" + model_name + ".h5"
    resp = requests.get(model_url)
    if resp.status_code != 200:
        raise Exception(f"Chiamata fallita: {resp.text} for url: {resp.url}")

    # TODO ADVERSARIAL METHOD FACTORY

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        tmp.write(resp.content)
        tmp.seek(0)
        model = models.load_model(tmp.name)
        X_adversarial = estimate_model_affidability(model,
                                                    X_train=X_train,
                                                    X_test=X_test,
                                                    y_train=y_train,
                                                    y_test=y_test)
        print(X_adversarial.shape)

    return sanic.json(dict(
        msg=f"Adversarial detection con metodo {detection_method} eseguita con successo per il modello {model_name}"))


# @doc.consumes(doc.JsonBody({"value": dict, "args": {"data_name": str, "internal_folder":str}}), location="body", content_type="application/json")
# @doc.consumes(doc.File(name="file"), location="formData", content_type="multipart/form-data")
# @extract_value_args(file=True)

@bp.post('/dataset/read')
@doc.consumes(doc.JsonBody({"value": dict, "args": {"data_type": str, "internal_folder": str}}), location="formData",
              content_type="multipart/form-data")
@doc.consumes(doc.File(name="file"), location="formData", content_type="multipart/form-data")
@extract_value_args(file=True)
async def read_dataset(file, args):
    file = file[0]
    internal_folder = args.get("internal_folder")
    data_type = args.get("data_type")
    with tempfile.NamedTemporaryFile(mode="wb", suffix="." + file.name.split(".")[-1]) as tt:
        tt.write(file.body)
        tt.seek(0)
        with open(tt.name, "rb") as f:
            data = extract_from_gzip(f)
            temp_npz = tempfile.NamedTemporaryFile(suffix=".npz")
            np.savez(temp_npz, **{data_type: data})
            file_name = file.name.split(".")[0] + ".npz"
            save_dataset(file=temp_npz, file_name=file_name, internal_folder=internal_folder)

    # logger.debug(f'ARGS: {args}')
    # logger.debug(f'JSON: {file[0].name}')
    # n = int(args.get('n'))
    # return sanic.json(dict(msg=f"{'#' * n} You have uploaded the file: {file.name}! {'#' * n}"))
    return sanic.json(dict(msg=f"You have uploaded the file: {file.name}!"))


@app.exception(Exception)
async def manage_exception(request, exception):
    e = dict(error=str(exception))
    if isinstance(exception, NotFound):
        return sanic.json(e, status=404)
    logger.error('TracebackERROR: \n' + traceback.format_exc() + '\n\n')
    status_code = exception.status_code or 500
    return sanic.json(e, status=status_code)


app.blueprint(bp)

app.run("0.0.0.0", port=8080, auto_reload=True)
