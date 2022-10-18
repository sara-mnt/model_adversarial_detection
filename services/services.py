import tempfile

import numpy as np
from utils.file_utils import extract_from_gzip, save_dataset
from utils.logger_utils import stream_logger
import sanic
import traceback
from sanic import Sanic, Blueprint
from sanic.exceptions import NotFound
from sanic_openapi import swagger_blueprint
from sanic_openapi.openapi2 import doc
from loko_extensions.business.decorators import extract_value_args

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


@bp.post('/myfirstservice')
@doc.consumes(doc.JsonBody({"value": dict, "args": {"data_name": str}}), location="body")
@extract_value_args()
async def f(value, args):
    logger.debug(f'ARGS: {args}')
    logger.debug(f'JSON: {value}')
    n = args.get('data_name')
    return sanic.json(dict(msg=f"{n} Hello world! {n}"))


@bp.post('/read_dataset')
@doc.consumes(doc.JsonBody({"value": dict, "args": {"new_model_name": str}}), location="formData",
              content_type="multipart/form-data")
@doc.consumes(doc.File(name="file"), location="formData", content_type="multipart/form-data")
# @extract_value_args(file=True)
async def f2(file, args):
    logger.debug(f'FILE: {file}')
    logger.debug(f'ARGS: {args}')
    file = file[0]
    internal_folder = args.get("internal_folder")
    data_name = args.get("data_name")
    with tempfile.NamedTemporaryFile(mode="wb", suffix="." + file.filename.split(".")[-1]) as tt:
        tt.write(file.stream.read())
        tt.seek(0)
        with open(tt.name, "rb") as f:
            data = extract_from_gzip(f)
            temp_npz = tempfile.NamedTemporaryFile(suffix=".npz")
            np.savez(temp_npz, **{data_name: data})
            file_name = file.filename.split(".")[0] + ".npz"
            save_dataset(file=temp_npz, file_name=file_name, internal_folder=internal_folder)

    logger.debug(f'ARGS: {args}')
    logger.debug(f'JSON: {file[0].name}')
    n = int(args.get('n'))
    return sanic.json(dict(msg=f"{'#' * n} You have uploaded the file: {file[0].name}! {'#' * n}"))


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
