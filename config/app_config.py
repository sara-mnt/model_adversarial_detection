

from ds4biz_commons.utils.config_utils import EnvInit
e=EnvInit()

#ORCHESTRATOR = f"{e.GATEWAY}/routes/orchestrator/"
ORCHESTRATOR = "http://localhost:9999/routes/orchestrator/"

PREDICTOR_EVALUATE_FOLDER = '/data/ts_evaluation'

DATASETS_FOLDER = "/data/datasets"
