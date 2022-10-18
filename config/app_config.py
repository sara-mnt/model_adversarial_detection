from ds4biz_commons.utils.config_utils import EnvInit

e = EnvInit()
GATEWAY = e.GATEWAY or "http://localhost:9999"
ORCHESTRATOR = f"{GATEWAY}/routes/orchestrator/"
# ORCHESTRATOR = "http://localhost:9999/routes/orchestrator/"

PREDICTOR_EVALUATE_FOLDER = '/data/ts_evaluation'

DATASETS_FOLDER = "/data/datasets"
