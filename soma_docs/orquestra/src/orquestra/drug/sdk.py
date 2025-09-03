import os

import mlflow
from orquestra.sdk.mlflow import get_tracking_token, get_tracking_uri


def configure_mlflow(experiment_name: str, config_name: str = "research") -> None:
    os.environ["MLFLOW_TRACKING_TOKEN"] = get_tracking_token(config_name=config_name)
    mlflow.set_tracking_uri(
        get_tracking_uri(
            workspace_id="drug-discovery-dwave-250cee", config_name=config_name
        )
    )
    mlflow.set_experiment(experiment_name)
