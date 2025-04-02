"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from kobe_ml.pipelines import data_processing as dp
from kobe_ml.pipelines import model_training as mt
from kobe_ml.pipelines import model_deployment as md

# Configuração do MLflow
from kedro.framework.hooks import _create_hook_manager
from kedro_mlflow.framework.hooks import MlflowHook

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Registrar os pipelines individuais
    data_processing_pipeline = dp.create_pipeline()
    model_training_pipeline = mt.create_pipeline()
    model_deployment_pipeline = md.create_pipeline()
    
    # Criar pipelines compostos
    return {
        "data_processing": data_processing_pipeline,
        "model_training": model_training_pipeline,
        "model_deployment": model_deployment_pipeline,
        "train": data_processing_pipeline + model_training_pipeline,
        "deploy": model_deployment_pipeline,
        "__default__": data_processing_pipeline + model_training_pipeline + model_deployment_pipeline,
    }

# Registrar o hook do MLflow
HOOKS = _create_hook_manager()
HOOKS.register(MlflowHook())

# Para ativar auto tracking:
MLFLOW_TRACKING_URI = "mlruns"