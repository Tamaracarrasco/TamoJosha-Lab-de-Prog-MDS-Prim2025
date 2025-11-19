# airflow/scripts/mlflow_utils.py

import os
from typing import Dict, Any

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

import config as cfg


def setup_mlflow() -> None:
    """
    Configura MLflow con el tracking URI y el experimento definidos en config.
    Si el experimento original está marcado como 'deleted', se crea/usa
    una variante con sufijo '_v2' para no romper la ejecución.
    """
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

    exp_name = cfg.MLFLOW_EXPERIMENT_NAME
    try:
        mlflow.set_experiment(exp_name)
    except MlflowException as e:
        if "deleted experiment" in str(e):
            new_name = exp_name + "_v2"
            print(
                f"[MLflow][WARN] El experimento '{exp_name}' está marcado como "
                f"borrado. Usando '{new_name}' en su lugar."
            )
            mlflow.set_experiment(new_name)
        else:
            # Si es otro tipo de error, lo propagamos
            raise


def log_params_flat(params: Dict[str, Any], prefix: str = "") -> None:
    """
    Loggea un diccionario de parámetros en MLflow, opcionalmente con prefijo.
    """
    for k, v in params.items():
        mlflow.log_param(f"{prefix}{k}", v)


def log_metrics_flat(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Loggea un diccionario de métricas en MLflow, opcionalmente con prefijo.
    """
    for k, v in metrics.items():
        mlflow.log_metric(f"{prefix}{k}", float(v))


def log_sklearn_pipeline(pipeline, artifact_path: str = "model") -> None:
    """
    Loggea un pipeline de sklearn (en este caso Pipeline + XGBClassifier) en MLflow.

    Si hay problemas de permisos al escribir los artefactos (por ejemplo, al
    intentar usar una ruta de artefactos como /mlflow sin permisos), se muestra
    un warning pero no se rompe la ejecución del pipeline.
    """
    try:
        mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path)
        print(
            f"[MLflow] Modelo sklearn registrado en MLflow bajo "
            f"artifact_path='{artifact_path}'."
        )
    except PermissionError as e:
        print(
            f"[MLflow][WARN] No se pudo registrar el modelo como artefacto "
            f"por problema de permisos: {e}"
        )
        print(
            "[MLflow] Las métricas del run siguen registradas, pero el modelo "
            "no se guardó como artefacto."
        )
    except Exception as e:
        print(
            f"[MLflow][WARN] Error inesperado al registrar el modelo en MLflow: {e}"
        )
        print("[MLflow] Se continúa sin guardar el modelo como artefacto.")

        # Todos estos warning y avisos o controles fueron porque tiraba error todo el rato.
