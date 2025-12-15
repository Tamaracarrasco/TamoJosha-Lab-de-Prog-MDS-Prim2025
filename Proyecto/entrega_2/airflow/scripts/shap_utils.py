# airflow/scripts/shap_utils.py

import os
from typing import List

import numpy as np
import pandas as pd


def compute_and_log_shap(
    pipeline,
    X_sample: pd.DataFrame,
    output_dir: str,
    artifact_subdir: str = "shap",
    max_features_dependence: int = 2,
) -> None:
    """
    Calcula valores SHAP para el pipeline XGBoost y guarda:
      - shap_summary.png
      - 1 a 2 dependence plots para las features más importantes.

    Si hay un run activo de MLflow, registra estos PNG como artefactos.
    """

    # Lazy import tambine por que se caia al quedarme sin RAM.
    import shap
    import mlflow
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Recuperar pasos del pipeline
    try:
        xgb_model = pipeline.named_steps["clasificador_xgb"]
        transformer = pipeline.named_steps["col_transformer"]
    except KeyError:
        raise ValueError(
            "El pipeline no contiene 'clasificador_xgb' o 'col_transformer'. "
            "Revisa los nombres de los pasos."
        )

    # Transformar X_sample con el ColumnTransformer
    X_trans = transformer.transform(X_sample)

    # Convertir a denso si viene como matriz dispersa
    if hasattr(X_trans, "toarray"):
        X_dense = X_trans.toarray()
    else:
        X_dense = X_trans

    # Nombres de features después del preprocesamiento
    try:
        feature_names: List[str] = transformer.get_feature_names_out().tolist()
    except AttributeError:
        feature_names = [f"feat_{i}" for i in range(X_dense.shape[1])]

    # TreeExplainer para XGBoost
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_dense)

    # ------------------------------------------------------------------
    # SHAP summary plot
    # ------------------------------------------------------------------
    summary_path = os.path.join(output_dir, "shap_summary.png")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_dense,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # Dependence plots para las features más relevantes
    # ------------------------------------------------------------------
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_idx = np.argsort(-mean_abs)[:max_features_dependence]

    dep_paths = []

    for idx in top_idx:
        feat_name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        safe_name = (
            feat_name.replace("[", "_")
            .replace("]", "_")
            .replace(" ", "_")
            .replace("/", "_")
        )

        dep_path = os.path.join(output_dir, f"shap_dependence_{safe_name}.png")

        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            idx,
            shap_values,
            X_dense,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(dep_path, dpi=150, bbox_inches="tight")
        plt.close()

        dep_paths.append(dep_path)

    # ------------------------------------------------------------------
    # Loggear artefactos en MLflow (si hay run activo)
    # ------------------------------------------------------------------
    active_run = mlflow.active_run()

    if active_run is not None:
        try:
            # Intentar loguear el resumen principal
            mlflow.log_artifact(summary_path, artifact_path=artifact_subdir)

            # Intentar loguear las dependencias
            for p in dep_paths:
                mlflow.log_artifact(p, artifact_path=artifact_subdir)

            print(f"[SHAP] Gráficos SHAP registrados en MLflow bajo '{artifact_subdir}'.")
        except Exception as e:
            # Captura cualquier error de permisos o escritura
            print(f"[SHAP][WARN] No se pudieron registrar artefactos en MLflow: {e}")
            print("[SHAP] Los archivos fueron guardados localmente, pero no en MLflow.")
    else:
        print("[SHAP] No hay run activo de MLflow: solo se guardaron los PNG localmente.")

        # De nuevo mucho error asi qeu hubi qe poner warning o aviso del error para poder trackear y solucionar
