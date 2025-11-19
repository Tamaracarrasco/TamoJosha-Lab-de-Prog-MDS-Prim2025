# airflow/scripts/modeling_xgb.py

import os
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from xgboost import XGBClassifier
import optuna
import mlflow

import config as cfg

from scripts.mlflow_utils import (
    setup_mlflow,
    log_params_flat,
    log_metrics_flat,
    log_sklearn_pipeline,
)

# ======================================================================
# SUBSAMPLING ESTRATIFICADO
# ======================================================================

def subsample_dataset(
    df_final: pd.DataFrame,
    fraction: float = 0.1,
    stratify_col: str = "y",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Submuestrea el dataset manteniendo la proporción de clases.
    
    """
    if fraction >= 1.0:
        return df_final
    
    from sklearn.model_selection import train_test_split
    
    print(f"\n{'='*60}")
    print(f"SUBSAMPLING ACTIVADO: {fraction*100:.1f}% de los datos")
    print(f"{'='*60}")
    print(f"  Dataset original: {len(df_final):,} registros")
    
    # Submuestreo estratificado
    df_sample, _ = train_test_split(
        df_final,
        train_size=fraction,
        stratify=df_final[stratify_col],
        random_state=random_state,
    )
    
    print(f"  Dataset muestreado: {len(df_sample):,} registros")
    print(f"  Proporción clase 1 (original): {df_final[stratify_col].mean():.3f}")
    print(f"  Proporción clase 1 (sample):   {df_sample[stratify_col].mean():.3f}")
    print(f"{'='*60}\n")
    
    return df_sample.reset_index(drop=True)


# ======================================================================
# Split temporal train / val / test
# ======================================================================

def temporal_train_val_test_split(
    df_final: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split temporal:
    - semanas ordenadas por semana_num
    - train: primeras cfg.N_TRAIN_WEEKS semanas
    - val:   siguientes cfg.N_VAL_WEEKS semanas
    - test:  resto
    """
    df_final = df_final.sort_values("semana_num").reset_index(drop=True)
    weeks_sorted = sorted(df_final["semana_num"].unique())

    n_train = cfg.N_TRAIN_WEEKS
    n_val = cfg.N_VAL_WEEKS

    train_weeks = weeks_sorted[:n_train]
    val_weeks = weeks_sorted[n_train : n_train + n_val]
    test_weeks = weeks_sorted[n_train + n_val :]

    train_df = df_final[df_final["semana_num"].isin(train_weeks)]
    val_df = df_final[df_final["semana_num"].isin(val_weeks)]
    test_df = df_final[df_final["semana_num"].isin(test_weeks)]

    drop_cols = ["y"]

    X_train = train_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_train = train_df["y"].astype(int).reset_index(drop=True)

    X_val = val_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_val = val_df["y"].astype(int).reset_index(drop=True)

    X_test = test_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_test = test_df["y"].astype(int).reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ======================================================================
# Optuna: búsqueda de hiperparámetros
# ======================================================================

def tune_hyperparams_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout_min: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Ejecuta Optuna para encontrar mejores hiperparámetros de XGBClassifier
    y del OneHotEncoder (min_frequency), usando F1 en validación.
    """

    drop_cols = ["customer_id", "product_id"]

    num_cols = ["X", "Y", "size", "num_deliver_per_week", "num_visit_per_week"]

    cat_cols = [
        "customer_type",
        "segment",
        "brand",
        "category",
        "sub_category",
        "zone_id",
        "region_id",
        "semana",
    ]

    for col in ["customer_type", "brand", "category", "sub_category", "segment", "package"]:
        for X in [X_train, X_val]:
            if col in X.columns:
                X[col] = X[col].astype("category")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos

    def objective(trial: optuna.Trial) -> float:
        params_xgb = {
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }

        params_ohe = {
            "min_frequency": trial.suggest_float("min_frequency", 0.0, 1.0),
        }

        num_pipeline = Pipeline(
            [
                ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("sc", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            [
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                        **params_ohe,
                    ),
                )
            ]
        )

        col_transformer = ColumnTransformer(
            [
                ("drop_ids", "drop", drop_cols),
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols),
            ],
            verbose_feature_names_out=False,
            remainder="drop",
        )

        xgb_clf = XGBClassifier(
            objective="binary:logistic",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            **params_xgb,
        )

        pipeline = Pipeline(
            steps=[
                ("col_transformer", col_transformer),
                ("clasificador_xgb", xgb_clf),
            ]
        )

        pipeline.fit(X_train, y_train)

        proba_val = pipeline.predict_proba(X_val)[:, 1]
        y_pred_val = (proba_val >= 0.5).astype(int)

        return f1_score(y_val, y_pred_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_min * 60,
        n_jobs=-1,
        show_progress_bar=False,
    )

    print(f"[Optuna] Número de trials: {len(study.trials)}")
    print(f"[Optuna] Mejor F1: {study.best_value:.4f}")
    print(f"[Optuna] Mejores parámetros: {study.best_params}")

    best = study.best_params

    best_params_xgb = {
        "learning_rate": best["learning_rate"],
        "n_estimators": best["n_estimators"],
        "max_depth": best["max_depth"],
        "min_child_weight": best["min_child_weight"],
        "reg_alpha": best["reg_alpha"],
        "reg_lambda": best["reg_lambda"],
    }

    best_params_ohe = {
        "min_frequency": best["min_frequency"],
    }

    return best_params_xgb, best_params_ohe


# ======================================================================
# Construcción del pipeline XGBoost
# ======================================================================

def build_xgb_best_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params_xgb: Optional[Dict[str, Any]] = None,
    params_ohe: Optional[Dict[str, Any]] = None,
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """
    Construye y ajusta el pipeline XGBoost.

    Si params_xgb / params_ohe son None, usa los hiperparámetros fijos de cfg.
    """

    drop_cols = ["customer_id", "product_id"]

    num_cols = ["X", "Y", "size", "num_deliver_per_week", "num_visit_per_week"]

    cat_cols = [
        "customer_type",
        "segment",
        "brand",
        "category",
        "sub_category",
        "zone_id",
        "region_id",
        "semana",
    ]

    for col in ["customer_type", "brand", "category", "sub_category", "segment", "package"]:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")

    if params_xgb is None:
        params_xgb = cfg.FIXED_XGB_PARAMS

    if params_ohe is None:
        params_ohe = cfg.FIXED_OHE_PARAMS

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos

    num_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("sc", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            **params_ohe,
        )),
    ])

    col_transformer = ColumnTransformer(
        [
            ("drop_ids", "drop", drop_cols),
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        verbose_feature_names_out=False,
        remainder="drop",
    )

    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        **params_xgb,
    )

    pipeline_xgb = Pipeline(
        steps=[
            ("col_transformer", col_transformer),
            ("clasificador_xgb", xgb_clf),
        ]
    )

    pipeline_xgb.fit(X_train, y_train)
    return pipeline_xgb, params_xgb, params_ohe


# ======================================================================
# Evaluación en validación + selección de umbral
# ======================================================================

def evaluate_and_select_threshold(
    pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series
) -> Tuple[float, Dict[str, float]]:
    """
    Selecciona el umbral que maximiza F1 en validación.
    """
    proba_val = pipeline.predict_proba(X_val)[:, 1]

    ths = np.linspace(cfg.THRESH_MIN, cfg.THRESH_MAX, cfg.THRESH_STEPS)
    f1_scores = []

    for t in ths:
        preds = (proba_val >= t).astype(int)
        f1_scores.append(f1_score(y_val, preds))

    best_idx = int(np.argmax(f1_scores))
    t_best = float(ths[best_idx])

    y_pred_val = (proba_val >= t_best).astype(int)

    metrics_val = {
        "f1": f1_score(y_val, y_pred_val),
        "precision": precision_score(y_val, y_pred_val, zero_division=0),
        "recall": recall_score(y_val, y_pred_val, zero_division=0),
        "accuracy": accuracy_score(y_val, y_pred_val),
    }

    print("=== Métricas en validación con umbral seleccionado ===")
    print("Umbral seleccionado:", t_best)
    print(classification_report(y_val, y_pred_val, digits=3))

    return t_best, metrics_val


# ======================================================================
# Predicciones en test
# ======================================================================

def predict_test_with_probabilities(
    pipeline: Pipeline,
    df_final: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Predicciones en test con probabilidades y ranking por cliente/semana.
    """
    proba_test = pipeline.predict_proba(X_test)[:, 1]

    meta_cols = [
        c
        for c in ["customer_id", "product_id", "week_t", "week_t_plus_1"]
        if c in df_final.columns
    ]
    meta_test = df_final.loc[X_test.index, meta_cols]

    tabla_probas = meta_test.copy()
    tabla_probas["proba_compra_t1"] = proba_test
    tabla_probas["pred_binaria"] = (proba_test >= threshold).astype(int)

    if "week_t" in tabla_probas.columns:
        tabla_probas["rank_semana"] = (
            tabla_probas.groupby(["customer_id", "week_t"])["proba_compra_t1"]
            .rank(method="dense", ascending=False)
        )

    return tabla_probas


def topN_por_cliente_semana(
    tabla_probas: pd.DataFrame, N: int = 5
) -> pd.DataFrame:
    """
    Extrae Top-N productos por cliente y semana.
    """
    if "week_t" in tabla_probas.columns:
        topN = (
            tabla_probas.sort_values(
                ["customer_id", "week_t", "proba_compra_t1"],
                ascending=[True, True, False],
            )
            .groupby(["customer_id", "week_t"], as_index=False)
            .head(N)
            .reset_index(drop=True)
        )
    else:
        topN = (
            tabla_probas.sort_values(
                ["customer_id", "proba_compra_t1"],
                ascending=[True, False],
            )
            .groupby("customer_id", as_index=False)
            .head(N)
            .reset_index(drop=True)
        )

    return topN


# ======================================================================
# Orquestador completo + MLflow + subsampling
# ======================================================================

def train_xgb_and_predict(
    df_final: pd.DataFrame,
    use_optuna: Optional[bool] = None,
    n_trials: Optional[int] = None,
    timeout_min: Optional[int] = None,
):
    """
    Orquesta todo el flujo:
      - (opcional) subsampling estratificado
      - split temporal
      - (opcional) tuning con Optuna
      - entrenamiento XGB
      - selección de umbral
      - evaluación en test
      - logging en MLflow
    """

    # Usar configuración de ambiente si no se especifica
    if use_optuna is None:
        use_optuna = cfg.USE_OPTUNA
    if n_trials is None:
        n_trials = cfg.OPTUNA_N_TRIALS
    if timeout_min is None:
        timeout_min = cfg.OPTUNA_TIMEOUT_MIN

    # ========== SUBSAMPLING (si está habilitado) ==========
    if cfg.SAMPLING_ENABLED:
        df_final = subsample_dataset(
            df_final,
            fraction=cfg.SAMPLE_FRACTION,
            stratify_col="y",
            random_state=42,
        )

    # ========== SPLIT TEMPORAL ==========
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_train_val_test_split(
        df_final
    )

    # ========== TUNING CON OPTUNA (opcional) ==========
    tuned_params_xgb = None
    tuned_params_ohe = None

    if use_optuna:
        print("[Optuna] Iniciando búsqueda de hiperparámetros...")
        tuned_params_xgb, tuned_params_ohe = tune_hyperparams_optuna(
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=n_trials,
            timeout_min=timeout_min,
        )
        print("[Optuna] Búsqueda finalizada.")
    else:
        print("[Optuna] Saltando tuning; se usan hiperparámetros fijos.")

    # ========== ENTRENAMIENTO ==========
    pipeline_xgb, params_xgb, params_ohe = build_xgb_best_pipeline(
        X_train,
        y_train,
        params_xgb=tuned_params_xgb,
        params_ohe=tuned_params_ohe,
    )

    for col in ["customer_type", "brand", "category", "sub_category", "segment", "package"]:
        for X in [X_val, X_test]:
            if col in X.columns:
                X[col] = X[col].astype("category")

    # ========== MLflow ==========
    setup_mlflow()

    with mlflow.start_run(run_name=f"xgb_{cfg.ENVIRONMENT}"):
        # Log ambiente
        mlflow.log_param("environment", cfg.ENVIRONMENT)
        mlflow.log_param("sampling_enabled", cfg.SAMPLING_ENABLED)
        mlflow.log_param("sample_fraction", cfg.SAMPLE_FRACTION)
        
        # Params del modelo
        log_params_flat(params_xgb, prefix="xgb_")
        log_params_flat(params_ohe, prefix="ohe_")

        split_info = {
            "n_train_rows": len(X_train),
            "n_val_rows": len(X_val),
            "n_test_rows": len(X_test),
            "use_optuna": use_optuna,
        }
        log_params_flat(split_info, prefix="split_")

        # Métricas val + threshold
        threshold, metrics_val = evaluate_and_select_threshold(
            pipeline_xgb, X_val, y_val
        )
        log_metrics_flat(metrics_val, prefix="val_")
        mlflow.log_param("best_threshold", threshold)

        # Métricas test
        proba_test = pipeline_xgb.predict_proba(X_test)[:, 1]
        y_pred_test = (proba_test >= threshold).astype(int)

        metrics_test = {
            "f1": f1_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test, zero_division=0),
            "recall": recall_score(y_test, y_pred_test, zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred_test),
        }
        log_metrics_flat(metrics_test, prefix="test_")

        print("=== Métricas en test con umbral seleccionado ===")
        print(classification_report(y_test, y_pred_test, digits=3))

        # Predicciones
        tabla_probas = predict_test_with_probabilities(
            pipeline_xgb, df_final, X_test, threshold
        )
        topN = topN_por_cliente_semana(
            tabla_probas, N=cfg.TOP_N_RECOMMENDATIONS
        )

        # SHAP
        sample_size = min(5000, len(X_val))
        if sample_size > 0:
            X_shap_sample = X_val.sample(n=sample_size, random_state=42)

            from scripts.shap_utils import compute_and_log_shap
            compute_and_log_shap(
                pipeline_xgb,
                X_shap_sample,
                output_dir=cfg.METRICS_DIR,
                artifact_subdir="shap",
            )

        # Log model
        log_sklearn_pipeline(pipeline_xgb, artifact_path="xgb_pipeline")

        os.makedirs(cfg.METRICS_DIR, exist_ok=True)
        topN_sample_path = os.path.join(cfg.METRICS_DIR, "topN_sample.csv")
        topN.head(1000).to_csv(topN_sample_path, index=False)
        mlflow.log_artifact(topN_sample_path, artifact_path="diagnostics")

    return pipeline_xgb, threshold, tabla_probas, topN