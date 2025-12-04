# airflow/dags/sodai_xgb_pipeline_dag.py

"""
DAG principal de SodAI Drinks con XGBoost.

Acá armo el pipeline productivo con:
- Detección de nueva data
- Detección de drift con PSI
- Reentrenamiento condicional
- Predicción operativa para t+2
- Lazy imports para que el parsing del DAG sea más liviano
"""

import os
import sys
from pathlib import Path
import pickle
from datetime import datetime, timedelta

import pandas as pd

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ------------------ PYTHONPATH SETUP -------------------
DAGS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DAGS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# -------------------------------------------------------


# -------------------------------------------------------------------
# Funciones Python usadas por las tareas
# -------------------------------------------------------------------

def extract_data_task(**context):
    """
    Acá reviso que estén todos los archivos base en DATA_DIR y
    marco si hay una nueva semana de transacciones para que el
    resto del DAG sepa si hay data nueva.
    """
    import config as cfg
    import json
    
    data_files = os.listdir(cfg.DATA_DIR)
    print(f"[extract_data_task] Archivos disponibles en DATA_DIR: {data_files}")
    
    # Validar archivos requeridos
    required_files = ["clientes.parquet", "productos.parquet", "transacciones.parquet"]
    missing_files = [f for f in required_files if f not in data_files]
    
    if missing_files:
        raise FileNotFoundError(
            f"[extract_data_task] Faltan archivos requeridos: {missing_files}"
        )
    
    # Detectar TODOS los archivos de nuevas transacciones
    new_tx_files = [
        f for f in data_files 
        if f.startswith(cfg.NEW_TRANSACCIONES_PREFIX) and f.endswith(".parquet")
    ]
    
    # Ordenar para procesamiento consistente
    new_tx_files.sort()
    
    has_new_data = len(new_tx_files) > 0
    
    ti = context["ti"]
    ti.xcom_push(key="has_new_data", value=has_new_data)
    
    if has_new_data:
        # Guardar LISTA de archivos (no solo el primero)
        new_tx_files_json = json.dumps(new_tx_files)
        Variable.set("NEW_TX_FILES_LIST", new_tx_files_json)
        
        print(f"[extract_data_task] Nueva data detectada: {new_tx_files}")
        print(f"[extract_data_task] Total de archivos nuevos: {len(new_tx_files)}")
        print(f"[extract_data_task] Variable NEW_TX_FILES_LIST seteada")
    else:
        # Limpiar la variable si no hay nueva data
        try:
            Variable.delete("NEW_TX_FILES_LIST")
        except:
            pass
        print("[extract_data_task]  No hay nueva data (solo histórico)")
    
    print("[extract_data_task]  Todos los archivos requeridos están presentes.")


def transform_data_task(**context):
    """
    Con esta tarea armo df_final a partir de los parquet crudos:
    cargo el histórico, si corresponde sumo la nueva semana (t+1),
    limpio/transformo los datos, construyo el panel semanal con
    el target y guardo df_final_latest.parquet.
    """
    import config as cfg
    import json
    from scripts.data_io import build_dataset_from_raw
    
    DF_FINAL_PATH = os.path.join(cfg.DATA_DIR, "df_final_latest.parquet")
    
    # Obtener LISTA de archivos de transacciones nuevas (si existe)
    try:
        new_tx_files_json = Variable.get("NEW_TX_FILES_LIST", default_var=None)
        if new_tx_files_json:
            new_transactions_files = json.loads(new_tx_files_json)
        else:
            new_transactions_files = None
    except:
        new_transactions_files = None
    
    print(f"[transform_data_task] Archivos nuevos: {new_transactions_files}")
    if new_transactions_files:
        print(f"[transform_data_task] Total archivos a procesar: {len(new_transactions_files)}")

    # Construir dataset (ahora acepta lista)
    df_final = build_dataset_from_raw(
        data_dir=cfg.DATA_DIR,
        new_transactions_filename=new_transactions_files,
    )
    
    # Guardar
    os.makedirs(os.path.dirname(DF_FINAL_PATH), exist_ok=True)
    df_final.to_parquet(DF_FINAL_PATH, index=False)

    # XCom push
    ti = context["ti"]
    ti.xcom_push(key="df_final_path", value=DF_FINAL_PATH)

    print(
        f"[transform_data_task]  df_final guardado en {DF_FINAL_PATH}. "
        f"Shape: {df_final.shape}"
    )
    return DF_FINAL_PATH


def check_drift(**context):
    """
    Acá reviso si hay drift usando PSI: comparo la última semana
    contra el histórico y, si el PSI pasa el umbral en alguna
    feature numérica, marco drift = True y guardo un pequeño reporte.
    """
    import config as cfg
    from scripts.drift import detect_drift_last_week

    ti = context["ti"]
    df_final_path = ti.xcom_pull(
        task_ids="transform_data_task",
        key="df_final_path",
    )

    if df_final_path is None or not os.path.exists(df_final_path):
        raise ValueError(
            f"[check_drift_task] No se encontró df_final_path: {df_final_path}"
        )

    print(f"[check_drift_task] Cargando df_final desde {df_final_path}")

    # Leer solo columnas necesarias para drift
    drift_cols = [
        "semana_num",
        "X", "Y", "size",
        "num_deliver_per_week", "num_visit_per_week",
        "purchased_count", "compra_o_no",
    ]

    try:
        df_final = pd.read_parquet(df_final_path, columns=drift_cols)
        print(
            "[check_drift_task] df_final cargado para drift "
            f"con shape: {df_final.shape}"
        )
    except Exception as e:
        print(
            f"[check_drift_task]  Error al filtrar columnas ({e}). "
            "Se carga df_final completo."
        )
        df_final = pd.read_parquet(df_final_path)
        print(
            "[check_drift_task] df_final completo cargado con "
            f"shape: {df_final.shape}"
        )

    # Features numéricas para PSI
    numeric_features = [
        "X", "Y", "size",
        "num_deliver_per_week", "num_visit_per_week",
    ]
    numeric_features = [f for f in numeric_features if f in df_final.columns]

    # ========== DETECCIÓN DE DRIFT ==========
    drift_flag, report = detect_drift_last_week(
        df_final,
        num_cols=numeric_features,
        psi_threshold=0.2,
        buckets=10,
    )

    # XCom push
    ti.xcom_push(key="drift_detected", value=drift_flag)
    ti.xcom_push(key="drift_report", value=report)

    if drift_flag:
        print("[check_drift_task] DRIFT DETECTADO")
    else:
        print("[check_drift_task]  Sin drift")

    print(f"[check_drift_task] Reporte PSI: {report}")

    return drift_flag


def branch_on_drift(**context):
    """
    Acá decido si reentreno o no. Miro tres cosas:
    - Si ya existe un modelo guardado
    - Si llegó data nueva
    - Si detecté drift

    Según eso mando el flujo a retrain_model_task o a skip_retrain.
    """
    import os
    import config as cfg

    MODEL_PATH = os.path.join(cfg.MODELS_DIR, "xgb_best_model.pkl")

    ti = context["ti"]
    
    # Pull de flags
    has_new_data = ti.xcom_pull(
        task_ids="extract_data_task",
        key="has_new_data",
    )
    
    drift_flag = ti.xcom_pull(
        task_ids="check_drift_task",
        key="drift_detected",
    )

    model_exists = os.path.exists(MODEL_PATH)

    print("\n" + "="*60)
    print("EVALUACIÓN DE REENTRENAMIENTO")
    print("="*60)
    print(f"  Modelo existe: {model_exists}")
    print(f"  Nueva data:    {has_new_data}")
    print(f"  Drift:         {drift_flag}")
    print("="*60 + "\n")

    # ========== DECISIÓN ==========
    
    # Caso 1: Primera vez (no existe modelo)
    if not model_exists:
        print("[branch_on_drift]  No existe modelo previo → REENTRENAR")
        return "retrain_model_task"

    # Caso 2: Hay nueva data
    if has_new_data:
        print("[branch_on_drift] Nueva data detectada → REENTRENAR")
        return "retrain_model_task"

    # Caso 3: Drift detectado
    if drift_flag:
        print("[branch_on_drift]  Drift detectado → REENTRENAR")
        return "retrain_model_task"

    # Caso 4: Todo igual, modelo vigente
    print("[branch_on_drift]  Modelo vigente (sin cambios) → SKIP")
    return "skip_retrain"


def retrain_model(**context):
    """
    Cuando toca reentrenar, cargo df_final completo, hago el split
    temporal, entreno XGBoost con Optuna, busco el mejor umbral,
    evalúo en test, registro todo en MLflow y guardo el modelo con
    su threshold y las predicciones del set de test.
    """
    import config as cfg
    from scripts.modeling_xgb import train_xgb_and_predict
    from scripts.data_io import save_predictions

    MODEL_PATH = os.path.join(cfg.MODELS_DIR, "xgb_best_model.pkl")
    PRED_BASE_PATH = os.path.join(cfg.PREDICTIONS_DIR, "predicciones_test.parquet")

    ti = context["ti"]
    df_final_path = ti.xcom_pull(
        task_ids="transform_data_task",
        key="df_final_path",
    )

    if df_final_path is None or not os.path.exists(df_final_path):
        raise FileNotFoundError(
            f"[retrain_model_task]  No se encontró df_final: {df_final_path}"
        )

    # ========== CARGA COMPLETA DEL DATASET ==========
    print(f"[retrain_model_task] Cargando df_final COMPLETO desde {df_final_path}")
    df_final = pd.read_parquet(df_final_path)
    print(f"[retrain_model_task] Shape total: {df_final.shape}")

    # ========== ENTRENAMIENTO CON OPTUNA ==========
    print("[retrain_model_task]  Iniciando entrenamiento con Optuna...")

    model, threshold, tabla_probas, topN = train_xgb_and_predict(
        df_final,
        use_optuna=True,
        n_trials=cfg.OPTUNA_N_TRIALS,
        timeout_min=cfg.OPTUNA_TIMEOUT_MIN,
    )

    # ========== SERIALIZACIÓN ==========
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "threshold": threshold}, f)

    print(f"[retrain_model_task]  Modelo guardado en {MODEL_PATH}")
    print(f"[retrain_model_task]  Threshold óptimo: {threshold:.4f}")

    # ========== GUARDAR PREDICCIONES TEST ==========
    save_predictions(tabla_probas, PRED_BASE_PATH)
    print(f"[retrain_model_task]  Predicciones test en {PRED_BASE_PATH}")

    # ========== XCOM ==========
    ti.xcom_push(key="threshold", value=threshold)
    ti.xcom_push(key="model_path", value=MODEL_PATH)
    ti.xcom_push(key="preds_path", value=PRED_BASE_PATH)


def predict_next_week(**context):
    """
    Con esta tarea genero las predicciones operativas para la semana t+2:
    cargo el modelo y su umbral, armo los candidatos desde los datos crudos,
    calculo probabilidades, aplico el threshold, rankeo por cliente y
    guardo predicciones_finales.parquet.
    """
    import config as cfg
    import json
    from scripts.data_preparation import build_next_week_candidates_from_raw
    
    MODEL_PATH = os.path.join(cfg.MODELS_DIR, "xgb_best_model.pkl")
    PRED_FINAL_PATH = os.path.join(cfg.PREDICTIONS_DIR, "predicciones_finales.parquet")
    
    # ========== CARGA DE MODELO ==========
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"[predict_next_week_task]  No existe modelo en {MODEL_PATH}. "
            "Ejecuta retrain_model_task primero."
        )

    with open(MODEL_PATH, "rb") as f:
        model_pack = pickle.load(f)

    model = model_pack["model"]
    threshold = model_pack["threshold"]

    print(
        f"[predict_next_week_task]  Modelo cargado. "
        f"Umbral: {threshold:.4f}"
    )

    # ========== CONSTRUCCIÓN DE CANDIDATOS t+2 ==========
    try:
        new_tx_files_json = Variable.get("NEW_TX_FILES_LIST", default_var=None)
        if new_tx_files_json:
            new_tx_files = json.loads(new_tx_files_json)
        else:
            new_tx_files = None
    except:
        new_tx_files = None
    
    print(f"[predict_next_week_task]  Archivos nueva semana: {new_tx_files}")

    df_candidates = build_next_week_candidates_from_raw(
        data_dir=cfg.DATA_DIR,
        new_transactions_filename=new_tx_files,
    )

    print(
        f"[predict_next_week_task]  Candidatos t+2 construidos. "
        f"Shape: {df_candidates.shape}"
    )

    # ========== PREDICCIÓN ==========
    proba = model.predict_proba(df_candidates)[:, 1]

    df_pred = df_candidates.copy()
    df_pred["proba_compra_t2"] = proba
    df_pred["pred_binaria"] = (proba >= threshold).astype(int)

    # Ranking por cliente
    if "customer_id" in df_pred.columns:
        df_pred["rank_cliente"] = (
            df_pred.groupby("customer_id")["proba_compra_t2"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

    # ========== GUARDAR ==========
    os.makedirs(os.path.dirname(PRED_FINAL_PATH), exist_ok=True)
    df_pred.to_parquet(PRED_FINAL_PATH, index=False)

    print(f"[predict_next_week_task]  Predicciones t+2 en {PRED_FINAL_PATH}")
    print(f"[predict_next_week_task] Muestra:\n{df_pred.head(10)}")
    
    # Estadísticas
    n_positivos = (df_pred["pred_binaria"] == 1).sum()
    print(
        f"[predict_next_week_task] Predicciones positivas: "
        f"{n_positivos:,} de {len(df_pred):,} "
        f"({100*n_positivos/len(df_pred):.2f}%)"
    )


# -------------------------------------------------------------------
# Definición del DAG
# -------------------------------------------------------------------

import config as cfg

default_args = {
    "owner": cfg.DEFAULT_DAG_OWNER,
    "retries": cfg.DEFAULT_DAG_RETRIES,
    "retry_delay": timedelta(minutes=cfg.DEFAULT_DAG_RETRY_DELAY_MIN),
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="sodai_xgb_pipeline",
    description="Pipeline productivo SodAI Drinks - XGBoost con drift detection",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["sodai", "xgboost", "mlops", "drift-detection"],
) as dag:

    # ========== TAREAS ==========
    
    extract_data = PythonOperator(
        task_id="extract_data_task",
        python_callable=extract_data_task,
        doc_md="""
        ### Extract Data
        Acá solo reviso que estén los archivos base y marco si llegó data nueva.
        """,
    )

    transform_data = PythonOperator(
        task_id="transform_data_task",
        python_callable=transform_data_task,
        doc_md="""
        ### Transform Data
        A partir de los parquet crudos armo df_final y lo guardo listo para modelar.
        """,
    )

    check_drift_task = PythonOperator(
        task_id="check_drift_task",
        python_callable=check_drift,
        doc_md="""
        ### Check Drift
        Reviso si las distribuciones cambiaron usando PSI y marco si hay drift.
        """,
    )

    branch_task = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=branch_on_drift,
        doc_md="""
        ### Branch Logic
        Con esta lógica decido si reentreno o no, mirando:
        1. Si ya hay un modelo guardado
        2. Si llegó data nueva
        3. Si detecté drift
        """,
    )

    retrain_model_task = PythonOperator(
        task_id="retrain_model_task",
        python_callable=retrain_model,
        doc_md="""
        ### Retrain Model
        Reentreno XGBoost con Optuna, registro en MLflow y guardo modelo + preds de test.
        """,
    )

    skip_retrain = EmptyOperator(
        task_id="skip_retrain",
        doc_md="""
        ### Skip Retrain
        No reentreno porque el modelo sigue vigente (sin cambios relevantes).
        """,
    )

    predict_next_week_task = PythonOperator(
        task_id="predict_next_week_task",
        python_callable=predict_next_week,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        doc_md="""
        ### Predict Next Week
        Genero las predicciones operativas para t+2 y las dejo guardadas en parquet.
        """,
    )

    # ========== DEPENDENCIAS ==========
    extract_data >> transform_data >> check_drift_task >> branch_task

    branch_task >> retrain_model_task >> predict_next_week_task
    branch_task >> skip_retrain >> predict_next_week_task