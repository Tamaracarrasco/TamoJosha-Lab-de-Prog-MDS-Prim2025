# airflow/config/config.py

import os

# ---------------------------------------------------------------------
# RUTAS BASE DEL PROYECTO
# ---------------------------------------------------------------------

AIRFLOW_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(AIRFLOW_DIR, "data")
MODELS_DIR = os.path.join(AIRFLOW_DIR, "models")
ARTIFACTS_DIR = os.path.join(AIRFLOW_DIR, "artifacts")
PREDICTIONS_DIR = os.path.join(ARTIFACTS_DIR, "predictions")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")

for _dir in [MODELS_DIR, ARTIFACTS_DIR, PREDICTIONS_DIR, METRICS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------
# NOMBRES DE ARCHIVOS
# ---------------------------------------------------------------------

CLIENTES_FILE = "clientes.parquet"
PRODUCTOS_FILE = "productos.parquet"
TRANSACCIONES_HIST_FILE = "transacciones.parquet"
NEW_TRANSACCIONES_PREFIX = "transacciones_"  # Prefijo para detectar nuevas semanas... entonces idealmente un nuevo archivo debe llamarse con algo similar o contener eso

# ---------------------------------------------------------------------
# COLUMNAS CLAVE
# ---------------------------------------------------------------------

ID_COL_CUSTOMER = "customer_id"
ID_COL_PRODUCT = "product_id"
DATE_COL = "purchase_date"
TARGET_COL = "y"
WEEK_COL = "semana"
WEEK_NUM_COL = "semana_num"

# ---------------------------------------------------------------------
# PARÁMETROS DEL SPLIT TEMPORAL
# ---------------------------------------------------------------------

N_TRAIN_WEEKS = 36
N_VAL_WEEKS = 11

# ---------------------------------------------------------------------
# PARÁMETROS DEL MODELO
# ---------------------------------------------------------------------

THRESH_MIN = 0.05
THRESH_MAX = 0.80
THRESH_STEPS = 40
TOP_N_RECOMMENDATIONS = 5

# ---------------------------------------------------------------------
#  ESTRATEGIA DE SUBSAMPLING Y OPTUNA POR AMBIENTE
# ---------------------------------------------------------------------

# Ambiente actual (configurar vía docker-compose)
ENVIRONMENT = os.getenv("ENVIRONMENT", "staging" ) # Modificar segun ambiente que se use "dev", "staging"  "prod" en docker-compose

# Configuración por ambiente
SAMPLING_CONFIG = {
    "dev": {
        "enabled": True,
        "sample_fraction": 0.05,      # 5% (~450K registros)
        "use_optuna": False,           # Params fijos
        "optuna_trials": 10,
        "optuna_timeout_min": 5,
    },
    "staging": {
        "enabled": True,
        "sample_fraction": 0.20,       # 20% (~1.8M registros)
        "use_optuna": True,
        "optuna_trials": 20,
        "optuna_timeout_min": 10,
    },
    "prod": {
        "enabled": False,              # SIN subsampling
        "sample_fraction": 1.0,        # 100% (9M registros)
        "use_optuna": True,
        "optuna_trials": 30,
        "optuna_timeout_min": 20,
    }
}

# Obtener configuración del ambiente actual
_config = SAMPLING_CONFIG.get(ENVIRONMENT, SAMPLING_CONFIG["staging" ])

SAMPLING_ENABLED = _config["enabled"]
SAMPLE_FRACTION = _config["sample_fraction"]
USE_OPTUNA = _config["use_optuna"]
OPTUNA_N_TRIALS = _config["optuna_trials"]
OPTUNA_TIMEOUT_MIN = _config["optuna_timeout_min"]

# ---------------------------------------------------------------------
# HIPERPARÁMETROS FIJOS (cuando USE_OPTUNA=False) -> "dev"
# ---------------------------------------------------------------------

FIXED_XGB_PARAMS = {
    "learning_rate": 0.06017425142691644,
    "n_estimators": 414,
    "max_depth": 9,
    "min_child_weight": 1,
    "reg_alpha": 0.6131484039416987,
    "reg_lambda": 0.6134461818710661,
}

FIXED_OHE_PARAMS = {
    "min_frequency": 0.13662759465397978,
}

# ---------------------------------------------------------------------
# CONFIGURACIÓN MLflow
# ---------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME = f"sodai_xgb_{ENVIRONMENT}"  # Experimentos separados por ambiente lo mismo de optuna en el fondo

# ---------------------------------------------------------------------
# CONFIGURACIÓN AIRFLOW
# ---------------------------------------------------------------------

DEFAULT_DAG_OWNER = "sodai_drinkers"
DEFAULT_DAG_RETRIES = 1
DEFAULT_DAG_RETRY_DELAY_MIN = 5

# ---------------------------------------------------------------------
# INFO DE AMBIENTE AL INICIO
# ---------------------------------------------------------------------

print("\n" + "="*70)
print(f"CONFIGURACIÓN PIPELINE SODAI - AMBIENTE: {ENVIRONMENT.upper()}")
print("="*70)
print(f"  Subsampling:       {'ACTIVO' if SAMPLING_ENABLED else 'DESACTIVADO'}")
print(f"  Fracción datos:    {SAMPLE_FRACTION*100:.1f}%")
print(f"  Usar Optuna:       {'SÍ' if USE_OPTUNA else 'NO (params fijos)'}")
if USE_OPTUNA:
    print(f"  Optuna trials:     {OPTUNA_N_TRIALS}")
    print(f"  Optuna timeout:    {OPTUNA_TIMEOUT_MIN} min")
print(f"  MLflow exp:        {MLFLOW_EXPERIMENT_NAME}")
print("="*70 + "\n")