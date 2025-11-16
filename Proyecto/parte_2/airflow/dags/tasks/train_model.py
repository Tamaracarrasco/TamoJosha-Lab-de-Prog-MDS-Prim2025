"""
En este scripts se intentará entrenar el modelo
pero no prometo nada
tengo sueño
- Pero de lo que se entiende: Igual se pueden usar los mejores hiperarametros
que se encontraron en la parte 1: o se puede correr de nuevo la parte 1 con
una nueva definición de los conjuntos de entrenamiento/validación y testeo.

- En todo caso, debe haber un reentrenamiento periodico y por lo tanto
se debe optimizar nuevamente el modelo.

Acá una diferencia: no se optimizarán parámetros del onehot encoder.
por ahora solo xgbclassifier..

- los hiperparámetros de van a optimizar de acuerdo a la métrica F1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime
import json
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import mlflow
import mlflow.xgboost
import shap
import matplotlib.pyplot as plt
import scipy.sparse as sp

from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Definir rutas base del proyecto

BASE_PATH = Path(__file__).parent.parent.parent.parent
PROCESSED_PATH = BASE_PATH / "data" / "run" / "processed"
MODELS_PATH = BASE_PATH / "models" / "saved_models"
REPORTS_PATH = BASE_PATH / "reports"
MLFLOW_PATH = BASE_PATH / "mlflow"


def setup_mlflow():
    """Configura MLflow tracking"""
    mlflow_uri = MLFLOW_PATH / "mlruns"
    mlflow_uri.mkdir(parents=True, exist_ok=True)
    
    mlflow.set_tracking_uri(f"file://{mlflow_uri.absolute()}")
    mlflow.set_experiment("customer_purchase_prediction")
    
    logger.info(f"MLflow configurado en: {mlflow_uri}")


def load_transformed_data() -> Tuple:
    """
    Carga datos transformados
    
    Returns:
    ------------
    tupla (X_train, y_train, X_val, y_val)
    """
    logger.info("== Cargando datos transformados ==")
    
    try:
        # Cargar X_train
        X_train_path = PROCESSED_PATH / "X_train_transformed.npz"
        try:
            X_train = sp.load_npz(X_train_path)
        except:
            data = np.load(X_train_path)
            X_train = data['data']
        
        # Cargar X_val
        X_val_path = PROCESSED_PATH / "X_val_transformed.npz"
        try:
            X_val = sp.load_npz(X_val_path)
        except:
            data = np.load(X_val_path)
            X_val = data['data']
        
        # Cargar y
        y_train = pd.read_parquet(PROCESSED_PATH / "y_train.parquet")['y'].values
        y_val = pd.read_parquet(PROCESSED_PATH / "y_val.parquet")['y'].values
        
        logger.info(f"✓ X_train: {X_train.shape}")
        logger.info(f"✓ y_train: {y_train.shape}")
        logger.info(f"✓ X_val: {X_val.shape}")
        logger.info(f"✓ y_val: {y_val.shape}")
        
        return X_train, y_train, X_val, y_val
        
    except Exception as e:
        logger.error(f"Error al cargar datos transformados: {e}")
        raise


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    Calcula scale_pos_weight para manejar desbalance
    ya que el modelo que escogimos no presenta ese hiperparámetro
    "class_weight" como lo de los otros modelos.
    
    Parámetros:
    --------------
    y_train: Array. Variable target
    
    Returns:
    -----------
    scale_pos_weight 
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    logger.info(f"\nBalance de clases:")
    logger.info(f"  - Negativos (y=0): {n_neg:,} ({n_neg/len(y_train)*100:.2f}%)")
    logger.info(f"  - Positivos (y=1): {n_pos:,} ({n_pos/len(y_train)*100:.2f}%)")
    logger.info(f"  - scale_pos_weight: {scale_pos_weight:.4f}")
    
    return scale_pos_weight


def get_baseline_params() -> Dict:
    """
    Retorna hiperparámetros base encontrados en la parte 1
    Estos servirán como punto de partida para Optuna
    
    Returns:
        Diccionario con parámetros base
    """
    # Ajusta estos valores según los mejores de tu parte 1
    baseline_params = {'learning_rate': 0.06017425142691644, 
                        'n_estimators': 414, 
                        'max_depth': 9, 
                        'min_child_weight': 1, 
                        'reg_alpha': 0.6131484039416987, 
                        'reg_lambda': 0.6134461818710661}
    
    logger.info("\nParámetros baseline (de parte 1):")
    for key, val in baseline_params.items():
        logger.info(f"  {key}: {val}")
    
    return baseline_params


def optimize_hyperparameters(
                            X_train, y_train, X_val, y_val,
                            scale_pos_weight: float,
                            n_trials: int = 50
                        ) -> Dict:
    """
    Optimiza hiperparámetros con Optuna maximizando F1-score
    
    Parámetros:
    ------------
    X_train, y_train: Datos de entrenamiento
    X_val, y_val: Datos de validación
    scale_pos_weight: Peso para balancear clases
    n_trials: Número de trials de Optuna
    
    Returns:
    ------------
    Diccionario con mejores parámetros
    """
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZANDO HIPERPARÁMETROS CON OPTUNA")
    logger.info("="*70)
    
    def objective(trial):
        # Espacio de búsqueda
        params = params_xgb = {
                                "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
                                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                                "max_depth": trial.suggest_int("max_depth", 3, 10),
                                "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
                                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1)
                                }
        
        # Entrenar modelo

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predecir
        y_pred = model.predict(X_val)
        
        # Calcular F1-score
        f1 = f1_score(y_val, y_pred)
        
        return f1
    
    # Estudio de optuna
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='xgb_f1_optimization'
    )
    
    # Optimizar
    logger.info(f"Iniciando optimización con {n_trials} trials.")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Resultados
    logger.info("\n" + "="*70)
    logger.info("RESULTADOS DE OPTIMIZACIÓN")
    logger.info("="*70)
    logger.info(f"Mejor F1-score: {study.best_value:.4f}")
    logger.info(f"Mejores hiperparámetros:")
    for key, val in study.best_params.items():
        logger.info(f"  {key}: {val}")
    
    # Agregar parámetros fijos
    best_params = study.best_params.copy()
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['tree_method'] = 'hist'
    best_params['eval_metric'] = 'logloss'
    
    return best_params, study


def train_final_model(X_train, y_train, X_val, y_val, params: Dict) -> xgb.XGBClassifier:
    """
    Entrena el modelo final con los mejores hiperparámetros
    
    Parámetros:
    -------------
    X_train, y_train: Datos de entrenamiento
    X_val, y_val: Datos de validación
    params: Hiperparámetros
    
    Returns:
    -----------
    Modelo entrenado
    """
    logger.info("\n" + "="*70)
    logger.info("ENTRENANDO MODELO FINAL")
    logger.info("="*70)
    
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )
    
    logger.info("Modelo entrenado SIUUUUUUUUU")
    
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val) -> Dict:
    """
    Evalúa el modelo en train y validación
    
    Parámetros:
    ---------------
    model: Modelo entrenado
    X_train, y_train: Datos de entrenamiento
    X_val, y_val: Datos de validación
    
    Returns:
    -----------
    Diccionario con métricas
    """
    logger.info("\n" + "="*70)
    logger.info("EVALUANDO MODELO")
    logger.info("="*70)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Métricas Train
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_proba)
    }
    
    # Métricas Val
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall': recall_score(y_val, y_val_pred),
        'f1': f1_score(y_val, y_val_pred),
        'roc_auc': roc_auc_score(y_val, y_val_proba)
    }
    
    # Log resultados
    logger.info("\nMÉTRICAS EN TRAIN:")
    for metric, value in train_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nMÉTRICAS EN VALIDACIÓN:")
    for metric, value in val_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Matriz de confusión
    cm_val = confusion_matrix(y_val, y_val_pred)
    logger.info(f"\nMatriz de confusión (Validación):\n{cm_val}")
    
    return {
        'train': train_metrics,
        'val': val_metrics,
        'confusion_matrix_val': cm_val.tolist()
    }


def create_feature_importance_plots(model, output_dir: Path) -> Dict[str, Path]:
    """
    Crea gráficos de feature importance
    
    Parámetros:
    --------------
    model: Modelo entrenado
    output_dir: Directorio de salida
    
    Returns:
    ------------
    Diccionario con rutas de gráficos
    """
    logger.info("\nCreando gráficos de feature importance...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    
    # Importance por gain, weight y cover
    for importance_type in ['gain', 'weight', 'cover']:
        fig, ax = plt.subplots(figsize=(10, 12))
        xgb.plot_importance(
            model,
            ax=ax,
            importance_type=importance_type,
            max_num_features=30,
            title=f'Feature Importance ({importance_type.capitalize()})'
        )
        plt.tight_layout()
        
        path = output_dir / f"feature_importance_{importance_type}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico guardado: {path.name}")
        paths[f'importance_{importance_type}'] = path
    
    return paths


def create_shap_summary_plot(model, X_val, y_val, output_dir: Path, sample_size: int = 20000) -> Path:
    """
    Crea summary plot de SHAP values
    
    Parámetros:
    -------------
    model: Modelo entrenado
    X_val: Features de validación
    y_val: Target de validación
    output_dir: Directorio de salida
    sample_size: Tamaño de muestra para SHAP (por costo computacional)
    
    Returns:
    --------------
    Path del gráfico
    """
    logger.info(f"\nCreando SHAP summary plot (muestra de {sample_size:,} registros)...")
    
    try:
        # Tomar muestra aleatoria
        n_samples = min(sample_size, X_val.shape[0])
        np.random.seed(42)
        sample_idx = np.random.choice(X_val.shape[0], n_samples, replace=False)
        
        if sp.issparse(X_val):
            X_sample = X_val[sample_idx].toarray()
        else:
            X_sample = X_val[sample_idx]
        
        # Calcular SHAP values
        logger.info("Calculando SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Crear plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values,
            X_sample,
            #plot_type="dot",
            show=False,
            max_display=30
        )
        
        path = output_dir / "shap_summary_plot.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP plot guardado: {path.name}")
        
        return path
        
    except Exception as e:
        logger.warning(f"No se pudo crear SHAP plot: {e}")
        return None


def log_to_mlflow(
    model,
    params: Dict,
    metrics: Dict,
    plots: Dict,
    shap_plot: Path,
    f1_val: float
):
    """
    Registra todo en MLflow
    
    Parámetros:
    --------------
    model: Modelo entrenado
    params: Hiperparámetros
    metrics: Métricas de evaluación
    plots: Diccionario con rutas de gráficos
    shap_plot: Ruta del SHAP plot
    f1_val: F1-score en validación
    """
    logger.info("\n" + "="*70)
    logger.info("REGISTRANDO EN MLFLOW")
    logger.info("="*70)
    
    # Crear run name identificable
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_f1_{f1_val:.4f}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parámetros
        mlflow.log_params(params)
        
        # Log métricas de train
        for metric, value in metrics['train'].items():
            mlflow.log_metric(f"train_{metric}", value)
        
        # Log métricas de val
        for metric, value in metrics['val'].items():
            mlflow.log_metric(f"val_{metric}", value)
        
        # Log modelo
        mlflow.xgboost.log_model(model, "model")
        
        # Log gráficos de feature importance
        for name, path in plots.items():
            mlflow.log_artifact(str(path), artifact_path="plots")
        
        # Log SHAP plot
        if shap_plot and shap_plot.exists():
            mlflow.log_artifact(str(shap_plot), artifact_path="plots")
        
        # Log metadata adicional
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("optimization", "Optuna")
        mlflow.set_tag("objective", "maximize_f1")
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Experimento registrado con ID: {run_id}")
        logger.info(f"Run name: {run_name}")


def save_model(model, params: Dict, metrics: Dict, f1_val: float) -> Path:
    """
    Guarda el modelo final
    
    Parámetros:
    -------------
    model: Modelo entrenado
    params: Hiperparámetros
    metrics: Métricas
    f1_val: F1-score en validación
    
    Returns:
    -----------
    Path del modelo guardado
    """
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgb_model_{timestamp}_f1_{f1_val:.4f}.pkl"
    model_path = MODELS_PATH / model_filename
    
    # Guardar modelo
    joblib.dump(model, model_path)
    
    # Guardar metadata
    metadata = {
        'timestamp': timestamp,
        'f1_score_val': f1_val,
        'params': params,
        'metrics': metrics
    }
    
    metadata_path = MODELS_PATH / f"model_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\n Modelo guardado: {model_path.name}")
    logger.info(f"Metadata guardado: {metadata_path.name}")
    
    return model_path


def train_model(**kwargs) -> dict:
    """
    Función principal que ejecuta todo el entrenamiento.
    
    Parámetros:
    --------------
    **kwargs: Argumentos adicionales de Airflow (ti, execution_date, etc.)
    
    Returns:
    -----------
    dict: Diccionario con información del proceso (para XCom en Airflow)
    """
    logger.info("\n" + "="*70)
    logger.info("INICIANDO ENTRENAMIENTO DE MODELO")
    logger.info("="*70)
    
    try:
        # 1. Setup MLflow
        setup_mlflow()
        
        # 2. Cargar datos
        X_train, y_train, X_val, y_val = load_transformed_data()
        
        # 3. Calcular scale_pos_weight
        scale_pos_weight = calculate_scale_pos_weight(y_train)
        
        # 4. Optimizar hiperparámetros
        best_params, study = optimize_hyperparameters(
            X_train, y_train, X_val, y_val,
            scale_pos_weight,
            n_trials=50  # Ajustar según necesidad
        )
        
        # 5. Entrenar modelo final
        model = train_final_model(X_train, y_train, X_val, y_val, best_params)
        
        # 6. Evaluar modelo
        metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
        
        # 7. Crear gráficos de feature importance
        plots_dir = REPORTS_PATH / "model_performance"
        plots = create_feature_importance_plots(model, plots_dir)
        
        # 8. Crear SHAP plot
        shap_plot = create_shap_summary_plot(model, X_val, y_val, plots_dir)
        
        # 9. Registrar en MLflow
        f1_val = metrics['val']['f1']
        log_to_mlflow(model, best_params, metrics, plots, shap_plot, f1_val)
        
        # 10. Guardar modelo
        model_path = save_model(model, best_params, metrics, f1_val)
        
        logger.info("\n" + "="*70)
        logger.info("ENTRENAMIENTO COMPLETADO SIUUUUUUUUU")
        logger.info("="*70)
        
        # Retornar información para XCom de Airflow
        return {
            'status': 'success',
            'model_path': str(model_path),
            'f1_score_train': metrics['train']['f1'],
            'f1_score_val': metrics['val']['f1'],
            'best_params': best_params,
            'n_optuna_trials': len(study.trials),
            'execution_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}:( revisa pls")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': str(e),
            'execution_time': datetime.now().isoformat()
        }


if __name__ == "__main__":
    """Ejecución directa del script"""
    result = train_model()
    print("\n" + "="*70)
    print("RESULTADO:", result)
    print("="*70)