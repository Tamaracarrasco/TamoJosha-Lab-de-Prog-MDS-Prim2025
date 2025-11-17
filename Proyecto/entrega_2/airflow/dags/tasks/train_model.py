"""
Script para entrenamiento de modelo con XGBoost, Optuna y MLflow
Incluye optimización de hiperparámetros, tracking y análisis SHAP

- De este script se espera hacer lo siguiente:

- Entrenar el modelo
- En caso de  reentrenamiento que se optimice
- Registrar todo con MLFLOW y guardar métricas.
- Generar gráficos de interpretabilidad:
    - shap values con una muestra reducida: quizas 20_000 para el conjunto de val.
    - métodos globales de xgboost (feature importance)
- por un tema de costo computacional: n_trial  fijado a 20 y timeout fijado
a 5 minutos para un estudio de optuna.
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
    
    # Convertir a URI compatible con MLflow en windows
    # Usar pathlib para manejar las barras correctamente
    tracking_uri = mlflow_uri.absolute().as_uri()
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("customer_purchase_prediction")
    
    logger.info(f"MLflow configurado en: {tracking_uri}")


def load_transformed_data() -> Tuple:
    """
    Carga datos transformados y el preprocessor para obtener nombres de features
    
    Returns:
    ------------
    Tupla (X_train, y_train, X_val, y_val, feature_names)
    """
    logger.info("Cargando datos transformados--")
    
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
        
        # Cargar preprocessor para obtener nombres de features
        preprocessor_path = MODELS_PATH / "preprocessor.pkl"
        try:
            import joblib
            preprocessor = joblib.load(preprocessor_path)
            feature_names = preprocessor.get_feature_names_out()
            logger.info(f"Nombres de features obtenidos: {len(feature_names)} features")
        except Exception as e:
            logger.warning(f"No se pudieron obtener nombres de features: {e}")
            feature_names = None
        
        logger.info(f"X_train: {X_train.shape}")
        logger.info(f"y_train: {y_train.shape}")
        logger.info(f"X_val: {X_val.shape}")
        logger.info(f"y_val: {y_val.shape}")
        
        return X_train, y_train, X_val, y_val, feature_names
        
    except Exception as e:
        logger.error(f"Error al cargar datos transformados: {e}")
        raise


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    Calcula scale_pos_weight para manejar desbalance (equivalente a class_weight='balanced')
    
    Parámetros:
    -------------
    y_train: Array de la variable target
    
    Returns:
    ------------
    scale_pos_weight value
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    
    logger.info(f"\n Balance de clases:")
    logger.info(f"  - Negativos (y=0): {n_neg:,} ({n_neg/len(y_train)*100:.2f}%)")
    logger.info(f"  - Positivos (y=1): {n_pos:,} ({n_pos/len(y_train)*100:.2f}%)")
    logger.info(f"  - scale_pos_weight: {scale_pos_weight:.4f}")
    
    return scale_pos_weight


def get_baseline_params() -> Dict:
    """
    Retorna hiperparámetros base encontrados en la parte 1
    Estos servirán como punto de partida para Optuna
    
    Returns:
    -------------
    Diccionario con parámetros base
    """
    # Hiperparámetros dps de ejecutar parte 1
    baseline_params = {'learning_rate': 0.06017425142691644, 
                        'n_estimators': 414, 
                        'max_depth': 9, 
                        'min_child_weight': 1, 
                        'reg_alpha': 0.6131484039416987, 
                        'reg_lambda': 0.6134461818710661}
    
    logger.info("\n Parámetros baseline (de parte 1):")
    for key, val in baseline_params.items():
        logger.info(f"  {key}: {val}")
    
    return baseline_params


def optimize_hyperparameters(
    X_train, y_train, X_val, y_val,
    scale_pos_weight: float,
    n_trials: int = 20
) -> Dict:
    """
    Optimiza hiperparámetros con Optuna maximizando F1-score
    
    Parámetros:
    --------------
    X_train, y_train: Datos de entrenamiento
    X_val, y_val: Datos de validación
    scale_pos_weight: Peso para balancear clases
    n_trials: Número de trials de Optuna
    
    Returns:
    ----------
    Diccionario con mejores parámetros
    """
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZANDO HIPERPARÁMETROS CON OPTUNA")
    logger.info("="*70)
    
    def objective(trial):
        # Espacio de búsqueda de la parte 1
        params = {
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
    
    # Crear estudio de Optuna
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='xgb_f1_optimization'
    )
    
    # Optimizar
    logger.info(f"Iniciando optimización con {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, timeout=5*60)
    
    # Resultados
    logger.info("\n" + "="*70)
    logger.info("RESULTADOS DE OPTIMIZACIÓN")
    logger.info("="*70)
    logger.info(f"Mejor F1-score: {study.best_value:.4f}")
    logger.info(f"Mejores hiperparámetros:")
    for key, val in study.best_params.items():
        logger.info(f"  {key}: {val}")
    
    # Parámetros fijos
    best_params = study.best_params.copy()
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['tree_method'] = 'hist'
    best_params['eval_metric'] = 'logloss'
    
    return best_params, study


def train_final_model(X_train, y_train, X_val, y_val, params: Dict, feature_names=None) -> xgb.XGBClassifier:
    """
    Entrena el modelo final con los mejores hiperparámetros
    
    Parámetros:
    --------------
    X_train, y_train: Datos de entrenamiento
    X_val, y_val: Datos de validación
    params: Hiperparámetros
    
    Returns:
    ----------
    Modelo entrenado
    """
    logger.info("\n" + "="*70)
    logger.info("ENTRENANDO MODELO FINAL")
    logger.info("="*70)
    
    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=0,
    )
    if feature_names is not None:
        model.get_booster().feature_names = list(feature_names)
    
    logger.info("Modelo entrenado exitosamente")
    
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val) -> Dict:
    """
    Evalúa el modelo en train y validación
    
    Parámetros:
    --------------
    model: Modelo entrenado
    X_train, y_train: Datos de entrenamiento
    X_val, y_val: Datos de validación
    
    Returns:
    ------------
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


def create_feature_importance_plots(model, output_dir: Path, feature_names=None) -> Dict[str, Path]:
    """
    Crea gráficos de feature importance
    
    Parámetros:
    --------------
    model: Modelo entrenado
    output_dir: Directorio de salida
    feature_names: Nombres reales de las features
    
    Returns:
    --------------
    Diccionario con rutas de gráficos
    """
    logger.info("\nCreando gráficos de feature importance...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    
    
    if feature_names is not None and len(feature_names) > 0:
        feature_mapping = {f'f{i}': name for i, name in enumerate(feature_names)}
    else:
        feature_mapping = {}
    
    # Importance por gain, weight y cover
    for importance_type in ['gain', 'weight', 'cover']:
        fig, ax = plt.subplots(figsize=(12, 14))
        
        # Se obtienen las  importancias
        importance_dict = model.get_booster().get_score(importance_type=importance_type)
        
        if importance_dict:

            # Si tenemos nombres de features personalizados, mapearlos
            if feature_names is not None and len(feature_names) > 0:

                # Crear mapeo de f0, f1, etc. a nombres reales
                feature_mapping = {f'f{i}': name for i, name in enumerate(feature_names)}
                importance_dict = {
                    feature_mapping.get(k, k): v 
                    for k, v in importance_dict.items()
                }
            
            # Ordenar por importancia
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            # Se toman las top 30
            top_features = sorted_importance[:30]
            features, scores = zip(*top_features)
            
            # Gráfico
            y_pos = np.arange(len(features))
            ax.barh(y_pos, scores)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features, fontsize=9)
            ax.set_xlabel('Importance', fontsize=11)
            ax.set_title(f'Feature Importance ({importance_type.capitalize()})', fontsize=13, fontweight='bold')
            ax.invert_yaxis()
            
        plt.tight_layout()
        
        path = output_dir / f"feature_importance_{importance_type}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico guardado: {path.name}")
        paths[f'importance_{importance_type}'] = path
    
    return paths


def create_shap_summary_plot(model, X_val, y_val, output_dir: Path, feature_names=None, sample_size: int = 20000) -> Path:
    """
    Crea summary plot de SHAP values
    
    Parámetros:
    -------------
    model: Modelo entrenado
    X_val: Features de validación
    y_val: Target de validación
    output_dir: Directorio de salida
    feature_names: Nombres reales de las features
    sample_size: Tamaño de muestra para SHAP (por costo computacional)
    
    """
    logger.info(f"\n Creando SHAP summary plot (muestra de {sample_size:,} registros)...")
    
    try:
        # Tomar muestra aleatoria
        n_samples = min(sample_size, X_val.shape[0])
        np.random.seed(42)
        sample_idx = np.random.choice(X_val.shape[0], n_samples, replace=False)
        
        if sp.issparse(X_val):

            X_sample = X_val[sample_idx].toarray()
        else:
            X_sample = X_val[sample_idx]
        
        # Se convierte a dfcon nombres de features si están disponibles

        if feature_names is not None and len(feature_names) > 0:

            X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
        else:
            X_sample_df = X_sample
        
        # Calcular SHAP values
        logger.info("Calculando SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_df)
        
        # Crear plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values,
            X_sample_df,
            plot_type="dot",
            show=False,
            max_display=30
        )
        
        # path = output_dir / "shap_summary_plot.png"
        # plt.savefig(path, dpi=150, bbox_inches='tight')
        # plt.close()
        
        # logger.info(f"SHAP plot guardado: {path.name}")
        
        # return path
        
    except Exception as e:

        logger.warning(f"No se pudo crear SHAP plot: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return None


def log_to_mlflow(
    model,
    params: Dict,
    metrics: Dict,
    plots: Dict,
    shap_plot: Path
):
    """
    Registra todo en MLflow de forma manual
    
    Parámetros:
    ------------
    model: Modelo entrenado
    params: Hiperparámetros
    metrics: Métricas de evaluación
    plots: Diccionario con rutas de gráficos
    shap_plot: Ruta del SHAP plot
    """
    logger.info("\n Registrando en MLflow...")
    
    try:
        # Log parámetros
        mlflow.log_params(params)
        logger.info("Parámetros registrados")
        
        # Log métricas de train
        for metric, value in metrics['train'].items():
            mlflow.log_metric(f"train_{metric}", value)
        
        # Log métricas de val
        for metric, value in metrics['val'].items():
            mlflow.log_metric(f"val_{metric}", value)
        logger.info("Métricas registradas")
        
        # Log modelo
        mlflow.xgboost.log_model(model, "model")
        logger.info("Modelo registrado")
        
        # Log gráficos
        for name, path in plots.items():
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="plots")
        
        if shap_plot and shap_plot.exists():
            mlflow.log_artifact(str(shap_plot), artifact_path="plots")
        logger.info("Gráficos registrados")
        
        # Tags
        mlflow.set_tag("model_type", "XGBClassifier")
        mlflow.set_tag("optimization", "Optuna")
        mlflow.set_tag("objective", "maximize_f1")
        
        logger.info("Todo registrado siuuu")
        
    except Exception as e:
        logger.warning(f"error al registrar en MLflow: {e}")


def save_model(model, params: Dict, metrics: Dict, f1_val: float) -> Path:
    """
    Guarda el modelo final
    
    Parámetros:
    -----------
    model: Modelo entrenado
    params: Hiperparámetros
    metrics: Métricas
    f1_val: F1-score en validación
    
    Returns:
    ----------
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
    Función principal que ejecuta todo el entrenamiento,
    para que se use en un operador de airflow.
    
    Parámetros:
    ------------
    **kwargs: Argumentos adicionales de Airflow (ti, execution_date, etc.)
    
    Returns:
    ----------
    dict: Diccionario con información del proceso (para XCom en Airflow)
    """
    logger.info("\n" + "="*70)
    logger.info("INICIANDO ENTRENAMIENTO DE MODELO")
    logger.info("="*70)
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Carga de datos 
        X_train, y_train, X_val, y_val, feature_names = load_transformed_data()
        
        # Calcular scale_pos_weight
        scale_pos_weight = calculate_scale_pos_weight(y_train)
        
        # Optimizar hiperparámetros
        best_params, study = optimize_hyperparameters(
            X_train, y_train, X_val, y_val,
            scale_pos_weight,
            n_trials=20
        )
        
        # Se crea run name identificable
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Iniciar run de MLflow
        with mlflow.start_run(run_name=f"{timestamp}"):
            
            # Entrenar modelo final (con nombres de features)
            model = train_final_model(X_train, y_train, X_val, y_val, best_params, feature_names)
            
            # Evaluación modelo
            metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
            
            # Crear gráficos de feature importance (con nombres reales)
            plots_dir = REPORTS_PATH / "model_performance"
            plots = create_feature_importance_plots(model, plots_dir, feature_names)
            
            # Crear SHAP plot (con nombres reales)
            shap_plot = create_shap_summary_plot(model, X_val, y_val, plots_dir, feature_names)
            
            # Registrar todo en MLflow
            log_to_mlflow(model, best_params, metrics, plots, shap_plot)
            
            # Actualizar run name con F1 score
            f1_val = metrics['val']['f1']
            mlflow.set_tag("mlflow.runName", f"{timestamp}_f1_{f1_val:.4f}")
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f"\n MLflow run ID: {run_id}")
        
        # Guardar modelo localmente
        model_path = save_model(model, best_params, metrics, f1_val)
        
        logger.info("\n" + "="*70)
        logger.info("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*70)
        
        # Retornar información para XCom de Airflow
        return {
            'status': 'success',
            'model_path': str(model_path),
            'mlflow_run_id': run_id,
            'f1_score_train': metrics['train']['f1'],
            'f1_score_val': metrics['val']['f1'],
            'best_params': best_params,
            'n_optuna_trials': len(study.trials),
            'n_features': len(feature_names) if feature_names else X_train.shape[1],
            'execution_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': str(e),
            'execution_time': datetime.now().isoformat()
        }


if __name__ == "__main__":
    """Ejecución directa del script (para testing)"""
    result = train_model()
    print("\n" + "="*70)
    print("RESULTADO:", result)
    print("="*70)