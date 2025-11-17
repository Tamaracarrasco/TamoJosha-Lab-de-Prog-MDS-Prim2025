"""
En este script, se buscará aplicar el mismo pre procesamiento
aplicado en la parte 1.

Se intentará guardar los datos trasnformados pero eso
a la larga no funciona.

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime
from typing import Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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


def get_column_definitions() -> Tuple[list, list, list]:
    """
    Define las columnas a eliminar, numéricas y categóricas
    
    Returns:
    -----------
    (drop_cols, num_cols, cat_cols)
    """
    drop_cols = [
        "customer_id", "product_id", "week_t", "week_t_plus_1",
        "semana", "semana_siguiente_str", "semana_num"
    ]
    
    num_cols = [
        "X", "Y", "size",
        "num_deliver_per_week", "num_visit_per_week",
        "purchased_count", "compra_o_no"
    ]
    
    cat_cols = [
        "customer_type", "segment", "brand", "category", 
        "sub_category", "zone_id", "region_id"
    ]
    
    return drop_cols, num_cols, cat_cols


def create_preprocessor() -> ColumnTransformer:
    """
    Crea el ColumnTransformer con pipelines de preprocesamiento
    
    Returns:
    -----------
    ColumnTransformer configurado
    """
    logger.info("Creando preprocessor...")
    
    drop_cols, num_cols, cat_cols = get_column_definitions()
    
    # Pipeline variables numéricas
    num_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="median")),  
        ("sc", StandardScaler())
    ])
    
    # Pipeline variables categóricas
    cat_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
    
    # ColumnTransformer

    col_transformer = ColumnTransformer(
        transformers=[
            ("drop_ids", "drop", drop_cols),
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    logger.info(f" Se creó el preprocesador SIUUU")
    logger.info(f"  - Columnas a eliminar: {len(drop_cols)}")
    logger.info(f"  - Columnas numéricas: {len(num_cols)}")
    logger.info(f"  - Columnas categóricas: {len(cat_cols)}")
    
    return col_transformer


def load_train_val_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Carga datos de entrenamiento y validación
    
    Returns:
    ------------
    (X_train, y_train, X_val, y_val)
    """
    logger.info("Cargando datos de entrenamiento y validación")
    
    try:
        X_train = pd.read_parquet(PROCESSED_PATH / "X_train.parquet")
        y_train = pd.read_parquet(PROCESSED_PATH / "y_train.parquet")['y']
        X_val = pd.read_parquet(PROCESSED_PATH / "X_val.parquet")
        y_val = pd.read_parquet(PROCESSED_PATH / "y_val.parquet")['y']
        
        logger.info(f"X_train: {X_train.shape}")
        logger.info(f"y_train: {y_train.shape}")
        logger.info(f"X_val: {X_val.shape}")
        logger.info(f"y_val: {y_val.shape}")
        
        return X_train, y_train, X_val, y_val
        
    except FileNotFoundError as e:
        logger.error(f"Error: Archivos de datos no encontrados - {e}. Pls revise")
        raise

    except Exception as e:
        logger.error(f"Error al cargar datos: {e}. Pls revise")
        raise


def fit_and_transform_data(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame
    ) -> Tuple:
    """
    Ajusta el preprocessor con train y transforma train y val
    
    Parámetros:
    preprocessor: ColumnTransformer a ajustar
    X_train: Datos de entrenamiento
    X_val: Datos de validación
    
    Returns:
    -----------
    Tupla (X_train_transformed, X_val_transformed, preprocessor_fitted)
    """
    logger.info("\n" + "="*70)
    logger.info("AJUSTANDO Y TRANSFORMANDO DATOS")
    logger.info("="*70)
    
    # Fit en train
    logger.info("== Ajustando preprocessor con training data ==")
    preprocessor.fit(X_train)
    logger.info("Listo")
    
    # Transform train

    logger.info("== Transformando datos de entrenamiento ==")
    X_train_transformed = preprocessor.transform(X_train)
    logger.info(f"X_train transformado: {X_train_transformed.shape}")
    
    # Transform val

    logger.info("== Transformando datos de validación ==")
    X_val_transformed = preprocessor.transform(X_val)
    logger.info(f"X_val transformado: {X_val_transformed.shape}")
    
    # Información sobre features
    try:
        feature_names = preprocessor.get_feature_names_out()
        logger.info(f"Total de features después de transformación: {len(feature_names)}")
    except:
        logger.warning("No se pudieron obtener nombres de features")
    
    return X_train_transformed, X_val_transformed, preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, filename: str = "preprocessor.pkl") -> Path:
    """
    Guarda el preprocessor ajustado
    
    Parámetros:
    --------------
    preprocessor: ColumnTransformer ajustado
    filename: Nombre del archivo
    
    Returns:
    ------------
    Path del archivo guardado
    """
    # Crear carpeta si no existe
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    output_path = MODELS_PATH / filename
    
    logger.info(f"\nGuardando preprocessor en {output_path}...")
    
    try:
        joblib.dump(preprocessor, output_path)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Preprocessor guardado: {file_size:.2f} MB")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error al guardar preprocessor: {e}")
        raise


def save_transformed_data(X_train_transformed, X_val_transformed,
    y_train: pd.Series,
    y_val: pd.Series
    ) -> dict:
    """
    Guarda los datos transformados
    
    Parámetros:
    -------------
    X_train_transformed: Features de train transformadas
    X_val_transformed: Features de val transformadas
    y_train: Target de train
    y_val: Target de val
    
    Returns:
    -------------
    Diccionario con rutas de archivos guardados
    """
    logger.info("\n" + "="*70)
    logger.info("GUARDANDO DATOS TRANSFORMADOS")
    logger.info("="*70)
    
    paths = {}
    
    # Guardar como archivos NPZ (más eficiente para sparse matrices)
    import scipy.sparse as sp
    
    # X_train
    train_path = PROCESSED_PATH / "X_train_transformed.npz"
    if sp.issparse(X_train_transformed):
        sp.save_npz(train_path, X_train_transformed)
    else:
        np.savez_compressed(train_path, data=X_train_transformed)
    logger.info(f"X_train_transformed guardado: {train_path.name}")
    paths['X_train_transformed'] = train_path
    
    # X_val
    val_path = PROCESSED_PATH / "X_val_transformed.npz"
    if sp.issparse(X_val_transformed):
        sp.save_npz(val_path, X_val_transformed)
    else:
        np.savez_compressed(val_path, data=X_val_transformed)
    logger.info(f"X_val_transformed guardado: {val_path.name}")
    paths['X_val_transformed'] = val_path
    
    # y_train y y_val (ya están guardados, pero por consistencia)
    # No es necesario guardarlos de nuevo
    
    return paths


def preprocess_data(**kwargs) -> dict:
    """
    Función principal que ejecuta todo el preprocesamiento.
    Esta función está diseñada para ser llamada desde un Airflow PythonOperator.
    
    Args:
        **kwargs: Argumentos adicionales de Airflow (ti, execution_date, etc.)
    
    Returns:
        dict: Diccionario con información del proceso (para XCom en Airflow)
    """
    logger.info("\n" + "="*70)
    logger.info("INICIANDO PREPROCESAMIENTO DE DATOS")
    logger.info("="*70)
    
    try:
        # 1. Cargar datos
        X_train, y_train, X_val, y_val = load_train_val_data()
        
        # 2. Crear preprocessor
        preprocessor = create_preprocessor()
        
        # 3. Ajustar y transformar
        X_train_transformed, X_val_transformed, preprocessor_fitted = fit_and_transform_data(
            preprocessor, X_train, X_val
        )
        
        # 4. Guardar preprocessor
        preprocessor_path = save_preprocessor(preprocessor_fitted)
        
        # 5. Guardar datos transformados
        transformed_paths = save_transformed_data(
            X_train_transformed, X_val_transformed, y_train, y_val
        )
        
        logger.info("\n" + "="*70)
        logger.info("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*70)
        
        # Retornar información para XCom de Airflow
        return {
            'status': 'success',
            'preprocessor_path': str(preprocessor_path),
            'train_shape_transformed': X_train_transformed.shape,
            'val_shape_transformed': X_val_transformed.shape,
            'n_features': X_train_transformed.shape[1],
            'is_sparse': str(type(X_train_transformed)),
            'execution_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en el preprocesamiento: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': str(e),
            'execution_time': datetime.now().isoformat()
        }


# Funciones auxiliares para cargar datos transformados

def load_transformed_train_data():
    """Carga datos de entrenamiento transformados"""
    import scipy.sparse as sp
    
    X_path = PROCESSED_PATH / "X_train_transformed.npz"
    y_train = pd.read_parquet(PROCESSED_PATH / "y_train.parquet")['y']
    
    try:
        X_train = sp.load_npz(X_path)
    except:
        data = np.load(X_path)
        X_train = data['data']
    
    return X_train, y_train


def load_transformed_val_data():
    """Carga datos de validación transformados"""
    import scipy.sparse as sp
    
    X_path = PROCESSED_PATH / "X_val_transformed.npz"
    y_val = pd.read_parquet(PROCESSED_PATH / "y_val.parquet")['y']
    
    try:
        X_val = sp.load_npz(X_path)
    except:
        data = np.load(X_path)
        X_val = data['data']
    
    return X_val, y_val


def load_preprocessor(filename: str = "preprocessor.pkl") -> ColumnTransformer:
    """Carga el preprocessor guardado"""
    path = MODELS_PATH / filename
    logger.info(f"Cargando preprocessor desde {path}...")
    preprocessor = joblib.load(path)
    logger.info("Preprocessor cargado")
    return preprocessor


if __name__ == "__main__":
    """Ejecución directa del script"""
    result = preprocess_data()
    print("\n" + "="*70)
    print("RESULTADO:", result)
    print("="*70)