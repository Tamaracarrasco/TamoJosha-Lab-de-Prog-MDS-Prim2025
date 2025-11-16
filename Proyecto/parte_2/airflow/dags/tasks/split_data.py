"""
En este script se buscará generar los conjuntos de entrenamiento,
validación y testeo.

Caso 1: En la parte 1, los conjuntos se definian así.
    - Enero a Septiembre: entrenamiento
    - Octubre - Noviembre: Validación.
    - Diciembre: Testing.
Caso 2: Se reciben nuevos datos 2025.
    - En caso de que hayan nuevos datos:
        - Enero a Octubre: Entrenamiento.
        - Validación: Noviembre - Diciembre.
        - Testing: Datos nuevos del 2025.

Se buscará crear estas funciones y guardar los conjuntos correspondientes.

**Comentario** Ahora los tamaños difieren a los conjuntos dados en la primera parte,
porque de enero a septiembre, se estimó que habian 36-37 semanas, cuando en realidad no
era así -> El conjunto de entrenamiento debía contener más semanas (estas semanas adicionales
se fueron al conjunto de validación)

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Tuple, Dict

# Configurar logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Definir rutas base del proyecto
BASE_PATH = Path(__file__).parent.parent.parent.parent
PROCESSED_PATH = BASE_PATH / "data" / "run" / "processed"


def load_base_final() -> pd.DataFrame:
    """
    Carga la base final procesada
    
    Returns:
    -------------
    DataFrame con la base final
    """
    logger.info(" = Cargando df_base_final.parquet =")
    
    try:
        df = pd.read_parquet(PROCESSED_PATH / "df_base_final.parquet")
        logger.info(f"Base final cargada: {len(df):,} registros, {len(df.columns)} columnas")
        return df

    except FileNotFoundError as e:
        logger.error(f"Error: df_base_final.parquet no encontrado en {PROCESSED_PATH}")
        raise

    except Exception as e:
        logger.error(f"Error al cargar base final: {e}")
        raise


def prepare_week_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna semana_num para ordenar cronológicamente
    Formato: YYYYWW (ej: 202401 = semana 1 de 2024)
    
    Parámetros:
    --------------
    df: DataFrame con columna 'semana' en formato YYYY-WW
    
    Returns:
    -------------
    DataFrame con columna semana_num agregada
    """
    logger.info("\n Creando columna semana_num para ordenamiento")
    
    df = df.copy()
    
    # Extraer año y semana de la columna "semana"

    df["semana_num"] = (
        df["semana"].str.split("-").str[0].astype(int) * 100 +
        df["semana"].str.split("-").str[1].astype(int)
    )
    
    logger.info(f"Rango de semanas: {df['semana_num'].min()} -> {df['semana_num'].max()}")
    
    return df


def detect_data_availability(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Detecta si hay datos de 2025 disponibles basándose en la
    fecha real (week_t)
    
    Parámetros:
    ------------
    df: DataFrame con columna week_t (fecha real)
    
    Returns:
    -----------
    Diccionario con información de disponibilidad
    """
    logger.info("\n" + "="*70)
    logger.info("DETECTANDO DISPONIBILIDAD DE DATOS")
    logger.info("="*70)
    
    # Convertir week_t a datetime

    if not pd.api.types.is_datetime64_any_dtype(df['week_t']):
        df['week_t'] = pd.to_datetime(df['week_t'])
    
    # Identificar años presentes basándose en fecha real

    years_present = df['week_t'].dt.year.unique()
    has_2024 = 2024 in years_present
    has_2025 = 2025 in years_present
    
    # Ordenar por semana_num
    weeks_sorted = sorted(df["semana_num"].unique())
    
    # Fecha mínima y máxima

    fecha_min = df['week_t'].min()
    fecha_max = df['week_t'].max()
    
    logger.info(f"Fecha mínima en los datos: {fecha_min.date()}")
    logger.info(f"Fecha máxima en los datos: {fecha_max.date()}")
    logger.info(f"Años detectados (por fecha real): {sorted(years_present)}")
    logger.info(f"¿Hay datos 2024?: {has_2024}")
    logger.info(f"¿Hay datos 2025?: {has_2025}")
    logger.info(f"Total de semanas únicas: {len(weeks_sorted)}")
    
    return {
        'has_2024': has_2024,
        'has_2025': has_2025,
        'years_present': sorted(years_present),
        'weeks_sorted': weeks_sorted,
        'n_weeks': len(weeks_sorted),
        'fecha_min': fecha_min,
        'fecha_max': fecha_max
    }


def split_with_2025_data(df: pd.DataFrame, weeks_sorted: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    # CASO 2: Divide los datos cuando HAY datos de 2025 disponibles
    - Train: Enero - Octubre 2024
    - Val: Noviembre - Diciembre 2024
    - Test: Datos 2025
    
    Parámetros:
    -------------
    df: DataFrame completo
    weeks_sorted: Lista ordenada de semanas
    
    Returns:
    -------------
    Tupla (train_df, val_df, test_df)
    """
    logger.info("\n" + "="*70)
    logger.info("ESTRATEGIA: HAY DATOS 2025 -> Train(Ene-Oct)/ Val(Nov-Dic)/ Test(2025)")
    logger.info("="*70)
    
    # Semanas de 2024 y 2025
    weeks_2024 = [w for w in weeks_sorted if w < 202500]
    weeks_2025 = [w for w in weeks_sorted if w >= 202500]
    
    # Mirando un calendario del W10
    # - Octubre termina en semana 44
    # - Noviembre termina en semana 48
    # Semana 202443: final de octubre
    # Semana 202452 como fin de diciembre
    
    # Train: hasta semana 43 de 2024 que ahi termina octubre
    train_weeks = [w for w in weeks_2024 if w <= 202443]
    
    # Val: semanas 45-53 de 2024 (nov y dic)
    val_weeks = [w for w in weeks_2024 if w > 202443]
    
    # Test: todas las semanas de 2025
    test_weeks = weeks_2025
      
    # Crear DataFrames

    train_df = df[df["semana_num"].isin(train_weeks)].copy()
    val_df = df[df["semana_num"].isin(val_weeks)].copy()
    test_df = df[df["semana_num"].isin(test_weeks)].copy()
    
    logger.info(f"\n Train: {len(train_df):,} filas, semanas {train_weeks[0]} --> {train_weeks[-1]}")
    logger.info(f" Val:   {len(val_df):,} filas, semanas {val_weeks[0]} --→ {val_weeks[-1]}")
    logger.info(f" Test:  {len(test_df):,} filas, semanas {test_weeks[0]} --> {test_weeks[-1]}")
    
    return train_df, val_df, test_df


def split_without_2025_data(df: pd.DataFrame, weeks_sorted: list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    # CASO 1: Como en la primera entrega de la parte 1

    Divide los datos cuando NO HAY datos de 2025 (solo 2024)
    - Train: Enero - Septiembre 2024
    - Val: Octubre - Noviembre 2024
    - Test: Diciembre 2024
    
    Parámetros:
    --------------
    df: DataFrame completo
    weeks_sorted: Lista ordenada de semanas
    
    Returns:
    -------------
    Tupla (train_df, val_df, test_df)
    """
    logger.info("\n" + "="*70)
    logger.info("CASO 2: SOLO 2024 -> Train(Ene-Sep)/ Val(Oct-Nov)/ Test(Dic)")
    logger.info("="*70)
    
    # Solo semanas 2024
    weeks_2024 = [w for w in weeks_sorted if w < 202500]
    
    # Train: hasta semana 40 de 2024 (final de septiembre, donde semana 01 es la primera de enero)
    train_weeks = [w for w in weeks_2024 if w <= 202440]
    
    # Val: semanas 41-48 de 2024 (octubre y noviembre)
    val_weeks = [w for w in weeks_2024 if 202440 < w <= 202448]
    
    # Test: el resto
    test_weeks = [w for w in weeks_2024 if w > 202448]

    
    # Crear DataFrames
    train_df = df[df["semana_num"].isin(train_weeks)].copy()
    val_df = df[df["semana_num"].isin(val_weeks)].copy()
    test_df = df[df["semana_num"].isin(test_weeks)].copy()
    
    logger.info(f"\nTrain: {len(train_df):,} filas, semanas {train_weeks[0]} -> {train_weeks[-1]}")
    logger.info(f" Val:   {len(val_df):,} filas, semanas {val_weeks[0]} -> {val_weeks[-1]}")
    logger.info(f" Test:  {len(test_df):,} filas, semanas {test_weeks[0]} -> {test_weeks[-1]}")
    
    return train_df, val_df, test_df


def create_train_val_test_sets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "y"
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separa X e y para cada conjunto
    
    Parámetros:
    -------------
    train_df: DataFrame de entrenamiento
    val_df: DataFrame de validación
    test_df: DataFrame de test
    target_col: Nombre de la columna target
    
    Returns:
    --------------
    (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info("\n" + "="*70)
    logger.info("SEPARANDO FEATURES (X) Y TARGET (y)")
    logger.info("="*70)
    
    # Columnas a eliminar
    drop_cols = [target_col]
    
    # Crear conjuntos X e y
    X_train = train_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_train = train_df[target_col].astype(int).reset_index(drop=True)
    
    X_val = val_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_val = val_df[target_col].astype(int).reset_index(drop=True)
    
    X_test = test_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_test = test_df[target_col].astype(int).reset_index(drop=True)
    
    logger.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    logger.info(f"X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    logger.info(f"X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    
    # Estadísticas de balance
    logger.info("\nBALANCE DE CLASES (y=1):")
    logger.info(f"  Train: {y_train.mean():.4f} ({y_train.sum():,} positivos)")
    logger.info(f"  Val:   {y_val.mean():.4f} ({y_val.sum():,} positivos)")
    logger.info(f"  Test:  {y_test.mean():.4f} ({y_test.sum():,} positivos)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_split_datasets(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Path]:
    """
    Guarda los conjuntos de datos divididos
    
    Parámetros:
    -------------
    X_train, y_train, X_val, y_val, X_test, y_test: Conjuntos de datos
    
    Returns:
    -------------
    Diccionario con las rutas de los archivos guardados
    """
    logger.info("\n" + "="*70)
    logger.info("GUARDANDO CONJUNTOS DE DATOS")
    logger.info("="*70)
    
    paths = {}
    
    # Guardar cada conjunto
    datasets = {
        'X_train': X_train,
        'y_train': y_train.to_frame('y'),
        'X_val': X_val,
        'y_val': y_val.to_frame('y'),
        'X_test': X_test,
        'y_test': y_test.to_frame('y')
    }
    
    for name, data in datasets.items():
        output_path = PROCESSED_PATH / f"{name}.parquet"
        data.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"{name}: {output_path.name} ({file_size:.2f} MB)")
        
        paths[name] = output_path
    
    return paths


def split_data(**kwargs) -> dict:
    """
    Función principal que ejecuta todo el proceso de división de datos,
    para que pueda ser llamada desde un P.operator
    Esta función está diseñada para ser llamada desde un Airflow PythonOperator.
    
    Según CASO 1 o CASO 2.

    Parámetros:
    -------------
    **kwargs: Argumentos adicionales de Airflow (ti, execution_date, etc.)
    
    Returns:
    ------------
    dict: Diccionario con información del proceso (para XCom en Airflow)
    """
    logger.info("\n" + "="*70)
    logger.info("INICIANDO DIVISIÓN DE DATOS EN TRAIN/VAL/TEST")
    logger.info("="*70)
    
    try:
        # 1. Cargar base final
        df = load_base_final()
        
        # 2. Preparar columna semana_num
        df = prepare_week_number(df)
        
        # 3. Detectar disponibilidad de datos
        data_info = detect_data_availability(df)
        
        # 4. Dividir según disponibilidad
        if data_info['has_2025']:
            train_df, val_df, test_df = split_with_2025_data(df, data_info['weeks_sorted'])
            strategy = "with_2025_data"
        else:
            train_df, val_df, test_df = split_without_2025_data(df, data_info['weeks_sorted'])
            strategy = "only_2024_data"
        
        # 5. Crear conjuntos X e y
        X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_sets(
            train_df, val_df, test_df
        )
        
        # 6. Guardar conjuntos
        saved_paths = save_split_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
        
        logger.info("\n" + "="*70)
        logger.info("== PROCESO LOGRADO SIUU ==")
        logger.info("="*70)
        
        # Retornar información para XCom de Airflow
        return {
            'status': 'success',
            'strategy': strategy,
            'has_2025_data': data_info['has_2025'],
            'train_shape': X_train.shape,
            'val_shape': X_val.shape,
            'test_shape': X_test.shape,
            'train_positive_rate': float(y_train.mean()),
            'val_positive_rate': float(y_val.mean()),
            'test_positive_rate': float(y_test.mean()),
            'saved_files': {k: str(v) for k, v in saved_paths.items()},
            'execution_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en el proceso: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'status': 'failed',
            'error': str(e),
            'execution_time': datetime.now().isoformat()
        }


# Funciones auxiliares para cargar datos divididos

def load_train_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Carga datos de entrenamiento"""
    X = pd.read_parquet(PROCESSED_PATH / "X_train.parquet")
    y = pd.read_parquet(PROCESSED_PATH / "y_train.parquet")['y']
    return X, y


def load_val_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Carga datos de validación"""
    X = pd.read_parquet(PROCESSED_PATH / "X_val.parquet")
    y = pd.read_parquet(PROCESSED_PATH / "y_val.parquet")['y']
    return X, y


def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Carga datos de test"""
    X = pd.read_parquet(PROCESSED_PATH / "X_test.parquet")
    y = pd.read_parquet(PROCESSED_PATH / "y_test.parquet")['y']
    return X, y


if __name__ == "__main__":
    """Ejecución directa del script (para testing)"""

    result = split_data()
    print("\n" + "="*70)
    print("RESULTADO:", result)
    print("="*70)