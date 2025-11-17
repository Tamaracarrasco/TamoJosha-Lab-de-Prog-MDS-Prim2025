"""
Script para crear el dataframe unificado (df_unified.parquet)
Combina transacciones 2024 + transacciones nuevas 2025 + clientes + productos.
En la carpeta /data/baseline/ se encuentran los datos del año anterior:
productos.parquet, clientes.parquet y transacciones.parquet.
También está el df utilizado en la entrega 1. Por ahora se considera
ya que servirá para detectar data drifting.

**Comentario para el jupyter**: La manera en que se aborda esto es que de manera inicial, se cuenta
con el df que se entrenó el modelo escogido en la parte 1 (contiene datos del todo el año 2024) y 
con los datos nuevos del año 2025, debe crearse de igual manera un df (igual que en la parte 1), con la diferencia
que ahora hay una concatenación de los dataframe de transacciones.

Esto servirá para identificar data drifting entre ambos conjuntos y
generar nuevos conjuntos de entrenamiento/validación/testeo.

### LO QUE SE ESPERA QUE HAGA ESTE SCRIPT:

- Que cargue los datos antiguos.
- Que cargue los datos nuevos, si es que los hay.
- Unión de dataframes de transacciones año 2024 y 2025
- Creación nuevo dataframe que contenga data 2024 + 2025 a través de 2 merges.
- Que igual se agregue un metadato al nuevo df 
- Guardado del dataframe.
- Función que ejecute todo lo de arriba.
- Extras: Que diga el tamaño archivo, cuánto pesa, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Tuple, Optional, List

# Configurar logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Definir rutas base del proyecto como variables globales.
BASE_PATH = Path(__file__).parent.parent.parent.parent
BASELINE_PATH = BASE_PATH / "data" / "baseline"
RUN_PATH = BASE_PATH / "data" / "run"
NEW_TRANSACTIONS_PATH = RUN_PATH / "new_transactions"


def load_baseline_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carga los datos baseline (clientes, productos, transacciones 2024)
    
    Returns:
        Tuple con (transacciones_2024, clientes, productos)
    """
    logger.info("Cargando datos baseline...")
    
    try:
        transacciones_2024 = pd.read_parquet(
            BASELINE_PATH / "transacciones.parquet"
        )
        logger.info(f"Transacciones 2024 cargadas: {len(transacciones_2024):,} registros")
        
        clientes = pd.read_parquet(
            BASELINE_PATH / "clientes.parquet"
        )
        logger.info(f"Clientes cargados: {len(clientes):,} registros")
        
        productos = pd.read_parquet(
            BASELINE_PATH / "productos.parquet"
        )
        logger.info(f"Productos cargados: {len(productos):,} registros")
        
        return transacciones_2024, clientes, productos
        
    except FileNotFoundError as e:
        logger.error(f"Error: Archivo no encontrado - {e}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar datos baseline: {e}")
        raise


def load_new_transactions() -> Optional[pd.DataFrame]:
    """
    Carga las nuevas transacciones 2025 desde la carpeta run/new_transactions
    
    Returns:
    ----------------
    DataFrame con transacciones nuevas o None si no hay archivos.
    """
    logger.info("\n Buscando nuevas transacciones en run/new_transactions...")
    
    # Buscar todos los archivos parquet en la carpeta

    parquet_files = list(NEW_TRANSACTIONS_PATH.glob("*.parquet"))
    
    if not parquet_files:
        logger.warning("No se encontraron archivos de transacciones nuevas")
        return None
    
    logger.info(f"Encontrados {len(parquet_files)} archivo(s) parquet")
    
    # Cargar y concatenar todos los archivos

    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            logger.info(f"Cargado {file.name}: {len(df):,} registros")
            dfs.append(df)

        except Exception as e:
            logger.error(f"Error al cargar {file.name}: {e}")
    
    if not dfs:
        return None
    
    # Concatenar todos los dataframes

    transacciones_nuevas = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total transacciones nuevas: {len(transacciones_nuevas):,} registros")
    
    return transacciones_nuevas


def concatenate_transactions(trans_2024: pd.DataFrame, 
                            trans_nuevas: Optional[pd.DataFrame]
                            ) -> pd.DataFrame:
    """
    Concatena transacciones 2024 con transacciones nuevas
    
    Parámetros:
    --------------
    trans_2024: DataFrame de transacciones 2024
    trans_nuevas: DataFrame de transacciones nuevas (puede ser None)
    
    Returns:
    --------------
    DataFrame concatenado
    """
    logger.info("\n Concatenando transacciones...")
    
    if trans_nuevas is None or len(trans_nuevas) == 0:
        logger.warning("No hay transacciones nuevas, usando solo datos año 2024")
        return trans_2024.copy()

    # Eliminación de transacciones duplicados año 2024:

    trans_2024 = trans_2024.drop_duplicates()
    
    # Verificar que las columnas coincidan

    cols_2024 = set(trans_2024.columns)
    cols_nuevas = set(trans_nuevas.columns)
    
    if cols_2024 != cols_nuevas:
        missing_in_nuevas = cols_2024 - cols_nuevas
        missing_in_2024 = cols_nuevas - cols_2024
        
        if missing_in_nuevas:
            logger.warning(f"Columnas faltantes en datos nuevos: {missing_in_nuevas}")

        if missing_in_2024:
            logger.warning(f"Columnas adicionales en datos nuevos: {missing_in_2024}")
    
    # Concatenar

    transacciones_all = pd.concat(
        [trans_2024, trans_nuevas], 
        ignore_index=True
    )
    
    logger.info(f"Transacciones totales: {len(transacciones_all):,} registros")
    logger.info(f"  - De 2024: {len(trans_2024):,}")
    logger.info(f"  - Nuevas: {len(trans_nuevas):,}")
    
    return transacciones_all


def merge_data(transacciones: pd.DataFrame, 
                clientes: pd.DataFrame, 
                productos: pd.DataFrame
                ) -> pd.DataFrame:
    """
    Realiza el cruce de información entre transacciones, clientes y productos
    tal como se hizo en la parte 1.
    
    Parámetros:
    -------------
    transacciones: DataFrame de transacciones
    clientes: DataFrame de clientes
    productos: DataFrame de productos
    
    Returns:
    --------------
    DataFrame unificado
    """
    logger.info("\n Realizando cruce de información...")

    # Antes de realizar el MERGE:
    # Se supone que ya no hay trans_2024 duplicadas.
    
    # Identificar columnas de join (ajustar según tu estructura real)
    # Estas son suposiciones comunes, ajusta según tus datos
    cliente_key = 'customer_id' # if 'cliente_id' in transacciones.columns else 'id_cliente'
    producto_key = 'product_id' # if 'producto_id' in transacciones.columns else 'id_producto'
    
    # Merge con clientes

    logger.info(f"Cruzando con clientes usando clave: {cliente_key}")

    df_unified = transacciones.merge(clientes, on=cliente_key, how='left',
        # suffixes=('', '_cliente')
    )

    logger.info(f"MERGE 1: transacciones con  clientes hecha: {len(df_unified):,} registros")
    
    # Merge con productos
    logger.info(f"Cruzando con productos usando clave: {producto_key}")

    df_unified = df_unified.merge(productos, on=producto_key, how='left',
        #suffixes=('', '_producto')
    )
    logger.info(f"MERGE 2: Después de merge con productos: {len(df_unified):,} registros")
    
    return df_unified


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega metadatos al dataframe
    
    Parámetros:
    ------------
    df: DataFrame a procesar
    
    Returns:
    ------------
    DataFrame con metadatos
    """
    df = df.copy()
    df['fecha_creacion'] = datetime.now()
    df['version_dataset'] = '2.0'
    
    logger.info("Metadatos agregados")
    return df


def save_unified_dataframe(df: pd.DataFrame, output_filename: str = "df_unified.parquet") -> Path:
    """
    Guarda el dataframe unificado
    
    Parámetros:
    --------------
    df: DataFrame a guardar
    output_filename: Nombre del archivo de salida
    
    Returns:
    --------------
    Path del archivo guardado
    """
    output_path = BASELINE_PATH / output_filename
    
    logger.info(f"\n Guardando dataframe unificado en {output_path}...")
    
    try:
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # Verificación tamaño del archivo: ojalá no pese TB

        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        logger.info(f"Dataframe guardado exitosamente")
        logger.info(f"  - Ruta: {output_path}")
        logger.info(f"  - Tamaño: {file_size:.2f} MB")
        logger.info(f"  - Registros: {len(df):,}")
        logger.info(f"  - Columnas: {len(df.columns)}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error al guardar dataframe: {e}")
        raise


def create_unified_dataframe(**kwargs) -> dict:
    """
    Función principal que ejecuta todo el proceso de creación del dataframe unificado.
    Esta función está diseñada para ser llamada desde un Airflow PythonOperator.
    
    Parámetros:
    ------------
    **kwargs: Argumentos adicionales de Airflow (ti, execution_date, etc.)
    
    Returns:
    -----------
    dict: Diccionario con información del proceso (para XCom en Airflow)
    """
    logger.info("="*70)
    logger.info("INICIANDO CREACIÓN DE DATAFRAME UNIFICADO")
    logger.info("="*70)
    
    try:
        # 1. Cargar datos baseline
        trans_2024, clientes, productos = load_baseline_data()
        
        # 2. Cargar transacciones nuevas
        trans_nuevas = load_new_transactions()
        
        # 3. Concatenar transacciones
        transacciones_all = concatenate_transactions(trans_2024, trans_nuevas)
        
        # 4. Realizar cruce de información
        df_unified = merge_data(transacciones_all, clientes, productos)
        
        # 5. Agregar metadatos
        df_unified = add_metadata(df_unified)
        
        # 6. Guardar resultado
        output_path = save_unified_dataframe(df_unified)
        
        logger.info("="*70)
        logger.info("PROCESO COMPLETADO EXITOSAMENTE")
        logger.info("="*70)
        
        # Resumen
        logger.info("\nRESUMEN DEL DATAFRAME UNIFICADO:")
        logger.info(f"  - Shape: {df_unified.shape}")
        logger.info(f"  - Memoria: {df_unified.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Retornar información para XCom de Airflow
        return {
            'status': 'success',
            'output_path': str(output_path),
            'n_rows': len(df_unified),
            'n_columns': len(df_unified.columns),
            'file_size_mb': output_path.stat().st_size / (1024 * 1024),
            'execution_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en el proceso: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'execution_time': datetime.now().isoformat()
        }


# Funciones auxiliares para uso en Airflow DAG

def get_baseline_path() -> str:
    """Retorna la ruta de la carpeta baseline"""
    return str(BASELINE_PATH)


def get_unified_df_path() -> str:
    """Retorna la ruta del dataframe unificado"""
    return str(BASELINE_PATH / "df_unified.parquet")


def check_new_transactions_exist() -> bool:
    """
    Verifica si existen nuevas transacciones para procesar.
    Útil para branching en Airflow.
    
    Returns:
        bool: True si hay archivos, False si no
    """
    parquet_files = list(NEW_TRANSACTIONS_PATH.glob("*.parquet"))
    has_files = len(parquet_files) > 0
    logger.info(f"Verificación de nuevas transacciones: {has_files}")
    return has_files


if __name__ == "__main__":
    """Ejecución directa del script (para testing)"""
    result = create_unified_dataframe()
    print("\n" + "="*70)
    print("RESULTADO:", result)
    print("="*70)