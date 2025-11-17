"""
create_base.py

En este script, se buscará armar la base de 10 millones de filas.
A partir de df_unified, lo ideal es que se parezca a df_parte_1.parquet.

df_parte_1.parquet: es las base de 10 millones de filas aprox, que se creó en la parte 1.
S
e conserva este df porque se planea implementar data drifting con métodos univariados y pca drift.
Luego se necesita un punto de comparación para cuando lleguen los primeros datos nuevos.


Se espera lo siguiente para la nueva base con datos nuevos.

- Creación de variable objetivo.
- Creacion de la base de datos.
- Idealmente no se debería guardar toda la base
pero si deberían guardarse los conjuntos de entrenamiento, validación y testeo.

- De igual forma, solo por ahora se guardaría la base con datos nueva. Esto es porque en 
un futuro, se crearán nuevos conjuntos de entrenamiento/validación/ testeo.
- que se muestren las dimensiones de la base.


    Ahora, antes de crear los conjuntos de train/val/testeo, se debe ver si hubo data drifting.

    ## Plan para detectar data drifting:
        - Si se van a entregar los datos de enero o de algún mes en específico del año 2025:
            - Esos datos deberían compararse con su contraparte con el año 2024.
            - Deberían compararse con los datos de Diciembre 2024 (conjunto testing)
        -> Si hay data drifting: debe haber un reentrenamiento.
        -> Si no hay data drifting: debe haber reentrenamiento períodico, 
        comenzando con los hiperparámetros que se obtuvieron en la parte 1.
        Ahora este re-entrenamiento periodico, se hará cada semana. por lo que debería haber un desplazamiento
        de semanas en los conjuntos de entrenamiento/ validacion/ testeo.

"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Tuple

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Definir rutas base del proyecto

BASE_PATH = Path(__file__).parent.parent.parent.parent
BASELINE_PATH = BASE_PATH / "data" / "baseline"
PROCESSED_PATH = BASE_PATH / "data" / "run" / "processed"


def load_unified_dataframe() -> pd.DataFrame:
    """
    Carga el dataframe unificado
    
    Returns:
    --------------
    DataFrame unificado.

    """
    logger.info("Cargando df_unified.parquet...")
    
    try:
        df = pd.read_parquet(BASELINE_PATH / "df_unified.parquet")
        logger.info(f"DataFrame cargado: {len(df):,} registros, {len(df.columns)} columnas")
        return df

    except FileNotFoundError as e:
        logger.error(f"Error: df_unified.parquet no encontrado en {BASELINE_PATH}")
        raise

    except Exception as e:
        logger.error(f"Error al cargar df_unified: {e}")
        raise


def create_panel_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea el panel de datos con resample semanal por par (cliente, producto)
    
    Parámetros:
    --------------
    df: DataFrame unificado
    
    Returns:

    Panel de datos con variables temporales y target.

    """
    logger.info("\n" + "="*70)
    logger.info("CREANDO PANEL DE DATOS")
    logger.info("="*70)
    
    df_ = df.copy()
    
    # Convertir purchase_date a datetime
    df_["purchase_date"] = pd.to_datetime(df_["purchase_date"], errors="coerce")
    df_ = df_.dropna(subset=["purchase_date"])

    logger.info(f"Registros después de limpiar fechas: {len(df_):,}")
    
    # Resample semanal por par (cliente, producto)
    logger.info("Realizando resample semanal (W-MON) por customer_id y product_id.")
    panel = (
        df_.set_index("purchase_date")
           .groupby(["customer_id", "product_id"])
           .resample("W-MON")  # semanas ancladas a lunes
           .size()
           .rename("purchased_count")
           .reset_index()
    )
    logger.info(f"Panel creado: {len(panel):,} registros")
    
    # Variables solicitadas

    logger.info("Creando variables parte 1")
    panel.rename(columns={"purchase_date": "week_t"}, inplace=True)
    panel["compra_o_no"] = (panel["purchased_count"] > 0).astype(int)
    panel = panel.sort_values(["customer_id", "product_id", "week_t"])
    
    # Target: compra en la próxima semana (y en t+1)

    logger.info("Generando variable objetivo")

    panel["y"] = panel.groupby(["customer_id", "product_id"])["compra_o_no"].shift(-1)
    panel = panel.dropna(subset=["y"]).copy()
    panel["y"] = panel["y"].astype(int)

    logger.info(f"Registros después de crear target: {len(panel):,}")
    logger.info(f"  - Positivos (y=1): {int(panel['y'].sum()):,}")
    logger.info(f"  - Tasa y=1: {panel['y'].mean():.4f}")
    
    # Semana t+1
    panel["week_t_plus_1"] = panel["week_t"] + pd.Timedelta(days=7)
    
    # Etiqueta "semana" estilo YYYY-ww (ISO)

    logger.info("Creando etiquetas de semana (formato ISO)")
    iso = panel["week_t"].dt.isocalendar()
    panel["semana"] = iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2)
    
    # Etiqueta para t+1
    iso1 = panel["week_t_plus_1"].dt.isocalendar()
    panel["semana_siguiente_str"] = iso1.year.astype(str) + "-" + iso1.week.astype(str).str.zfill(2)
    
    # Dataset final del panel

    df_copy = panel[[
        "customer_id", "product_id",
        "week_t", "week_t_plus_1", "semana", "semana_siguiente_str",
        "purchased_count", "compra_o_no", "y"
    ]].reset_index(drop=True)
    
    logger.info(f"Panel final: {len(df_copy):,} registros")
    
    return df_copy


def create_full_cartesian_base(df: pd.DataFrame, df_copy: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la base completa con producto cartesiano de 
    clientes, productos y semanas.
    
    Parámetros:
    --------------
    df: DataFrame unificado original
    df_copy: Panel de datos creado previamente
    
    Returns:
    -------------
    DataFrame completo con todas las combinaciones
    """
    logger.info("\n" + "="*70)
    logger.info("CREANDO BASE CARTESIANA COMPLETA")
    logger.info("="*70)
    
    # Obtener listas únicas
    logger.info("Extrayendo clientes, productos y semanas únicos...")
    customers = (
        df["customer_id"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )
    logger.info(f"Clientes únicos: {len(customers):,}")
    
    products = (
        df["product_id"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )
    logger.info(f"Productos únicos: {len(products):,}")
    
    weeks = (
        pd.to_datetime(df_copy["week_t"])
        .drop_duplicates()
        .sort_values()
        .to_list()
    )
    logger.info(f"Semanas únicas: {len(weeks)}")
    
    # Crear producto cartesiano por semana
    logger.info("\nGenerando producto cartesiano")
    frames = []
    n_prod = len(products)
    n_cust = len(customers)
    
    for idx, wk in enumerate(weeks, 1):
        if idx % 10 == 0 or idx == len(weeks):
            logger.info(f"  Procesando semana {idx}/{len(weeks)}...")
            
        wk_ts = pd.Timestamp(wk)
        wk_plus1 = wk_ts + pd.Timedelta(days=7)
        
        # Etiquetas YYYY-ww para semana actual
        iso0 = wk_ts.isocalendar()
        iso1 = wk_plus1.isocalendar()
        
        # Manejar tanto tupla como objeto IsoCalendarDate: acá me tiraba error isocalendar
                                                            # tira una tupla en versiones de pandas
        if hasattr(iso0, 'year'):
            semana_str = f"{iso0.year}-{int(iso0.week):02d}"
            semana_next_str = f"{iso1.year}-{int(iso1.week):02d}"
        else:
            # Si es tupla (year, week, weekday)
            semana_str = f"{iso0[0]}-{int(iso0[1]):02d}"
            semana_next_str = f"{iso1[0]}-{int(iso1[1]):02d}"
        
        # Producto cartesiano clientes x productos SOLO para esta semana
        base = pd.DataFrame({
            "customer_id": np.repeat(customers, n_prod),
            "product_id": np.tile(products, n_cust),
            "week_t": wk_ts,
            "week_t_plus_1": wk_plus1,
            "semana": semana_str,
            "semana_siguiente_str": semana_next_str
        })
        
        # Filas reales de esa semana desde df_copy
        wk_rows = df_copy.loc[
            df_copy["week_t"].eq(wk_ts),
            ["customer_id", "product_id", "purchased_count", "compra_o_no", "y"]
        ]
        
        # MERGE
        merged = base.merge(wk_rows, on=["customer_id", "product_id"], how="left")
        
        # Se rellena 0's donde no hubo compra
        for c in ["purchased_count", "compra_o_no", "y"]:
            merged[c] = merged[c].fillna(0).astype(int)
        
        frames.append(merged)
    
    # Concatenar todas las semanas
    df_full = pd.concat(frames, ignore_index=True)
    logger.info(f"Base cartesiana creada: {len(df_full):,} registros")
    
    return df_full


def add_dimension_features(df: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega las features de clientes y productos a la base completa
    
    Parámetros:
    -------------
    df: DataFrame unificado original
    df_full: Base cartesiana completa
    
    Returns:
    -------------
    DataFrame final con todas las features
    """
    logger.info("\n" + "="*70)
    logger.info("AGREGANDO FEATURES DE DIMENSIONES")
    logger.info("="*70)
    
    # Columnas de clientes
    logger.info("Preparando dimensión de clientes")

    cols_cli = ["customer_id", "customer_type", "X", "Y", "zone_id",
                "region_id", "num_deliver_per_week", "num_visit_per_week"]
    cols_cli = [c for c in cols_cli if c in df.columns]

    logger.info(f"  Columnas encontradas: {cols_cli}")
    
    # Dimensión clientes: 1 fila por customer_id
    dim_clientes = (
        df[cols_cli]
          .drop_duplicates(subset=["customer_id"])
          .reset_index(drop=True)
    )
    logger.info(f"Dimensión clientes: {len(dim_clientes):,} registros")
    
    # Columnas de productos
    logger.info("Preparando dimensión de productos..")
    cols_prod = ["product_id", "brand", "category", "sub_category",
                 "segment", "package", "size"]
    cols_prod = [c for c in cols_prod if c in df.columns]
    logger.info(f"  Columnas encontradas: {cols_prod}")
    
    # Dimensión productos: 1 fila por product_id
    dim_productos = (
        df[cols_prod]
          .drop_duplicates(subset=["product_id"])
          .reset_index(drop=True)
    )
    logger.info(f"Dimensión productos: {len(dim_productos):,} registros")
    
    # MERGE
    logger.info("\nRealizando merge con dimensiones..")
    df_final = (
        df_full
        .merge(dim_clientes, on="customer_id", how="left")
        .merge(dim_productos, on="product_id", how="left")
    )
    logger.info(f"DataFrame final: {len(df_final):,} registros, {len(df_final.columns)} columnas")
    
    return df_final


def save_final_base(df_final: pd.DataFrame, filename: str = "df_base_final.parquet") -> Path:
    """
    Guarda la base final procesada
    
    Parámetros:
    -------------
    df_final: DataFrame final a guardar
    filename: Nombre del archivo de salida
    
    Returns:
    -------------
    Path del archivo guardado
    """
    # Crear carpeta processed si no existe
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    
    output_path = PROCESSED_PATH / filename
    
    logger.info(f"\nGuardando base final en {output_path}...")
    
    try:
        df_final.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        # Verificar tamaño del archivo
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        logger.info(f" Base final guardada exitosamente")
        logger.info(f"  - Ruta: {output_path}")
        logger.info(f"  - Tamaño: {file_size:.2f} MB")
        logger.info(f"  - Registros: {len(df_final):,}")
        logger.info(f"  - Columnas: {len(df_final.columns)}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error al guardar base final: {e}")
        raise


def print_final_statistics(df_copy: pd.DataFrame, df_full: pd.DataFrame, df_final: pd.DataFrame):
    """
    Imprime estadísticas finales del proceso
    
    Parámetros:
    --------------

    df_copy: Panel observado
    df_full: Base cartesiana
    df_final: Base final con features
    """
    logger.info("\n" + "="*70)
    logger.info("ESTADÍSTICAS FINALES")
    logger.info("="*70)
    logger.info(f"Filas df_copy (panel observado): {len(df_copy):,}")
    logger.info(f"Filas df_full (cartesiano por semana): {len(df_full):,}")
    logger.info(f"Filas df_final (con features): {len(df_final):,}")
    logger.info(f"Positivos (y=1): {int(df_final['y'].sum()):,}")
    logger.info(f"Tasa y=1: {df_final['y'].mean():.6f}")
    logger.info(f"Columnas totales: {len(df_final.columns)}")
    logger.info(f"Memoria utilizada: {df_final.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info("="*70)


def create_base(**kwargs) -> dict:
    """
    Función principal que ejecuta todo el proceso de creación de la base.
    Para que se llame desde un operador de Airflow
    
    Parámetros:
    ------------
    **kwargs: Argumentos adicionales de Airflow (ti, execution_date, etc.)
    
    Returns:
    -----------
    dict: Diccionario con información del proceso (para XCom en Airflow)
    """
    logger.info("\n" + "="*70)
    logger.info("INICIANDO CREACIÓN DE BASE FINAL")
    logger.info("="*70)
    
    try:
        # 1. Cargar df_unified
        df = load_unified_dataframe()
        
        # 2. Crear panel de datos con variable objetivo
        df_copy = create_panel_data(df)
        
        # 3. Crear base cartesiana completa
        df_full = create_full_cartesian_base(df, df_copy)
        
        # 4. Agregar features de dimensiones
        df_final = add_dimension_features(df, df_full)
        
        # 5. Guardar resultado
        output_path = save_final_base(df_final)
        
        # 6. Imprimir estadísticas
        print_final_statistics(df_copy, df_full, df_final)
        
        logger.info("\n" + "="*70)
        logger.info("PROCESO COMPLETADO EXITOSAMENTE")
        logger.info("="*70)
        
        # Retornar información para XCom de Airflow
        return {
            'status': 'success',
            'output_path': str(output_path),
            'n_rows': len(df_final),
            'n_columns': len(df_final.columns),
            'n_positives': int(df_final['y'].sum()),
            'positive_rate': float(df_final['y'].mean()),
            'file_size_mb': output_path.stat().st_size / (1024 * 1024),
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


# Funciones auxiliares para uso en Airflow DAG

def get_base_final_path() -> str:
    """Retorna la ruta de la base final"""
    return str(PROCESSED_PATH / "df_base_final.parquet")


def load_base_final() -> pd.DataFrame:
    """
    Carga la base final para uso en otras tareas
    
    Returns:
    ------------
    DataFrame con la base final
    """
    path = PROCESSED_PATH / "df_base_final.parquet"
    logger.info(f"Cargando base final desde {path}...")
    df = pd.read_parquet(path)
    logger.info(f"Base cargada: {len(df):,} registros")
    return df


if __name__ == "__main__":
    """Ejecución directa del script (para testing)"""
    result = create_base()
    print("\n" + "="*70)
    print("RESULTADO:", result)
    print("="*70)