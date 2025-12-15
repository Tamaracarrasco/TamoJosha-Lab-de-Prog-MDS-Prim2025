# airflow/scripts/data_io.py

import os
from typing import Optional, Union, List

import pandas as pd


def build_dataset_from_raw(
    data_dir: str,
    new_transactions_filename: Optional[Union[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Wrapper fino sobre build_model_dataset.
    Construye df_final a partir de los parquet crudos (y opcionalmente
    un archivo nuevo de transacciones o una lista de archivos).
    
    Args:
        data_dir: directorio con los parquet
        new_transactions_filename: puede ser None, un string (archivo único),
                                   o una lista de strings (múltiples archivos)
    """
    # Lazy import porque me quedaba sin RAM
    from scripts.data_preparation import build_model_dataset
    
    return build_model_dataset(
        data_dir=data_dir,
        new_transactions_filename=new_transactions_filename,
    )


def save_predictions(df: pd.DataFrame, output_path: str) -> None:
    """
    Guarda las predicciones generadas por el modelo en formato parquet.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}")