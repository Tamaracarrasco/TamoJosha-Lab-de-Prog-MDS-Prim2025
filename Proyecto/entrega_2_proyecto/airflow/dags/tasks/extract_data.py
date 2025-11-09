# acá se define la función para extraer la data 
# que está en la carpeta baseline, dentro de la carpeta data.
"""
Extracción y preparación de datos.
- Lee data/baseline/ (df de la parte 1) si existe.
- Si no existe, lee transactions.parquet + customers.parquet + products.parquet y reconstruye
el "df_full" (cartesiano por semana) similar a la Parte 1.
- Guarda archivos en data/run/<date>/raw/
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging


logger = logging.getLogger(__name__)



def read_baseline_if_available(baseline_path: str = "./data/baseline/df_final_entrega_1.parquet") -> pd.DataFrame:
    p = Path(baseline_path)
    if p.exists():
        logger.info(f"Leyendo baseline desde {p}")
        return pd.read_parquet(p)
    logger.warning("Baseline not found: %s" % baseline_path)
    return None