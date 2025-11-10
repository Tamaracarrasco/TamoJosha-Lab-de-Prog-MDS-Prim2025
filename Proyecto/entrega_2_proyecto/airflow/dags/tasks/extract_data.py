# # acá se define la función para extraer la data 
# # que está en la carpeta baseline, dentro de la carpeta data.
# """
# Extracción y preparación de datos.
# - Lee data/baseline/ (df de la parte 1) si existe.
# - Si no existe, lee transactions.parquet + customers.parquet + products.parquet y reconstruye
# el "df_full" (cartesiano por semana) similar a la Parte 1.
# - Guarda archivos en data/run/<date>/raw/
# """
# import pandas as pd
# from pathlib import Path
# from datetime import datetime
# import logging


# logger = logging.getLogger(__name__)


# def extract_data(**context):
#     """
#     Extrae datos base desde data/baseline/ y guarda una copia en data/run/{ds}/raw/
#     """
#     base = Path(__file__).resolve().parents[3]  # /proyecto/
#     baseline = base / "data" / "baseline"
#     run_dir = base / "data" / "run" / context["ds"] / "raw"
#     run_dir.mkdir(parents=True, exist_ok=True)

#     # Se lee los archivos parquet
#     clientes = pd.read_parquet(baseline / "clientes.parquet")
#     productos = pd.read_parquet(baseline / "productos.parquet")
#     transacciones = pd.read_parquet(baseline / "transacciones.parquet")

#     # Se guarda copia en la extracción.
#     clientes.to_parquet(run_dir / "clientes.parquet", index=False)
#     productos.to_parquet(run_dir / "productos.parquet", index=False)
#     transacciones.to_parquet(run_dir / "transacciones.parquet", index=False)

#     print(f"Datos extraídos y guardados en {run_dir}")