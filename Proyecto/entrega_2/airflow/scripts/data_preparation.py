# airflow/scripts/data_preparation.py


# Importo librerias
import os
from typing import Tuple, Optional, Union, List

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. Carga de datos crudos (histórico + si es que hay nuevas transacciones)
# ----------------------------------------------------------------------

def load_raw_data(
    data_dir: str,
    extra_transactions_path: Optional[Union[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lee clientes.parquet, productos.parquet y transacciones.parquet
    desde el directorio data_dir.

    Si extra_transactions_path no es None, concatena uno o más archivos de
    transacciones (nueva semana, p.ej. t+1) al histórico antes de devolverlo.
    
    Args:
        data_dir: directorio con los archivos base
        extra_transactions_path: puede ser None, un string (archivo único),
                                 o una lista de strings (múltiples archivos)
    """

    # Clientes y productos se asumen estables en el tiempo
    clientes = pd.read_parquet(os.path.join(data_dir, "clientes.parquet"))
    productos = pd.read_parquet(os.path.join(data_dir, "productos.parquet"))

    # Histórico de transacciones
    transacciones_hist = pd.read_parquet(
        os.path.join(data_dir, "transacciones.parquet")
    )

    if extra_transactions_path is not None:
        # Convertir a lista si es string único
        if isinstance(extra_transactions_path, str):
            extra_files = [extra_transactions_path]
        else:
            extra_files = extra_transactions_path
        
        # Leer y concatenar TODOS los archivos nuevos
        new_dfs = [transacciones_hist]  # Empezar con histórico
        
        for filepath in extra_files:
            if os.path.exists(filepath):
                transacciones_new = pd.read_parquet(filepath)
                new_dfs.append(transacciones_new)
                print(f"  [load_raw_data] Cargado: {os.path.basename(filepath)} con {len(transacciones_new):,} registros")
            else:
                print(f"  [load_raw_data] ADVERTENCIA: No se encontró {filepath}")
        
        # Concatenar histórico + todos los nuevos
        transacciones = pd.concat(new_dfs, ignore_index=True)
        print(f"  [load_raw_data] Total transacciones: {len(transacciones):,} registros")
    else:
        transacciones = transacciones_hist
        print(f"  [load_raw_data] Solo histórico: {len(transacciones):,} registros")

    return clientes, productos, transacciones


# ----------------------------------------------------------------------
# 2. Casting de tipos + limpieza básica de transacciones
# ----------------------------------------------------------------------

def cast_and_clean_raw_tables(
    clientes: pd.DataFrame,
    productos: pd.DataFrame,
    transacciones: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aplica los mismos cambios de tipos de datos y limpieza básica
    que en la entrega 1.
    """

    # --- dtypes clientes ---
    clientes = clientes.astype(
        {
            "customer_id": "string",
            "region_id": "string",
            "zone_id": "string",
            "customer_type": "category",
        }
    )

    # --- dtypes productos ---
    productos = productos.astype(
        {
            "product_id": "string",
            "brand": "category",
            "category": "category",
            "sub_category": "category",
            "segment": "category",
            "package": "category",
        }
    )

    # --- dtypes transacciones ---
    transacciones = transacciones.astype(
        {
            "customer_id": "string",
            "product_id": "string",
            "order_id": "string",
        }
    )

    # Asegurar que purchase_date es datetime
    transacciones["purchase_date"] = pd.to_datetime(
        transacciones["purchase_date"], errors="coerce"
    )

    # Componentes de fecha
    transacciones["year"] = transacciones["purchase_date"].dt.year
    transacciones["month"] = transacciones["purchase_date"].dt.month
    transacciones["day"] = transacciones["purchase_date"].dt.day

    return clientes, productos, transacciones


def deduplicate_and_fix_transactions(
    transacciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replica la lógica de deduplicación y consolidación de items de la Entrega 1:
    - elimina duplicados completos
    - consolida registros con misma (customer_id, product_id, order_id, purchase_date)
      sumando items
    - descarta registros con items <= 0
    """

    # Elimina duplicados completos
    transacciones = transacciones.drop_duplicates()

    cols_clave = ["customer_id", "product_id", "order_id", "purchase_date"]

    # Detecta misma clave con más de un valor de items
    mask_parcial = (
        transacciones.groupby(cols_clave)["items"].transform("nunique") > 1
    )

    df_parcial = transacciones[mask_parcial].copy()
    df_no_parcial = transacciones[~mask_parcial].copy()

    # Consolidar parciales sumando items
    df_parcial_fix = (
        df_parcial.groupby(cols_clave, as_index=False)
        .agg(
            items=("items", "sum"),
            year=("year", "first"),
            month=("month", "first"),
            day=("day", "first"),
        )
    )

    transacciones_fix = pd.concat(
        [df_no_parcial, df_parcial_fix], ignore_index=True
    )
    transacciones_fix = transacciones_fix.sort_values(cols_clave).reset_index(
        drop=True
    )

    # Descarta netos <= 0
    transacciones_fix = transacciones_fix[transacciones_fix["items"] > 0]

    return transacciones_fix


# ----------------------------------------------------------------------
# 3. Merge a nivel transacción -> df
# ----------------------------------------------------------------------

def build_transaction_level_df(
    clientes: pd.DataFrame,
    productos: pd.DataFrame,
    transacciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Hace el merge de transacciones con clientes y productos,
    y aplica los mismos filtros finales que en la entrega 1.
    """

    # Merge transacciones + clientes + productos
    df = transacciones.merge(clientes, on="customer_id", how="left")
    df = df.merge(productos, on="product_id", how="left")

    # Filtrar items > 0 por si algo quedo
    df = df[df["items"] > 0]

    # Convertir ciertas categóricas a string
    for c in ["customer_type", "brand", "category", "zone_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df


# ----------------------------------------------------------------------
# 4. Construcción del panel semanal y target "y"
# ----------------------------------------------------------------------

def build_weekly_panel_with_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del df a nivel de transacción:
    - genera un panel semanal por (customer_id, product_id)
    - define compra_o_no en semana t
    - define target y = compra en semana t+1
    - expande a cartesiano completo cliente-producto por semana
    - agrega dimensiones de cliente y producto
    - construye semana_num para splits temporales
    Todo heredado de lo que se hizo en la entrega 1.
    """

    df_ = df.copy()
    df_["purchase_date"] = pd.to_datetime(df_["purchase_date"], errors="coerce")
    df_ = df_.dropna(subset=["purchase_date"])

    # Resample semanal por par (cliente, producto)
    panel = (
        df_.set_index("purchase_date")
        .groupby(["customer_id", "product_id"])
        .resample("W-MON")
        .size()
        .rename("purchased_count")
        .reset_index()
    )

    panel.rename(columns={"purchase_date": "week_t"}, inplace=True)
    panel["compra_o_no"] = (panel["purchased_count"] > 0).astype(int)
    panel = panel.sort_values(["customer_id", "product_id", "week_t"])

    # Target: compra en la próxima semana
    panel["y"] = panel.groupby(["customer_id", "product_id"])["compra_o_no"].shift(
        -1
    )
    panel = panel.dropna(subset=["y"]).copy()
    panel["y"] = panel["y"].astype(int)

    # Semana t+1
    panel["week_t_plus_1"] = panel["week_t"] + pd.Timedelta(days=7)

    # Etiquetas YYYY-ww 
    iso = panel["week_t"].dt.isocalendar()
    panel["semana"] = (
        iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2)
    )

    iso1 = panel["week_t_plus_1"].dt.isocalendar()
    panel["semana_siguiente_str"] = (
        iso1.year.astype(str) + "-" + iso1.week.astype(str).str.zfill(2)
    )

    # Dataset panel observado
    df_copy = panel[
        [
            "customer_id",
            "product_id",
            "week_t",
            "week_t_plus_1",
            "semana",
            "semana_siguiente_str",
            "purchased_count",
            "compra_o_no",
            "y",
        ]
    ].reset_index(drop=True)

    # --------------------------------------------------------------
    # Expansión a cartesiano cliente-producto por cada semana
    # --------------------------------------------------------------
    customers = (
        df["customer_id"].drop_duplicates().sort_values().to_numpy()
    )
    products = df["product_id"].drop_duplicates().sort_values().to_numpy()

    weeks = (
        pd.to_datetime(df_copy["week_t"])
        .drop_duplicates()
        .sort_values()
        .to_list()
    )

    frames = []
    n_prod = len(products)
    n_cust = len(customers)

    for wk in weeks:
        wk_ts = pd.Timestamp(wk)
        wk_plus1 = wk_ts + pd.Timedelta(days=7)

        # Etiquetas semana actual y siguiente
        iso0 = wk_ts.isocalendar()
        iso1 = wk_plus1.isocalendar()
        semana_str = f"{iso0.year}-{int(iso0.week):02d}"
        semana_next_str = f"{iso1.year}-{int(iso1.week):02d}"

        # Producto cartesiano clientes x productos para esta semana
        base = pd.DataFrame(
            {
                "customer_id": np.repeat(customers, n_prod),
                "product_id": np.tile(products, n_cust),
                "week_t": wk_ts,
                "week_t_plus_1": wk_plus1,
                "semana": semana_str,
                "semana_siguiente_str": semana_next_str,
            }
        )

        # Filas reales de esa semana desde df_copy
        wk_rows = df_copy.loc[
            df_copy["week_t"].eq(wk_ts),
            ["customer_id", "product_id", "purchased_count", "compra_o_no", "y"],
        ]

        # Merge por semana 
        merged = base.merge(wk_rows, on=["customer_id", "product_id"], how="left")

        # Rellenar con 0 donde no hubo compra
        for c in ["purchased_count", "compra_o_no", "y"]:
            merged[c] = merged[c].fillna(0).astype(int)

        frames.append(merged)

    df_full = pd.concat(frames, ignore_index=True)

    # --------------------------------------------------------------
    # Agregar features de clientes y productos
    # --------------------------------------------------------------
    cols_cli = [
        "customer_id",
        "customer_type",
        "X",
        "Y",
        "zone_id",
        "region_id",
        "num_deliver_per_week",
        "num_visit_per_week",
    ]
    cols_cli = [c for c in cols_cli if c in df.columns]

    cols_prod = [
        "product_id",
        "brand",
        "category",
        "sub_category",
        "segment",
        "package",
        "size",
    ]
    cols_prod = [c for c in cols_prod if c in df.columns]

    dim_clientes = (
        df[cols_cli]
        .drop_duplicates(subset=["customer_id"])
        .reset_index(drop=True)
    )
    dim_productos = (
        df[cols_prod]
        .drop_duplicates(subset=["product_id"])
        .reset_index(drop=True)
    )

    df_final = (
        df_full.merge(dim_clientes, on="customer_id", how="left")
        .merge(dim_productos, on="product_id", how="left")
    )

    # semana_num = YYYY * 100 + WW (para splits temporales)
    df_final["semana_num"] = (
        df_final["semana"].str.split("-").str[0].astype(int) * 100
        + df_final["semana"].str.split("-").str[1].astype(int)
    )

    return df_final


# ----------------------------------------------------------------------
# 5. Orquestador: de crudo → df_final
# ----------------------------------------------------------------------

def build_model_dataset(
    data_dir: str,
    new_transactions_filename: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Pipeline completo de preparación:
    - carga parquet histórico (clientes, productos, transacciones)
    - opcionalmente concatena uno o más archivos extra de transacciones (nueva semana)
    - cast dtypes
    - dedup transacciones
    - merge a nivel transacción
    - panel semanal y target

    Args:
        data_dir: directorio con los archivos base
        new_transactions_filename: puede ser None, un string (archivo único),
                                   o una lista de strings (múltiples archivos)
    """

    # Construir paths completos si hay archivos nuevos
    extra_paths = None
    if new_transactions_filename is not None:
        if isinstance(new_transactions_filename, str):
            extra_paths = os.path.join(data_dir, new_transactions_filename)
        else:
            # Es una lista
            extra_paths = [os.path.join(data_dir, f) for f in new_transactions_filename]

    clientes, productos, transacciones = load_raw_data(
        data_dir, extra_transactions_path=extra_paths
    )
    clientes, productos, transacciones = cast_and_clean_raw_tables(
        clientes, productos, transacciones
    )
    transacciones = deduplicate_and_fix_transactions(transacciones)
    df = build_transaction_level_df(clientes, productos, transacciones)
    df_final = build_weekly_panel_with_target(df)
    return df_final


def build_next_week_candidates_from_raw(
    data_dir: str,
    new_transactions_filename: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Construye el dataset de candidatos para predecir la próxima semana (t+2).

    Lógica:
    - Carga clientes, productos, transacciones (histórico + opcional uno o más archivos nuevos).
    - Limpia/castea y deduplica transacciones (mismas funciones que build_model_dataset).
    - Calcula la serie semanal (W-MON) por (customer_id, product_id).
    - Identifica la última semana disponible (t+1).
    - Construye todas las combinaciones cliente-producto para esa última semana (t+1),
      con sus features semanales (purchased_count, compra_o_no) y features estáticas,
      pero dejando week_t = t+1 y week_t_plus_1 = t+2.
    - Devuelve un df con la misma estructura de features que df_final, pero sin "y",
      pensado para alimentar al pipeline XGBoost y predecir t+2.
      
    Args:
        data_dir: directorio con los archivos base
        new_transactions_filename: puede ser None, un string (archivo único),
                                   o una lista de strings (múltiples archivos)
    """

    # Construir paths completos si hay archivos nuevos
    extra_paths = None
    if new_transactions_filename is not None:
        if isinstance(new_transactions_filename, str):
            extra_paths = os.path.join(data_dir, new_transactions_filename)
        else:
            # Es una lista
            extra_paths = [os.path.join(data_dir, f) for f in new_transactions_filename]

    clientes, productos, transacciones = load_raw_data(
        data_dir=data_dir,
        extra_transactions_path=extra_paths,
    )
    clientes, productos, transacciones = cast_and_clean_raw_tables(
        clientes, productos, transacciones
    )
    transacciones = deduplicate_and_fix_transactions(transacciones)

    # df a nivel transacción (con features estáticas)
    df = build_transaction_level_df(clientes, productos, transacciones)

    # Panel semanal solo para construir features de la última semana
    df_ = df.copy()
    df_["purchase_date"] = pd.to_datetime(df_["purchase_date"], errors="coerce")
    df_ = df_.dropna(subset=["purchase_date"])

    weekly = (
        df_.set_index("purchase_date")
        .groupby(["customer_id", "product_id"])
        .resample("W-MON")
        .size()
        .rename("purchased_count")
        .reset_index()
    )
    weekly["compra_o_no"] = (weekly["purchased_count"] > 0).astype(int)

    # Última semana observada (t+1) en los datos
    last_week = weekly["purchase_date"].max()
    next_week = last_week + pd.Timedelta(days=7)  # esto será t+2

    # Etiquetas tipo YYYY-ww
    iso_last = last_week.isocalendar()
    semana_str = f"{iso_last.year}-{int(iso_last.week):02d}"

    iso_next = next_week.isocalendar()
    semana_next_str = f"{iso_next.year}-{int(iso_next.week):02d}"

    # Producto cartesiano clientes × productos para esa última semana
    customers = df["customer_id"].drop_duplicates().sort_values().to_numpy()
    products = df["product_id"].drop_duplicates().sort_values().to_numpy()

    n_cust = len(customers)
    n_prod = len(products)

    base = pd.DataFrame(
        {
            "customer_id": np.repeat(customers, n_prod),
            "product_id": np.tile(products, n_cust),
            "week_t": last_week,          # t+1
            "week_t_plus_1": next_week,   # t+2 (la semana a predecir)
            "semana": semana_str,
            "semana_siguiente_str": semana_next_str,
        }
    )

    # Merge con las features semanales de la última semana (t+1)
    wk_rows = weekly.loc[
        weekly["purchase_date"].eq(last_week),
        ["customer_id", "product_id", "purchased_count", "compra_o_no"],
    ]

    candidates = base.merge(
        wk_rows,
        on=["customer_id", "product_id"],
        how="left",
    )

    for c in ["purchased_count", "compra_o_no"]:
        candidates[c] = candidates[c].fillna(0).astype(int)

    # Agregar dimensiones de clientes y productos (igual que df_final)
    cols_cli = [
        "customer_id",
        "customer_type",
        "X",
        "Y",
        "zone_id",
        "region_id",
        "num_deliver_per_week",
        "num_visit_per_week",
    ]
    cols_cli = [c for c in cols_cli if c in df.columns]

    cols_prod = [
        "product_id",
        "brand",
        "category",
        "sub_category",
        "segment",
        "package",
        "size",
    ]
    cols_prod = [c for c in cols_prod if c in df.columns]

    dim_clientes = (
        df[cols_cli]
        .drop_duplicates(subset=["customer_id"])
        .reset_index(drop=True)
    )
    dim_productos = (
        df[cols_prod]
        .drop_duplicates(subset=["product_id"])
        .reset_index(drop=True)
    )

    candidates = (
        candidates.merge(dim_clientes, on="customer_id", how="left")
        .merge(dim_productos, on="product_id", how="left")
    )

    # semana_num coherente con df_final (para trazabilidad / logging)
    candidates["semana_num"] = (
        candidates["semana"].str.split("-").str[0].astype(int) * 100
        + candidates["semana"].str.split("-").str[1].astype(int)
    )

    return candidates