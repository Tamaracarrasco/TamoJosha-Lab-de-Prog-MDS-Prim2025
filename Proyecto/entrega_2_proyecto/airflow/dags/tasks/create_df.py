# acá se define la función para transformar la data

from pathlib import Path
import pandas as pd
import numpy as np
import shutil

def create_dataframe(**context):
    """
    Une clientes, productos y transacciones de
    la misma forma que se hizo en la parte 1
    """
    base = Path(__file__).resolve().parents[2]
    baseline = base / "data" / "baseline"
    run_dir = base / "data" / "run" / context["ds"]
    raw_dir = run_dir / "raw"
    prep_dir = run_dir / "preprocessed"

    if not (raw_dir / "clientes.parquet").exists():
        for f in ["clientes.parquet", "productos.parquet", "transacciones.parquet"]:
            shutil.copy(baseline / f, raw_dir / f)

    prep_dir.mkdir(parents=True, exist_ok=True)

    # Leer archivos
    clientes = pd.read_parquet(raw_dir / "clientes.parquet")
    productos = pd.read_parquet(raw_dir / "productos.parquet")
    transacciones = pd.read_parquet(raw_dir / "transacciones.parquet")

    # dropear duplicados de transacciones

    transacciones = transacciones.drop_duplicates() 

    # Cruce de información
    df = transacciones.merge(clientes, on="customer_id", how="left")
    df = df.merge(productos, on="product_id", how="left")

    # Eliminación de posibles valores negativos residuales
    df = df[df["items"] > 0]

    # Convertir categóricas a string si aún no lo están
    for c in ["customer_type", "brand", "category", "zone_id"]:
        df[c] = df[c].astype(str)

    # Creacion variable objetivo y variables explicativas

    df_ = df.copy()
    df_["purchase_date"] = pd.to_datetime(df_["purchase_date"], errors="coerce")
    df_ = df_.dropna(subset=["purchase_date"])

    # Resample semanal por par (cliente, producto)
    panel = (
    df_.set_index("purchase_date")
       .groupby(["customer_id","product_id"])
       .resample("W-MON")                    # semanas ancladas a lunes
       .size()
       .rename("purchased_count")
       .reset_index()
)

    # Variables solicitadas
    panel.rename(columns={"purchase_date":"week_t"}, inplace=True)
    panel["compra_o_no"] = (panel["purchased_count"] > 0).astype(int)
    panel = panel.sort_values(["customer_id","product_id","week_t"])

    # Target: compra en la próxima semana (y en t+1)
    panel["y"] = panel.groupby(["customer_id","product_id"])["compra_o_no"].shift(-1)
    panel = panel.dropna(subset=["y"]).copy()
    panel["y"] = panel["y"].astype(int)

    # Semana t+1
    panel["week_t_plus_1"] = panel["week_t"] + pd.Timedelta(days=7)

    # Etiqueta "semana" estilo YYYY-ww (ISO)
    iso = panel["week_t"].dt.isocalendar()
    panel["semana"] = iso.year.astype(str) + "-" + iso.week.astype(str).str.zfill(2)

    # Etiqueta para t+1
    iso1 = panel["week_t_plus_1"].dt.isocalendar()
    panel["semana_siguiente_str"] = iso1.year.astype(str) + "-" + iso1.week.astype(str).str.zfill(2)

    # Dataset final
    df_copy = panel[[
        "customer_id","product_id",
        "week_t","week_t_plus_1","semana","semana_siguiente_str",
        "purchased_count","compra_o_no","y"
    ]].reset_index(drop=True)

    # Construcción final de la base

    customers = (
    df["customer_id"]
    .drop_duplicates()
    .sort_values()
    .to_numpy()
    )
    products = (
        df["product_id"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    # Semanas 
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

        # Etiquetas YYYY-ww para semana actual 
        iso0 = wk_ts.isocalendar()
        iso1 = wk_plus1.isocalendar()
        semana_str = f"{iso0.year}-{int(iso0.week):02d}"
        semana_next_str = f"{iso1.year}-{int(iso1.week):02d}"

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
            ["customer_id","product_id","purchased_count","compra_o_no","y"]
        ]

        # Merge por semana (left = todos los pares; right = observados)
        merged = base.merge(wk_rows, on=["customer_id","product_id"], how="left")

        # Relleno con 0 donde no hubo compra 
        for c in ["purchased_count","compra_o_no","y"]:
            merged[c] = merged[c].fillna(0).astype(int)

        frames.append(merged)

    # Se concatenan todas las semanas

    df_full = pd.concat(frames, ignore_index=True)

    # Acá añadimos las features de clientes y productos

    cols_cli = ["customer_id","customer_type","X","Y","zone_id",
                "region_id","num_deliver_per_week","num_visit_per_week"]
    cols_cli = [c for c in cols_cli if c in df.columns]  # por si falta alguna

    cols_prod = ["product_id","brand","category","sub_category",
                "segment","package","size"]
    cols_prod = [c for c in cols_prod if c in df.columns]

    # Dimensión clientes (1 fila por customer_id)
    dim_clientes = (
        df[cols_cli]
        .drop_duplicates(subset=["customer_id"])
        .reset_index(drop=True)
    )

    # Dimensión productos (1 fila por product_id)
    dim_productos = (
        df[cols_prod]
        .drop_duplicates(subset=["product_id"])
        .reset_index(drop=True)
    )

    # Merge de dimensiones (LEFT, para no perder combinaciones)
    df_final = (
        df_full
        .merge(dim_clientes, on="customer_id", how="left")
        .merge(dim_productos, on="product_id", how="left")
    )

    df_final.to_parquet(prep_dir / "df_unified.parquet", index=False)
    print(f"Datos transformados y guardados en {prep_dir}")



def split_data(**context):
    """Acá se busca crear los conjuntos de
    entrenamiento, validación y testeo"""

    ds = context["ds"]
    run_dir = Path(__file__).resolve().parents[3] / "data" / "run" / ds
    prep_file = run_dir / "preprocessed" / "df_unified.parquet"

    df_final = pd.read_csv(prep_file)

    df_final["semana_num"] = (
    df_final["semana"].str.split("-").str[0].astype(int) * 100 +
    df_final["semana"].str.split("-").str[1].astype(int)
    )

    df_final = df_final.sort_values("semana_num")

    weeks_sorted = sorted(df_final["semana_num"].unique())

    # Decidimos que hasta septiembre fuera entrenamiento
    train_weeks = weeks_sorted[:36]
    val_weeks = weeks_sorted[36:36+11] # validación meses de octubre y noviembre
    test_weeks = weeks_sorted[36+11:] #  diciembre como muestra de testeo.

    train_df = df_final[df_final["semana_num"].isin(train_weeks)]
    val_df = df_final[df_final["semana_num"].isin(val_weeks)]
    test_df = df_final[df_final["semana_num"].isin(test_weeks)]

    # print(f"Train: {train_df.shape[0]} filas, semanas {train_weeks[0]} → {train_weeks[-1]}")
    # print(f"Val: {val_df.shape[0]} filas, semanas {val_weeks[0]} → {val_weeks[-1]}")
    # print(f"Test: {test_df.shape[0]} filas, semanas {test_weeks[0]} → {test_weeks[-1]}")

    drop_cols = ["y"]

    X_train = train_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_train = train_df["y"].astype(int).reset_index(drop=True)

    X_val   = val_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_val   = val_df["y"].astype(int).reset_index(drop=True)

    X_test  = test_df.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)
    y_test  = test_df["y"].astype(int).reset_index(drop=True)

    # guardar
    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    X_train.assign(y=y_train).to_csv(splits_dir / "train.csv", index=False)
    X_val.assign(y=y_val).to_csv(splits_dir / "val.csv", index=False)
    X_test.assign(y=y_test).to_csv(splits_dir / "test.csv", index=False)

    print(f"[split_data] Guardados: {splits_dir / 'train.csv'} , {splits_dir / 'val.csv'} y {splits_dir / 'test.csv'}")