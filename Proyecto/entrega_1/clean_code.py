# =============================================================================
# ENTREGA 1 CORREGIDA - SIN DATA LEAKAGE + FEATURES TEMPORALES
# =============================================================================

import os
import time
from datetime import datetime, time as dt_time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

from pandas.api.types import is_numeric_dtype

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, r2_score
)

from xgboost import XGBClassifier

import shap
import optuna
from sklearn.linear_model import LinearRegression

pio.renderers.default = "notebook"
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# 1. EXTRACCIÓN DE DATOS
# =============================================================================

clientes = pd.read_parquet("clientes.parquet")
productos = pd.read_parquet("productos.parquet")
transacciones = pd.read_parquet("transacciones.parquet")

print("="*10 + " Primeros 3 registros del dataset Clientes "+ "="*10)
display(clientes.head(3))

print("="*10 + " Primeros 3 registros del dataset Productos "+ "="*10)
display(productos.head(3))

print("="*10 + " Primeros 3 registros del dataset Transacciones "+ "="*10)
display(transacciones.head(3))

print("="*10 + " Dimensiones de los datasets " + "="*10)
print("Clientes: ", clientes.shape)
print("Productos: ", productos.shape)
print("Transacciones: ", transacciones.shape)

client_columns = clientes.columns.tolist()
products_columns = productos.columns.tolist()
transactions_columns = transacciones.columns.tolist()

print("Columnas del archivo clientes: ", client_columns)
print("Columnas del archivo productos: ", products_columns)
print("Columnas del archivo transacciones: ", transactions_columns)

print("="*30 + " Tipo de valores en las columnas (clientes) "+ "="*30)
print(clientes.dtypes)
print("="*30 + " Tipo de valores en las columnas (productos) "+ "="*30)
print(productos.dtypes)
print("="*30 + " Tipo de valores en las columnas (transacciones) "+ "="*30)
print(transacciones.dtypes)
print("="*80)


# =============================================================================
# 2. LIMPIEZA Y TRANSFORMACIÓN
# =============================================================================

# 2.1 Ajuste de tipos de datos
clientes = clientes.astype({
    "customer_id": "string",
    "region_id": "string",
    "zone_id": "string",
    "customer_type": "category"
})

productos = productos.astype({
    "product_id": "string",
    "brand": "category",
    "category": "category",
    "sub_category": "category",
    "segment": "category",
    "package": "category"
})

transacciones = transacciones.astype({
    "customer_id": "string",
    "product_id": "string",
    "order_id": "string"
})

print("="*30 + " Tipo de valores después de casting "+ "="*30)
print("\nClientes dtypes:")
print(clientes.dtypes)
print("\nProductos dtypes:")
print(productos.dtypes)
print("\nTransacciones dtypes:")
print(transacciones.dtypes)

# 2.2 Estadística descriptiva básica
print("="*10 + " Estadística descriptiva clientes (numérico) " + "="*10)
print(clientes.describe())

print("="*10 + " Estadística descriptiva productos (numérico) " + "="*10)
print(productos.describe())

print("="*10 + " Estadística descriptiva transacciones (numérico) " + "="*10)
print(transacciones.describe())

print("="*10 + " Estadística descriptiva clientes (string) " + "="*10)
print(clientes.describe(include="string"))

print("="*10 + " Estadística descriptiva productos (string) " + "="*10)
print(productos.describe(include="string"))

print("="*10 + " Estadística descriptiva transacciones (string) " + "="*10)
print(transacciones.describe(include="string"))

print("="*10 + " Estadística descriptiva clientes (category) " + "="*10)
print(clientes.describe(include="category"))

print("="*10 + " Estadística descriptiva productos (category) " + "="*10)
print(productos.describe(include="category"))

# 2.3 Manejo columna de fecha
transacciones["purchase_date"] = pd.to_datetime(transacciones["purchase_date"], errors="coerce")

print("Tipo de dato purchase_date:", transacciones["purchase_date"].dtype)
print(transacciones["purchase_date"].head(3))

print("Min:", transacciones["purchase_date"].min())
print("Max:", transacciones["purchase_date"].max())
print("Unique:", transacciones["purchase_date"].nunique())
print("Nulos:", transacciones["purchase_date"].isna().sum())

print("Cantidad de valores únicos en .dt.time:",
      transacciones["purchase_date"].dt.time.nunique())

transacciones["year"] = transacciones["purchase_date"].dt.year
transacciones["month"] = transacciones["purchase_date"].dt.month
transacciones["day"] = transacciones["purchase_date"].dt.day

# 2.4 Identificación de valores nulos
print("Identificación de valores nulos por columna de cada dataset")
print("="*30 + " Dataset clientes "+ "="*30)
print(clientes.isna().sum())
print("="*30 + " Dataset productos "+ "="*30)
print(productos.isna().sum())
print("="*30 + " Dataset transacciones "+ "="*30)
print(transacciones.isna().sum())


# -----------------------------------------------------------------------------
# Función auxiliar para revisar subconjuntos problemáticos
# -----------------------------------------------------------------------------
def show_count_and_sample(df, mask, cols=None, sample_n=10, title=""):
    cnt = int(mask.sum())
    print(f"\n{title} -> {cnt:,} fila(s)")
    if cnt > 0:
        if cols:
            display(df.loc[mask, cols].head(sample_n))
        else:
            display(df.loc[mask].head(sample_n))


# 2.5 Checks de calidad de datos (igual que tu código original)
cols_clave = ["customer_id", "product_id", "order_id", "purchase_date"]
mask_dups_clave = transacciones.duplicated(subset=cols_clave, keep=False)
show_count_and_sample(
    transacciones,
    mask_dups_clave,
    cols=cols_clave + ["items"],
    title="(0) Duplicados por clave (customer_id, product_id, order_id, purchase_date)"
)

ITEMS_MAX = 1_000
mask_items_no_pos = transacciones["items"] <= 0
mask_items_muy_altos = transacciones["items"] > ITEMS_MAX

show_count_and_sample(
    transacciones, mask_items_no_pos,
    cols=["customer_id", "product_id", "order_id", "purchase_date", "items"],
    title="(1) items ≤ 0 (devoluciones/anulaciones o error)"
)
show_count_and_sample(
    transacciones, mask_items_muy_altos,
    cols=["customer_id", "product_id", "order_id", "purchase_date", "items"],
    title=f"(1b) items > {ITEMS_MAX} (posible carga errónea)"
)

mask_cli_fk = ~transacciones["customer_id"].isin(clientes["customer_id"])
mask_prod_fk = ~transacciones["product_id"].isin(productos["product_id"])

show_count_and_sample(
    transacciones, mask_cli_fk,
    cols=["customer_id", "order_id", "purchase_date"],
    title="(2) customer_id en transacciones que NO existen en clientes"
)
show_count_and_sample(
    transacciones, mask_prod_fk,
    cols=["product_id", "order_id", "purchase_date"],
    title="(2b) product_id en transacciones que NO existen en productos"
)

cli_por_order = transacciones.groupby("order_id")["customer_id"].nunique()
mask_order_multi_cli = transacciones["order_id"].isin(
    cli_por_order[cli_por_order > 1].index
)
show_count_and_sample(
    transacciones, mask_order_multi_cli,
    cols=["order_id", "customer_id", "product_id", "purchase_date", "items"],
    title="(3) order_id con múltiples customer_id (inconsistencia grave)"
)

fecha_min, fecha_max = transacciones["purchase_date"].min(), transacciones["purchase_date"].max()
mask_fecha_na = transacciones["purchase_date"].isna()
show_count_and_sample(
    transacciones, mask_fecha_na,
    title="(4) purchase_date nulo(s)"
)
print(f"Rango de fechas: {fecha_min} → {fecha_max}")

if {"Y", "X"}.issubset(clientes.columns):
    mask_lat_inv = (clientes["Y"] < -90) | (clientes["Y"] > 90)
    mask_lon_inv = (clientes["X"] < -180) | (clientes["X"] > 180)

    show_count_and_sample(
        clientes, mask_lat_inv,
        cols=["customer_id", "Y", "X"],
        title="(5) Latitudes fuera de rango (-90,90)"
    )
    show_count_and_sample(
        clientes, mask_lon_inv,
        cols=["customer_id", "Y", "X"],
        title="(5b) Longitudes fuera de rango (-180,180)"
    )

for col, name in [
    ("num_deliver_per_week", "entregas/semana"),
    ("num_visit_per_week", "visitas/semana")
]:
    if col in clientes.columns and is_numeric_dtype(clientes[col]):
        mask_neg = clientes[col] < 0
        mask_gt7 = clientes[col] > 7
        mask_float = (clientes[col] % 1 != 0)

        show_count_and_sample(
            clientes, mask_neg,
            cols=["customer_id", col],
            title=f"(6) {col} < 0 (inválido)"
        )
        show_count_and_sample(
            clientes, mask_gt7,
            cols=["customer_id", col],
            title=f"(6b) {col} > 7 (inválido si es semanal)"
        )
        show_count_and_sample(
            clientes, mask_float,
            cols=["customer_id", col],
            title=f"(6c) {col} no entero (revisar si deben ser enteros)"
        )

if "size" in productos.columns and is_numeric_dtype(productos["size"]):
    SIZE_MAX = 20
    mask_size_le0 = productos["size"] <= 0
    mask_size_gt = productos["size"] > SIZE_MAX

    show_count_and_sample(
        productos, mask_size_le0,
        cols=["product_id", "brand", "package", "size"],
        title="(7) size ≤ 0 (inválido)"
    )
    show_count_and_sample(
        productos, mask_size_gt,
        cols=["product_id", "brand", "package", "size"],
        title=f"(7b) size > {SIZE_MAX}L (posible error)"
    )

for c in ["brand", "category", "sub_category", "segment", "package"]:
    if c in productos.columns:
        mask_na = productos[c].isna() | (productos[c].astype(str).str.strip() == "")
        show_count_and_sample(
            productos, mask_na,
            cols=["product_id", c],
            title=f"(8) {c} nulo o vacío"
        )

mask_many_decimals = (transacciones["items"].round(4) != transacciones["items"])
show_count_and_sample(
    transacciones, mask_many_decimals,
    cols=["customer_id", "product_id", "order_id", "purchase_date", "items"],
    title="(9) items con >4 decimales (revisar si es esperado)"
)

# 2.6 Manejo de duplicados y consolidación
print("Identificación de registros duplicados por dataset")
print("="*30 + " Dataset clientes "+ "="*30)
print(clientes.duplicated(["customer_id", "X", "Y"]).sum())

print("="*30 + " Dataset productos "+ "="*30)
print(productos.duplicated().sum())

print("="*30 + " Dataset transacciones "+ "="*30)
print(transacciones.duplicated().sum())

transacciones = transacciones.drop_duplicates()
print("Dimensión transacciones tras eliminar duplicados: ", transacciones.shape)

cols_clave = ["customer_id", "product_id", "order_id", "purchase_date"]
mask_parcial = transacciones.groupby(cols_clave)["items"].transform("nunique") > 1

df_parcial = transacciones[mask_parcial].copy()
df_no_parcial = transacciones[~mask_parcial].copy()

df_parcial_fix = (
    df_parcial
    .groupby(cols_clave, as_index=False)
    .agg(
        items=("items", "sum"),
        year=("year", "first"),
        month=("month", "first"),
        day=("day", "first")
    )
)

transacciones_fix = pd.concat([df_no_parcial, df_parcial_fix], ignore_index=True)
transacciones_fix = transacciones_fix.sort_values(cols_clave).reset_index(drop=True)
transacciones_fix = transacciones_fix[transacciones_fix["items"] > 0]

assert (transacciones_fix.groupby(cols_clave)["items"].nunique() <= 1).all()
assert transacciones_fix.duplicated(subset=cols_clave, keep=False).sum() == 0

n_parciales_filas = len(df_parcial)
n_parciales_grupos = df_parcial.drop_duplicates(cols_clave).shape[0]
n_final = len(transacciones_fix)

print(f"Filas tras deduplicado total previo: {len(transacciones):,}")
print(f"Filas involucradas en duplicados parciales: {n_parciales_filas:,}")
print(f"Duplicados parciales (grupos clave): {n_parciales_grupos:,}")
print(f"Filas después de consolidar parciales: {n_final:,}")

transacciones = transacciones_fix.copy()


# =============================================================================
# 3. CONSTRUCCIÓN DE BASE PARA MODELADO (EDA igual que tu código)
# =============================================================================

print("Total clientes únicos: ", clientes['customer_id'].nunique())
print("Total productos únicos: ", productos['product_id'].nunique())

print(
    transacciones.groupby('customer_id')['order_id']
    .nunique()
    .describe()
)

prod_por_cli = (
    transacciones
    .groupby("customer_id")["product_id"]
    .nunique()
    .rename("n_productos_distintos")
    .reset_index()
)

promedio_global = prod_por_cli["n_productos_distintos"].mean()
print(f"Promedio de productos distintos por cliente: {promedio_global:.3f}")
print("\nDistribución de productos/cliente:")
print(prod_por_cli["n_productos_distintos"].describe())

print("\nTop 10 clientes por variedad de productos:")
print(
    prod_por_cli
    .set_index("customer_id")["n_productos_distintos"]
    .sort_values(ascending=False)
    .head(10)
)

tx_sorted = (
    transacciones
    .sort_values(["customer_id", "product_id", "purchase_date"])
    .loc[:, ["customer_id", "product_id", "purchase_date"]]
    .copy()
)

tx_sorted["days_since_last"] = (
    tx_sorted
    .groupby(["customer_id", "product_id"])["purchase_date"]
    .diff()
    .dt.days
)

recompras_validas = tx_sorted.dropna(subset=["days_since_last"]).copy()

rep_compra_por_sku = (
    recompras_validas
    .groupby("product_id")["days_since_last"]
    .agg(
        repurchase_mean_days="mean",
        repurchase_median_days="median",
        n_intervals="count"
    )
    .reset_index()
    .sort_values("repurchase_median_days")
)

print("Recompra promedio (por SKU) — primeras 10 filas:")
display(rep_compra_por_sku.head(10))

if "product_id" in productos.columns:
    rep_compra_por_sku = rep_compra_por_sku.merge(productos, on="product_id", how="left")

# Merge master df
df = transacciones.merge(clientes, on="customer_id", how="left")
df = df.merge(productos, on="product_id", how="left")

print("Dimensiones del dataset mergeado df: ", df.shape)
print("Valores Nulos en dataset df")
print(df.isna().sum())

print("Clientes en clientes:", clientes["customer_id"].nunique())
print("Clientes en transacciones:", df["customer_id"].nunique())
print("Clientes faltantes:", len(set(clientes["customer_id"]) - set(df["customer_id"])))

print("Productos en productos:", productos["product_id"].nunique())
print("Productos en transacciones:", df["product_id"].nunique())
print("Productos faltantes:", len(set(productos["product_id"]) - set(df["product_id"])))

print("Filas duplicadas en df mergeado:", df.duplicated().sum())

clientes_unicos = df["customer_id"].nunique()
productos_unicos = df["product_id"].nunique()
print(f"Hay un total de {clientes_unicos} clientes únicos")
print(f"Hay un total de {productos_unicos} productos únicos")

fecha_inicio = df["purchase_date"].min()
fecha_final = df["purchase_date"].max()
print("Fecha inicio de observaciones: ", fecha_inicio)
print("Fecha final de observaciones: ", fecha_final)

# 3.6 Series temporales agregadas (igual que tu código)
df2 = df.set_index("purchase_date").copy()

frecuencia_diaria = df2.resample('D').size().reset_index(name='frecuencia')

fig = px.line(
    frecuencia_diaria, x="purchase_date", y="frecuencia",
    markers=True, title="Frecuencia diaria de transacciones"
)
fig.show()

inicio = frecuencia_diaria["purchase_date"].min()
frecuencia_diaria["semana"] = (
    (frecuencia_diaria["purchase_date"] - inicio).dt.days // 7
)
frecuencia_diaria["dia_semana"] = frecuencia_diaria["purchase_date"].dt.weekday
mapa = {0: "Lun", 1: "Mar", 2: "Mie", 3: "Jue", 4: "Vie", 5: "Sab", 6: "Dom"}
frecuencia_diaria["dia_semana"] = frecuencia_diaria["dia_semana"].map(mapa)

pivot = frecuencia_diaria.pivot(
    index="semana", columns="dia_semana", values="frecuencia"
)
cols = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
pivot = pivot.reindex(columns=cols).fillna(0)

week_start_dates = inicio + pd.to_timedelta(pivot.index * 7, unit="D")

plt.figure(figsize=(16, 6))
for col in pivot.columns:
    plt.plot(week_start_dates, pivot[col], marker='o', label=col)
plt.title("Transacciones por día de la semana")
plt.xlabel(f"Semana (desde {inicio.date()})")
plt.ylabel("Transacciones")
plt.legend(title="Día")
plt.grid(alpha=0.5)
plt.show()

frecuencia_semanal = df2.resample('W').size().reset_index(name='frecuencia')
plt.figure(figsize=(12, 6))
sns.lineplot(data=frecuencia_semanal, x="purchase_date", y="frecuencia", marker="o")
plt.title("Frecuencia semanal de transacciones")
plt.grid(alpha=0.5)
plt.show()

frecuencia_mensual = df2.resample("ME").size().reset_index(name="frecuencia")
plt.figure(figsize=(12, 6))
sns.lineplot(data=frecuencia_mensual, x="purchase_date", y="frecuencia", marker="o")
plt.title("Frecuencia mensual de transacciones")
plt.grid(alpha=0.5)
plt.show()

# 3.7 Distribuciones de variables numéricas
plt.figure()
plt.title("Distribución de Items")
sns.histplot(data=df, x="items", log_scale=True, bins=20)
plt.yscale("log")
plt.show()

plt.figure()
plt.title("Distribución de tamaño del producto")
sns.histplot(data=df, x="size", log_scale=True, bins=20)
plt.yscale("log")
plt.show()

plt.figure()
plt.title("Distribución de Número de entregas por semana")
sns.histplot(data=df, x="num_deliver_per_week", bins=20)
plt.yscale("log")
plt.show()

df["customer_id"] = df["customer_id"].astype(str)

copia = df.sort_values(["customer_id", "purchase_date"]).reset_index(drop=True)
visitas = (
    copia
    .groupby(["customer_id", "X", "Y", "customer_type"], observed=True)
    .agg(
        primera_visita=("purchase_date", "min"),
        ultima_visita=("purchase_date", "max"),
        transacciones_tot=("order_id", "nunique"),
        productos_tot=("product_id", "count")
    )
    .reset_index()
)

visitas["num_semanas"] = (
    (visitas["ultima_visita"] - visitas["primera_visita"]).dt.days / 7
).round(1).apply(lambda x: max(1, np.ceil(x)))

visitas["promedio_productos_por_semana"] = (
    visitas["productos_tot"] / visitas["num_semanas"]
)

customer_type_list = [
    'ABARROTES', 'CANAL FRIO', 'MAYORISTA', 'MINIMARKET',
    'RESTAURANT', 'SUPERMERCADO', 'TIENDA DE CONVENIENCIA'
]

for tipo in customer_type_list:
    print(f"========= estadísticas del tipo de cliente {tipo} ==========")
    print(
        visitas.loc[
            visitas["customer_type"] == tipo,
            "promedio_productos_por_semana"
        ].describe()
    )
    print("\n")

copia["week"] = copia["purchase_date"].dt.to_period("W").apply(
    lambda r: r.start_time
)

weekly_freq = (
    copia
    .groupby(
        ["customer_id", "customer_type", "product_id", "week",
         "segment", "brand", "package"],
        observed=True
    )["product_id"]
    .count()
    .reset_index(name="productos_comprados")
)

print("Dimensiones de weekly_freq:", weekly_freq.shape)

freq_summary = (
    weekly_freq
    .groupby(
        ["customer_id", "customer_type", "product_id",
         "segment", "brand", "package"],
        observed=True
    )["productos_comprados"]
    .mean()
    .reset_index(name="promedio_productos_por_semana")
)

fig = px.scatter(
    visitas, x="X", y="Y",
    size="promedio_productos_por_semana", color="customer_type"
)
fig.show()

visitas_geo = visitas[
    (visitas["X"] <= -100) & (visitas["Y"] >= -60)
]
fig = px.scatter(
    visitas_geo, x="X", y="Y",
    size="promedio_productos_por_semana", color="customer_type",
    title="Distribución espacial de los clientes según tipo"
)
fig.show()

top_prod = df.groupby("product_id")["items"].sum().sort_values(ascending=False).head(10)
top_cli = df.groupby("customer_id")["order_id"].nunique().sort_values(ascending=False).head(10)

print("Top 10 productos más vendidos:\n", top_prod)
print("\nTop 10 clientes con más transacciones:\n", top_cli)

cross = pd.crosstab(df["customer_type"], df["segment"])
sns.heatmap(cross, cmap="Reds", annot=False)
plt.title("Volumen de productos por tipo de cliente y segmento")
plt.show()

sns.boxplot(data=df, x="items")
plt.title("Outliers en número de ítems por transacción")
plt.show()

df = df[df["items"] > 0]

for c in ["customer_type", "brand", "category", "zone_id", "region_id"]:
    if c in df.columns:
        df[c] = df[c].astype(str)

print("Dataset final listo para modelado: ", df.shape)


# =============================================================================
# 3.13 CONSTRUCCIÓN DEL PANEL SEMANAL COMPLETO (CORREGIDO)
# =============================================================================

print("\n" + "="*80)
print("CONSTRUCCIÓN DEL PANEL SEMANAL - VERSIÓN CORREGIDA")
print("="*80)

df_ = df.copy()
df_["purchase_date"] = pd.to_datetime(df_["purchase_date"], errors="coerce")
df_ = df_.dropna(subset=["purchase_date"])

# Paso 1: Identificar universo completo
customers = df_["customer_id"].drop_duplicates().sort_values().to_numpy()
products = df_["product_id"].drop_duplicates().sort_values().to_numpy()

# Generar rango COMPLETO de semanas (sin gaps)
fecha_min = df_["purchase_date"].min()
fecha_max = df_["purchase_date"].max()

primer_lunes = fecha_min - pd.Timedelta(days=fecha_min.weekday())
ultimo_lunes = fecha_max - pd.Timedelta(days=fecha_max.weekday())

weeks_completo = pd.date_range(
    start=primer_lunes,
    end=ultimo_lunes,
    freq="W-MON"
)

print(f"Clientes: {len(customers):,}")
print(f"Productos: {len(products):,}")
print(f"Semanas: {len(weeks_completo)}")
print(f"Combinaciones totales: {len(customers) * len(products) * len(weeks_completo):,}")

# Paso 2: Calcular compras observadas por semana
df_tx = df_.copy()
df_tx["week_start"] = (
    df_tx["purchase_date"] - pd.to_timedelta(df_tx["purchase_date"].dt.weekday, unit='D')
)

compras_por_semana = (
    df_tx
    .groupby(["customer_id", "product_id", "week_start"])
    .agg(
        items_comprados=("items", "sum"),
        n_ordenes=("order_id", "nunique")
    )
    .reset_index()
)

compras_por_semana["compro"] = 1

# Paso 3: Construir cartesiano COMPLETO por semana
print("\nGenerando panel cartesiano completo...")
frames = []
n_prod = len(products)
n_cust = len(customers)

for i, wk in enumerate(weeks_completo):
    if i % 10 == 0:
        print(f"Procesando semana {i+1}/{len(weeks_completo)}...", end="\r")
    
    wk_ts = pd.Timestamp(wk)
    
    # Base cartesiana completa
    base = pd.DataFrame({
        "customer_id": np.repeat(customers, n_prod),
        "product_id": np.tile(products, n_cust),
        "week_t": wk_ts,
    })
    
    # Merge con compras observadas
    compras_esta_semana = compras_por_semana[
        compras_por_semana["week_start"] == wk_ts
    ][["customer_id", "product_id", "items_comprados", "compro"]]
    
    merged = base.merge(
        compras_esta_semana,
        on=["customer_id", "product_id"],
        how="left"
    )
    
    # Rellenar con 0 donde no hubo compra
    merged["items_comprados"] = merged["items_comprados"].fillna(0).astype(int)
    merged["compro"] = merged["compro"].fillna(0).astype(int)
    
    frames.append(merged)

print("\nConcatenando semanas...")
panel = pd.concat(frames, ignore_index=True)

# Paso 4: Ordenar y crear target
panel = panel.sort_values(
    ["customer_id", "product_id", "week_t"]
).reset_index(drop=True)

# CORRECCIÓN CRÍTICA: Target mirando la siguiente semana
panel["y"] = (
    panel
    .groupby(["customer_id", "product_id"])["compro"]
    .shift(-1)
)

# Eliminar última semana (no tiene target)
panel = panel.dropna(subset=["y"]).copy()
panel["y"] = panel["y"].astype(int)

# Paso 5: Agregar metadatos temporales
panel["week_t_plus_1"] = panel["week_t"] + pd.Timedelta(days=7)

iso = panel["week_t"].dt.isocalendar()
panel["year"] = iso.year
panel["week_of_year"] = iso.week
panel["month"] = panel["week_t"].dt.month
panel["quarter"] = panel["week_t"].dt.quarter
panel["semana_num"] = iso.year * 100 + iso.week

print(f"\n Panel completo generado: {panel.shape}")
print(f"Tasa de compra actual (compro): {panel['compro'].mean():.4f}")
print(f"Tasa de target (y): {panel['y'].mean():.4f}")


# =============================================================================
# 3.14 FEATURE ENGINEERING TEMPORAL (SIN LEAKAGE) 
# =============================================================================

print("\n" + "="*80)
print("CREACIÓN DE FEATURES TEMPORALES (SIN LEAKAGE)")
print("="*80)

# ============================================
# FEATURES LAG (mirando hacia atrás)
# ============================================
print("→ Creando lags de compras...")

for lag in [1, 2, 3, 4]:
    panel[f"compro_t_minus_{lag}"] = (
        panel
        .groupby(["customer_id", "product_id"])["compro"]
        .shift(lag)
        .fillna(0)
        .astype(int)
    )

# ============================================
# VENTANAS MÓVILES (rolling windows)
# ============================================
print("→ Creando ventanas móviles...")

for window in [4, 8, 12]:
    # Número de compras en ventana
    panel[f"num_compras_last_{window}w"] = (
        panel
        .groupby(["customer_id", "product_id"])["compro"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    ).fillna(0)
    
    # Items promedio en ventana
    panel[f"avg_items_last_{window}w"] = (
        panel
        .groupby(["customer_id", "product_id"])["items_comprados"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    ).fillna(0)

# ============================================
# RECENCIA (weeks since last purchase)
# ============================================
print("→ Calculando recencia...")

def calcular_recencia(grupo):
    """Calcula semanas desde última compra para un (cliente, producto)."""
    recencia = []
    ultima_compra = -999

    for i, compro in enumerate(grupo["compro"]):
        if i == 0:
            # Primera observación: no hay historia previa
            recencia.append(999)
        else:
            if ultima_compra == -999:
                # Nunca ha comprado antes
                recencia.append(999)
            else:
                # Distancia en semanas desde la última compra
                recencia.append(i - ultima_compra)

        if compro == 1:
            ultima_compra = i

    # Devolver una Serie con el MISMO índice que el grupo
    return pd.Series(recencia, index=grupo.index)

panel["semanas_desde_ultima_compra"] = (
    panel
    .groupby(["customer_id", "product_id"], group_keys=False)
    .apply(calcular_recencia)
    .astype(int)
)


# ============================================
# FRECUENCIA HISTÓRICA
# ============================================
print("→ Calculando frecuencia de compra...")

panel["tasa_compra_historica"] = (
    panel
    .groupby(["customer_id", "product_id"])["compro"]
    .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
).fillna(0)

# ============================================
# FEATURES A NIVEL CLIENTE (cross-product)
# ============================================
print("→ Features a nivel cliente...")

for window in [4, 8]:
    temp = (
        panel[panel["compro"] == 1]
        .groupby(["customer_id", "week_t"])["product_id"]
        .nunique()
        .reset_index(name="num_prod_distintos")
    )
    
    panel = panel.merge(temp, on=["customer_id", "week_t"], how="left")
    
    panel[f"num_prod_distintos_last_{window}w"] = (
        panel
        .groupby("customer_id")["num_prod_distintos"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    ).fillna(0)
    
    panel = panel.drop(columns=["num_prod_distintos"])

# ============================================
# FEATURES A NIVEL PRODUCTO (cross-customer)
# ============================================
print("→ Features a nivel producto...")

for window in [4, 8]:
    temp = (
        panel[panel["compro"] == 1]
        .groupby(["product_id", "week_t"])
        .size()
        .reset_index(name="veces_vendido")
    )
    
    panel = panel.merge(temp, on=["product_id", "week_t"], how="left")
    
    panel[f"popularidad_producto_last_{window}w"] = (
        panel
        .groupby("product_id")["veces_vendido"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    ).fillna(0)
    
    panel = panel.drop(columns=["veces_vendido"])

# ============================================
# Ya tenemos features temporales creadas:
# - year, week_of_year, month, quarter
# ============================================
print("→ Features de estacionalidad ya creadas (year, week_of_year, month, quarter)")

# Días desde inicio del dataset (tendencia)
panel["dias_desde_inicio"] = (panel["week_t"] - panel["week_t"].min()).dt.days

print(f"\nFeatures temporales creadas: {panel.shape[1]} columnas totales")


# =============================================================================
# 3.15 AGREGAR FEATURES ESTÁTICAS
# =============================================================================

print("\n" + "="*80)
print("AGREGANDO FEATURES ESTÁTICAS")
print("="*80)

# Features de clientes
cols_cli = [
    "customer_id", "customer_type", "X", "Y", "zone_id",
    "region_id", "num_deliver_per_week", "num_visit_per_week"
]
cols_cli = [c for c in cols_cli if c in df.columns]

dim_clientes = (
    df[cols_cli]
    .drop_duplicates(subset=["customer_id"])
    .reset_index(drop=True)
)

# Features de productos
cols_prod = [
    "product_id", "brand", "category", "sub_category",
    "segment", "package", "size"
]
cols_prod = [c for c in cols_prod if c in df.columns]

dim_productos = (
    df[cols_prod]
    .drop_duplicates(subset=["product_id"])
    .reset_index(drop=True)
)

# Merge
df_final = (
    panel
    .merge(dim_clientes, on="customer_id", how="left")
    .merge(dim_productos, on="product_id", how="left")
)

print(f" Dataset final con features: {df_final.shape}")


# =============================================================================
# 3.16 VERIFICACIÓN DE NO-LEAKAGE
# =============================================================================

print("\n" + "="*80)
print("VERIFICACIÓN DE AUSENCIA DE DATA LEAKAGE")
print("="*80)

# Verificar que 'compro' e 'items_comprados' NO se usarán como features
leakage_cols = ["compro", "items_comprados"]
leakage_encontrado = [c for c in leakage_cols if c in df_final.columns]

print(f"Columnas con potencial leakage detectadas: {leakage_encontrado}")
print(" IMPORTANTE: Estas columnas NO deben usarse como features en el modelo")
print(" Solo se usarán features que miran hacia atrás (lags, ventanas, recencia, etc.)")

# Verificar lags en un ejemplo
ejemplo_cliente = customers[0]
ejemplo_producto = products[0]

subset = df_final[
    (df_final["customer_id"] == ejemplo_cliente) &
    (df_final["product_id"] == ejemplo_producto)
].sort_values("week_t").head(15)

print(f"\n Verificación visual (Cliente {ejemplo_cliente}, Producto {ejemplo_producto}):")
print(subset[["week_t", "compro", "compro_t_minus_1", "num_compras_last_4w", "semanas_desde_ultima_compra", "y"]].to_string())

# Verificar que compro_t_minus_1 efectivamente viene de la semana anterior
ejemplo_compro_actual = subset["compro"].values
ejemplo_lag_1 = subset["compro_t_minus_1"].values[1:]
ejemplo_compro_anterior = ejemplo_compro_actual[:-1]

print(f"\n Verificación de lags:")
print(f"  ¿compro_t_minus_1 coincide con compro de semana anterior? {np.array_equal(ejemplo_lag_1, ejemplo_compro_anterior)}")

# Verificar consecutividad de semanas
diffs = subset["week_t"].diff().dt.days
non_weekly = diffs[(diffs > 0) & (diffs != 7)]

if len(non_weekly) == 0:
    print("   Semanas consecutivas sin gaps")
else:
    print(f"   ADVERTENCIA: Hay {len(non_weekly)} gaps en las semanas")


# =============================================================================
# 4. SPLIT TRAIN / VALIDACIÓN / TEST
# =============================================================================

print("\n" + "="*80)
print("SPLIT TEMPORAL")
print("="*80)

df_final = df_final.sort_values("semana_num").reset_index(drop=True)
weeks_sorted = sorted(df_final["semana_num"].unique())

train_weeks = weeks_sorted[:36]
val_weeks = weeks_sorted[36:36+11]
test_weeks = weeks_sorted[36+11:]

train_df = df_final[df_final["semana_num"].isin(train_weeks)]
val_df = df_final[df_final["semana_num"].isin(val_weeks)]
test_df = df_final[df_final["semana_num"].isin(test_weeks)]

print(f"Train: {train_df.shape[0]:,} filas, semanas {train_weeks[0]} → {train_weeks[-1]}")
print(f"Val:   {val_df.shape[0]:,} filas, semanas {val_weeks[0]} → {val_weeks[-1]}")
print(f"Test:  {test_df.shape[0]:,} filas, semanas {test_weeks[0]} → {test_weeks[-1]}")

# Preparar matrices X, y
#  CRÍTICO: Eliminamos columnas con leakage
drop_cols_target = ["y", "compro", "items_comprados"]  # ← SIN LEAKAGE

X_train = train_df.drop(columns=drop_cols_target, errors="ignore").reset_index(drop=True)
y_train = train_df["y"].astype(int).reset_index(drop=True)

X_val = val_df.drop(columns=drop_cols_target, errors="ignore").reset_index(drop=True)
y_val = val_df["y"].astype(int).reset_index(drop=True)

X_test = test_df.drop(columns=drop_cols_target, errors="ignore").reset_index(drop=True)
y_test = test_df["y"].astype(int).reset_index(drop=True)

print(f"\n Dimensiones finales:")
print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape} | y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}")

print("\n Balance por split:")
for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"{name}: {y.mean():.4f} (clase 1)")

# Visualización de distribución por semana y split
train_set = set(train_weeks)
val_set = set(val_weeks)
test_set = set(test_weeks)

df_plot = df_final.copy()
df_plot["split"] = np.where(
    df_plot["semana_num"].isin(train_set), "train",
    np.where(df_plot["semana_num"].isin(val_set), "val", "test")
)

weekly_counts = (
    df_plot
    .groupby(["semana_num", "split"])
    .size()
    .unstack("split", fill_value=0)
    .reindex(weeks_sorted, fill_value=0)
)

ax = weekly_counts.plot(kind="bar", figsize=(14, 4))
ax.set_title("Distribución semanal por split")
ax.set_xlabel("Semana (YYYYWW)")
ax.set_ylabel("# filas")
ax.legend(title="Split")
plt.tight_layout()
plt.show()

# Guardar dataset procesado
df_final.to_parquet("df_final_entrega_1_CORREGIDO.parquet", index=False)
print("\n Dataset guardado: df_final_entrega_1_CORREGIDO.parquet")


# =============================================================================
# 5. DEFINICIÓN DE PIPELINES Y MODELOS
# =============================================================================

print("\n" + "="*80)
print("DEFINICIÓN DE FEATURES PARA EL MODELO")
print("="*80)

# Columnas a eliminar (metadata)
drop_cols = [
    "customer_id", "product_id",
    "week_t", "week_t_plus_1",
    "semana_num", "year"  # year es redundante con week_of_year
]

#  FEATURES NUMÉRICAS (SIN LEAKAGE)
num_cols = [
    # Estáticas de cliente
    "X", "Y",
    "num_deliver_per_week", "num_visit_per_week",
    
    # Estáticas de producto
    "size",
    
    #  FEATURES TEMPORALES (todas miran hacia atrás)
    "compro_t_minus_1", "compro_t_minus_2", "compro_t_minus_3", "compro_t_minus_4",
    "num_compras_last_4w", "num_compras_last_8w", "num_compras_last_12w",
    "avg_items_last_4w", "avg_items_last_8w", "avg_items_last_12w",
    "semanas_desde_ultima_compra",
    "tasa_compra_historica",
    "num_prod_distintos_last_4w", "num_prod_distintos_last_8w",
    "popularidad_producto_last_4w", "popularidad_producto_last_8w",
    "week_of_year", "month", "quarter", "dias_desde_inicio"
]

# Verificar que existen
num_cols = [c for c in num_cols if c in X_train.columns]

# Features categóricas
cat_cols = [
    "customer_type", "segment", "brand",
    "category", "sub_category",
    "zone_id", "region_id"
]

cat_cols = [c for c in cat_cols if c in X_train.columns]

print(f"Features numéricas: {len(num_cols)}")
print(f"Features categóricas: {len(cat_cols)}")
print(f"Total features: {len(num_cols) + len(cat_cols)}")

print("\n IMPORTANTE: Todas las features numéricas miran hacia atrás (sin leakage)")
print("   - Lags: compro_t_minus_1 a compro_t_minus_4")
print("   - Ventanas: num_compras_last_Xw, avg_items_last_Xw")
print("   - Recencia: semanas_desde_ultima_compra")
print("   - Frecuencia: tasa_compra_historica")
print("   - Estacionalidad: week_of_year, month, quarter")

# Pipelines de transformación
num_pipeline = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

col_transformer = ColumnTransformer(
    transformers=[
        ("drop_ids", "drop", drop_cols),
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Balance de clases
n_neg = int((y_train == 0).sum())
n_pos = int((y_train == 1).sum())
scale_pos_weight = n_neg / n_pos

print(f"\n Balance de clases:")
print(f"  Negativos: {n_neg:,} ({n_neg/len(y_train)*100:.1f}%)")
print(f"  Positivos: {n_pos:,} ({n_pos/len(y_train)*100:.1f}%)")
print(f"  Scale pos weight: {scale_pos_weight:.2f}")


# =============================================================================
# 5.1 MODELO BASELINE: REGRESIÓN LOGÍSTICA
# =============================================================================

print("\n" + "="*80)
print("MODELO BASELINE: REGRESIÓN LOGÍSTICA")
print("="*80)

pipeline_reg_log = Pipeline(
    steps=[
        ("col_transformer", col_transformer),
        ("regressor", LogisticRegression(
            random_state=42, class_weight="balanced", n_jobs=-1, max_iter=1000
        ))
    ]
)

start_t_reg_log = time.time()
pipeline_reg_log.fit(X_train, y_train)
end_t_reg_log = time.time()

y_pred_val_reg_log = pipeline_reg_log.predict(X_val)

print(f" Tiempo entrenamiento: {end_t_reg_log - start_t_reg_log:.2f}s")
print("\nReporte en validación:")
print(classification_report(y_val, y_pred_val_reg_log, digits=3))


# =============================================================================
# 5.2 OTROS MODELOS BASE
# =============================================================================

print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS BASE")
print("="*80)

# SGD
pipeline_sgd = Pipeline(
    steps=[
        ("col_transformer", col_transformer),
        ("clasificador", SGDClassifier(
            random_state=42, max_iter=1000, class_weight="balanced"
        ))
    ]
)
start_t_sgd = time.time()
pipeline_sgd.fit(X_train, y_train)
end_t_sgd = time.time()
y_pred_val_sgd = pipeline_sgd.predict(X_val)

print(f"\n SGD Classifier")
print(f" Tiempo: {end_t_sgd - start_t_sgd:.2f}s")
print(classification_report(y_val, y_pred_val_sgd, digits=3))

# XGBoost base
pipeline_xgb = Pipeline(
    steps=[
        ("col_transformer", col_transformer),
        ("clasificador", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            objective="binary:logistic",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight
        ))
    ]
)
start_t_xgb = time.time()
pipeline_xgb.fit(X_train, y_train)
end_t_xgb = time.time()
y_pred_val_xgb = pipeline_xgb.predict(X_val)

print(f"\n XGBoost Base")
print(f" Tiempo: {end_t_xgb - start_t_xgb:.2f}s")
print(classification_report(y_val, y_pred_val_xgb, digits=3))

# Árbol de decisión
pipeline_tree = Pipeline(
    steps=[
        ("col_transformer", col_transformer),
        ("clasificador", DecisionTreeClassifier(
            random_state=42, max_depth=10, min_samples_leaf=50,
            class_weight="balanced"
        ))
    ]
)
start_t_tree = time.time()
pipeline_tree.fit(X_train, y_train)
end_t_tree = time.time()
y_pred_val_tree = pipeline_tree.predict(X_val)

print(f"\n Decision Tree")
print(f" Tiempo: {end_t_tree - start_t_tree:.2f}s")
print(classification_report(y_val, y_pred_val_tree, digits=3))


# =============================================================================
# 6. GUARDADO DE MODELOS BASE
# =============================================================================

base_path = os.path.join(os.getcwd(), "Proyecto", "proyecto", "data")
os.makedirs(base_path, exist_ok=True)

import pickle

pd.DataFrame({"predicciones": y_pred_val_reg_log}).to_csv(
    os.path.join(base_path, "predicciones_regresion_logistica.csv"),
    index=False
)
with open(os.path.join(base_path, "reg_log_model.sav"), "wb") as f:
    pickle.dump(pipeline_reg_log, f)

pd.DataFrame({"predicciones": y_pred_val_sgd}).to_csv(
    os.path.join(base_path, "predicciones_sgd.csv"),
    index=False
)
with open(os.path.join(base_path, "sgd_model.sav"), "wb") as f:
    pickle.dump(pipeline_sgd, f)

pd.DataFrame({"predicciones": y_pred_val_xgb}).to_csv(
    os.path.join(base_path, "predicciones_xgb.csv"),
    index=False
)
with open(os.path.join(base_path, "xgb_model.sav"), "wb") as f:
    pickle.dump(pipeline_xgb, f)

pd.DataFrame({"predicciones": y_pred_val_tree}).to_csv(
    os.path.join(base_path, "predicciones_tree.csv"),
    index=False
)
with open(os.path.join(base_path, "tree_model.sav"), "wb") as f:
    pickle.dump(pipeline_tree, f)

print("\n Modelos guardados en:", base_path)


# =============================================================================
# 7. OPTIMIZACIÓN CON OPTUNA (XGBoost)
# =============================================================================

print("\n" + "="*80)
print("OPTIMIZACIÓN CON OPTUNA")
print("="*80)

seed = 42

# Definir columnas para optimización
drop_cols_opt = ["customer_id", "product_id"]

# Convertir categóricas
for col in ['customer_type', 'brand', 'category', 'sub_category', 'segment']:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')

def objective(trial):
    params_xgb = {
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
    }
    params_ohe = {
        "min_frequency": trial.suggest_float("min_frequency", 0, 0.5)
    }

    num_pipeline_opt = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("sc", StandardScaler())
    ])
    
    cat_pipeline_opt = Pipeline([
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            **params_ohe
        ))
    ])

    col_transformer_opt = ColumnTransformer([
        ("drop_ids", "drop", drop_cols_opt),
        ("numerical", num_pipeline_opt, num_cols),
        ("categorical", cat_pipeline_opt, cat_cols)
    ],
    verbose_feature_names_out=False,
    remainder="drop")

    pipeline = Pipeline(steps=[
        ("col_transformer", col_transformer_opt),
        ("clasificador_xgb", XGBClassifier(
            objective="binary:logistic",
            random_state=seed,
            scale_pos_weight=scale_pos_weight,
            **params_xgb
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    
    return f1_score(y_val, y_pred)

print(" Iniciando búsqueda de hiperparámetros (5 min)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, timeout=5*60, n_jobs=-1, n_trials=50, show_progress_bar=True)

print(f"\n Optuna finalizado:")
print(f"  Trials: {len(study.trials)}")
print(f"  Mejor F1: {study.best_value:.4f}")
print(f"  Mejores parámetros:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")


# =============================================================================
# 8. MODELO FINAL OPTIMIZADO
# =============================================================================

print("\n" + "="*80)
print("MODELO FINAL OPTIMIZADO")
print("="*80)

params_xgb_best = {k: v for k, v in study.best_params.items() if k != "min_frequency"}
params_ohe_best = {"min_frequency": study.best_params.get("min_frequency", 0.0)}

num_pipeline_best = Pipeline([
    ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
    ("sc", StandardScaler())
])

cat_pipeline_best = Pipeline([
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        **params_ohe_best
    ))
])

col_transformer_best = ColumnTransformer([
    ("drop_ids", "drop", drop_cols_opt),
    ("numerical", num_pipeline_best, num_cols),
    ("categorical", cat_pipeline_best, cat_cols)
],
verbose_feature_names_out=False,
remainder="drop")

pipeline_xgb_op = Pipeline(steps=[
    ("col_transformer", col_transformer_best),
    ("clasificador_xgb", XGBClassifier(
        objective="binary:logistic",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        **params_xgb_best
    ))
])

print(" Entrenando modelo optimizado...")
start_t = time.time()
pipeline_xgb_op.fit(X_train, y_train)
end_t = time.time()

y_pred_val_op = pipeline_xgb_op.predict(X_val)

print(f" Tiempo: {end_t - start_t:.2f}s")
print("\n Reporte OPTIMIZADO en validación:")
print(classification_report(y_val, y_pred_val_op, digits=3))


# =============================================================================
# 9. EVALUACIÓN EN TEST
# =============================================================================

print("\n" + "="*80)
print("EVALUACIÓN FINAL EN TEST")
print("="*80)

# Búsqueda de umbral óptimo en validación
proba_val = pipeline_xgb_op.predict_proba(X_val)[:, 1]
ths = np.linspace(0.05, 0.8, 40)
scores = [f1_score(y_val, (proba_val >= t).astype(int)) for t in ths]
t_best = ths[np.argmax(scores)]

print(f" Umbral óptimo seleccionado: {t_best:.4f}")

# Predicciones en test
proba_test = pipeline_xgb_op.predict_proba(X_test)[:, 1]
y_pred_test = (proba_test >= t_best).astype(int)

print("\n Reporte FINAL en TEST:")
print(classification_report(y_test, y_pred_test, digits=3))


# =============================================================================
# 10. GENERAR TABLAS DE RANKING (TOP-N)
# =============================================================================

print("\n" + "="*80)
print("GENERACIÓN DE TABLAS DE RANKING")
print("="*80)

meta_cols = [c for c in ["customer_id", "product_id", "week_t", "week_t_plus_1"] 
             if c in df_final.columns]
meta_test = test_df[meta_cols].reset_index(drop=True)

tabla_probas = meta_test.copy()
tabla_probas["proba_compra_t1"] = proba_test
tabla_probas["pred_binaria"] = (proba_test >= t_best).astype(int)

if "week_t" in tabla_probas.columns:
    tabla_probas["rank_semana"] = (
        tabla_probas
        .groupby(["customer_id", "week_t"])["proba_compra_t1"]
        .rank(method="dense", ascending=False)
    )

N = 5
if "week_t" in tabla_probas.columns:
    topN = (
        tabla_probas
        .sort_values(
            ["customer_id", "week_t", "proba_compra_t1"],
            ascending=[True, True, False]
        )
        .groupby(["customer_id", "week_t"], as_index=False)
        .head(N)
        .reset_index(drop=True)
    )
else:
    topN = (
        tabla_probas
        .sort_values(
            ["customer_id", "proba_compra_t1"],
            ascending=[True, False]
        )
        .groupby("customer_id", as_index=False)
        .head(N)
        .reset_index(drop=True)
    )

print(f"Umbral seleccionado: {t_best:.4f}")
print(f"\n Tabla de probabilidades: {tabla_probas.shape}")
print("\n== Muestra (10 primeras filas) ==")
display(tabla_probas.head(10))

print(f"\n Top-{N} productos por cliente/semana: {topN.shape}")
print("\n== Muestra (10 primeras filas) ==")
display(topN.head(10))


# =============================================================================
# 11. INTERPRETABILIDAD CON SHAP
# =============================================================================

print("\n" + "="*80)
print("INTERPRETABILIDAD CON SHAP")
print("="*80)

xgb_model = pipeline_xgb_op.named_steps["clasificador_xgb"]
transformer = pipeline_xgb_op.named_steps["col_transformer"]

# Escalabilidad SHAP
sample_sizes = np.array([1000, 5000, 10000, 20000, 50000, 100000])
times_shap = []

print(" Analizando escalabilidad de SHAP...")

for size in sample_sizes:
    print(f"  Calculando SHAP para {size:,} registros...", end=" ")
    idx_sample = np.random.choice(X_val.index, size=min(size, len(X_val)), replace=False)
    X_sample = X_val.loc[idx_sample]
    X_sample_transformed = transformer.transform(X_sample)

    explainer = shap.TreeExplainer(xgb_model)
    start_time = time.time()
    shap_values = explainer.shap_values(X_sample_transformed)
    end_time = time.time()

    elapsed = end_time - start_time
    times_shap.append(elapsed)
    print(f"{elapsed:.2f}s")

plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, times_shap, marker='o')
plt.xlabel("Tamaño de muestra")
plt.ylabel("Tiempo de cálculo SHAP (s)")
plt.title("Escalabilidad del cálculo de SHAP values")
plt.grid(True)
plt.show()

# Ajuste power-law
ln_n = np.log(sample_sizes).reshape(-1, 1)
ln_t = np.log(times_shap)
lr = LinearRegression().fit(ln_n, ln_t)
b = lr.coef_[0]
a = np.exp(lr.intercept_)
preds_shap = a * sample_sizes**b
r2 = r2_score(times_shap, preds_shap)

print(f"\nModelo ajustado: t = {a:.4e} * n^{b:.3f}")
print(f"R² = {r2:.5f}")

new_sizes = np.array([2e5, 5e5, 1e6, 2e6], dtype=int)
preds_new = a * new_sizes**b
print("\nPredicciones de tiempo:")
for n, t_pred in zip(new_sizes, preds_new):
    print(f"  {n:,} muestras: {t_pred:.2f}s ({t_pred/60:.2f} min)")

plt.figure(figsize=(8, 5))
plt.scatter(sample_sizes, times_shap, label="Medido")
plt.plot(sample_sizes, preds_shap, '--', label="Ajuste (power law)")
plt.plot(new_sizes, preds_new, 'o--', label="Predicción")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Tamaño de muestra (n)")
plt.ylabel("Tiempo (s)")
plt.title("Escalabilidad SHAP: tiempo vs tamaño de muestra")
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

# SHAP summary plot
print("\n Generando gráficos SHAP...")
idx_sample = np.random.choice(X_val.index, size=min(50_000, len(X_val)), replace=False)
X_val_sample = X_val.loc[idx_sample]
X_val_sample_transformed = transformer.transform(X_val_sample)

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_val_sample_transformed)

feature_names = transformer.get_feature_names_out()

shap.summary_plot(shap_values, X_val_sample_transformed, feature_names=feature_names)

# Dependence plots para features clave
for feat in ["semanas_desde_ultima_compra", "tasa_compra_historica", 
             "num_compras_last_4w", "size"]:
    if feat in feature_names:
        shap.dependence_plot(
            feat, shap_values, X_val_sample_transformed,
            feature_names=feature_names
        )


# =============================================================================
# 12. FEATURE IMPORTANCE
# =============================================================================

print("\n" + "="*80)
print("IMPORTANCIA DE VARIABLES")
print("="*80)

xgb_model_base = pipeline_xgb.named_steps["clasificador"]
feature_names_base = pipeline_xgb.named_steps["col_transformer"].get_feature_names_out().tolist()
xgb_model_base.get_booster().feature_names = feature_names_base

importance_gain = xgb_model_base.get_booster().get_score(importance_type='gain')

importance_df_gain = (
    pd.DataFrame.from_dict(importance_gain, orient='index', columns=['importance'])
    .sort_values('importance', ascending=False)
)
importance_df_gain.reset_index(inplace=True)
importance_df_gain.rename(columns={'index': 'feature'}, inplace=True)

print(" Top 15 features por gain:")
display(importance_df_gain.head(15))

plt.figure(figsize=(10, 6))
plt.barh(importance_df_gain['feature'][:15][::-1], importance_df_gain['importance'][:15][::-1])
plt.xlabel("Gain")
plt.ylabel("Feature")
plt.title("Importancia de características XGBoost (Gain)")
plt.tight_layout()
plt.show()

importance_weight = xgb_model_base.get_booster().get_score(importance_type='weight')

importance_df_weight = (
    pd.DataFrame.from_dict(importance_weight, orient='index', columns=['importance'])
    .sort_values('importance', ascending=False)
)
importance_df_weight.reset_index(inplace=True)
importance_df_weight.rename(columns={'index': 'feature'}, inplace=True)

print("\n Top 15 features por weight:")
display(importance_df_weight.head(15))

plt.figure(figsize=(10, 6))
plt.barh(importance_df_weight['feature'][:15][::-1], importance_df_weight['importance'][:15][::-1])
plt.xlabel("Weight")
plt.ylabel("Feature")
plt.title("Importancia de características XGBoost (Weight)")
plt.tight_layout()
plt.show()


# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "="*80)
print(" PIPELINE CORREGIDO COMPLETADO")
print("="*80)

print("\n Correcciones principales aplicadas:")
print("  1.  Eliminado data leakage (compro, items_comprados)")
print("  2.  Cartesiano completo ANTES de calcular target")
print("  3.  Semanas sin gaps (rango continuo)")
print("  4.  Target calculado con shift(-1) sobre panel completo")
print("  5.  Features temporales agregadas:")
print("       - Lags (1-4 semanas)")
print("       - Ventanas móviles (4, 8, 12 semanas)")
print("       - Recencia (semanas desde última compra)")
print("       - Frecuencia histórica de compra")
print("       - Features a nivel cliente y producto")
print("       - Estacionalidad (week_of_year, month, quarter)")
print("  6.  Verificación de consecutividad de semanas")
print("  7.  Todas las features miran hacia atrás (sin leakage)")

print("\n Próximos pasos:")
print("  - Analizar feature importance en detalle")
print("  - Responder preguntas de interpretabilidad con SHAP")
print("  - Calcular métricas de ranking (Precision@N, NDCG)")
print("  - Para Entrega 2: usar este código como base")

print("\n" + "="*80)