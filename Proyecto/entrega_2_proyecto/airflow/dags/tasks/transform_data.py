# acá se define la función para transformar la data

from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

def transform_data(**context):
    """
    Une clientes, productos y transacciones; aplica encoding y escalado 
    usando artefactos baseline.
    """
    base = Path(__file__).resolve().parents[3]
    baseline = base / "data" / "baseline"
    run_dir = base / "data" / "run" / context["ds"]
    raw_dir = run_dir / "raw"
    prep_dir = run_dir / "preprocessed"
    prep_dir.mkdir(parents=True, exist_ok=True)

    # Leer archivos
    clientes = pd.read_parquet(raw_dir / "clientes.parquet")
    productos = pd.read_parquet(raw_dir / "productos.parquet")
    transacciones = pd.read_parquet(raw_dir / "transacciones.parquet")

    # Merge ejemplo
    df = transacciones.merge(clientes, on="id_cliente", how="left")
    df = df.merge(productos, on="id_producto", how="left")

    # Cargar encoder y scaler del baseline
    encoder = joblib.load(baseline / "encoder.joblib")
    scaler = joblib.load(baseline / "scaler.joblib")

    # Aplicar transformaciones
    cat_cols = encoder.feature_names_in_
    num_cols = [c for c in df.columns if c not in cat_cols + ["target"]]

    df_encoded = encoder.transform(df[cat_cols])
    df_scaled = scaler.transform(df[num_cols])

    df_final = pd.DataFrame(df_scaled, columns=num_cols)
    df_final[cat_cols] = df_encoded
    df_final["target"] = df["target"]

    df_final.to_parquet(prep_dir / "df_preprocessed.parquet", index=False)
    print(f"✅ Datos transformados y guardados en {prep_dir}")