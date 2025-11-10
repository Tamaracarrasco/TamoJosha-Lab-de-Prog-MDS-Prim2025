# En este scripts se busca definir las funciones
# que permitirán realizar el procesamiento

import joblib
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Variables globales

DROP_COLS = [
    "customer_id", "product_id", "week_t", "week_t_plus_1",
    "semana", "semana_siguiente_str", "semana_num"
]

NUM_COLS = [
    "X", "Y", "size",
    "num_deliver_per_week", "num_visit_per_week",
    "purchased_count", "compra_o_no"
]

CAT_COLS = [
    "customer_type", "segment", "brand",
    "category", "sub_category", "zone_id", "region_id"
]

def preprocess(**context):
    """
    Construye el ColumnTransformer con las transformaciones
    de la parte 1.

    Retorna el column_transformer
    """

    num_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    params_ohe = {'min_frequency': 0.13662759465397978}

    cat_pipeline = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True, **params_ohe))
    ])
    
    col_transformer = ColumnTransformer(
        transformers=[
            ("drop_ids", "drop", DROP_COLS),
            ("num", num_pipeline, NUM_COLS),
            ("cat", cat_pipeline, CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return col_transformer


def fit_preprocessor(df, save_path: Path):
    """
    Ajusta el preprocesador con los datos de entrenamiento y lo guarda.

    Parámetros:
    ---------------
    df: (pandas.dataframe) dataset de entrenamiento
    save_path (Path): ruta donde guardar el preprocesador
    
    Returns:
    ---------------
    
    retorna el fitted preprocessor
    """
    preprocessor = preprocess()
    preprocessor.fit(df)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_path)
    print(f"El preprocesador se guardó en : {save_path}")
    return preprocessor

## por si acaso

def load_preprocessor(load_path: Path):
    """
    Carga un preprocesador previamente guardado con joblib.
    """
    preprocessor = joblib.load(load_path)
    print(f"Preprocesador cargado desde: {load_path}")
    return preprocessor

def transform_data(df, preprocessor):
    """
    Aplica el preprocesamiento al DataFrame.

    Parámetros:
    -----------------
    df: pandas.dataframe
    preprocessor: objeto ColumnTransformer ajustado
    
    Returns:
    -----------------
    matriz o arreglo con las features transformadas
    """
    X = preprocessor.transform(df)
    return X

if __name__ == "__main__":
    df_path = Path("../../../../data/baseline/df_final.parquet")
    df = pd.read_parquet(df_path)

    preprocessor_path = Path("../../../../data/baseline/preprocessor.joblib")

    preproc = fit_preprocessor(df, preprocessor_path)
    X = transform_data(df, preproc)
    print(f"Shape de X transformado: {X.shape}")