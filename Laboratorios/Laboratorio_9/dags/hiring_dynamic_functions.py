## Desarrollo 2.1 Preparando un Nuevo Pipeline

# hiring_dynamic_functions.py

#Librerías
from pathlib import Path
import shutil
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score


# Primera definición: create_folders(**context) --> Crea YYYY-MM-DD/(raw, preprocessed, splits, models)

def create_folders(**context):
    ds = context["ds"]                               
    dags_dir = Path(__file__).resolve().parent       
    run_dir  = dags_dir / ds

    for sub in ["raw", "preprocessed", "splits", "models"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

# Segunda definición: load_and_merge(**context) --> Lee raw/data_1.csv (+ data_2.csv si existe), concat y guarda preprocessed/merged.csv

def load_and_merge(**context):
    ds = context["ds"]
    base = Path(__file__).resolve().parent / ds
    raw_dir = base / "raw"
    prep_dir = base / "preprocessed"
    prep_dir.mkdir(parents=True, exist_ok=True)

    p1 = raw_dir / "data_1.csv"
    p2 = raw_dir / "data_2.csv"

    dfs = []
    if p1.exists():
        dfs.append(pd.read_csv(p1))
        print(f"[load_and_merge] Leído: {p1}")
    if p2.exists():
        dfs.append(pd.read_csv(p2))
        print(f"[load_and_merge] Leído: {p2}")

    merged = pd.concat(dfs, ignore_index=True)
    out = prep_dir / "merged.csv"
    merged.to_csv(out, index=False)
    print(f"[load_and_merge] Guardado: {out} (n={len(merged)})")


# Tercera definición: split_data(**context) --> Lee preprocessed/merged.csv y hace hold-out 80/20 (seed=42)

def split_data(**context):
    ds = context["ds"]
    base = Path(__file__).resolve().parent / ds
    merged_path = base / "preprocessed" / "merged.csv"

    df = pd.read_csv(merged_path)

    y = df["HiringDecision"]
    X = df.drop(columns=["HiringDecision"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    splits_dir = base / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (X_train.assign(HiringDecision=y_train)).to_csv(splits_dir / "train.csv", index=False)
    (X_test.assign(HiringDecision=y_test)).to_csv(splits_dir / "test.csv", index=False)


# Cuarta definición: train_model(model, model_name=None, **context) --> Entrena pipeline(preproc + modelo) y guarda models/<nombre>.joblib
# Acá también defino el preprocessor de las bases .csv

def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Devuelve un ColumnTransformer minimalista (compatible con sklearn 1.3.x).
    """
    num_cols = [
        "Age", "ExperienceYears", "PreviousCompanies", "DistanceFromCompany",
        "InterviewScore", "SkillScore", "PersonalityScore",
    ]
    cat_cols = ["Gender", "EducationLevel", "RecruitmentStrategy"]

    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

def train_model(model, model_name: str | None = None, **context):
    """
    Entrena (preprocesamiento + modelo) sobre <ds>/splits/train.csv
    y guarda el pipeline en <ds>/models/<nombre>.joblib
    - model: estimador sklearn ya instanciado (clasificador)
    - model_name: nombre del archivo .joblib (opcional). Si no, usa la clase del modelo.
    """
    ds = context["ds"]
    base = Path(__file__).resolve().parent / ds
    splits_dir = base / "splits"
    models_dir = base / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_path = splits_dir / "train.csv"

    train_df = pd.read_csv(train_path)
    target_col = "HiringDecision"

    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])

    preprocessor = _build_preprocessor(X_train)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    base_name = model_name or model.__class__.__name__
    out_path = models_dir / f"{base_name}.joblib"
    joblib.dump(pipe, out_path)
    print(f"[train_model] Modelo entrenado y guardado en: {out_path}")


# Quinta definición: evaluate_models(**context) --> Evalúa todos los .joblib en models/ con accuracy en test y selecciona el mejor y lo guarda como models/best_model.joblib

def evaluate_models(**context):
    ds = context["ds"]
    base = Path(__file__).resolve().parent / ds
    splits_dir = base / "splits"
    models_dir = base / "models"

    test_path = splits_dir / "test.csv"

    test_df = pd.read_csv(test_path)

    y_test = test_df["HiringDecision"]
    X_test = test_df.drop(columns=["HiringDecision"])

    candidates = sorted([p for p in models_dir.glob("*.joblib") if p.name != "best_model.joblib"])

    best_name = None
    best_score = -1.0
    best_pipe = None

    for path in candidates:
        pipe = joblib.load(path)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[evaluate_models] {path.name}: accuracy={acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_name = path.name
            best_pipe = pipe

    out_best = models_dir / "best_model.joblib"
    joblib.dump(best_pipe, out_best)
    print(f"[evaluate_models] Mejor modelo: {best_name} | accuracy={best_score:.4f}")
    print(f"[evaluate_models] Guardado como: {out_best}")
