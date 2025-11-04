## Desarrollo parte 1.1 Preparando el Pipeline

# Librerías
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


# Primera definición: create_folders(**context) --> crea estructura y copia data_1.csv

def create_folders(**context):
    """
    Crea la carpeta de ejecución YYYY-MM-DD con subcarpetas raw/splits/models
    y copia dags/data_1.csv a <ds>/raw/data_1.csv si existe.
    """
    ds = context["ds"]                             
    dags_dir = Path(__file__).resolve().parent    
    run_dir  = dags_dir / ds

    for sub in ["raw", "splits", "models"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    src = dags_dir / "data_1.csv"                
    dst = run_dir / "raw" / "data_1.csv"

    if src.exists():
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"[create_folders] Copiado: {src} -> {dst}")
        else:
            print(f"[create_folders] Ya existe: {dst} (no se copia)")

# Segunda definición: split_data(**context)--> hold-out 80/20 estratificado

def split_data(**context):
    """
    Lee <ds>/raw/data_1.csv, aplica hold-out 80/20 estratificado (seed=42)
    y guarda train/test en <ds>/splits.
    """
    ds = context["ds"]
    base = Path(__file__).resolve().parent / ds
    raw_file = base / "raw" / "data_1.csv"

    df = pd.read_csv(raw_file)

    y = df["HiringDecision"]
    X = df.drop(columns=["HiringDecision"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    splits_dir = base / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    X_train.assign(HiringDecision=y_train).to_csv(splits_dir / "train.csv", index=False)
    X_test.assign(HiringDecision=y_test).to_csv(splits_dir / "test.csv", index=False)

    print(f"[split_data] Guardados: {splits_dir / 'train.csv'} y {splits_dir / 'test.csv'}")


# Tercera definición: preprocess_and_train(**context) --> preprocesa, entrena y evalúa

def preprocess_and_train(**context):
    """
    - Lee <ds>/splits/train.csv y test.csv
    - Preprocesa: imputación median (num) / most_frequent + OneHot (cat)
    - Entrena RandomForest
    - Imprime accuracy y f1 (clase positiva=1)
    - Guarda pipeline en <ds>/models/pipeline.joblib
    """
    ds = context["ds"]
    base = Path(__file__).resolve().parent / ds
    splits_dir = base / "splits"
    models_dir = base / "models"

    train_df = pd.read_csv(splits_dir / "train.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    y_train = train_df["HiringDecision"]
    X_train = train_df.drop(columns=["HiringDecision"])
    y_test = test_df["HiringDecision"]
    X_test = test_df.drop(columns=["HiringDecision"])

    num_cols = [
        "Age", "ExperienceYears", "PreviousCompanies", "DistanceFromCompany",
        "InterviewScore", "SkillScore", "PersonalityScore"
    ]
    cat_cols = ["Gender", "EducationLevel", "RecruitmentStrategy"]

    num_cols = [c for c in num_cols if c in X_train.columns]
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf),
    ])

    # entrenamiento
    pipe.fit(X_train, y_train)

    # evaluación
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_pos = f1_score(y_test, y_pred, pos_label=1)

    print(f"[preprocess_and_train] Test Accuracy: {acc:.4f}")
    print(f"[preprocess_and_train] Test F1-score (clase 1 - contratado): {f1_pos:.4f}")

    # guardar modelo
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "pipeline.joblib"
    joblib.dump(pipe, model_path)
    print(f"[preprocess_and_train] Modelo guardado en: {model_path}")


# Cuarta definicion: gradio_interface(**kwargs)--> UI mínima para predecir desde JSON

def predict(file, model_path):
    """
    Carga el modelo y predice a partir de un archivo JSON con las features.
    Soporta gr.File (con .name) o una ruta de archivo (str/Path).
    """
    pipeline = joblib.load(model_path)
    path = file.name if hasattr(file, "name") else file
    input_data = pd.read_json(path)
    preds = pipeline.predict(input_data)
    labels = ["No contratado" if p == 0 else "Contratado" for p in preds]
    return {"Predicción": labels[0]}


def gradio_interface(**kwargs):
    """
    Interfaz Gradio que carga el modelo desde <ds>/models/pipeline.joblib.
    """
    import gradio as gr

    run_folder_name = kwargs["ds"]
    dags_dir = Path(__file__).resolve().parent
    model_path = dags_dir / run_folder_name / "models" / "pipeline.joblib"

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description=(
            "Sube un archivo JSON con las características de entrada "
            "para predecir si Vale será contratada o no."
        ),
        allow_flagging="never",
    )

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,    
        inbrowser=False,
    )


