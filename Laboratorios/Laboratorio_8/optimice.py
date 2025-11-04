######### Este es un intento de optimice.py

### Cómo es un archivo.py, yo lo ejecuto de la siguiente manera:
# antes de correr esto, se abre terminal y se escoge cmd
# y se escribe "ipython --matplotlib" + enter
# escribir > run "p1.py" para ejecutar el archivo.
# y se llaman las funciones.
# instalar mlflow en caso de que no esté: pip install mlflow --user

### importaciones de librerías importantes:

# carga de los datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# modelo
import xgboost
from xgboost import XGBClassifier

# métricas

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report

# librerías extras: os, pickle, mlflow
import os
import pickle
import mlflow
import mlflow.sklearn
import optuna
import platform

########################################

# Carga de los datos:

def carga_data(path: str = "water_potability.csv"):
    """Esta función carga los datos 
    del dataset a utilizar y muestra cosas básicas del
    dataframe.

    Parameters:
    -----------------
    path: (str) nombre del archivo.

    Returns:
    ------------------
    pandas dataframe"""

    df = pd.read_csv(path)
    print(f"Dimensiones del dataframe: ", df.shape)
    print(f"Columnas del dataframe: ", df.columns.tolist())
    return df


# División en el conjunto de entrenamiento y testeo.

def split_features_target(df: pd.DataFrame, target_col: str = "Potability"):
    """Esta función separa el dataframe
    en variables predictoras y variable dependiente.
    
    Parameters:
    ---------------
    df: pandas dataframe.
    target_col: (str) variable target. 
    Valor default "Potability".

    Returns:
    --------------
    X: variables predictoras
    y: variable dependiente.
    """
    X = df.drop(columns=target_col)
    y = df[target_col]
    return X, y

def split_train_test(X, y, test_size=0.2, random_state=42):
    """Esta función divide los datos en 
    entrenamiento y validación. Usa la estratificación, porque
    hay desbalance en los datos.
    
    Parameters:
    ------------
    X: variables predictoras (df)
    y: variable dependiente (1d series)
    test_size: fracción del tamaño del conjunto test.
    random_state: semilla para reproductibilidad. (int)
    
    Returns:
    -----------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True)


# pipeline 

def const_preprocessor():
    """Esta función crea un ColumnTransformer 
    con imputación (mediana) para los valores faltantes y 
    realiza escalado estándar.
    
    Returns:
    ------------
    col_transformer: Columntransformer class.
    """

    num_features = [
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity",
    ]
    
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    col_transformer = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features)
        ]
    )
    return col_transformer

# pipeline con el modelo

def const_pipeline():
    """Esta función intenta crear el pipeline completo 
    con preprocesamiento más el modelo base.
    Modelo base es XGBoost
    
    Returns:
    ----------
    
    pipeline."""
    
    prep = const_preprocessor()

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("col_transformer", prep),
        ("clasificador", model)
    ])
    return pipeline

## OPTIMIZACIÓN

def optimize_model(X_train, X_test, y_train, y_test, n_trials=30):
    """
    Esta funcion optimiza el modelo XGboost y 
    registra los resultados usando MLflow.
    Parameters:
    ------------
    X_train, X_test, y_train, y_test: datos de entrenamiento y testeo.
    """
    # se hacen las carpetas que piden
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # se conf mlflow
    experiment_name = "XGBoost_WaterPotability"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    ## existe desbalance en los datos:

    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = neg / pos

    def objective(trial): # PUNTO 2
        """
        Esta función corresponde a el objetivo a 
        maximizar/minimizar
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "n_jobs": -1
        }

        run_name = f"XGB_lr_{params['learning_rate']:.3f}_depth_{params['max_depth']}"

        with mlflow.start_run(experiment_id= experiment_id, run_name= run_name): # acá se inicia un run
        
            preprocessor = const_preprocessor()

            model = XGBClassifier(
                objective = "binary:logistic",
                eval_metric = "logloss",
                use_label_encoder = False,
                random_state = 42,
                scale_pos_weight= scale_pos_weight,
                **params
            )

            pipeline = Pipeline([
                ("col_transformer", preprocessor),
                ("clasificador", model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            f1 = f1_score(y_test, y_pred)

            # Log
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)
            mlflow.sklearn.log_model(pipeline, "model")

            model_path = f"models/model_trial_{trial.number}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(pipeline, f)
            mlflow.log_artifact(model_path, artifact_path="models")

            # PUNTO 6)
            mlflow.log_param("python_version", platform.python_version())
            mlflow.log_param("xgboost_version", xgboost.__version__)
            mlflow.log_param("optuna_version", optuna.__version__)
            mlflow.log_param("mlflow_version", mlflow.__version__)

            return f1
    
    # optuna
    study = optuna.create_study(direction= "maximize")
    study.optimize(objective, n_trials= n_trials)

    # gráficos de optuna (PUNTO 3)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html("plots/optimization_history.html")
    mlflow.log_artifact("plots/optimization_history.html", artifact_path= "plots")

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_html("plots/param_importance.html")
    mlflow.log_artifact("plots/param_importance.html", artifact_path="plots")

    # Mejor modelo (PUNTO 4)

    def get_best_model(experiment_id):
        """
        Esta función devuelve el mejor modelo.
        """
        runs = mlflow.search_runs(experiment_id)
        best_model_id = runs.sort_values("metrics.valid_f1", ascending= False)["run_id"].iloc[0]
        best_model = mlflow.sklearn.load_model("runs:/"+ best_model_id +
        "/model")
        return best_model
    
    # Guardar el mejor modelo (PUNTO 5?)

    best_model = get_best_model(experiment_id)

    final_model_path = "models/best_model.pkl"

    with open(final_model_path, "wb") as f:
        pickle.dump(best_model, f)

    mlflow.log_artifact(final_model_path, artifact_path= "models")

    # Respaldo de las configuraciones e importancia de las features (punto 7)
    feature_names = best_model.named_steps["col_transformer"].transformers_[0][2]
    importances = best_model.named_steps["clasificador"].feature_importances_

    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances)
    plt.title("Importancia de variables (mejor modelo)")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png")
    mlflow.log_artifact("plots/feature_importance.png", artifact_path= "plots")

    return best_model



# Ejecución del script

if __name__ == "__main__":

    # Se cargan los datos
    df = carga_data()

    # se separan las variables predictoras de la target
    X, y = split_features_target(df)

    # separación de conjunto train y test
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # optimización
    best_model = optimize_model(X_train, X_test, y_train, y_test, n_trials=30)

    # metricas de desempeño
    y_pred_final = best_model.predict(X_test)
    print("Reporte final en validación:\n")
    print(classification_report(y_test, y_pred_final))