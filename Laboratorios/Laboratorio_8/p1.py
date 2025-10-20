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

# preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# modelo
from xgboost import XGBClassifier

# librerías extras: os, pickle, mlflow
import os
import pickle
import mlflow
import mlflow.sklearn
import optuna

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

def build_preprocessor():
    """Esta función crea un ColumnTransformer 
    con imputación (mediana) para los valores faltantes y 
    realiza escalado estándar.
    
    Returns:
    ------------
    preprocessor: Columntrasnformer class.
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

def build_pipeline():
    """Esta función intenta crear el pipeline completo 
    con preprocesamiento más el modelo base.
    Modelo base es XGBoost
    
    Returns:
    ----------
    
    pipeline."""
    
    prep = build_preprocessor()

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", prep),
        ("classifier", model)
    ])
    return pipeline

# Ejecución del script

if __name__ == "__main__":

    # Se cargan los datos
    df = carga_data()

    # se separan las variables predictoras de la target
    X, y = split_features_target(df)

    # separación de conjunto train y test
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # instanciando el pipeline del modelo
    pipeline = build_pipeline()

    # funciona?
    print("tamos trabajando pa usted... si pasa algo malo, revise código")
    pipeline.fit(X_train, y_train)

    score = pipeline.score(X_test, y_test)
    print(f"Exactitud base en validación: {score:.4f}")