# main.py
from fastapi import FastAPI
import uvicorn
import pickle
import os
import pandas as pd

# Se crea la aplicación
app = FastAPI(
    title = "TamoJosha Water potability API",
    description = "API para predecir la potabilidad del agua usando XGBoost para ayudar a SMAPina",
    version = "1.0"
)

# ruta donde se guardó el mejor modelo

MODEL_PATH = os.path.join("models", "best_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Se define la ruta GET

@app.get("/")
def home():
    """
    Ruta principal que describe brevemente el modelo.
    """
    descripcion = {
        "TamoJosha_model": "Clasificador de potabilidad del agua",
        "descripcion": "Modelo de machine learning (XGBoost) que predice si una muestra de agua es potable o no.",
        "entrada": ["ph", 
                    "Hardness", 
                    "Solids", 
                    "Chloramines",
                    "Sulfate", 
                    "Conductivity", 
                    "Organic_carbon",
                    "Trihalomethanes", 
                    "Turbidity"],
        "salida": {
            "potabilidad": "0 = No potable, 1 = Potable"
        }
    }
    return descripcion

# Se define la ruta post

@app.post("/potabilidad/")

def predecir_potabilidad(data: dict): # porque recibe un JSON
    """
   Esta función recibe un diccionario con características
   químicas del agua y determina si e spotable o no.
   Predicción realizada con modelo de XGBoost.
   Ojo que el orden de las variables importa
    """
    try: 
        features = [
            "ph", "Hardness", "Solids", "Chloramines",
            "Sulfate", "Conductivity", "Organic_carbon",
            "Trihalomethanes", "Turbidity"
        ]

        # se transforma en array 2d: 1 fila y 9 columnas
        x_input = pd.DataFrame([data], columns=features)

        prediccion = model.predict(x_input)[0]

        return {"potabilidad": int(prediccion)}

    except KeyError as error_1:
        return {"error": f"falta la columna {str(error_1)}"}
    
    except Exception as error_2:
        return {"error": str(error_2)}

# pasos q hiceco: 
# 1) ejecutar main.py en Ipython (para activar Ipython-> abrir terminal -> escribir ipython --matplotlib)
# 2) escribir uvicorn.run("main:app", port = 8000) en la terminal.
# 3) ctrl + click en http://127.0.0.1:8000
# 4) se abre una pestaña en el navegador con la descripción que sale en home()as
# 5) en la url escribir http://127.0.0.1:8000/docs
# 6) en post -> apretar try out y colocar el valores de las caracteristicas en el orden que correspondan
# 7) ejecutar y ver la predicción.
