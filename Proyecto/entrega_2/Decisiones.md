# Decisiones tomadas en la entrega 2:

## Visión general del proyecto

Este proyecto desarrolla un pipeline productivo de Machine Learning utilizando **Apache Airflow** para predecir si un cliente comprará un producto durante la semana siguiente. La solución está diseñada con enfoque **MLOps**, considerando todo el ciclo de vida de un modelo en producción.

El sistema incorpora:

- **Procesamiento incremental de datos**: semana histórica (t) y nueva semana entrante (t+1).
- **Predicción operativa**: generación de predicciones para la semana siguiente (t+2).
- **Reentrenamiento condicionado** mediante detección de **data drift** (PSI).
- **Optimización automática** de hiperparámetros con **Optuna**.
- **Registro completo** del modelo, métricas y artefactos mediante **MLflow**.
- **Interpretabilidad** del modelo utilizando **SHAP**.
- **Estrategia de subsampling** por ambiente (dev/staging/prod).

El objetivo es emular un entorno productivo real, donde los datos cambian semana a semana y el sistema debe adaptarse, monitorear y entregar predicciones confiables.

---

## Estructura del proyecto

```
airflow/
│
├── dags/
│   └── sodai_xgb_pipeline_dag.py       # DAG principal del pipeline
│
├── scripts/
│   ├── data_preparation.py             # Limpieza, panel semanal y generación de target
│   ├── data_io.py                      # Funciones de carga/guardado y wrapper del dataset
│   ├── modeling_xgb.py                 # XGBoost, Optuna, splits y predicciones
│   ├── drift.py                        # Detección de drift mediante PSI semanal
│   ├── shap_utils.py                   # Gráficos de interpretabilidad SHAP
│   └── mlflow_utils.py                 # Funciones de tracking y registro en MLflow
│
├── config/
│   ├── config.py                       # Rutas, parámetros globales y configuración
│   └── __init__.py                     # Permite import config as cfg
│
├── data/
│   ├── clientes.parquet                # Dimensión clientes (estática)
│   ├── productos.parquet               # Dimensión productos (estática)
│   ├── transacciones.parquet           # Histórico de transacciones (t)
│   ├── transacciones_2025W01.parquet   # [OPCIONAL] Nueva semana (t+1) en caso que se agregue nueva data
│   └── df_final_latest.parquet         # [AUTO] Dataset procesado para el modelo
│
├── models/
│   └── xgb_best_model.pkl              # [AUTO] Modelo más reciente entrenado
│
├── artifacts/
│   ├── predictions/
│   │   ├── predicciones_test.parquet   # [AUTO] Predicciones del set de test
│   │   └── predicciones_finales.parquet # [AUTO] Predicciones operativas t+2
│   └── metrics/
│       ├── shap_summary.png            # [AUTO] Gráfico SHAP summary
│       └── topN_sample.csv             # [AUTO] Top-N recomendaciones por cliente
│
├── logs/                               # [AUTO] Logs de Airflow
│
├── docker-compose.yml                  # Configuración de servicios Docker
├── Dockerfile                          # Imagen de Airflow personalizada
└── requirements.txt                    # Dependencias Python
``` 

---

# Pipeline
Se crearon scripts para el levantamiento del pipeline. A continuación se describen a grandes rasgos lo que se espera el script.

1. ``data_preparation.py``

En este script se definen las siguientes funciones para cumplir lo siguiente: Craga de datos -> limpieza y transformaciones -> generar base de datos de entrega 1 -> generar base de datos para predicciones.

- **load_raw_data()**: Carga datasets disponibles de cliente.parquet, productos.parquet y transacciones.parquet En caso de que hayan transacciones nuevas, se concatenaran a las transacciones antiguas. 

- **cast_and_clean_raw_tables()**: Esta función se encarga de replicar la limpieza y cambios realizados a los datos definidos de la misma forma en la tarea 1, tales como transformar las columans de fechas al formato correspondiente (datetime).

- **deduplicate_and_fix_transactions()**: Se encarga de eliminar registros duplicados y consolida registros con mina id de clientes, productos, orden y fecha de compra. También se descartan items que presentaban cantidades negativas.

- **build_transaction_level_df()**: genera el merge de transacciones con clientes y productos, aplicando los mismos filtros que en la entrega 1.

- **build_weekly_panel_with_target()**: A partir del df que se obtuvo en la función anterior, se definen columnas de comptra o no en semana t y define variable target en semana t+1. También se agregan características clientes y productos, generando el dataframe final.
- **build_model_dataset()**: función que define todo el pipeline de preparación, haciendo uso de las funciones anteriormente descritas.
- **build_next_week_candidates_from_raw()**: Construye el dataset de candidatos para predecir semana t + 2.


2. ``drift.py``: En este script se busca aplicar el método univariado de detección de data drifting utilizando PSI.
La lógica para detectar si hubo data drifting será que en caso de que hayan datos nuevos, lo cual corresponderían a datos de la última semana, esta se compara con toda la data histórica. Se define la función auxiliar **_compute_psi()** que calcula el coeficiente PSI.

Cabe mencionar en este punto la decisión de reentrenamiento:
- En caso de primera ejecución, se realiza el primer reentrenamiento (por que no existe aún un modelo)
- Si hay una nueva semana, hay un reentrenamiento.
- Si hay data drifting, hay reentrenamiento.

3. ``modeling_xgb.py``: En este scripts se definen las funciones principales para realizar el holdout, entrenamiento, optimización de hiperparámetros y predicciones. Se ejecutará la función **train_xgb_predict()** en caso de que se cumpla alguna condición de reentrenamiento. Dependiendo del ambiente, si es developing, staging, se obtendrá una submuestra aleatoria estratificada, por tema de costo computacional.

Se tienen las siguientes funciones:
- **subsample_dataset()**: se genera submuestra aleatoria.

- **temporal_train_val_test_split()**: holdout considerando que los splits son temporales.

- **tune_hyperparams_optuna()**: Se utiliza optuna para optimizar los mismos hiperparámetros utilizados para XGBClassifier y también el mismo hiperparámetro de OneHotEncoder. Tal como en la entrega 1, esta optimización se realiza maximizando la métrica F1-Score, utilizando el conjunto de validación.

- **build_xgb_best_pipeline():** Se construye el pipeline del modelo considerando los mejores hiperparámetros encontrados.

- **evaluate_and_select_threshold()**: Selecciona el umbral que maximiza F1-Score en el conjunto de validación.

- **predict_test_with_probabilities()**: Se realizan predicciones en el conjunto de test y se genera ranking por cliente/semana.

- **topN_por_cliente_semana()**: Extrae TopN productos por cliente por semana (N = 5) por default.

- **train_xgb_predict()**: función que ejecuta todas las funciones anteriores.

### Extras
* ``shap_utils.py``: Se obtienen los valores shap para el pipeline del modelo. 
* ``mlflow_utils.py``: En este script se definen funciones que configuran Mlflow con el tracking URI y el experimento. También se definen funciones que loggean diccionarios para el pipeline, métricas y parámetros.
* ``data_io.py``: Este script se encarga de extraer los datos crudos disponibles en ../data/ y detecta si hay archivos correspondiente a la data nueva.También guarda las predicciones generadas por el modelo en formato parquet.

### Se recomienda leer el readme.md que se dejó en ../airflow/

------------
# DESCRIPCIÓN DEL DAG

El dag llamado ``sodai_xgb_pipeline_dag.py`` está ubicado en ../airflow/dags/.

El objetivo principal del dags es cumplir con las siguientes tareas:
- Detección de nueva data.
- Detección de drift con PSI.
- Reentrenamiento condicional.
- Predicción operativa para t+2.
- Lazy imports para que el parsing del DAG sea más liviano.

Se definen las siguientes tareas:

- **extract_data_task()**: Se revisa que estén los archivos base y además se marca por si hay una semana nueva (datos nuevos)
- **transform_data_task()** Esta tarea arma el dataframe final a partir de los datos crudos. En caso de que corresponda, se agrega la nueva semana t+1. También se limpian y transforman los datos, se construye la variable target y se guarda el dataframe más reciente.
- **check_drift()**: Se revisa si hay data drifting. Se compara la última semana con los datos históricos y si hay data drifting en características numéricas, se guarda un reporte y que marca drift = TRUE.
- **brach_on_drift()**: Aquí se decide el reentrenamiento bajo las 3 condiciones mencionadas anteriormente.
- **retrain_model()**: En caso de reentrenamiento, se carga el dataframe final con la variable target y se realiza el holdout, optimización de hiperparámetros con optuna, se busca el mejor umbral, evaluación en conjunto test y se registra todo en MLFLOW. Se guarda el mejor modelo con su umbral y las predicciones en el conjunto test.
- **predict_next_week()**: Esta tareagenera las predicciones para la semana t + 2. Para ello se carga el modelo con su umbral. Se arman los candidatos desde los datos crudos, se calculan las probabilidades y a partir del umbral, se realiza el ranking por clientes y se guardan las predicciones  finales en formato parquet.


## Definición del DAG

Se define el dag con los siguientes parámetros:
- dag_id: identificador único del DAG en Airflow.

- description: descripción que sale en la UI.

- default_args: aplicados por defecto a cada tarea.

- schedule_interval=None: ejecución manual

- start_date = datetime(2025,1,1): fecha de inicio del DAG. 

- catchup=False: no hacer catchup (no ejecutar runs pendientes del pasado).

- tags: etiquetas para filtrar en UI.

### Operadores

- **extract_data**: Ejecuta la función extract_data_task(), cuyo propósito es comprobar los archivos base y verificar si hay datos nuevos.
- **transform_data**: Ejecuta la función transform_data_task(). EL propósito es leer los datos.parquet crudos, aplica limpieza, trasnformación de los datos y escribe el df_final listo para que el modelo lo use.
- **check_drift_task**: Ejecuta la función check_drift() para checkear el data drifting.
- **branch_task**: Operador tpo branch. Este operador retorna el task_id para ejecutar las tareas a continuación, en particular, mirando las condiciones de reentrenamiento:
    - Si no hay modelo guardado $\rightarrow$ retornar "retrain_model_task".
    - Si hay drift y hay data nueva $\rightarrow$ "retrain_model_task".
    - Si no hay drift o no hay data nueva $\rightarrow$ "skip_retrain".
- **retrain_model_task**: Este operador se encarga del entrenamiento con el modelo XGBoost, Otimización de Hiperparams con Optuna, registra run con mlflow y se guardan artefactos (modelo, métricas, predicciones)

- **skip_retrain**: Empty operator que sirve para representar la rama donde no se va a reentrenar.

- **predict_next_week_task**: genera las predicciones para la semana t + 2 y las guarda en archivo.parquet.
Acá es importante destacar la trigger_rule ``TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS``; La tarea se ejecutará si ninguna de las tareas anteriores falló y si al menos una tuvo éxito.

El razonamiento lo planteamos así: como **predict_next_week_task** tiene los tareas anteriores (**retrain_model_taks** y **skip_retrain**) y una de esas ramas será marcada como skipped por **branch_task**. Luego por la trigger rule definida, se asegura que si una tarea fue skipeada y la otra fue exitosa, la tarea corre.

## Diagrama DAG

Acá se muestra el diagrama del DAG

```text
┌───────────────────────────┐
│       Extract Data        │
│  - Validar archivos       │
│  - Detectar nuevos        │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│       Transform Data      │
│  - Cargar parquets        │
│  - Limpiar / dedup        │
│  - Panel semanal          │
│  - Target                 │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│        Check Drift        │
│  - PSI última semana      │
│  - vs histórico           │
└──────────────┬────────────┘
               │
               ▼
┌───────────────────────────┐
│      Branch on Drift      │
│       ¿Reentrenar?        │
└───────────┬───────┬───────┘
            │       │
            │ NO    │ SÍ
            │       │
            ▼       ▼
┌────────────────┐  ┌─────────────────────────┐
│     Skip       │  │     Retrain Model       │
│    Retrain     │  │  - Optuna               │
└───────┬────────┘  │  - XGBoost              │
        │           │  - Threshold tuning     │
        │           │  - MLflow logging       │
        │           └───────────┬─────────────┘
        │                       │
        └──────────────┬────────┘
                       │
                       ▼
        ┌───────────────────────────┐
        │     Predict Next Week     │
        │  - Candidatos t+2         │
        │  - Scoring                │
        │  - Ranking                │
        └───────────────────────────┘
```

## Representación visual del DAG en la interfaz de Airflow.

<img src="../entrega_2/pipeline_airflow_dag.png">