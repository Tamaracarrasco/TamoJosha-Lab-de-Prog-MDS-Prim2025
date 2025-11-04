## Desarrollo parte 1.2 Creando Nuestro DAG

## dag_lineal.py

# Librerías
from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from hiring_functions import (create_folders, split_data, preprocess_and_train, gradio_interface)

# DAG con ejecución manual, sin backfill, start 2024-10-01.

with DAG(
    dag_id="hiring_lineal",
    start_date=datetime(2024, 10, 1),
    schedule=None,         
    catchup=False,        # esto es lo del backfill
    tags=["lab9"],
    is_paused_upon_creation=False,
) as dag:

# Marcador de posición que indica el inicio del pipeline.

    start = EmptyOperator(task_id="start")

# Creación carpetas  para el pipeline y subcarpetas raw, splits y models con la función create_folders()

    mk_run_folders = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"ds": "{{ ds }}"},
    )

# Descarga de data_1.csv para guardar en carpeta raw de la ejecución

    download_raw = BashOperator(
    task_id="download_raw",
    bash_command=(
        "mkdir -p {{ dag.folder }}/{{ ds }}/raw && "
        "curl -sSL -o {{ dag.folder }}/{{ ds }}/raw/data_1.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    ),
)

# Aplico hold out con la función split_data()

    do_split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"ds": "{{ ds }}"},
    )

# Aplico preprocesamiento y entrenamiento del modelo con la función preprocess_and_train()

    train_model = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        op_kwargs={"ds": "{{ ds }}"},
    )

# Interfaz de gradio para cargar archivo json

    launch_gradio = PythonOperator(
        task_id="gradio_interface",
        python_callable=gradio_interface,
        op_kwargs={"ds": "{{ ds }}"},
    )

    start >> mk_run_folders >> download_raw >> do_split >> train_model >> launch_gradio

