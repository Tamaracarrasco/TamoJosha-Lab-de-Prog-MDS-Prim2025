## Desarrollo parte 1.2 Creando Nuestro DAG

# Librerías
from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from hiring_functions import (
    create_folders, split_data, preprocess_and_train, gradio_interface
)

with DAG(
    dag_id="hiring_lineal",
    start_date=datetime(2024, 10, 1),
    schedule=None,         
    catchup=False,        
    tags=["lab9"],
    is_paused_upon_creation=False,
) as dag:

    start = EmptyOperator(task_id="start")

    mk_run_folders = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"ds": "{{ ds }}"},
    )

    download_raw = BashOperator(
    task_id="download_data",
    bash_command=(
        "mkdir -p {{ dag.folder }}/{{ ds }}/raw && "
        "cp {{ dag.folder }}/data_1.csv {{ dag.folder }}/{{ ds }}/raw/data_1.csv"
    ),
)

    do_split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"ds": "{{ ds }}"},
    )

    train_model = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        op_kwargs={"ds": "{{ ds }}"},
    )

    launch_gradio = PythonOperator(
        task_id="gradio_interface",
        python_callable=gradio_interface,
        op_kwargs={"ds": "{{ ds }}"},
    )

    start >> mk_run_folders >> download_raw >> do_split >> train_model >> launch_gradio


