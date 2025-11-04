## Desarrollo Parte 2.2 Componiendo un nuevo DAG

# dag_dynamic.py

# Librerías
from datetime import datetime
import pendulum
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from hiring_dynamic_functions import (create_folders, load_and_merge, split_data, train_model, evaluate_models)


# DAG --> corre el día 5 de cada mes a las 15:00 UTC: backfill HABILITADO (catchup=True) y start_date = 2024-10-01

with DAG(
    dag_id="hiring_dynamic",               
    start_date=datetime(2024, 10, 1),
    schedule="0 15 5 * *",                   # 5 de cada mes, 15:00 UTC
    catchup=True,                            # habilita backfill
    tags=["lab9", "airflow", "dynamic"],
    is_paused_upon_creation=False,
) as dag:

# Marcador de inicio
    start = EmptyOperator(task_id="start")

# Carpetas YYYY-MM-DD/{raw, preprocessed, splits, models}
    t_create = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"ds": "{{ ds }}"},
    )

# Branching por fecha de ejecución, antes del 2024-11-01: solo data_1.csv y desde 2024-11-01 inclusive: data_1.csv y data_2.csv
def branching(**context):
    logical_date = context["logical_date"].in_timezone("UTC")
    cutoff = pendulum.datetime(2024, 11, 1, tz="UTC")
    if logical_date < cutoff:
        return "download_data1"

    return ["download_data1", "download_data2"]

branch = BranchPythonOperator(
    task_id="branch_by_date",
    python_callable=branching,
)

# Descargo de als baases de datos según branching
t_dl1 = BashOperator(
    task_id="download_data1",
    bash_command=(
        "mkdir -p {{ dag.folder }}/{{ ds }}/raw && "
        "curl -sSL -o {{ dag.folder }}/{{ ds }}/raw/data_1.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    ),
)

t_dl2 = BashOperator(
    task_id="download_data2",
    bash_command=(
        "mkdir -p {{ dag.folder }}/{{ ds }}/raw && "
        "curl -sSL -o {{ dag.folder }}/{{ ds }}/raw/data_2.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
    ),
)

# Merge de datasets disponibles, debe correr si hay al menos UNO disponible → TriggerRule.ONE_SUCCESS
t_merge = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        op_kwargs={"ds": "{{ ds }}"},
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

# split hold-out (80/20 con semilla) → splits/train.csv y splits/test.csv
t_split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        op_kwargs={"ds": "{{ ds }}"},
    )

# Tres entrenamientos en paralelo
t_train_rf = PythonOperator(
        task_id="train_rf",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model": RandomForestClassifier(n_estimators=300, random_state=42),
            "model_name": "RandomForest",
        },
    )

t_train_lr = PythonOperator(
        task_id="train_lr",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model": LogisticRegression(max_iter=1000),
            "model_name": "LogReg",
        },
    )

t_train_dt = PythonOperator(
        task_id="train_dt",
        python_callable=train_model,
        op_kwargs={
            "ds": "{{ ds }}",
            "model": DecisionTreeClassifier(random_state=42),
            "model_name": "DecisionTree",
        },
    )

# Evaluación del mejor modelo, aca se va a correr solo si los 3 entrenamientos terminaron OK → ALL_SUCCESS si no, no porque falla
t_eval = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_kwargs={"ds": "{{ ds }}"},
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

end = EmptyOperator(task_id="end")

# Estructura del DAG
start >> t_create >> branch
branch >> t_dl1
branch >> t_dl2
[t_dl1, t_dl2] >> t_merge >> t_split >> [t_train_rf, t_train_lr, t_train_dt] >> t_eval >> end

