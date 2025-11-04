# Dockerfile.dockerfile
# version de python recomendada en el enunciado
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    AIRFLOW_HOME=/opt/airflow

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "apache-airflow==2.10.2" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.2/constraints-3.10.txt"

RUN pip install --no-cache-dir \
    pandas scikit-learn joblib numpy \
    gradio huggingface_hub

RUN mkdir -p ${AIRFLOW_HOME}/dags ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins
WORKDIR ${AIRFLOW_HOME}
COPY dags/ ./dags/

EXPOSE 8080 7860
CMD ["bash", "-lc", "airflow standalone"]




