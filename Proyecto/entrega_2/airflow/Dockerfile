# airflow/Dockerfile

FROM apache/airflow:2.8.0-python3.9

USER root

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Instalar dependencias Python
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt

# Variables de entorno
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV PYTHONPATH=/opt/airflow:$PYTHONPATH

WORKDIR ${AIRFLOW_HOME}

# Crear directorios necesarios
RUN mkdir -p logs dags plugins data models artifacts/predictions artifacts/metrics

# Expose webserver port
EXPOSE 8080

# Default command (se sobreescribe en docker-compose)
CMD ["bash"]