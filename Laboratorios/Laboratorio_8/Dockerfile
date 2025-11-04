# Imagen base 
FROM python:3.11-slim

# Evita prompts interactivos y mejora logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependencias del sistema 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Instala dependencias de python
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Documenta el puerto que usa la API
EXPOSE 8000

# Arranque del servidor (FastAPI con Uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]