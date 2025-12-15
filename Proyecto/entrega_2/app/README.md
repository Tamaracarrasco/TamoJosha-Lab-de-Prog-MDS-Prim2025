# SodAI Drinks - Aplicación Web de Predicción

Sistema completo de predicción de compras desarrollado por el equipo Deep Drinkers para SodAI Drinks, integrado con el pipeline productivo de Airflow y XGBoost.

## Descripción General

Esta aplicación web consume el modelo XGBoost entrenado por el pipeline de Airflow y permite realizar predicciones en tiempo real sobre la probabilidad de que un cliente compre un producto específico la próxima semana.

El sistema está compuesto por:

- **Backend (FastAPI):** API REST que carga el modelo entrenado desde `/airflow/models/xgb_best_model.pkl`
- **Frontend (Gradio):** Interfaz web amigable para interactuar con las predicciones
- **Docker:** Containerización completa con integración directa al pipeline de Airflow

## Integración con Airflow

La aplicación trabaja en conjunto con el pipeline de Airflow:

1. Airflow entrena el modelo y lo guarda en `/airflow/models/xgb_best_model.pkl`
2. El backend monta los volúmenes `/airflow/models/` y `/airflow/data/`
3. El backend carga las dimensiones desde `clientes.parquet` y `productos.parquet`
4. El frontend ofrece la interfaz para predicciones en tiempo real

## Estructura del Proyecto
```
app/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   └── utils.py
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
├── .gitignore
├── docker-compose.yml
├── Levanta_APP.md
├── test_prediccion.csv
└── README.md
```

## Requisitos Previos

- Docker (versión 20.10 o superior)
- Docker Compose (versión 1.29 o superior)
- Modelo entrenado guardado como `xgb_best_model.pkl`

## Instalación

### Verificar archivos necesarios
```bash
ls -lh airflow/models/xgb_best_model.pkl
ls -lh airflow/data/clientes.parquet
ls -lh airflow/data/productos.parquet
```

### Construir y levantar contenedores
```bash
cd app
docker-compose build
docker-compose up -d
```

### Verificar el estado
```bash
sleep 30
curl http://localhost:8000/health
docker-compose ps
```

### Acceder a la aplicación

- Frontend: http://localhost:7860
- Backend API: http://localhost:8000
- Documentación: http://localhost:8000/docs

## Uso

### Predicción Individual

1. Accede a http://localhost:7860
2. Ve a la pestaña "Predicción Individual"
3. Ingresa el ID del cliente y el ID del producto
4. Obtén el resultado con la probabilidad de compra

### Predicción por Lotes

1. Prepara un CSV con formato:
```csv
cliente_id,producto_id
12345,67890
23456,78901
```
2. Ve a la pestaña "Predicción por Lotes"
3. Sube el archivo y obtén los resultados

### API REST

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Predicción Individual:**
```bash
curl -X POST "http://localhost:8000/prediccion" \
  -H "Content-Type: application/json" \
  -d '{"cliente_id": 12345, "producto_id": 67890}'
```

**Predicción por Lotes:**
```bash
curl -X POST "http://localhost:8000/prediccion/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "datos": [
      {"cliente_id": 12345, "producto_id": 67890},
      {"cliente_id": 23456, "producto_id": 78901}
    ]
  }'
```

**Recargar Modelo:**
```bash
curl -X POST "http://localhost:8000/modelo/reload"
```

## Comandos Útiles
```bash
# Ver logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Reiniciar servicios
docker-compose restart

# Detener contenedores
docker-compose down

# Reconstruir imágenes
docker-compose build --no-cache
```

## Configuración

### Variables de Entorno

Modifica estas variables en `docker-compose.yml` según necesites:

- `MODEL_PATH`: Ruta al archivo del modelo
- `BACKEND_URL`: URL del backend para el frontend
- `MLFLOW_TRACKING_URI`: URI del servidor MLflow (opcional)

### Cambiar Puertos

Si los puertos están en uso, modifica en `docker-compose.yml`:
```yaml
services:
  backend:
    ports:
      - "8080:8000"
  frontend:
    ports:
      - "7870:7860"
```

## Resolución de Problemas

**Modelo no encontrado:**
- Verifica que `xgb_best_model.pkl` existe en `airflow/models/`
- Revisa los logs: `docker-compose logs backend`

**Error de conexión:**
- Verifica que ambos contenedores estén corriendo
- Revisa la variable `BACKEND_URL` en el frontend

**Puerto en uso:**
- Cambia los puertos en `docker-compose.yml`
- O detén el proceso que usa el puerto

## Mantenimiento

### Actualizar el Modelo

Recarga sin reiniciar:
```bash
curl -X POST "http://localhost:8000/modelo/reload"
```

O reinicia el backend:
```bash
docker-compose restart backend
```

### Actualizar el Código
```bash
docker-compose build
docker-compose up -d
```

## Monitoreo
```bash
# Ver recursos utilizados
docker stats sodai-backend sodai-frontend
```

El backend incluye un health check automático cada 30 segundos.

---

**Versión:** 1.0.0  
**Fecha:** Noviembre 2025  
**Desarrollado por:** Deep Drinkers