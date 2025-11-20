# Pipeline de Predicción Semanal - SodAI Drinks

Pipeline de Machine Learning con Apache Airflow que predice si un cliente comprará un producto la próxima semana.

## Qué Hace

El sistema:
1. Procesa datos de transacciones semanalmente
2. Detecta cambios en los datos (drift)
3. Reentrena el modelo cuando es necesario
4. Genera predicciones para la semana siguiente
5. Registra todo en MLflow

## Estructura

```
airflow/
├── dags/                    # DAG principal del pipeline
├── scripts/                 # Lógica de procesamiento y ML
├── config/                  # Configuración del sistema
├── data/                    # Datos de entrada
├── models/                  # Modelo entrenado (auto)
├── artifacts/               # Predicciones y gráficos (auto)
└── docker-compose.yml       # Configuración Docker
```

## Requisitos

- Docker Desktop (Windows/Mac) o Docker Engine (Linux)
- Docker Compose v2.0+
- RAM mínima: 8GB (12GB recomendado)

### Solo Windows con Docker Desktop y WSL2

Creo el archivo `.wslconfig` en mi carpeta de usuario si no tengo la suficiente RAM:

Primero:
```powershell
notepad $env:USERPROFILE\.wslconfig
```

Luego agrego:
```ini
[wsl2]
memory=12GB
processors=4
swap=8GB
```

Reinicio WSL:
```powershell
wsl --shutdown
```

## Instalación

### Paso 1: Ir a la carpeta

```bash
cd airflow
```

### Paso 2: Verificar datos

Debo tener estos archivos en `data/`:
- `clientes.parquet`
- `productos.parquet`
- `transacciones.parquet`

```bash
ls data/
```

### Paso 3: Configurar ambiente

Edito `docker-compose.yml` y busco la línea `ENVIRONMENT`:

```yaml
environment:
  ENVIRONMENT: "dev"  # dev, staging o prod según la etapa
```

**Ambientes disponibles:**
- `dev`: 5% de datos, entrenamiento rápido (3-5 min)
- `staging`: 20% de datos, validación (20-30 min)
- `prod`: 100% de datos, producción (50-90 min)

Recomiendo empezar con `dev`.

### Paso 4: Configurar permisos de MLflow

```bash
docker-compose up mlflow-init
```

Espero ver: `Permisos de MLflow configurados` y presiono Ctrl+C.

### Paso 5: Levantar servicios

```bash
docker-compose up -d
```

### Paso 6: Verificar inicio

```bash
docker-compose logs -f
```

Busco estas líneas:
```
Admin user admin created
Airflow webserver is ready
Started process to run scheduler
CONFIGURACIÓN PIPELINE SODAI - AMBIENTE: DEV
```

Presiono Ctrl+C para salir.

## Usar el Pipeline

### Paso 1: Abrir Airflow

Voy a http://localhost:8080

Usuario: `admin`  
Contraseña: `admin`

### Paso 2: Activar el DAG

1. Busco `sodai_xgb_pipeline` en la lista
2. Click en el toggle a la izquierda (se pone azul)

### Paso 3: Ejecutar

Click en el botón "▶ Trigger DAG" arriba a la derecha.

### Paso 4: Monitorear

Click en el nombre del DAG para ver el progreso:
- Verde: Completado
- Verde claro: Ejecutando
- Amarillo: En cola
- Rojo: Error
- Gris: Omitido

## Verificar Resultados

### Modelo entrenado

```bash
ls -lh models/xgb_best_model.pkl
```

### Predicciones

```bash
ls -lh artifacts/predictions/
```

Debo ver:
- `predicciones_test.parquet` (evaluación)
- `predicciones_finales.parquet` (predicciones operativas)

### Métricas en MLflow

Voy a http://localhost:5000

Busco el experimento `sodai_xgb_dev` y veo:
- Parámetros del modelo
- Métricas (F1, Precision, Recall)
- Gráficos SHAP

## Agregar Nueva Semana

Si tengo nuevos datos:

1. Copio el archivo a `data/`:
```bash
cp nueva_data.parquet data/transacciones_2025W01.parquet
```

2. En Airflow UI: Admin → Variables → Add
   - Key: `NEW_TX_FILENAME`
   - Val: `transacciones_2025W01.parquet`

3. Ejecuto el DAG nuevamente

El sistema detectará la nueva data y reentrenará automáticamente.

## Cambiar de Ambiente

Para usar `staging` o `prod`:

```bash
# Detener servicios
docker-compose down

# Editar docker-compose.yml
# Cambiar ENVIRONMENT a "staging" o "prod"

# Reconstruir y levantar
docker-compose build --no-cache
docker-compose up -d
```

## Comandos Útiles

```bash
# Ver estado
docker-compose ps

# Ver logs
docker-compose logs -f

# Reiniciar
docker-compose restart

# Detener todo
docker-compose down

# Limpiar completamente
docker-compose down -v
```

## Problemas Comunes

### Puerto 8080 ocupado

```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8080
kill -9 <PID>
```

### MLflow no inicia

```bash
docker-compose down -v
docker-compose up mlflow-init
docker-compose up -d
```

### DAG no aparece

```bash
docker-compose logs airflow-scheduler | grep "Failed to import"
```

### Archivos de data no encontrados

```bash
docker exec -it <scheduler-container> ls -lh /opt/airflow/data/
```

## URLs

- **Airflow:** http://localhost:8080
- **MLflow:** http://localhost:5000

## Tiempos Estimados

- **dev:** 3-5 minutos
- **staging:** 20-30 minutos
- **prod:** 50-90 minutos

---

Desarrollado por Deep Drinkers para SodAI Drinks