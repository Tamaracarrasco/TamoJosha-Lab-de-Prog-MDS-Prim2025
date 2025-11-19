# README – Pipeline de Predicción Semanal
**SodAI Drinks – Entrega 2**

---

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

## Ciclo de vida del pipeline

### **1. EXTRACT – Llegada de datos**
Airflow revisa la carpeta `data/` y valida la existencia de:

- `clientes.parquet`
- `productos.parquet`
- `transacciones.parquet`

Opcionalmente **detecta nuevas semanas** buscando archivos que empiecen con `transacciones_*` (ej: `transacciones_2025W01.parquet`).

**Responsable**: `extract_data_task`  
No transforma datos; solo detecta disponibilidad y comunica `has_new_data` vía XCom.

---

### **2. TRANSFORM – Construcción del dataset semanal**

Se ejecuta `build_dataset_from_raw()`:

1. `load_raw_data()` - Concatena histórico (t) + opcional nueva semana (t+1)
2. `cast_and_clean_raw_tables()` - Asegura tipos de datos correctos
3. `deduplicate_and_fix_transactions()` - Elimina duplicados y consolida registros
4. `build_transaction_level_df()` - Merge con dimensiones (clientes, productos)
5. `build_weekly_panel_with_target()` - Panel semanal por (cliente, producto) con target `y`

**Resultado guardado**: `data/df_final_latest.parquet`

**Responsable**: `transform_data_task`

---

### **3. DRIFT – Detección de cambio en los datos**

Se utiliza **PSI (Population Stability Index)** para comparar la **última semana** vs **histórico completo**:

- Calcula PSI para cada feature numérica (X, Y, size, num_deliver_per_week, num_visit_per_week)
- Si **alguna** feature tiene `PSI > 0.2` → **DRIFT detectado**

**Resultado**:
- `drift_detected`: True/False
- `drift_report`: Dict con PSI por feature

**Responsable**: `check_drift_task`

---

### **4. BRANCH – Decisión de reentrenamiento**

Lógica de decisión basada en **3 factores**:

```python
if not model_exists:        # Primera ejecución
    → ENTRENAR
    
elif has_new_data:          # Nueva semana detectada
    → ENTRENAR
    
elif drift_detected:        # Distribución cambió
    → ENTRENAR
    
else:                       # Todo igual
    → SKIP (solo predecir)
```

**Responsable**: `branch_on_drift`

---

### **5. TRAIN – Reentrenamiento condicionado**

Si se cumple alguna condición de reentrenamiento, se ejecuta `train_xgb_and_predict()`:

**Pipeline completo**:
1. **Subsampling estratificado** (si ambiente = dev/staging)
2. **Split temporal** train/val/test
3. **Optimización Optuna** (si ambiente ≠ dev)
   - Búsqueda de hiperparámetros XGBoost
   - Búsqueda de `min_frequency` para OneHotEncoder
4. **Entrenamiento XGBoost** con mejores params
5. **Selección de umbral óptimo** (maximiza F1 en validación)
6. **Evaluación en test**
7. **Tracking completo en MLflow**:
   - Parámetros del modelo
   - Métricas (F1, Precision, Recall, Accuracy)
   - Gráficos SHAP
   - Pipeline serializado
8. **Guardado de artefactos**:
   - `models/xgb_best_model.pkl`
   - `artifacts/predictions/predicciones_test.parquet`

**Responsable**: `retrain_model_task`

---

### **6. PREDICT – Predicción operativa para t+2**

Se ejecuta `build_next_week_candidates_from_raw()`:

1. Carga modelo serializado + threshold
2. Construye **todas las combinaciones** cliente-producto para la última semana disponible (t+1)
3. Agrega features semanales y estáticas
4. Aplica pipeline XGBoost
5. Calcula probabilidades y predicciones binarias
6. Genera ranking por cliente

**Resultado guardado**: `artifacts/predictions/predicciones_finales.parquet`

**Responsable**: `predict_next_week_task`

---

## Configuración del sistema

### **Estrategia por ambiente**

El pipeline soporta 3 ambientes configurables vía variable de entorno `ENVIRONMENT`:

| Ambiente | % Datos | Optuna | Trials | Timeout | Propósito |
|----------|---------|--------|--------|---------|-----------|
| **dev** | 5% (~450K) | NO - Params fijos | - | - | Testing rápido |
| **staging** | 20% (~1.8M) | SI | 20 | 10 min | Validación pre-prod |
| **prod** | 100% (~9M) | SI | 30 | 20 min | Modelo final |

**Configuración actual**: Se define en `docker-compose.yml`:

```yaml
environment:
  ENVIRONMENT: "dev"  # Cambiar a "staging" o "prod" segun capacidad y etapa
```

### **Parámetros clave** (en `config/config.py`):

- **Split temporal**: 36 semanas train, 11 semanas val, resto test
- **Threshold search**: Rango 0.05-0.80 en 40 pasos
- **Drift PSI threshold**: 0.2
- **MLflow tracking**: `http://mlflow:5000`
- **Experimento MLflow**: `sodai_xgb_{ENVIRONMENT}`

---

## Ejecución del pipeline en Airflow

**Interfaces web**:
- **Airflow**: http://localhost:8080 (usuario: `admin` / contraseña: `admin`)
- **MLflow**: http://localhost:5000

**DAG principal**: `sodai_xgb_pipeline`

---

## Resultado final del pipeline

Cada ejecución produce:

**Modelo actualizado** (si hubo reentrenamiento):
- `models/xgb_best_model.pkl`

**Predicciones operativas**:
- `artifacts/predictions/predicciones_test.parquet` (evaluación en test)
- `artifacts/predictions/predicciones_finales.parquet` (predicciones t+2)

**Artefactos MLflow**:
- Parámetros del modelo
- Métricas de evaluación
- Gráficos SHAP
- Pipeline serializado

**Reportes de drift**:
- PSI por feature en logs

---

# INSTRUCCIONES DE EJECUCIÓN

## Requisitos previos

### Software necesario:
- **Docker Desktop** (Windows/Mac) o Docker Engine (Linux)
- **Docker Compose** v2.0+
- **Git** (para clonar el repositorio)

### Especificaciones de hardware recomendadas:

| Ambiente | RAM | CPU | Disco |
|----------|-----|-----|-------|
| **dev** | 4 GB | 2 cores | 10 GB |
| **staging** | 8 GB | 4 cores | 20 GB |
| **prod** | 16 GB | 8 cores | 30 GB |

---

## 0️Configuración previa (solo Windows + Docker Desktop con WSL2)

Si usas **Docker Desktop en Windows con WSL2**, es crítico aumentar la RAM asignada:

### Paso 1: Crear/editar `.wslconfig`

```powershell
# Abrir editor
notepad $env:USERPROFILE\.wslconfig
```

### Paso 2: Agregar configuración

```ini
[wsl2]
memory=12GB
processors=4
swap=8GB
```

### Paso 3: Aplicar cambios

```powershell
# Reiniciar WSL
wsl --shutdown

# Esperar 10 segundos y verificar
docker info | findstr /C:"Memory"
```

Deberías aparecer: `Total Memory: 12.00 GB`

---

## Preparación del entorno

### 1.1. Clonar repositorio y navegar al proyecto

```bash
cd airflow/
```

### 1.2. Verificar estructura de datos

Revisar qeu los archivos que necesito están en `data/`:

```bash
ls data/

# Debe mostrar al menos:
# - clientes.parquet
# - productos.parquet
# - transacciones.parquet
```

**IMPORTANTE**: Si no tienes estos archivos, el pipeline fallará en `extract_data_task`.

---

## Configurar ambiente de ejecución

### 2.1. Editar `docker-compose.yml`

Abre `docker-compose.yml` y localiza la variable `ENVIRONMENT`:

```yaml
x-airflow-common: &airflow-common
  build: .
  environment: &airflow-common-env
    # ... otras variables ...
    ENVIRONMENT: "dev"  # CAMBIAR AQUÍ
```

**Opciones**:
- `"dev"` → Testing rápido (5% datos, sin Optuna)
- `"staging"` → Validación (20% datos, Optuna ligero) 
- `"prod"` → Producción (100% datos, Optuna completo) 

**Recomendación**: Empezar con `"dev"` para la primera ejecución.

---

## Levantar servicios Docker

### **IMPORTANTE**: Ejecutar en orden secuencial

#### 3.1. Encender Docker Desktop

```bash
# Windows: Abrir Docker Desktop desde menú inicio
# Linux: Verificar que Docker esté corriendo
sudo systemctl status docker
```

#### 3.2. Detener servicios previos (si existen)

```bash
docker-compose down -v
```

> **Nota**: El flag `-v` elimina volúmenes. Usar si quiero empezar desde cero.

#### 3.3. Configurar permisos de MLflow (solo primera vez)

```bash
docker-compose up mlflow-init
```

**Salida esperada**:
```
mlflow-init_1  | Permisos de MLflow configurados
mlflow-init_1 exited with code 0
```

> Presiona `Ctrl+C` después de ver este mensaje.

#### 3.4. Levantar todos los servicios

```bash
docker-compose up -d
```

**Servicios iniciados**:
- `postgres` - Base de datos de Airflow
- `mlflow` - Servidor de tracking MLflow
- `airflow-init` - Inicializa DB y crea usuario admin
- `airflow-webserver` - Interfaz web de Airflow
- `airflow-scheduler` - Ejecutor de tareas

#### 3.5. Monitorear el inicio

```bash
docker-compose logs -f
```

**Busca estas líneas clave**:

```
 airflow-init_1     | Admin user admin created
 mlflow_1           | [INFO] Listening at: http://0.0.0.0:5000
 airflow-webserver_1| Airflow webserver is ready
 airflow-scheduler_1| Started process (PID=xxx) to run scheduler
```

**Línea de configuración de ambiente**:
```
 CONFIGURACIÓN PIPELINE SODAI - AMBIENTE: DEV
 Subsampling:       ACTIVO
 Fracción datos:    5.0%
 Usar Optuna:       NO (params fijos)
```

> Presiona `Ctrl+C` para salir de los logs (los servicios siguen corriendo).

---

## Acceder a las interfaces web

### 4.1. Airflow

**URL**: http://localhost:8080

**Credenciales**:
- Usuario: `admin`
- Contraseña: `admin`

**Verificación**:
1. Deberías ver el DAG `sodai_xgb_pipeline` en la lista
2. El DAG debería estar **pausado** (toggle OFF)

### 4.2. MLflow

**URL**: http://localhost:5000

**Verificación**:
- Página de inicio muestra "MLflow Tracking"
- El experimento `sodai_xgb_dev` (o staging/prod) debería aparecer después de la primera ejecución

---

## Ejecutar el pipeline

### 5.1. Activar el DAG

En la interfaz de Airflow:

1. Localiza `sodai_xgb_pipeline` en la lista
2. Click en el **toggle** a la izquierda del nombre (debe ponerse en azul)
3. El DAG ahora está activado

### 5.2. Trigger manual (primera ejecución)

**Opción A - Desde la UI**:
1. Click en el nombre del DAG `sodai_xgb_pipeline`
2. Click en el botón **"▶ Trigger DAG"** (esquina superior derecha)
3. Click en **"Trigger"** en el modal

**Opción B - Desde CLI**:
```bash
docker exec -it <scheduler-container-id> \
  airflow dags trigger sodai_xgb_pipeline
```

### 5.3. Monitorear la ejecución

**En Airflow UI**:
1. Click en el DAG `sodai_xgb_pipeline`
2. Click en la ejecución más reciente (columna "Runs")
3. Vista de **Graph** muestra el progreso en tiempo real

**Colores de las tareas**:
-  Verde claro: En ejecución
-  Verde oscuro: Completada exitosamente
-  Amarillo: En cola
-  Rojo: Falló
-  Gris: Omitida (skipped)
-  Morado: Upstream fallido

**Desde terminal**:
```bash
# Logs generales
docker-compose logs -f airflow-scheduler

# Logs de una tarea específica
docker-compose logs -f airflow-scheduler | grep "extract_data_task"
```

---

## Verificar resultados

### 6.1. Modelo entrenado

```bash
# Verificar existencia
ls -lh models/xgb_best_model.pkl

# Debería mostrar algo como:
# -rw-r--r-- 1 airflow airflow 2.3M Nov 19 15:30 xgb_best_model.pkl
```

### 6.2. Predicciones generadas

```bash
ls -lh artifacts/predictions/

# Debería mostrar:
# predicciones_test.parquet      (evaluación en test)
# predicciones_finales.parquet   (predicciones t+2 operativas)
```

### 6.3. Métricas en MLflow

1. Abrir http://localhost:5000
2. Click en el experimento `sodai_xgb_dev` (o staging/prod)
3. Deberías ver un **nuevo run** con:
   - **Parameters**: xgb_*, ohe_*, split_*, environment, sampling_*
   - **Metrics**: val_f1, val_precision, test_f1, test_recall, etc.
   - **Artifacts**: 
     - `xgb_pipeline/` - Modelo serializado
     - `shap/` - Gráficos SHAP
     - `diagnostics/topN_sample.csv`

### 6.4. Logs de decisión de reentrenamiento

```bash
docker-compose logs airflow-scheduler | grep "EVALUACIÓN DE REENTRENAMIENTO"
```

**Primera ejecución (sin modelo previo)**:
```
============================================================
EVALUACIÓN DE REENTRENAMIENTO
============================================================
  Modelo existe: False
  Nueva data:    False
  Drift:         False
============================================================
[branch_on_drift]  No existe modelo previo → REENTRENAR
```

**Ejecución posterior (sin cambios)**:
```
============================================================
EVALUACIÓN DE REENTRENAMIENTO
============================================================
  Modelo existe: True
  Nueva data:    False
  Drift:         False
============================================================
[branch_on_drift]  Modelo vigente (sin cambios) → SKIP
```

---

##  Simulación de nueva semana de datos

Para testear el flujo completo con nueva data:

### 7.1. Agregar archivo de nueva semana

```bash
# Copiar archivo de ejemplo (debe existir)
cp data/nueva_semana_ejemplo.parquet data/transacciones_2025W01.parquet
```

### 7.2. Configurar variable en Airflow

1. En Airflow UI: **Admin** → **Variables**
2. Click **"+"** (Add a new record)
3. Llenar:
   - **Key**: `NEW_TX_FILENAME`
   - **Val**: `transacciones_2025W01.parquet`
4. Click **Save**

### 7.3. Ejecutar el DAG nuevamente

El pipeline debería:
1. Detectar nueva data en `extract_data_task`
2. Decidir reentrenar en `branch_on_drift`
3. Entrenar con datos ampliados (t + t+1)
4. Generar predicciones para t+2

**Logs esperados**:
```
[extract_data_task] Nueva data detectada: ['transacciones_2025W01.parquet']
[branch_on_drift] Nueva data detectada → REENTRENAR
```

---

## Cambio de ambiente (dev → staging → prod)

### 8.1. Detener servicios

```bash
docker-compose down
```

### 8.2. Editar `docker-compose.yml`

```yaml
ENVIRONMENT: "staging"  # o "prod"
```

### 8.3. Rebuild y reiniciar

```bash
docker-compose build --no-cache
docker-compose up -d
```

### 8.4. Verificar configuración

```bash
docker-compose logs airflow-scheduler | grep "CONFIGURACIÓN PIPELINE"
```

**Staging**:
```
 CONFIGURACIÓN PIPELINE SODAI - AMBIENTE: STAGING
 Subsampling:       ACTIVO
 Fracción datos:    20.0%
 Usar Optuna:       SÍ
 Optuna trials:     20
  Optuna timeout:    10 min
```

**Prod**:
```
 CONFIGURACIÓN PIPELINE SODAI - AMBIENTE: PROD
 Subsampling:       DESACTIVADO
 Fracción datos:    100.0%
 Usar Optuna:       SÍ
 Optuna trials:     30
  Optuna timeout:    20 min
```

---

## Troubleshooting - Problemas que tuvimos y costo INFINITO solucionar

### Problema 1: "Bind for 0.0.0.0:8080 failed: port is already allocated"

**Causa**: Puerto 8080 ocupado por otro servicio.

**Solución**:
```bash
# Windows
netstat -ano | findstr :8080
taskkill /PID <PID> /F

# Linux/Mac
lsof -i :8080
kill -9 <PID>
```

### Problema 2: MLflow no inicia (permiso denegado)

**Causa**: Permisos incorrectos en volumen de MLflow.

**Solución**:
```bash
docker-compose down -v
docker-compose up mlflow-init  # Volver a ejecutar
docker-compose up -d
```

### Problema 3: Airflow scheduler se reinicia constantemente

**Causa**: RAM insuficiente (común en WSL2).

**Solución**:
1. Verificar `.wslconfig` tiene al menos 8GB
2. Reiniciar WSL: `wsl --shutdown`
3. Aumentar swap si es necesario

### Problema 4: DAG no aparece en la UI

**Causa**: Error de sintaxis en el DAG o imports faltantes.

**Solución**:
```bash
# Ver errores de parsing
docker-compose logs airflow-scheduler | grep "Failed to import"

# Validar sintaxis del DAG
docker exec -it <scheduler-container> python /opt/airflow/dags/sodai_xgb_pipeline_dag.py
```

### Problema 5: Error "No such file: df_final_latest.parquet"

**Causa**: Archivos de data faltantes o ruta incorrecta.

**Solución**:
```bash
# Verificar archivos en data/
docker exec -it <scheduler-container> ls -lh /opt/airflow/data/

# Debería mostrar:
# clientes.parquet
# productos.parquet
# transacciones.parquet
```

### Problema 6: MLflow no registra artefactos

**Causa**: Permisos del volumen o URI mal configurado.

**Solución**:
```bash
# Verificar URI
docker exec -it <scheduler-container> env | grep MLFLOW

# Debería mostrar:
# MLFLOW_TRACKING_URI=http://mlflow:5000

# Test de conectividad
docker exec -it <scheduler-container> curl http://mlflow:5000/health
```

---

## Comandos útiles

### Gestión de servicios

```bash
# Ver estado de servicios
docker-compose ps

# Reiniciar un servicio específico
docker-compose restart airflow-scheduler

# Ver logs en tiempo real
docker-compose logs -f <service-name>

# Entrar a un contenedor
docker exec -it <container-id> bash

# Detener todo y limpiar
docker-compose down -v
docker system prune -a
```

### Gestión de Airflow

```bash
# Listar DAGs
docker exec -it <scheduler-container> airflow dags list

# Trigger manual de DAG
docker exec -it <scheduler-container> airflow dags trigger sodai_xgb_pipeline

# Ver estado de ejecuciones
docker exec -it <scheduler-container> airflow dags list-runs -d sodai_xgb_pipeline

# Ver logs de una tarea específica
docker exec -it <scheduler-container> airflow tasks logs sodai_xgb_pipeline extract_data_task <execution-date>
```

### Verificación de archivos

```bash
# Ver tamaño del modelo
docker exec -it <scheduler-container> ls -lh /opt/airflow/models/

# Ver predicciones generadas
docker exec -it <scheduler-container> head /opt/airflow/artifacts/predictions/predicciones_finales.parquet

# Ver estructura de carpetas
docker exec -it <scheduler-container> tree /opt/airflow/artifacts/
```

---

## Tiempos de ejecución estimados

| Ambiente | Registros | Optuna | Tiempo total | Bottleneck |
|----------|-----------|--------|--------------|------------|
| **dev** | ~450K | NO | ~3-5 min | Transformación de datos |
| **staging** | ~1.8M | SI - 20 trials | ~20-30 min | Optuna |
| **prod** | ~9M | SI - 30 trials | ~50-90 min | Optuna + Entrenamiento |

**Desglose típico en prod**:
- Extract: 10 seg
- Transform: 5-8 min
- Drift detection: 1-2 min
- Optuna: 30-40 min
- Entrenamiento final: 8-12 min
- SHAP: 3-5 min
- Predicción t+2: 2-3 min

---

##  Documentación adicional

- **Airflow**: https://airflow.apache.org/docs/
- **MLflow**: https://mlflow.org/docs/latest/index.html
- **Optuna**: https://optuna.readthedocs.io/
- **XGBoost**: https://xgboost.readthedocs.io/
- **SHAP**: https://shap.readthedocs.io/

---

## Soporte

Para problemas o preguntas:
1. Revisar sección **Troubleshooting** arriba
2. Verificar logs: `docker-compose logs -f`
3. Consultar documentación de Airflow/MLflow

---

## Notas finales

- **Primera ejecución**: Siempre usar ambiente `dev` para validar
- **Producción**: Solo usar ambiente `prod` cuando el pipeline esté validado
- **Nuevas semanas**: Configurar variable `NEW_TX_FILENAME` en Airflow UI
- **Drift threshold**: Ajustable en `config.py` (default: 0.2)
- **Experimentos MLflow**: Se crean separados por ambiente

---



   


