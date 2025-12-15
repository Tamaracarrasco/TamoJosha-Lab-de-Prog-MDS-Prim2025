# Sistema de Recomendación de Productos - SodAI Drinks

Sistema que recomienda los 5 mejores productos para cualquier cliente usando el modelo XGBoost de Airflow.

## Qué Hace

Ingreso el ID de un cliente y el sistema:
1. Calcula la probabilidad de que compre cada producto disponible
2. Ordena los productos por probabilidad
3. Me muestra los 5 productos con mayor probabilidad

## Estructura

```
recsys/
├── backend/         (API FastAPI - puerto 8001)
├── frontend/        (Interfaz Gradio - puerto 7861)
└── docker-compose.yml
```

## Requisitos

Antes de empezar, verifico que tengo:
- Docker y Docker Compose instalados
- El modelo entrenado en `../airflow/models/xgb_best_model.pkl`
- Los datos en `../airflow/data/clientes.parquet` y `../airflow/data/productos.parquet`

## Instalación

### Paso 1: Ir a la carpeta correcta

Desde la raíz del proyecto:

```bash
cd recsys
```

Si estoy en `airflow/` u otro:
```bash
cd ../recsys
```

### Paso 2: Construir y levantar

```bash
docker-compose build
docker-compose up -d
```

### Paso 3: Esperar y verificar

```bash
# Esperar 1 minuto
sleep 60

# Verificar que funciona
curl http://localhost:8001/health
```

Debo ver `"modelo_cargado": true` y `"datos_cargados": true`.

### Paso 4: Abrir la aplicación

**Interfaz Web:** http://localhost:7861

## Uso

### Opción 1: Interfaz Web

1. Abro http://localhost:7861
2. Veo los clientes de ejemplo en el panel derecho
3. Ingreso un ID de cliente (ej: 61353)
4. Click en "Obtener Recomendaciones"
5. Veo la tabla con los 5 productos recomendados

### Opción 2: API

```bash
curl http://localhost:8001/recomendaciones/61353?top_n=5
```

## Comandos Útiles

```bash
# Ver logs
docker-compose logs -f

# Detener
docker-compose down

# Reiniciar
docker-compose restart

# Ver estado
docker-compose ps
```

## Problemas Comunes

### No encuentra el modelo

Verifico que existe:
```bash
ls -lh ../airflow/models/xgb_best_model.pkl
```

Si no existe, ejecuto primero el DAG de Airflow.

### Cliente no encontrado

Obtengo una lista de clientes válidos:
```bash
curl http://localhost:8001/clientes/lista
```

Uso uno de esos IDs.

### Puerto en uso

Si el puerto 8001 o 7861 está ocupado, edito `docker-compose.yml` y cambio:

```yaml
ports:
  - "8002:8001"  # Cambiar 8001 por 8002
```

## URLs

- **Interfaz:** http://localhost:7861
- **API:** http://localhost:8001
- **Docs:** http://localhost:8001/docs
- **Health:** http://localhost:8001/health

## Notas

- Este sistema es independiente de la app principal
- Puede correr simultáneamente con la app de predicción
- Lee directamente del modelo de Airflow (no copia archivos)
- Los puertos son diferentes para evitar conflictos:
  - App principal: 8000 y 7860
  - Recsys: 8001 y 7861

---

Desarrollado por Deep Drinkers para SodAI Drinks