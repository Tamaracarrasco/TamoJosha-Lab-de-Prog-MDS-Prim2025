"""
Sistema de Recomendación de Productos para SodAI Drinks
Backend FastAPI que recomienda productos basándose en el modelo de predicción XGBoost
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import pickle
import logging
from datetime import datetime
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SodAI Drinks - Sistema de Recomendación",
    description="API para recomendar productos basándose en predicciones de compra",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
modelo = None
df_clientes = None
df_productos = None
threshold = 0.6134

MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/xgb_best_model.pkl')
DATA_DIR = os.getenv('DATA_DIR', '/app/data')


class RecomendacionResponse(BaseModel):
    cliente_id: str
    recomendaciones: List[Dict]
    timestamp: str


def cargar_modelo():
    """Carga el modelo XGBoost"""
    global modelo
    try:
        logger.info(f"Cargando modelo desde: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            modelo = pickle.load(f)
        logger.info("Modelo cargado exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error al cargar modelo: {e}")
        return False


def cargar_datos_dimensionales():
    """Carga los datos de clientes y productos desde df_final_latest.parquet"""
    global df_clientes, df_productos
    try:
        # Cargar desde df_final_latest que tiene todas las columnas
        df_final_path = os.path.join(DATA_DIR, 'df_final_latest.parquet')
        
        logger.info(f"Cargando datos desde: {df_final_path}")
        df_full = pd.read_parquet(df_final_path)
        
        # Extraer clientes únicos con sus atributos
        columnas_clientes = ['customer_id', 'region_id', 'customer_type', 'zone_id', 
                            'num_deliver_per_week', 'num_visit_per_week']
        df_clientes = df_full[columnas_clientes].drop_duplicates(subset=['customer_id']).copy()
        df_clientes['cliente_id'] = df_clientes['customer_id'].astype(str)
        
        # Extraer productos únicos con sus atributos
        columnas_productos = ['product_id', 'category', 'brand', 'size', 
                             'package', 'segment', 'sub_category']
        df_productos = df_full[columnas_productos].drop_duplicates(subset=['product_id']).copy()
        df_productos['producto_id'] = df_productos['product_id'].astype(str)
        
        # Agregar nombre de producto (combinando categoría y marca)
        df_productos['nombre'] = df_productos['brand'] + ' ' + df_productos['category']
        
        logger.info(f"Datos cargados: {len(df_clientes)} clientes, {len(df_productos)} productos")
        return True
    except Exception as e:
        logger.error(f"Error al cargar datos dimensionales: {e}")
        logger.error(f"Traceback: ", exc_info=True)
        return False


def preparar_features_para_prediccion(cliente_id: str, producto_id: str) -> pd.DataFrame:
    """Prepara las features necesarias para la predicción"""
    
    # Obtener datos del cliente
    cliente = df_clientes[df_clientes['cliente_id'] == cliente_id]
    if cliente.empty:
        raise ValueError(f"Cliente {cliente_id} no encontrado")
    
    # Obtener datos del producto
    producto = df_productos[df_productos['producto_id'] == producto_id]
    if producto.empty:
        raise ValueError(f"Producto {producto_id} no encontrado")
    
    # Crear diccionario con las features usando nombres de columnas reales
    features = {
        'customer_id': int(cliente['customer_id'].values[0]),
        'product_id': int(producto['product_id'].values[0]),
        'region_id': cliente['region_id'].values[0],
        'customer_type': cliente['customer_type'].values[0],
        'zone_id': cliente['zone_id'].values[0],
        'num_deliver_per_week': cliente['num_deliver_per_week'].values[0],
        'num_visit_per_week': cliente['num_visit_per_week'].values[0],
        'category': producto['category'].values[0],
        'brand': producto['brand'].values[0],
        'size': producto['size'].values[0],
        'package': producto['package'].values[0],
        'segment': producto['segment'].values[0],
        'sub_category': producto['sub_category'].values[0]
    }
    
    df = pd.DataFrame([features])
    return df


def predecir_probabilidad(cliente_id: str, producto_id: str) -> float:
    """Predice la probabilidad de compra para un par cliente-producto"""
    try:
        df_features = preparar_features_para_prediccion(cliente_id, producto_id)
        probabilidad = modelo['model'].predict_proba(df_features)[0][1]
        return float(probabilidad)
    except Exception as e:
        logger.warning(f"Error al predecir {cliente_id}-{producto_id}: {e}")
        return 0.0


def obtener_recomendaciones(cliente_id: str, top_n: int = 5) -> List[Dict]:
    """
    Genera recomendaciones de productos para un cliente
    Calcula la probabilidad de compra para todos los productos y retorna los top N
    """
    if modelo is None or df_productos is None:
        raise ValueError("Modelo o datos no cargados")
    
    # Verificar que el cliente existe
    if cliente_id not in df_clientes['cliente_id'].values:
        raise ValueError(f"Cliente {cliente_id} no encontrado")
    
    recomendaciones = []
    
    # Calcular probabilidad para cada producto
    for _, producto in df_productos.iterrows():
        producto_id = str(producto['producto_id'])
        
        try:
            probabilidad = predecir_probabilidad(cliente_id, producto_id)
            
            recomendaciones.append({
                'producto_id': producto_id,
                'nombre': producto.get('nombre', f"Producto {producto_id}"),
                'categoria': producto['category'],
                'marca': producto['brand'],
                'volumen': float(producto['size']),
                'tipo_envase': producto['package'],
                'probabilidad': round(probabilidad * 100, 2),
                'recomendado': probabilidad >= threshold
            })
        except Exception as e:
            logger.warning(f"Error procesando producto {producto_id}: {e}")
            continue
    
    # Ordenar por probabilidad y tomar top N
    recomendaciones.sort(key=lambda x: x['probabilidad'], reverse=True)
    top_recomendaciones = recomendaciones[:top_n]
    
    return top_recomendaciones


@app.on_event("startup")
async def startup_event():
    """Inicialización al arrancar el servidor"""
    logger.info("Iniciando servidor de recomendaciones...")
    
    modelo_ok = cargar_modelo()
    datos_ok = cargar_datos_dimensionales()
    
    if not modelo_ok or not datos_ok:
        logger.warning("El servidor se inició pero faltan componentes")
    else:
        logger.info("Servidor de recomendaciones listo")


@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "servicio": "SodAI Drinks - Sistema de Recomendación",
        "version": "1.0.0",
        "estado": "activo"
    }


@app.get("/health")
async def health_check():
    """Verifica el estado del servicio"""
    return {
        "status": "healthy" if modelo is not None and df_productos is not None else "degraded",
        "modelo_cargado": modelo is not None,
        "datos_cargados": df_productos is not None and df_clientes is not None,
        "num_productos": len(df_productos) if df_productos is not None else 0,
        "num_clientes": len(df_clientes) if df_clientes is not None else 0,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/recomendaciones/{cliente_id}")
async def recomendar_productos(cliente_id: str, top_n: int = 5):
    """
    Genera recomendaciones de productos para un cliente
    
    Parámetros:
    - cliente_id: ID del cliente
    - top_n: Número de recomendaciones a retornar (default: 5)
    """
    if modelo is None or df_productos is None:
        raise HTTPException(
            status_code=503,
            detail="El servicio no está disponible. Modelo o datos no cargados."
        )
    
    if top_n < 1 or top_n > 50:
        raise HTTPException(
            status_code=400,
            detail="El parámetro top_n debe estar entre 1 y 50"
        )
    
    try:
        recomendaciones = obtener_recomendaciones(cliente_id, top_n)
        
        return {
            "cliente_id": cliente_id,
            "total_recomendaciones": len(recomendaciones),
            "recomendaciones": recomendaciones,
            "timestamp": datetime.now().isoformat()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generando recomendaciones: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.get("/clientes/lista")
async def listar_clientes(limite: int = 20):
    """
    Lista clientes disponibles
    
    Parámetros:
    - limite: Número máximo de clientes a retornar (default: 20)
    """
    if df_clientes is None:
        raise HTTPException(status_code=503, detail="Datos no disponibles")
    
    clientes_sample = df_clientes.head(limite)[['cliente_id', 'region_id', 'customer_type', 'zone_id']].to_dict('records')
    
    return {
        "total_clientes": len(df_clientes),
        "muestra": clientes_sample,
        "limite": limite
    }


@app.get("/productos/lista")
async def listar_productos(limite: int = 20):
    """
    Lista productos disponibles
    
    Parámetros:
    - limite: Número máximo de productos a retornar (default: 20)
    """
    if df_productos is None:
        raise HTTPException(status_code=503, detail="Datos no disponibles")
    
    productos_sample = df_productos.head(limite).to_dict('records')
    
    return {
        "total_productos": len(df_productos),
        "muestra": productos_sample,
        "limite": limite
    }


@app.post("/modelo/reload")
async def reload_modelo():
    """Recarga el modelo y los datos sin reiniciar el servidor"""
    try:
        modelo_ok = cargar_modelo()
        datos_ok = cargar_datos_dimensionales()
        
        if modelo_ok and datos_ok:
            return {
                "status": "success",
                "mensaje": "Modelo y datos recargados exitosamente",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Error al recargar modelo o datos"
            )
    except Exception as e:
        logger.error(f"Error al recargar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)