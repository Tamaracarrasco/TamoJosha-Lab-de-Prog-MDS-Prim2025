# entrega_2/app/backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SodAI Drinks API",
    description="API para predicciones de compra de productos con XGBoost",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerProduct(BaseModel):
    customer_id: str = Field(..., description="ID del cliente")
    product_id: str = Field(..., description="ID del producto")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "12345",
                "product_id": "67890"
            }
        }

class PrediccionRequest(BaseModel):
    datos: List[CustomerProduct] = Field(..., description="Lista de pares customer-product")
    
    class Config:
        json_schema_extra = {
            "example": {
                "datos": [
                    {"customer_id": "12345", "product_id": "67890"},
                    {"customer_id": "23456", "product_id": "78901"}
                ]
            }
        }

class PrediccionResponse(BaseModel):
    customer_id: str
    product_id: str
    prediccion: int
    probabilidad: float
    interpretacion: str

class ModeloInfo(BaseModel):
    nombre: str
    version: str
    tipo: str
    threshold: Optional[float]
    fecha_carga: Optional[str]

pipeline_xgb = None
threshold = None
clientes_df = None
productos_df = None
modelo_info = {
    "nombre": "XGBoost Productivo SodAI",
    "version": "2.0",
    "tipo": "Pipeline sklearn + XGBoost",
    "threshold": None,
    "fecha_carga": None
}

def cargar_datos_dimensionales():
    """
    Cargo las tablas de clientes y productos para tener las features completas
    """
    global clientes_df, productos_df
    
    data_dir = os.getenv('DATA_DIR', '/app/data')
    
    try:
        clientes_path = os.path.join(data_dir, 'clientes.parquet')
        productos_path = os.path.join(data_dir, 'productos.parquet')
        
        if os.path.exists(clientes_path):
            clientes_df = pd.read_parquet(clientes_path)
            clientes_df['customer_id'] = clientes_df['customer_id'].astype(str)
            logger.info(f"Clientes cargados: {len(clientes_df)} registros")
        else:
            logger.warning(f"No se encontró {clientes_path}")
            clientes_df = None
            
        if os.path.exists(productos_path):
            productos_df = pd.read_parquet(productos_path)
            productos_df['product_id'] = productos_df['product_id'].astype(str)
            logger.info(f"Productos cargados: {len(productos_df)} registros")
        else:
            logger.warning(f"No se encontró {productos_path}")
            productos_df = None
            
    except Exception as e:
        logger.error(f"Error cargando datos dimensionales: {e}")
        clientes_df = None
        productos_df = None

def cargar_modelo():
    global pipeline_xgb, threshold, modelo_info
    
    try:
        model_path = os.getenv('MODEL_PATH', '/app/models/xgb_best_model.pkl')
        
        if not os.path.exists(model_path):
            logger.error(f"No existe el archivo {model_path}")
            return False
            
        logger.info(f"Cargando modelo desde {model_path}")
        
        with open(model_path, 'rb') as f:
            model_pack = pickle.load(f)
        
        pipeline_xgb = model_pack.get("model")
        threshold = model_pack.get("threshold", 0.5)
        
        if pipeline_xgb is None:
            logger.error("El pickle no contiene 'model'")
            return False
            
        logger.info(f"Pipeline XGBoost cargado exitosamente")
        logger.info(f"Threshold óptimo: {threshold:.4f}")
        
        modelo_info["threshold"] = float(threshold)
        modelo_info["fecha_carga"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cargar_datos_dimensionales()
        
        return True
        
    except Exception as e:
        logger.error(f"Error al cargar modelo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando aplicación SodAI Drinks...")
    if not cargar_modelo():
        logger.warning("La aplicación se inició sin un modelo cargado")

@app.get("/")
async def root():
    return {
        "mensaje": "Bienvenido a la API de SodAI Drinks - Sistema Productivo XGBoost",
        "version": "2.0.0",
        "modelo": "Pipeline XGBoost con detección de drift",
        "endpoints": {
            "health": "/health",
            "modelo_info": "/modelo/info",
            "prediccion": "/prediccion",
            "prediccion_batch": "/prediccion/batch",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    modelo_cargado = pipeline_xgb is not None
    datos_dimensionales = clientes_df is not None and productos_df is not None
    
    status = "healthy" if modelo_cargado else "degraded"
    if modelo_cargado and not datos_dimensionales:
        status = "partial"
    
    return {
        "status": status,
        "modelo_cargado": modelo_cargado,
        "datos_dimensionales_cargados": datos_dimensionales,
        "threshold": float(threshold) if threshold is not None else None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/modelo/info", response_model=ModeloInfo)
async def obtener_info_modelo():
    if pipeline_xgb is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado actualmente"
        )
    
    return modelo_info

def preparar_features_para_prediccion(
    customer_id: str,
    product_id: str
) -> pd.DataFrame:
    """
    Preparo un DataFrame con todas las features que espera el modelo XGBoost.
    Uso las tablas de clientes y productos para obtener las dimensiones.
    """
    
    if clientes_df is None or productos_df is None:
        raise HTTPException(
            status_code=503,
            detail="Datos dimensionales no disponibles. El backend necesita acceso a clientes.parquet y productos.parquet"
        )
    
    cliente_info = clientes_df[clientes_df['customer_id'] == customer_id]
    producto_info = productos_df[productos_df['product_id'] == product_id]
    
    if cliente_info.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Cliente {customer_id} no encontrado en la base de datos"
        )
    
    if producto_info.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Producto {product_id} no encontrado en la base de datos"
        )
    
    cliente_info = cliente_info.iloc[0]
    producto_info = producto_info.iloc[0]
    
    semana_actual = datetime.now().isocalendar()
    semana_str = f"{semana_actual.year}-{int(semana_actual.week):02d}"
    
    features = {
        'customer_id': str(customer_id),
        'product_id': str(product_id),
        'semana': semana_str,
        'purchased_count': 0,
        'compra_o_no': 0,
    }
    
    features['customer_type'] = str(cliente_info.get('customer_type', 'UNKNOWN'))
    features['X'] = float(cliente_info.get('X', 0.0))
    features['Y'] = float(cliente_info.get('Y', 0.0))
    features['zone_id'] = str(cliente_info.get('zone_id', 'UNKNOWN'))
    features['region_id'] = str(cliente_info.get('region_id', 'UNKNOWN'))
    features['num_deliver_per_week'] = int(cliente_info.get('num_deliver_per_week', 0))
    features['num_visit_per_week'] = int(cliente_info.get('num_visit_per_week', 0))
    
    features['brand'] = str(producto_info.get('brand', 'UNKNOWN'))
    features['category'] = str(producto_info.get('category', 'UNKNOWN'))
    features['sub_category'] = str(producto_info.get('sub_category', 'UNKNOWN'))
    features['segment'] = str(producto_info.get('segment', 'UNKNOWN'))
    features['package'] = str(producto_info.get('package', 'UNKNOWN'))
    features['size'] = float(producto_info.get('size', 0.0))
    
    for col in ['customer_type', 'brand', 'category', 'sub_category', 'segment', 'package']:
        if col in features:
            features[col] = str(features[col])
    
    df = pd.DataFrame([features])
    
    return df

def interpretar_prediccion(probabilidad: float) -> str:
    if probabilidad >= 0.7:
        return "Alta probabilidad de compra - Cliente muy interesado en este producto"
    elif probabilidad >= 0.5:
        return "Probabilidad moderada de compra - Cliente potencialmente interesado"
    elif probabilidad >= 0.3:
        return "Baja probabilidad de compra - Cliente poco interesado"
    else:
        return "Muy baja probabilidad de compra - Cliente no muestra interés"

@app.post("/prediccion", response_model=PrediccionResponse)
async def realizar_prediccion(item: CustomerProduct):
    if pipeline_xgb is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Por favor, intenta más tarde."
        )
    
    try:
        X = preparar_features_para_prediccion(item.customer_id, item.product_id)
        
        proba = pipeline_xgb.predict_proba(X)[0, 1]
        
        prediccion = 1 if proba >= threshold else 0
        
        interpretacion = interpretar_prediccion(proba)
        
        return PrediccionResponse(
            customer_id=item.customer_id,
            product_id=item.product_id,
            prediccion=int(prediccion),
            probabilidad=round(float(proba), 4),
            interpretacion=interpretacion
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(e)}"
        )

@app.post("/prediccion/batch", response_model=List[PrediccionResponse])
async def realizar_prediccion_batch(request: PrediccionRequest):
    if pipeline_xgb is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Por favor, intenta más tarde."
        )
    
    try:
        resultados = []
        
        dfs_features = []
        items_validos = []
        
        for item in request.datos:
            try:
                X = preparar_features_para_prediccion(item.customer_id, item.product_id)
                dfs_features.append(X)
                items_validos.append(item)
            except HTTPException as e:
                logger.warning(f"Customer {item.customer_id} o producto {item.product_id} no encontrado: {e.detail}")
                resultados.append(PrediccionResponse(
                    customer_id=item.customer_id,
                    product_id=item.product_id,
                    prediccion=0,
                    probabilidad=0.0,
                    interpretacion="Error: Cliente o producto no encontrado"
                ))
        
        if dfs_features:
            X_batch = pd.concat(dfs_features, ignore_index=True)
            
            probas = pipeline_xgb.predict_proba(X_batch)[:, 1]
            
            for i, (item, proba) in enumerate(zip(items_validos, probas)):
                prediccion = 1 if proba >= threshold else 0
                interpretacion = interpretar_prediccion(proba)
                
                resultados.append(PrediccionResponse(
                    customer_id=item.customer_id,
                    product_id=item.product_id,
                    prediccion=int(prediccion),
                    probabilidad=round(float(proba), 4),
                    interpretacion=interpretacion
                ))
        
        return resultados
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar las predicciones: {str(e)}"
        )

@app.post("/modelo/reload")
async def recargar_modelo():
    try:
        if cargar_modelo():
            return {
                "mensaje": "Modelo recargado exitosamente",
                "threshold": float(threshold) if threshold is not None else None,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="No se pudo recargar el modelo"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al recargar modelo: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)