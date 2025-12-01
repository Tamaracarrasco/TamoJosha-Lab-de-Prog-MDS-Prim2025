# entrega_2/app/frontend/app.py

import gradio as gr
import requests
import pandas as pd
import os
from typing import List, Tuple
import json
import tempfile

BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8000')

def verificar_conexion_backend() -> Tuple[bool, str]:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('modelo_cargado', False):
                return True, "Conexión establecida y modelo cargado correctamente"
            else:
                return False, "Backend conectado pero modelo no disponible"
        else:
            return False, f"Backend respondió con código {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"No se pudo conectar con el backend: {str(e)}"

def predecir_individual(customer_id: str, product_id: str) -> str:
    if not customer_id or not product_id:
        return "Por favor, ingresa tanto el ID del cliente como el ID del producto."
    
    try:
        # Convertir a string y limpiar espacios
        customer_id_str = str(customer_id).strip()
        product_id_str = str(product_id).strip()
        
        response = requests.post(
            f"{BACKEND_URL}/prediccion",
            json={
                "customer_id": customer_id_str,
                "product_id": product_id_str
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            resultado = f"""
## Resultado de la Predicción

**Customer ID:** {data['customer_id']}  
**Product ID:** {data['product_id']}

---

### Predicción
{' **SÍ COMPRARÁ**' if data['prediccion'] == 1 else ' **NO COMPRARÁ**'}

### Probabilidad de Compra
**{data['probabilidad']*100:.2f}%**

### Interpretación
{data['interpretacion']}

---
*Predicción generada por el modelo XGBoost de Deep Drinkers*
            """
            return resultado
        else:
            error_detail = response.json().get('detail', 'Error desconocido')
            return f"Error al obtener predicción: {error_detail}"
            
    except requests.exceptions.RequestException as e:
        return f"Error de conexión: {str(e)}"
    except Exception as e:
        return f"Error inesperado: {str(e)}"

def predecir_batch(archivo) -> Tuple[pd.DataFrame, str, str]:
    if archivo is None:
        return None, "Por favor, sube un archivo CSV con las columnas: customer_id, product_id", None
    
    try:
        df = pd.read_csv(archivo.name)
        
        if 'customer_id' not in df.columns or 'product_id' not in df.columns:
            return None, "El archivo debe contener las columnas: customer_id y product_id", None
        
        # Convertir IDs a strings (importante para compatibilidad con el backend)
        df['customer_id'] = df['customer_id'].astype(str).str.strip()
        df['product_id'] = df['product_id'].astype(str).str.strip()
        
        datos = df[['customer_id', 'product_id']].to_dict('records')
        
        response = requests.post(
            f"{BACKEND_URL}/prediccion/batch",
            json={"datos": datos},
            timeout=60
        )
        
        if response.status_code == 200:
            resultados = response.json()
            
            df_resultados = pd.DataFrame(resultados)
            df_resultados['prediccion_texto'] = df_resultados['prediccion'].apply(
                lambda x: 'SÍ COMPRARÁ' if x == 1 else 'NO COMPRARÁ'
            )
            df_resultados['probabilidad_porcentaje'] = (df_resultados['probabilidad'] * 100).round(2)
            
            # DataFrame para mostrar en la UI
            df_display = df_resultados[[
                'customer_id', 
                'product_id', 
                'prediccion_texto', 
                'probabilidad_porcentaje',
                'interpretacion'
            ]].copy()
            
            df_display.columns = [
                'Customer ID',
                'Product ID', 
                'Predicción',
                'Probabilidad (%)',
                'Interpretación'
            ]
            
            # CSV para descargar (solo customer_id y product_id, sin encabezados)
            df_descarga = df_resultados[['customer_id', 'product_id']].copy()
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
            df_descarga.to_csv(temp_file.name, index=False, header=False)
            temp_file.close()
            
            resumen = f"""
## Resumen de Predicciones

**Total de predicciones:** {len(df_resultados)}  
**Predicciones positivas:** {(df_display['Predicción'] == 'SÍ COMPRARÁ').sum()}  
**Predicciones negativas:** {(df_display['Predicción'] == 'NO COMPRARÁ').sum()}  
**Probabilidad promedio:** {df_display['Probabilidad (%)'].mean():.2f}%

---
*Usa el botón "Descargar CSV" para obtener el archivo con customer_id y product_id*
            """
            
            return df_display, resumen, temp_file.name
        else:
            return None, f"Error al obtener predicciones: {response.json().get('detail', 'Error desconocido')}", None
            
    except Exception as e:
        return None, f"Error al procesar el archivo: {str(e)}", None

def obtener_info_modelo() -> str:
    try:
        response = requests.get(f"{BACKEND_URL}/modelo/info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            info = f"""
## Información del Modelo

**Nombre:** {data.get('nombre', 'N/A')}  
**Versión:** {data.get('version', 'N/A')}  
**Fecha de entrenamiento:** {data.get('fecha_entrenamiento', 'N/A')}

---
*Modelo desarrollado por el equipo Deep Drinkers para SodAI Drinks*
            """
            return info
        else:
            return "No se pudo obtener información del modelo"
            
    except Exception as e:
        return f"Error al obtener información: {str(e)}"

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="SodAI Drinks - Predictor de Compras"
) as demo:
    
    gr.Markdown(
        """
        # 🥤 SodAI Drinks - Sistema de Predicción de Compras
        
        Bienvenido al sistema de predicción desarrollado por **Deep Drinkers**. 
        Esta aplicación te permite predecir si un cliente comprará un producto específico la próxima semana.
        
        ---
        """
    )
    
    with gr.Tab("Predicción Individual"):
        gr.Markdown(
            """
            ## Predicción para un Cliente-Producto
            
            Ingresa el ID del cliente y el ID del producto para obtener una predicción individual.
            El sistema analizará los patrones históricos y te indicará la probabilidad de compra.
            """
        )
        
        with gr.Row():
            with gr.Column():
                customer_input = gr.Textbox(
                    label="Customer ID",
                    placeholder="Ejemplo: CLI001234",
                    info="Ingresa el identificador único del cliente"
                )
                product_input = gr.Textbox(
                    label="Product ID",
                    placeholder="Ejemplo: PRO005678",
                    info="Ingresa el identificador único del producto"
                )
                predecir_btn = gr.Button("🔮 Realizar Predicción", variant="primary")
            
            with gr.Column():
                resultado_individual = gr.Markdown(label="Resultado")
        
        predecir_btn.click(
            fn=predecir_individual,
            inputs=[customer_input, product_input],
            outputs=resultado_individual
        )
        
        gr.Markdown(
            """
            ### Cómo interpretar los resultados:
            
            - **Predicción:** Indica si el modelo predice que el cliente comprará o no el producto
            - **Probabilidad:** Porcentaje de confianza del modelo en su predicción
            - **Interpretación:** Explicación contextual de la predicción según el rango de probabilidad
            """
        )
    
    with gr.Tab("Predicción por Lotes"):
        gr.Markdown(
            """
            ## Predicción para Múltiples Cliente-Producto
            
            Sube un archivo CSV con las columnas `customer_id` y `product_id` para obtener predicciones masivas.
            El archivo debe tener el siguiente formato:
            
            ```
            customer_id,product_id
            12345,67890
            23456,78901
            34567,89012
            ```
            """
        )
        
        with gr.Row():
            with gr.Column():
                archivo_input = gr.File(
                    label="Archivo CSV",
                    file_types=[".csv"],
                    type="filepath"
                )
                predecir_batch_btn = gr.Button("📊 Realizar Predicciones por Lote", variant="primary")
        
        resultado_batch = gr.Dataframe(
            label="Resultados de las Predicciones",
            interactive=False
        )
        
        resumen_batch = gr.Markdown(label="Resumen")
        
        # Botón de descarga CSV
        with gr.Row():
            download_btn = gr.DownloadButton(
                label="📥 Descargar CSV (customer_id, product_id)",
                visible=False
            )
        
        # Estado para guardar el archivo temporal
        csv_file_state = gr.State()
        
        def actualizar_descarga(df, resumen, csv_path):
            if csv_path is not None:
                return df, resumen, gr.DownloadButton(visible=True, value=csv_path), csv_path
            return df, resumen, gr.DownloadButton(visible=False), None
        
        predecir_batch_btn.click(
            fn=predecir_batch,
            inputs=archivo_input,
            outputs=[resultado_batch, resumen_batch, csv_file_state]
        ).then(
            fn=actualizar_descarga,
            inputs=[resultado_batch, resumen_batch, csv_file_state],
            outputs=[resultado_batch, resumen_batch, download_btn, csv_file_state]
        )
        
        gr.Markdown(
            """
            ### Consejos para predicciones por lotes:
            
            - Asegúrate de que el archivo CSV tenga las columnas `customer_id` y `product_id`
            - Los IDs deben ser cadenas de texto o números
            - No hay límite en el número de predicciones, pero archivos muy grandes pueden tardar más
            - El archivo descargado contendrá solo `customer_id` y `product_id` (sin encabezados)
            """
        )
    
    with gr.Tab("Información del Modelo"):
        gr.Markdown(
            """
            ## Detalles del Modelo Predictivo
            
            Aquí puedes consultar información sobre el modelo actual en producción.
            """
        )
        
        info_btn = gr.Button("📈 Obtener Información del Modelo")
        info_output = gr.Markdown()
        
        info_btn.click(
            fn=obtener_info_modelo,
            inputs=None,
            outputs=info_output
        )
        
        gr.Markdown(
            """
            ---
            
            ### Sobre el Sistema
            
            Este sistema utiliza técnicas avanzadas de Machine Learning para predecir el comportamiento de compra 
            de los clientes. El modelo ha sido entrenado con datos históricos de transacciones y se actualiza 
            periódicamente para mantener su precisión.
            
            **Tecnologías utilizadas:**
            - Backend: FastAPI
            - Frontend: Gradio
            - Orquestación: Apache Airflow
            - Tracking: MLflow
            - Containerización: Docker
            
            **Desarrollado por:** Deep Drinkers  
            **Cliente:** SodAI Drinks 
            """
        )
    
    with gr.Tab("Ayuda"):
        gr.Markdown(
            """
            ## Guía de Uso
            
            ### Predicción Individual
            1. Ve a la pestaña "Predicción Individual"
            2. Ingresa el Customer ID en el primer campo
            3. Ingresa el Product ID en el segundo campo
            4. Haz clic en "Realizar Predicción"
            5. Observa los resultados que incluyen la predicción, probabilidad e interpretación
            
            ### Predicción por Lotes
            1. Prepara un archivo CSV con las columnas `customer_id` y `product_id`
            2. Ve a la pestaña "Predicción por Lotes"
            3. Sube tu archivo CSV usando el botón de carga
            4. Haz clic en "Realizar Predicciones por Lote"
            5. Revisa los resultados en la tabla
            6. Usa el botón "Descargar CSV" para obtener el archivo con `customer_id` y `product_id` (sin encabezados)
            
            ### Información del Modelo
            1. Ve a la pestaña "Información del Modelo"
            2. Haz clic en "Obtener Información del Modelo"
            3. Consulta los detalles del modelo actual en producción
            
            ---
            
            ## Preguntas Frecuentes
            
            **¿Qué significa la probabilidad?**  
            Es el nivel de confianza del modelo en su predicción, expresado en porcentaje. Valores más altos 
            indican mayor confianza.
            
            **¿Con qué frecuencia se actualiza el modelo?**  
            El modelo se reentrena automáticamente cuando se detecta drift en los datos o de forma periódica 
            según la configuración del pipeline.
            
            **¿Qué formato tiene el CSV descargable?**  
            El CSV descargable contiene solo dos columnas (`customer_id` y `product_id`) sin encabezados, 
            listo para ser usado en otros sistemas.
            
            **¿Qué hago si obtengo un error?**  
            Verifica que los IDs sean válidos y que el backend esté funcionando correctamente. Si el problema 
            persiste, contacta al equipo de soporte.
            
            **¿Puedo confiar en las predicciones?**  
            El modelo ha sido validado con métricas rigurosas, pero siempre debe usarse como una herramienta 
            de apoyo para la toma de decisiones, no como la única fuente de información.
            
            ---
            
            ## Contacto y Soporte
            
            Para reportar problemas o sugerencias, contacta al equipo **Deep Drinkers**.
            """
        )
    
    gr.Markdown(
        """
        ---
        
        ### Estado del Sistema
        """
    )
    
    estado_conexion = gr.Markdown()
    
    def actualizar_estado():
        conectado, mensaje = verificar_conexion_backend()
        if conectado:
            return f"**Estado:** {mensaje}"
        else:
            return f"**Estado:** {mensaje}"
    
    demo.load(fn=actualizar_estado, outputs=estado_conexion)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )