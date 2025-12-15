"""
Frontend Gradio para el Sistema de Recomendación de SodAI Drinks
Interfaz amigable para obtener recomendaciones de productos
"""

import gradio as gr
import requests
import pandas as pd
import os
from typing import Tuple

BACKEND_URL = os.getenv('BACKEND_URL', 'http://backend:8001')


def verificar_backend() -> bool:
    """Verifica que el backend esté disponible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def obtener_recomendaciones(cliente_id: str, num_recomendaciones: int = 5) -> Tuple[pd.DataFrame, str]:
    """
    Obtiene recomendaciones de productos para un cliente
    
    Args:
        cliente_id: ID del cliente
        num_recomendaciones: Número de productos a recomendar
    
    Returns:
        DataFrame con las recomendaciones y mensaje de resumen
    """
    if not cliente_id or not cliente_id.strip():
        return None, "Por favor, ingresa un ID de cliente válido."
    
    try:
        cliente_id_str = str(cliente_id).strip()
        
        response = requests.get(
            f"{BACKEND_URL}/recomendaciones/{cliente_id_str}",
            params={"top_n": num_recomendaciones},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            recomendaciones = data['recomendaciones']
            
            if not recomendaciones:
                return None, f"No se encontraron recomendaciones para el cliente {cliente_id_str}"
            
            df = pd.DataFrame(recomendaciones)
            
            df['recomendado_texto'] = df['recomendado'].apply(
                lambda x: '⭐ Sí' if x else 'No'
            )
            
            df_display = df[[
                'producto_id',
                'nombre',
                'categoria',
                'marca',
                'volumen',
                'tipo_envase',
                'probabilidad',
                'recomendado_texto'
            ]].copy()
            
            df_display.columns = [
                'ID Producto',
                'Nombre',
                'Categoría',
                'Marca',
                'Volumen',
                'Tipo Envase',
                'Probabilidad (%)',
                'Recomendado'
            ]
            
            recomendados = df[df['recomendado']].shape[0]
            prob_promedio = df['probabilidad'].mean()
            prob_max = df['probabilidad'].max()
            mejor_producto = df.loc[df['probabilidad'].idxmax()]
            
            resumen = f"""
## Recomendaciones para Cliente {cliente_id_str}

**Total de productos analizados:** {len(df)}  
**Productos altamente recomendados:** {recomendados} (probabilidad ≥ 61.34%)  
**Probabilidad promedio:** {prob_promedio:.2f}%  
**Mejor recomendación:** {mejor_producto['nombre']} ({prob_max:.2f}%)

### Interpretación

Los productos marcados con ⭐ tienen alta probabilidad de compra y son especialmente recomendados para este cliente.

El producto con mayor probabilidad es **{mejor_producto['nombre']}** de la categoría **{mejor_producto['categoria']}**.
            """
            
            return df_display, resumen
            
        elif response.status_code == 404:
            return None, f"Cliente {cliente_id_str} no encontrado en la base de datos."
        else:
            error_detail = response.json().get('detail', 'Error desconocido')
            return None, f"Error al obtener recomendaciones: {error_detail}"
            
    except requests.exceptions.Timeout:
        return None, "Tiempo de espera agotado. El servidor está procesando muchas solicitudes."
    except requests.exceptions.ConnectionError:
        return None, "No se pudo conectar con el servidor. Verifica que el backend esté corriendo."
    except Exception as e:
        return None, f"Error inesperado: {str(e)}"


def obtener_clientes_ejemplo() -> str:
    """Obtiene una lista de clientes de ejemplo"""
    try:
        response = requests.get(
            f"{BACKEND_URL}/clientes/lista",
            params={"limite": 10},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            clientes = data['muestra']
            
            texto = "## Clientes de Ejemplo\n\n"
            texto += "Puedes probar con cualquiera de estos IDs:\n\n"
            
            for i, cliente in enumerate(clientes, 1):
                texto += f"{i}. **{cliente['cliente_id']}** - Región: {cliente['region_id']} - Tipo: {cliente['customer_type']}\n"
            
            texto += f"\n*Total de clientes disponibles: {data['total_clientes']}*"
            
            return texto
        else:
            return "No se pudieron cargar los clientes de ejemplo."
            
    except Exception as e:
        return f"Error al obtener clientes: {str(e)}"


def verificar_estado_sistema() -> str:
    """Verifica el estado del sistema de recomendación"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            estado_icono = "🟢" if data['status'] == 'healthy' else "🟡"
            modelo_icono = "✅" if data['modelo_cargado'] else "❌"
            datos_icono = "✅" if data['datos_cargados'] else "❌"
            
            texto = f"""
## Estado del Sistema {estado_icono}

**Estado general:** {data['status']}  
**Modelo cargado:** {modelo_icono}  
**Datos cargados:** {datos_icono}  
**Productos disponibles:** {data['num_productos']}  
**Clientes disponibles:** {data['num_clientes']}

**Última verificación:** {data['timestamp']}
            """
            
            return texto
        else:
            return "⚠️ No se pudo verificar el estado del sistema."
            
    except Exception as e:
        return f"❌ Error de conexión: {str(e)}"


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="SodAI Drinks - Sistema de Recomendación"
) as app:
    
    gr.Markdown("""
    # 🥤 SodAI Drinks - Sistema de Recomendación de Productos
    
    Obtén las mejores recomendaciones de productos para tus clientes basadas en inteligencia artificial.
    
    Este sistema analiza todos los productos disponibles y calcula la probabilidad de compra para cada uno,
    mostrándote los productos con mayor probabilidad de ser adquiridos por el cliente.
    """)
    
    with gr.Tab("Recomendaciones"):
        gr.Markdown("""
        ### Obtener Recomendaciones
        
        Ingresa el ID de un cliente y selecciona cuántos productos deseas que se recomienden.
        El sistema calculará la probabilidad de compra para cada producto y te mostrará los mejores resultados.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                cliente_input = gr.Textbox(
                    label="ID del Cliente",
                    placeholder="Ej: 61353",
                    info="Ingresa el identificador único del cliente"
                )
                
                num_rec_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Número de Recomendaciones",
                    info="¿Cuántos productos quieres recomendar?"
                )
                
                recomendar_btn = gr.Button(
                    "🔍 Obtener Recomendaciones",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                ejemplos_clientes = gr.Markdown(
                    obtener_clientes_ejemplo(),
                    label="Clientes de Ejemplo"
                )
                
                actualizar_ejemplos_btn = gr.Button(
                    "🔄 Actualizar Ejemplos",
                    size="sm"
                )
        
        resumen_output = gr.Markdown(label="Resumen")
        tabla_output = gr.Dataframe(
            label="Recomendaciones de Productos",
            wrap=True
        )
        
        recomendar_btn.click(
            fn=obtener_recomendaciones,
            inputs=[cliente_input, num_rec_slider],
            outputs=[tabla_output, resumen_output]
        )
        
        actualizar_ejemplos_btn.click(
            fn=obtener_clientes_ejemplo,
            outputs=ejemplos_clientes
        )
    
    with gr.Tab("Estado del Sistema"):
        gr.Markdown("""
        ### Monitoreo del Sistema
        
        Verifica el estado del backend y la disponibilidad de los datos.
        """)
        
        estado_output = gr.Markdown(
            verificar_estado_sistema(),
            label="Estado"
        )
        
        verificar_btn = gr.Button("🔄 Verificar Estado")
        
        verificar_btn.click(
            fn=verificar_estado_sistema,
            outputs=estado_output
        )
    
    with gr.Tab("Ayuda"):
        gr.Markdown("""
        ## Cómo Usar el Sistema
        
        ### Paso 1: Obtener ID de Cliente
        
        - Ve a la pestaña "Recomendaciones"
        - Mira la sección "Clientes de Ejemplo" para ver IDs válidos
        - O usa cualquier ID de cliente de tu base de datos
        
        ### Paso 2: Configurar Recomendaciones
        
        - Ingresa el ID del cliente en el campo correspondiente
        - Selecciona cuántos productos deseas recomendar (1-20)
        - Por defecto se recomiendan 5 productos
        
        ### Paso 3: Obtener Resultados
        
        - Haz clic en "Obtener Recomendaciones"
        - Espera unos segundos mientras el sistema analiza todos los productos
        - Verás una tabla con las mejores recomendaciones ordenadas por probabilidad
        
        ### Interpretación de Resultados
        
        - **Probabilidad (%):** Indica qué tan probable es que el cliente compre ese producto
        - **Recomendado:** Productos con ⭐ tienen alta probabilidad (≥61.34%) y son especialmente recomendados
        - **Categoría y Marca:** Información adicional del producto para tomar mejores decisiones
        
        ### Tips
        
        - Los productos están ordenados de mayor a menor probabilidad
        - Puedes solicitar hasta 20 recomendaciones
        - Los productos marcados con ⭐ son los más recomendados
        - Si un cliente no tiene productos altamente recomendados, considera factores adicionales
        
        ### Verificar Estado
        
        - Ve a la pestaña "Estado del Sistema" para verificar que todo funciona correctamente
        - Debe mostrar que el modelo y los datos están cargados
        - Si hay algún problema, contacta al administrador del sistema
        
        ## API REST
        
        También puedes usar el backend directamente:
        
        ```bash
        # Obtener recomendaciones
        curl http://localhost:8001/recomendaciones/{cliente_id}?top_n=5
        
        # Ver estado del sistema
        curl http://localhost:8001/health
        
        # Listar clientes de ejemplo
        curl http://localhost:8001/clientes/lista?limite=10
        ```
        
        ## Contacto
        
        Desarrollado por Deep Drinkers para SodAI Drinks.
        """)
    
    gr.Markdown("""
    ---
    
    **Sistema de Recomendación v1.0.0** | Powered by XGBoost + FastAPI + Gradio
    """)


if __name__ == "__main__":
    print("Iniciando Sistema de Recomendación de SodAI Drinks...")
    print(f"Conectando a backend: {BACKEND_URL}")
    
    if verificar_backend():
        print("✓ Backend disponible")
    else:
        print("⚠ Advertencia: Backend no disponible")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )