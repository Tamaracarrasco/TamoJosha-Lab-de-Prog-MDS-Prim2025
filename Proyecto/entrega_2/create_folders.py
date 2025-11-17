"""
Script para crear la estructura de carpetas del proyecto.
"""

import os
from pathlib import Path
import logging

# Configurar logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_project_structure(base_path: str = "."):
    """
    Crea la estructura de carpetas del proyecto
    
    Parámetros:
    ----------------
    base_path: Ruta base donde se creará la estructura (por defecto directorio actual)
    """
    
    # Definir estructura de carpetas
    folders = [
        # Carpetas principales de datos
        "data/baseline",
        "data/run/new_transactions",
        "data/run/processed",
        "data/run/predictions",
        "data/run/drift_reports",
        
        # Carpetas de MLflow
        "mlflow/artifacts",
        "mlflow/mlruns",
        
        # Carpetas de Airflow (ya existen pero por si acaso)
        "airflow/dags/tasks",
        "airflow/logs",
        "airflow/plugins",
        
        # Carpetas adicionales que podrían servinos
        "models/saved_models",
        "models/checkpoints",
        "reports/shap_values",
        "reports/model_performance",
        "reports/drift_analysis",
    ]
    
    base_path = Path(base_path)
    created_folders = []
    existing_folders = []
    
    # Crear cada carpeta
    for folder in folders:

        folder_path = base_path / folder
        
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            if folder_path.exists():
                created_folders.append(str(folder_path))
                logger.info(f"Carpeta creada/verificada: {folder_path}")

            else:
                existing_folders.append(str(folder_path))
        
        except Exception as e:
            logger.error(f"Error al crear carpeta {folder_path}: {str(e)}")
    
    # Crear archivos .gitkeep para mantener carpetas vacías en git
    gitkeep_folders = [
        "data/run/new_transactions",
        "data/run/processed",
        "data/run/predictions",
        "mlflow/artifacts",
        "reports/shap_values",
    ]
    
    for folder in gitkeep_folders:
        gitkeep_path = base_path / folder / ".gitkeep"
        try:
            gitkeep_path.touch(exist_ok=True)
            logger.info(f"Archivo .gitkeep creado en: {folder}")
        except Exception as e:
            logger.error(f"Error al crear .gitkeep en {folder}: {str(e)}")
    
    # Resumen
    logger.info("\n" + "="*60)
    logger.info("RESUMEN DE CREACIÓN DE ESTRUCTURA")
    logger.info("="*60)
    logger.info(f"Total de carpetas procesadas: {len(folders)}")
    logger.info(f"Carpetas creadas/verificadas: {len(created_folders)}")
    logger.info("="*60)
    
    return created_folders


def verify_baseline_data(base_path: str = "."):
    """
    Verifica que existan los archivos baseline necesarios
    
    Parámetros:
    ----------------
    base_path: Ruta base del proyecto
    """
    base_path = Path(base_path)
    baseline_path = base_path / "data" / "baseline"
    
    required_files = [
        "clientes.parquet",
        "productos.parquet",
        "df_parte_1.parquet",
        "transacciones.parquet"
    ]
    
    logger.info("\n" + "="*60)
    logger.info("VERIFICACIÓN DE ARCHIVOS BASELINE")
    logger.info("="*60)
    
    missing_files = []
    existing_files = []
    
    for file in required_files:
        file_path = baseline_path / file
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            existing_files.append(file)
            logger.info(f" {file} - {file_size:.2f} MB")
        else:
            missing_files.append(file)
            logger.warning(f"{file} - NO ENCONTRADO")
    
    logger.info("="*60)
    
    if missing_files:
        logger.warning(f"\n Archivos faltantes: {', '.join(missing_files)}")
        logger.warning("Asegúrate de copiar estos archivos a data/baseline/")
    else:
        logger.info("\n Todos los archivos baseline están presentes")
    
    return existing_files, missing_files


if __name__ == "__main__":

    logger.info("Iniciando creación de estructura de carpetas...")
    
    # Crear estructura de carpetas
    created = create_project_structure()
    
    # Verificar archivos baseline
    existing, missing = verify_baseline_data()
    
    logger.info("\nProceso completado exitosamente")