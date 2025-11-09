# acá se define la función para crear las carpetas.
from pathlib import Path
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


DEFAULT_DIRS = [
"./data/baseline",
"./data/run",
"./data/run/raw",
"./data/run/processed",
"./data/run/models",
"./data/run/outputs",
"./airflow/logs",
"./airflow/plugins",
"./mlflow",
]




def create_base_folders(base_dirs: list = None):
    """Crea las carpetas base necesarias si no existen."""
    dirs = base_dirs or DEFAULT_DIRS
    for d in dirs:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured folder exists: {p.resolve()}")




def create_run_folder(run_date: str = None):
    """Crea la carpeta de ejecución para la fecha dada (YYYY-MM-DD) y subcarpetas.


    Retorna la ruta Path del run_dir.
    """
    if run_date is None:
        run_date = datetime.utcnow().strftime("%Y-%m-%d")
    run_dir = Path("./data/run") / run_date
    for sub in ["raw", "processed", "models", "outputs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run folder: {run_dir}")
    return run_dir

if __name__ == "__main__":
    create_base_folders()
    create_run_folder()