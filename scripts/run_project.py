"""Ejecuta el flujo principal del proyecto."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.split_data import main as split_main
from src.train import main as train_main


if __name__ == "__main__":
    print("Generando partición train/test...")
    split_main()
    print("\nEntrenando modelo...")
    train_main()
