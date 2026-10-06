"""Entrada principal del proyecto de churn telco."""

from pathlib import Path

from src.train import main as train_main


def main() -> None:
    print("Iniciando pipeline de modelado para Telco Customer Churn...")
    train_main()


if __name__ == "__main__":
    main()
