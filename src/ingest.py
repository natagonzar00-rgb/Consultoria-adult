# src/ingest.py

import pandas as pd
from pathlib import Path
import json


RAW_PATH = Path("data/raw")
OUTPUT_PATH = Path("data/raw")  # seguimos guardando en raw, solo cambiamos formato
ARTIFACTS_PATH = Path("artifacts")

ARTIFACTS_PATH.mkdir(exist_ok=True, parents=True)


def ingest():
    print("Iniciando ingesta de datos...")

    features_file = RAW_PATH / "features.csv"
    targets_file = RAW_PATH / "targets.csv"

    if not features_file.exists() or not targets_file.exists():
        raise FileNotFoundError("No se encontraron los CSV en data/raw/")

    # -------------------------
    # Leer datos
    # -------------------------
    X = pd.read_csv(features_file)
    y = pd.read_csv(targets_file)

    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")

    # -------------------------
    # Validación mínima estructural
    # -------------------------
    if len(X) != len(y):
        raise ValueError("Features y Targets tienen diferente número de filas")

    if X.empty:
        raise ValueError("Features está vacío")

    if y.empty:
        raise ValueError("Targets está vacío")

    # -------------------------
    # Guardar en formato PARQUET (artefacto versionable)
    # -------------------------
    X.to_parquet(OUTPUT_PATH / "features.parquet", index=False)
    y.to_parquet(OUTPUT_PATH / "targets.parquet", index=False)

    print("Archivos convertidos a Parquet.")

    # -------------------------
    # Metadata (esto es clave en MLOps)
    # -------------------------
    report = {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "columns": list(X.columns),
        "target_name": list(y.columns),
        "missing_values_features": X.isnull().sum().to_dict(),
        "missing_values_target": y.isnull().sum().to_dict()
    }

    with open(ARTIFACTS_PATH / "ingest_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Reporte de ingesta generado.")


if __name__ == "__main__":
    ingest()