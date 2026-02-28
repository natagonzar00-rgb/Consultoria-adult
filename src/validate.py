# src/validate.py

import pandas as pd
import pandera as pa
from pandera import Column, Check
from pathlib import Path
import json


RAW_PATH = Path("data/raw")
ARTIFACTS_PATH = Path("artifacts")

ARTIFACTS_PATH.mkdir(exist_ok=True, parents=True)


def validate():

    print("Validando datos...")

    # -------------------------------------------------
    # Cargar datos
    # -------------------------------------------------
    X = pd.read_parquet(RAW_PATH / "features.parquet")
    y = pd.read_parquet(RAW_PATH / "targets.parquet")

    # -------------------------------------------------
    # Eliminar columnas basura tipo 'Unnamed'
    # -------------------------------------------------
    X = X.loc[:, ~X.columns.str.contains("^Unnamed")]
    y = y.loc[:, ~y.columns.str.contains("^Unnamed")]

    print("Columnas del target:", y.columns.tolist())

    # -------------------------------------------------
    # Identificar columna target
    # -------------------------------------------------
    if "income" in y.columns:
        target_col = "income"
    else:
        target_col = y.columns[-1]

    print(f"Usando columna target: {target_col}")

    # -------------------------------------------------
    # Limpieza FUERTE del target (Adult suele venir sucio)
    # -------------------------------------------------
    y[target_col] = (
        y[target_col]
        .astype(str)
        .str.replace(r"\.", "", regex=True)   # quitar puntos
        .str.replace(r"\s+", "", regex=True)  # quitar todos los espacios
        .str.replace(r"\r|\n", "", regex=True)  # quitar saltos de línea
    )

    print("Clases únicas después de limpiar:", y[target_col].unique())

    # -------------------------------------------------
    # Verificación binaria estricta
    # -------------------------------------------------
    unique_classes = set(y[target_col].unique())
    expected_classes = {"<=50K", ">50K"}

    if not unique_classes.issubset(expected_classes):
        raise ValueError(
            f"Se detectaron clases inesperadas: {unique_classes}"
        )

    # -------------------------------------------------
    # Validación estructural básica
    # -------------------------------------------------
    if len(X) != len(y):
        raise ValueError("Features y Targets tienen diferente número de filas")

    # -------------------------------------------------
    # Schema Adult (features)
    # -------------------------------------------------
    schema = pa.DataFrameSchema({
        "age": Column(int, Check.in_range(17, 90)),
        "workclass": Column(str, nullable=True),
        "fnlwgt": Column(int),
        "education": Column(str),
        "education-num": Column(int, Check.in_range(1, 16)),
        "marital-status": Column(str),
        "occupation": Column(str, nullable=True),
        "relationship": Column(str),
        "race": Column(str),
        "sex": Column(str),
        "capital-gain": Column(int, Check.ge(0)),
        "capital-loss": Column(int, Check.ge(0)),
        "hours-per-week": Column(int, Check.in_range(1, 100)),
        "native-country": Column(str, nullable=True)
    })

    schema.validate(X, lazy=True)

    print("Schema válido.")

    # -------------------------------------------------
    # Distribución del target (YA LIMPIO)
    # -------------------------------------------------
    target_distribution = (
        y[target_col]
        .value_counts()
        .to_dict()
    )

    # -------------------------------------------------
    # Construcción del reporte serializable
    # -------------------------------------------------
    report = {
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "duplicates": int(X.duplicated().sum()),
        "null_percentage": {
            str(k): float(v)
            for k, v in (X.isnull().mean() * 100).items()
        },
        "target_distribution": {
            str(k): int(v)
            for k, v in target_distribution.items()
        }
    }

    with open(ARTIFACTS_PATH / "validation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Reporte de validación generado correctamente.")
    print("Distribución final:", target_distribution)


if __name__ == "__main__":
    validate()