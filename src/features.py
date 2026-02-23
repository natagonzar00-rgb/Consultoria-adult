# src/features.py

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer
)

# ==============================
# DEFINICIÓN DE COLUMNAS
# ==============================

NUM_COLS = [
    "age",
    "hours-per-week"
]

LOG_COLS = [
    "capital-gain",
    "capital-loss"
]

CAT_COLS = [
    "workclass",
    "occupation",
    "education",
    "marital-status",
    "relationship",
    "race",
    "sex",
    "native-country"
]

# ==============================
# TRANSFORMACIONES
# ==============================

def log_transform(x):
    return np.log1p(x)


def build_preprocessor():

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    log_pipeline = Pipeline([
        ("log", FunctionTransformer(log_transform)),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUM_COLS),
        ("log", log_pipeline, LOG_COLS),
        ("cat", categorical_pipeline, CAT_COLS)
    ])

    return preprocessor


# ==============================
# EJECUCIÓN
# ==============================

if __name__ == "__main__":

    df = pd.read_csv("data/raw/features.csv")

    # Eliminar columna irrelevante
    if "fnlwgt" in df.columns:
        df = df.drop(columns=["fnlwgt"])

    preprocessor = build_preprocessor()

    X_processed = preprocessor.fit_transform(df)

    # Guardar artefacto
    joblib.dump(preprocessor, "artifacts/preprocessor.joblib")

    print("✔ Preprocesador entrenado y guardado en artifacts/preprocessor.joblib")
    print(f"Shape después de OneHotEncoding: {X_processed.shape}")
    