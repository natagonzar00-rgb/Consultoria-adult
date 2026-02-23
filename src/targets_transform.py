# src/targets_transform.py

import pandas as pd
import joblib


def transform_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Transforma el target a formato binario (0,1).
    Funciona si viene como texto ('<=50K', '>50K')
    o si ya viene como numérico.
    """

    # Convertir todo a string y limpiar espacios
    df[target_col] = df[target_col].astype(str).str.strip()

    # Mapeo robusto
    mapping = {
        "<=50K": 0,
        ">50K": 1,
        "<=50K.": 0,
        ">50K.": 1,
        "0": 0,
        "1": 1
    }

    df[target_col] = df[target_col].map(mapping)

    # Validación
    if df[target_col].isna().any():
        raise ValueError(
            f"Valores no reconocidos en el target: {df[target_col].unique()}"
        )

    return df


if __name__ == "__main__":

    print("🔄 Cargando targets...")

    # Leer archivo ignorando índice guardado
    df = pd.read_csv("data/raw/targets.csv")

    # Eliminar columna índice si existe
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Confirmar que existe la columna income
    if "income" not in df.columns:
        raise ValueError(
            f"No se encontró la columna 'income'. Columnas actuales: {df.columns}"
        )

    TARGET_COL = "income"

    df_transformed = transform_target(df, TARGET_COL)

    # Guardar versión procesada SIN índice
    df_transformed.to_csv(
        "data/raw/targets_processed.csv",
        index=False
    )

    # Guardar artefacto serializado
    joblib.dump(
        df_transformed,
        "artifacts/target.joblib"
    )

    print("✔ Target transformado correctamente")
    print("\nDistribución del target:")
    print(df_transformed[TARGET_COL].value_counts())