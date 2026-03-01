# src/train.py

import os
import json
import joblib
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RAW_PATH = Path("data/raw")


def train():

    print("Iniciando entrenamiento...")

    # -------------------------------------------------
    # Cargar datos
    # -------------------------------------------------
    X = pd.read_parquet(RAW_PATH / "features.parquet")
    y = pd.read_parquet(RAW_PATH / "targets.parquet")["income"]

    print("Distribución del target:")
    print(y.value_counts())

    # -------------------------------------------------
    # Identificar columnas
    # -------------------------------------------------
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    print("Columnas categóricas:", list(categorical_cols))
    print("Columnas numéricas:", list(numeric_cols))

    # -------------------------------------------------
    # Preprocesamiento
    # -------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # -------------------------------------------------
    # Modelo
    # -------------------------------------------------
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    # -------------------------------------------------
    # Cross Validation
    # -------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "f1": make_scorer(f1_score),
        "accuracy": "accuracy",
        "roc_auc": "roc_auc"
    }

    # -------------------------------------------------
    # MLflow
    # -------------------------------------------------
    mlflow.set_experiment("adult-income")

    with mlflow.start_run():

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        pipeline.fit(X, y)

        # -------------------------
        # Métricas promedio
        # -------------------------
        metrics = {
            "f1_mean": cv_results["test_f1"].mean(),
            "accuracy_mean": cv_results["test_accuracy"].mean(),
            "roc_auc_mean": cv_results["test_roc_auc"].mean()
        }

        # -------------------------
        # Parámetros
        # -------------------------
        params = pipeline.named_steps["model"].get_params()

        # -------------------------
        # Tags
        # -------------------------
        tags = {
            "dataset": "adult",
            "model_type": "GradientBoosting",
            "cv_folds": 5
        }

        # -------------------------
        # Log MLflow
        # -------------------------
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        mlflow.set_tags(tags)
        mlflow.sklearn.log_model(pipeline, "model")

        # -------------------------------------------------
        # Guardado local estructurado
        # -------------------------------------------------
        base_path = Path("artifacts") / run_id
        (base_path / "model").mkdir(parents=True, exist_ok=True)
        (base_path / "metrics").mkdir(parents=True, exist_ok=True)
        (base_path / "params").mkdir(parents=True, exist_ok=True)
        (base_path / "tags").mkdir(parents=True, exist_ok=True)

        # Guardar métricas
        with open(base_path / "metrics" / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Guardar parámetros
        with open(base_path / "params" / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        # Guardar tags
        with open(base_path / "tags" / "tags.json", "w") as f:
            json.dump(tags, f, indent=4)

        # Guardar modelo
        joblib.dump(pipeline, base_path / "model" / "model.joblib")

        # Opcional: subir carpeta completa como artifact a MLflow
        mlflow.log_artifacts(str(base_path))

    print("Entrenamiento finalizado y todo guardado correctamente.")


if __name__ == "__main__":
    train()