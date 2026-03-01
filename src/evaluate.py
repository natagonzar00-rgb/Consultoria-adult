# src/evaluate.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


RAW_PATH = Path("data/raw")
MODELS_PATH = Path("models")


def get_latest_model_path():
    runs = sorted(MODELS_PATH.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        raise ValueError("No hay modelos en la carpeta models/")
    return runs[0]


def evaluate():

    print("Iniciando evaluación...")

    # -------------------------------------------------
    # Obtener último modelo
    # -------------------------------------------------
    model_dir = get_latest_model_path()
    run_id = model_dir.name
    print(f"Evaluando modelo: {run_id}")

    model_path = model_dir / "model" / "model.joblib"
    model = joblib.load(model_path)

    # -------------------------------------------------
    # Cargar datos
    # -------------------------------------------------
    X = pd.read_parquet(RAW_PATH / "features.parquet")
    y = pd.read_parquet(RAW_PATH / "targets.parquet")["income"]

    # -------------------------------------------------
    # Predicciones
    # -------------------------------------------------
    y_pred = model.predict(X)

    # -------------------------------------------------
    # Matriz de confusión
    # -------------------------------------------------
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")

    eval_path = model_dir / "evaluation"
    eval_path.mkdir(exist_ok=True)

    plt.savefig(eval_path / "confusion.png")
    plt.close()

    # -------------------------------------------------
    # Classification report HTML
    # -------------------------------------------------
    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_html = report_df.to_html()

    with open(eval_path / "report.html", "w") as f:
        f.write("<h1>Reporte de Clasificación</h1>")
        f.write(report_html)

    print("Evaluación completada.")
    print(f"Archivos guardados en: {eval_path}")


if __name__ == "__main__":
    evaluate()