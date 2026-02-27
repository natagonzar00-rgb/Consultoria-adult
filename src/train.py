import pandas as pd
from pathlib import Path
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import subprocess
import matplotlib.pyplot as plt
import os

# -------------------- Utilidades --------------------
def get_git_commit():
    """Obtiene el hash del commit actual o 'unknown' si falla."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit
    except Exception:
        return "unknown"

def load_data():
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'

    # --- Cargar features ---
    X = pd.read_parquet(data_dir / 'features.parquet')

    # --- Cargar targets ---
    y_df = pd.read_csv(data_dir / 'targets.csv', index_col=0)
    if 'income' not in y_df.columns:
        raise ValueError("No se encontró la columna 'income' en targets.csv")
    y = y_df['income'].copy()

    # --- Limpiar filas con NaN en y ---
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # --- Manejar NaN en X rellenando con median ---
    for col in X.select_dtypes(include='number').columns:
        X[col] = X[col].fillna(X[col].median())

    # --- Codificar y ---
    le = LabelEncoder()
    y = le.fit_transform(y)

    # --- Codificar variables categóricas ---
    cat_cols = X.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        print(f"Columnas categóricas a codificar: {list(cat_cols)}")
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y

# -------------------- Entrenamiento --------------------
def train(X_train, y_train, params: dict):
    mlflow.set_experiment('adult-income')
    with mlflow.start_run():
        clf = GradientBoostingClassifier(**params)

        # --- Validación cruzada ---
        scores = cross_val_score(
            clf, X_train, y_train,
            cv=5,
            scoring='f1_macro',
            error_score='raise'
        )

        # --- Entrenamiento final ---
        clf.fit(X_train, y_train)

        # --- Logging MLflow ---
        mlflow.log_params(params)
        mlflow.log_metric('f1_cv', scores.mean())
        mlflow.sklearn.log_model(clf, 'model')

        # --- Tags / metadata ---
        mlflow.set_tag('dataset_version', 'v1.0')
        mlflow.set_tag('git_commit', get_git_commit())
        mlflow.set_tag('run_timestamp', datetime.utcnow().isoformat())

        # --- Feature importance opcional ---
        fi = pd.Series(clf.feature_importances_, index=X_train.columns)
        fi = fi.sort_values(ascending=False)
        plt.figure(figsize=(8,6))
        fi.head(20).plot(kind='barh')
        plt.title('Top 20 Feature Importances')
        plt.gca().invert_yaxis()
        os.makedirs("artifacts", exist_ok=True)
        fi_path = "artifacts/feature_importance.png"
        plt.savefig(fi_path)
        mlflow.log_artifact(fi_path)
        plt.close()

        print(f"Run finished with mean F1 score: {scores.mean():.4f}")

# -------------------- Main --------------------
if __name__ == '__main__':
    X_train, y_train = load_data()

    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42,
    }

    train(X_train, y_train, params)