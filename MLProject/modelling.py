"""
modelling.py
Final CI version for Kriteria 3 Advanced.

Fungsi utama:
1. Melatih model Random Forest menggunakan dataset hasil preprocessing.
2. Melakukan manual logging ke MLflow.
3. Menyimpan artifact tambahan.
4. Mengekspor model ke folder model_export agar menghasilkan:
   - MLmodel
   - conda.yaml
   - model.pkl
   - python_env.yaml
   - requirements.txt
5. Siap digunakan oleh GitHub Actions dan mlflow models build-docker.

Struktur folder yang diasumsikan:
MLProject/
├── modelling.py
├── requirements.txt
├── titanic_preprocessing/
│   ├── train_preprocessed.csv
│   └── test_preprocessed.csv
└── .github/workflows/ci-cd.yml

Cara run lokal untuk CI mode:
PowerShell:
$env:MLFLOW_TRACKING_MODE="ci"
python modelling.py

Git Bash:
MLFLOW_TRACKING_MODE=ci python modelling.py
"""

from pathlib import Path
import os
import json
import shutil
import warnings
import logging

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.utils import estimator_html_repr


# =========================
# CONFIGURATION
# =========================
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "titanic_preprocessing"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "model"
MODEL_EXPORT_DIR = BASE_DIR / "model_export"

TRAIN_PATH = DATA_DIR / "train_preprocessed.csv"
TEST_PATH = DATA_DIR / "test_preprocessed.csv"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# MLFLOW SETUP
# =========================
def setup_mlflow():
    """
    Tracking mode:
    - ci      : file-based MLflow tracking for GitHub Actions.
    - local   : localhost MLflow UI, requires mlflow ui running on 127.0.0.1:5000.
    - dagshub : online MLflow tracking with DagsHub.
    """
    tracking_mode = os.getenv("MLFLOW_TRACKING_MODE", "ci").lower()

    if tracking_mode == "dagshub":
        import dagshub

        dagshub.init(
            repo_owner="entiondhi",
            repo_name="Eksperimen_SML_Keshin_Ryandy",
            mlflow=True
        )
        logging.info("MLflow tracking connected to DagsHub.")

    elif tracking_mode == "local":
        mlflow.set_tracking_uri("http://127.0.0.1:5000/")
        logging.info("MLflow tracking URI set to http://127.0.0.1:5000/")

    else:
        # Mode utama untuk GitHub Actions CI.
        # Tidak membutuhkan server MLflow localhost.
        mlflow.set_tracking_uri("file:./mlruns")
        logging.info("MLflow tracking URI set to local file store: file:./mlruns")

    mlflow.set_experiment("Titanic Random Forest CI")


# =========================
# DATA FUNCTIONS
# =========================
def load_data():
    """Load preprocessed train and test dataset."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"File train tidak ditemukan: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"File test tidak ditemukan: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    logging.info("Train shape: %s", train_df.shape)
    logging.info("Test shape : %s", test_df.shape)

    return train_df, test_df


def prepare_data(train_df, test_df):
    """Split features and target, then align test columns with train columns."""
    if "Survived" not in train_df.columns:
        raise ValueError("Kolom target 'Survived' tidak ditemukan pada train_preprocessed.csv")

    X = train_df.drop(columns=["Survived"])
    y = train_df["Survived"]

    X_test = test_df.copy()
    passenger_id = None

    if "PassengerId" in X_test.columns:
        passenger_id = X_test["PassengerId"]
        X_test = X_test.drop(columns=["PassengerId"])

    extra_cols = set(X_test.columns) - set(X.columns)
    missing_cols = set(X.columns) - set(X_test.columns)

    if extra_cols:
        logging.warning("Kolom ekstra pada test dataset akan dihapus: %s", extra_cols)
        X_test = X_test.drop(columns=list(extra_cols))

    if missing_cols:
        raise ValueError(f"Kolom berikut tidak ditemukan di test dataset: {missing_cols}")

    X_test = X_test[X.columns]

    return X, y, X_test, passenger_id


# =========================
# MODEL FUNCTIONS
# =========================
def evaluate_model(model, X_valid, y_valid):
    """Evaluate trained model on validation data."""
    y_pred = model.predict(X_valid)

    metrics = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred, zero_division=0),
        "recall": recall_score(y_valid, y_pred, zero_division=0),
        "f1_score": f1_score(y_valid, y_pred, zero_division=0),
    }

    return y_pred, metrics


def save_manual_artifacts(model, X_valid, y_valid, y_pred, feature_names):
    """
    Save additional artifacts for manual MLflow logging.
    These artifacts satisfy the advanced requirement beyond default model logging.
    """
    artifact_paths = []

    # 1. estimator.html
    estimator_path = ARTIFACT_DIR / "estimator.html"
    with open(estimator_path, "w", encoding="utf-8") as file:
        file.write(estimator_html_repr(model))
    artifact_paths.append(estimator_path)

    # 2. metric_info.json
    metric_info_path = ARTIFACT_DIR / "metric_info.json"
    metric_payload = {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred, zero_division=0),
        "recall": recall_score(y_valid, y_pred, zero_division=0),
        "f1_score": f1_score(y_valid, y_pred, zero_division=0),
    }

    with open(metric_info_path, "w", encoding="utf-8") as file:
        json.dump(metric_payload, file, indent=4)
    artifact_paths.append(metric_info_path)

    # 3. training_confusion_matrix.png
    confusion_matrix_path = ARTIFACT_DIR / "training_confusion_matrix.png"
    ConfusionMatrixDisplay.from_predictions(y_valid, y_pred)
    plt.title("Training Confusion Matrix - Random Forest")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()
    artifact_paths.append(confusion_matrix_path)

    # 4. classification_report.txt
    report_path = ARTIFACT_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(classification_report(y_valid, y_pred, zero_division=0))
    artifact_paths.append(report_path)

    # 5. feature_importance.csv and feature_importance.png
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feature_importance_csv_path = ARTIFACT_DIR / "feature_importance.csv"
    feature_importance.to_csv(feature_importance_csv_path, index=False)
    artifact_paths.append(feature_importance_csv_path)

    feature_importance_png_path = ARTIFACT_DIR / "feature_importance.png"
    plt.figure(figsize=(8, 5))
    plt.barh(feature_importance["feature"], feature_importance["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - Random Forest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(feature_importance_png_path, dpi=150)
    plt.close()
    artifact_paths.append(feature_importance_png_path)

    return artifact_paths


def export_mlflow_model(model):
    """
    Export model as an MLflow model directory.
    This folder is required by mlflow models build-docker.
    """
    if MODEL_EXPORT_DIR.exists():
        shutil.rmtree(MODEL_EXPORT_DIR)

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(MODEL_EXPORT_DIR)
    )

    required_files = [
        MODEL_EXPORT_DIR / "MLmodel",
        MODEL_EXPORT_DIR / "conda.yaml",
        MODEL_EXPORT_DIR / "model.pkl",
        MODEL_EXPORT_DIR / "python_env.yaml",
        MODEL_EXPORT_DIR / "requirements.txt",
    ]

    missing_files = [str(file) for file in required_files if not file.exists()]

    if missing_files:
        raise FileNotFoundError(f"Model export tidak lengkap. File hilang: {missing_files}")

    logging.info("MLflow model exported to: %s", MODEL_EXPORT_DIR)
    logging.info("Exported files: %s", [file.name for file in required_files])


# =========================
# MAIN PIPELINE
# =========================
def main():
    setup_mlflow()

    train_df, test_df = load_data()
    X, y, X_test, passenger_id = prepare_data(train_df, test_df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    params = {
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    }

    with mlflow.start_run(run_name="ci_random_forest_model"):
        # Manual parameter logging
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Manual metric logging
        train_score = model.score(X_train, y_train)
        y_pred_valid, metrics = evaluate_model(model, X_valid, y_valid)
        metrics["train_score"] = train_score

        mlflow.log_metrics(metrics)

        # Manual artifact logging
        artifact_paths = save_manual_artifacts(
            model=model,
            X_valid=X_valid,
            y_valid=y_valid,
            y_pred=y_pred_valid,
            feature_names=X.columns
        )

        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path))

        # Save local model with joblib
        model_path = MODEL_DIR / "random_forest_ci.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        # Log MLflow model artifact
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        # Export MLflow model folder for Docker build
        export_mlflow_model(model)
        mlflow.log_artifacts(str(MODEL_EXPORT_DIR), artifact_path="model_export")

        # Save test prediction artifact
        test_prediction = model.predict(X_test)

        if passenger_id is not None:
            submission_df = pd.DataFrame({
                "PassengerId": passenger_id,
                "Survived": test_prediction
            })
        else:
            submission_df = pd.DataFrame({
                "Survived": test_prediction
            })

        submission_path = ARTIFACT_DIR / "submission_random_forest_ci.csv"
        submission_df.to_csv(submission_path, index=False)
        mlflow.log_artifact(str(submission_path))

        logging.info("Train score: %.4f", train_score)
        logging.info("Validation metrics: %s", metrics)
        logging.info("Manual artifacts logged: %s", [path.name for path in artifact_paths])

    logging.info("CI modelling pipeline completed successfully.")


if __name__ == "__main__":
    main()
