import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))

print("Python import path:")
print(PROJECT_ROOT)

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    brier_score_loss,
)

from database import Database

OUTPUT_PATH = Path("outputs/tables/table3_performance_metrics.csv")


def load_predictions():

    db = Database()

    try:
        db.connect()

        query = """
        SELECT
            patient_id,
            prediction_time,
            risk_probability
        FROM patient_risk_predictions
        WHERE model_version = 'SepsiGuard_TCN_v1'
        """

        df = db.fetch_dataframe(query)

    finally:
        db.disconnect()

    return df


def load_ground_truth():

    db = Database()

    try:
        db.connect()

        query = """
        SELECT
            tw.patient_id,
            tw.prediction_iculos,
            tw.y_12,
            ps.split
        FROM trajectory_windows tw
        JOIN patient_splits ps
        ON tw.patient_id = ps.patient_id
        WHERE ps.split = 'test'
        """

        df = db.fetch_dataframe(query)

    finally:
        db.disconnect()

    return df


def main():

    print("Loading predictions...")
    predictions = load_predictions()

    print("Prediction rows:", len(predictions))

    print("Loading labels...")
    labels = load_ground_truth()

    print("Label rows:", len(labels))

    print("Merging datasets...")

    merged = predictions.merge(
        labels,
        left_on=["patient_id", "prediction_time"],
        right_on=["patient_id", "prediction_iculos"],
        how="inner",
    )

    print("Matched rows:", len(merged))

    y_true = merged["y_12"].values

    y_prob = merged["risk_probability"].values

    y_pred = (y_prob >= 0.5).astype(int)

    print("\nEvaluation")
    print("================")

    metrics = {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Brier_score": brier_score_loss(y_true, y_prob),
    }

    cm = confusion_matrix(y_true, y_pred)

    print(metrics)

    print("\nConfusion Matrix")
    print(cm)

    output = pd.DataFrame([metrics])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
