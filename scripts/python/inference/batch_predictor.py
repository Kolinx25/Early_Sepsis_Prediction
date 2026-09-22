import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


import torch
import numpy as np

from models.tcn import SepsisTCN
from database import Database


CHECKPOINT = "models/best_tcn_y12.pt"
MODEL_VERSION = "SepsiGuard_TCN_v1"

BATCH_SIZE = 4096

# Testing only
# Set to None for full inference
DEBUG_LIMIT = None


def load_model(device):

    model = SepsisTCN(input_channels=69)

    checkpoint = torch.load(CHECKPOINT, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("✓ TCN model loaded")

    return model


def risk_category(prob):

    if prob >= 0.75:
        return "CRITICAL"

    elif prob >= 0.40:
        return "WATCH"

    else:
        return "STABLE"


def prepare_tensor(sequence_list):
    """
    PostgreSQL JSONB
        ↓
    numpy float32
        ↓
    torch tensor

    Expected:
    (batch,12,69)
    """

    array = np.asarray(sequence_list, dtype=np.float32)

    # Safety against invalid numerical values
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.from_numpy(array)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    model = load_model(device)

    db = Database()

    try:
        db.connect()

        print("Loading trajectories...")

        query = """
        SELECT
            patient_id,
            prediction_iculos,
            x_sequence
        FROM trajectory_windows
        ORDER BY
            patient_id,
            prediction_iculos
        """

        if DEBUG_LIMIT:
            query += f"""
            LIMIT {DEBUG_LIMIT}
            """

        df = db.fetch_dataframe(query)

        total = len(df)

        print(f"Trajectory windows: {total}")

        predictions = []

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)

            batch = df.iloc[start:end]

            X = prepare_tensor(batch["x_sequence"].tolist())

            X = X.to(device)

            with torch.no_grad():
                logits = model(X)

                probabilities = torch.sigmoid(logits)

                probabilities = probabilities.cpu().numpy()

            # Final safety check
            if np.isnan(probabilities).any():
                raise RuntimeError("NaN probabilities detected")

            probabilities = np.clip(probabilities, 0.0, 1.0)

            for i, prob in enumerate(probabilities):
                row = batch.iloc[i]

                predictions.append(
                    (
                        int(row.patient_id),
                        int(row.prediction_iculos),
                        float(prob),
                        risk_category(float(prob)),
                        MODEL_VERSION,
                    )
                )

            progress = (end / total) * 100

            print(f"Processed {end}/{total} ({progress:.1f}%)")

        print("Preparing database insert...")

        #
        # Remove previous predictions
        # from same model version
        #
        cleanup_query = f"""
        DELETE FROM patient_risk_predictions
        WHERE model_version = '{MODEL_VERSION}'
        """

        db.fetch_dataframe(cleanup_query)

        print("Old predictions removed")

        print("Old predictions removed")

        insert_query = """

        INSERT INTO patient_risk_predictions
        (
            patient_id,
            prediction_time,
            risk_probability,
            risk_category,
            model_version
        )

        VALUES %s

        """

        db.bulk_insert(insert_query, predictions)

        print("✓ Predictions inserted:", len(predictions))

    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
