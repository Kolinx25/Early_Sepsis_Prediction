import math
import json

from psycopg2.extras import execute_values

from feature_config import (
    CLINICAL_FEATURES,
    MISSINGNESS_FEATURES,
    TEMPORAL_FEATURES,
    TARGET_COLUMN,
    LOOKBACK,
    HORIZONS,
)


FEATURE_COLUMNS = CLINICAL_FEATURES + MISSINGNESS_FEATURES + TEMPORAL_FEATURES


def clean_json_values(sequence):
    """
    Convert NaN values to None for PostgreSQL JSONB compatibility.

    Missing clinical measurements are preserved as null values.
    Missingness indicators remain separate features.
    """

    cleaned = []

    for row in sequence:
        cleaned_row = []

        for value in row:
            if isinstance(value, float) and math.isnan(value):
                cleaned_row.append(None)

            else:
                cleaned_row.append(value)

        cleaned.append(cleaned_row)

    return cleaned


def create_trajectory_windows(feature_table):
    """
    Convert hourly feature table into database-stored trajectory windows.

    Each generated sample contains:

        patient_id
        prediction_iculos
        lookback_hours
        y_6
        y_12
        X_sequence (12 x 69)
    """

    window_records = []

    # Ensure chronological patient trajectories

    feature_table = feature_table.sort_values(["patient_id", "iculos"]).reset_index(
        drop=True
    )

    for patient_id, patient_data in feature_table.groupby("patient_id"):
        patient_data = patient_data.reset_index(drop=True)

        values = patient_data[FEATURE_COLUMNS].values

        labels = patient_data[TARGET_COLUMN].values

        for t in range(LOOKBACK - 1, len(patient_data)):
            # 12-hour historical window

            window = values[t - LOOKBACK + 1 : t + 1]

            # Future prediction horizons

            future_6 = labels[t + 1 : t + 1 + HORIZONS["secondary"]]

            future_12 = labels[t + 1 : t + 1 + HORIZONS["primary"]]

            # Require complete future horizon

            if len(future_6) < HORIZONS["secondary"]:
                continue

            if len(future_12) < HORIZONS["primary"]:
                continue

            window_records.append(
                (
                    int(patient_id),
                    int(patient_data.loc[t, "iculos"]),
                    LOOKBACK,
                    int(future_6.max()),
                    int(future_12.max()),
                    json.dumps(clean_json_values(window.tolist())),
                )
            )

    print("Trajectory windows created:", len(window_records))

    return window_records


def save_windows(db, window_records):
    """
    Bulk insert trajectory windows into PostgreSQL.
    """

    sql = """
    INSERT INTO trajectory_windows
    (
        patient_id,
        prediction_iculos,
        lookback_hours,
        y_6,
        y_12,
        X_sequence
    )

    VALUES %s;
    """

    execute_values(db.cursor, sql, window_records, page_size=5000)

    db.connection.commit()

    print(f"Inserted {len(window_records)} trajectory windows.")
