import numpy as np
from psycopg2.extras import execute_values

from database import Database


RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def create_patient_splits(patient_ids):

    patient_ids = np.array(patient_ids)

    rng = np.random.default_rng(RANDOM_SEED)

    shuffled_ids = rng.permutation(patient_ids)

    total_patients = len(shuffled_ids)

    train_end = int(total_patients * TRAIN_RATIO)

    validation_end = train_end + int(total_patients * VALIDATION_RATIO)

    train_ids = shuffled_ids[:train_end]

    validation_ids = shuffled_ids[train_end:validation_end]

    test_ids = shuffled_ids[validation_end:]

    split_records = []

    for patient_id in train_ids:
        split_records.append((int(patient_id), "train"))

    for patient_id in validation_ids:
        split_records.append((int(patient_id), "validation"))

    for patient_id in test_ids:
        split_records.append((int(patient_id), "test"))

    return split_records


def save_patient_splits(db, split_records):

    sql = """
    INSERT INTO patient_splits
    (
        patient_id,
        split
    )

    VALUES %s

    ON CONFLICT (patient_id)
    DO UPDATE SET

        split = EXCLUDED.split;
    """

    execute_values(db.cursor, sql, split_records, page_size=5000)

    db.connection.commit()


def main():

    db = Database()

    try:
        db.connect()

        patient_df = db.fetch_dataframe(
            """
            SELECT DISTINCT patient_id
            FROM trajectory_windows
            ORDER BY patient_id;
            """
        )

        patient_ids = patient_df["patient_id"].tolist()

        print("Patients available for splitting:", len(patient_ids))

        split_records = create_patient_splits(patient_ids)

        save_patient_splits(db, split_records)

        print("Patient splits created:", len(split_records))

    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
