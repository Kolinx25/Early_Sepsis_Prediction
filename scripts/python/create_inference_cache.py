import os
import numpy as np

from database import Database


OUTPUT = "data/cache"

BATCH_SIZE = 4096


def main():

    os.makedirs(OUTPUT, exist_ok=True)

    db = Database()

    db.connect()

    print("Counting trajectories...")

    count_query = """
    SELECT COUNT(*)
    FROM trajectory_windows;
    """

    total = int(db.fetch_dataframe(count_query).iloc[0, 0])

    print("Total rows:", total)

    X_path = f"{OUTPUT}/inference_X.npy"
    patient_path = f"{OUTPUT}/inference_patient_id.npy"
    iculos_path = f"{OUTPUT}/inference_iculos.npy"

    # Create disk-backed arrays
    X_mem = np.lib.format.open_memmap(
        X_path, mode="w+", dtype=np.float32, shape=(total, 12, 69)
    )

    patient_mem = np.lib.format.open_memmap(
        patient_path, mode="w+", dtype=np.int64, shape=(total,)
    )

    iculos_mem = np.lib.format.open_memmap(
        iculos_path, mode="w+", dtype=np.int64, shape=(total,)
    )

    offset = 0

    print("Loading trajectories in batches...")

    for start in range(0, total, BATCH_SIZE):
        query = f"""
        SELECT
            patient_id,
            prediction_iculos,
            x_sequence

        FROM trajectory_windows

        ORDER BY
            patient_id,
            prediction_iculos

        OFFSET {start}
        LIMIT {BATCH_SIZE};
        """

        df = db.fetch_dataframe(query)

        batch_X = np.array(df["x_sequence"].tolist(), dtype=np.float32)

        batch_X = np.nan_to_num(batch_X, nan=0.0, posinf=0.0, neginf=0.0)

        end = offset + len(df)

        X_mem[offset:end] = batch_X

        patient_mem[offset:end] = df["patient_id"].values

        iculos_mem[offset:end] = df["prediction_iculos"].values

        offset = end

        print(f"Processed {offset}/{total}")

    db.disconnect()

    print("Inference cache created")


if __name__ == "__main__":
    main()
