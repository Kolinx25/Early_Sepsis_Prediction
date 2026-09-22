from database import Database
from build_features import build_feature_table
from create_windows import create_trajectory_windows


def main():

    db = Database()

    try:
        db.connect()

        feature_table = db.fetch_dataframe(
            """
            SELECT *
            FROM feature_table
            ORDER BY patient_id, iculos;
            """
        )

        print("Feature table loaded:", feature_table.shape)

        windows = create_trajectory_windows(feature_table)

        save_windows(db, windows)

    finally:
        db.disconnect()


if __name__ == "__main__":
    main()
