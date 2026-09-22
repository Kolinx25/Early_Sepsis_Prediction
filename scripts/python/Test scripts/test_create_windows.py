from database import Database

from build_features import build_feature_table

from create_windows import create_trajectory_windows


db = Database()

try:
    db.connect()

    # Load only one patient first
    measurements = db.fetch_dataframe(
        """
        SELECT *
        FROM clinical_measurements
        WHERE patient_id = 9
        ORDER BY iculos;
        """
    )

    print("Measurements:", measurements.shape)

    feature_table = build_feature_table(measurements)

    print("Feature table:", feature_table.shape)

    X, y_6, y_12 = create_trajectory_windows(feature_table)
    print("\nLast 20 ICU labels:")
    print(feature_table[["iculos", "sepsis_label"]].tail(20))

    print("\nPositive targets:")
    print("y_6 positives:", y_6.sum())

    print("y_12 positives:", y_12.sum())

    print("X shape:", X.shape)
    print("y_6 shape:", y_6.shape)
    print("y_12 shape:", y_12.shape)


finally:
    db.disconnect()

print("\nFirst 20 sepsis labels:")
print(feature_table[["iculos", "sepsis_label"]].head(20))

positive_hours = feature_table[feature_table["sepsis_label"] == 1]["iculos"]

print("\nFirst positive ICU hour:")
print(positive_hours.iloc[0])
