from database import Database

from build_features import build_feature_table, validate_feature_table


db = Database()

try:
    db.connect()

    measurements = db.fetch_dataframe(
        """
        SELECT *
        FROM clinical_measurements;
        """
    )

    print("Measurements loaded:", measurements.shape)

    feature_table = build_feature_table(measurements)

    print("Feature table:", feature_table.shape)

    validate_feature_table(feature_table)


finally:
    db.disconnect()
