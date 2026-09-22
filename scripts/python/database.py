"""
database.py

Handles all communication with PostgreSQL.

Responsibilities
----------------
- Connect to PostgreSQL
- Execute SQL scripts
- Insert patient records
- Insert clinical measurements
- Bulk insert prediction outputs
- Fetch dataframe results
- Close database connections
"""

import pandas as pd
import psycopg2

from psycopg2.extras import execute_values

from config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    SCHEMA_FILE,
)


class Database:
    """
    Simple wrapper around PostgreSQL connection.
    """

    def __init__(self):

        self.connection = None
        self.cursor = None

    def connect(self):
        """
        Establish PostgreSQL connection.
        """

        try:
            self.connection = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )

            self.cursor = self.connection.cursor()

            print("✓ Connected to PostgreSQL successfully.")

        except Exception as error:
            print("\nConnection Failed")
            print("----------------------------")
            print(error)

            raise

    def insert_patient(self, patient):

        sql = """

        INSERT INTO patients
        (
            patient_id,
            age,
            gender,
            unit1,
            unit2,
            hosp_adm_time
        )

        VALUES
        (
            %s,%s,%s,%s,%s,%s
        )

        ON CONFLICT(patient_id)
        DO NOTHING;

        """

        values = (
            patient["Patient_id"],
            patient["Age"],
            patient["Gender"],
            patient["Unit1"],
            patient["Unit2"],
            patient["HospAdmTime"],
        )

        try:
            self.cursor.execute(sql, values)

            self.connection.commit()

        except Exception as error:
            self.connection.rollback()

            print(error)

            raise

    def insert_patients_bulk(self, df):

        sql = """

        INSERT INTO patients
        (
            patient_id,
            age,
            gender,
            unit1,
            unit2,
            hosp_adm_time
        )

        VALUES %s

        ON CONFLICT(patient_id)
        DO NOTHING;

        """

        values = []

        for _, row in df.iterrows():
            values.append(
                (
                    row["Patient_id"],
                    row["Age"],
                    row["Gender"],
                    row["Unit1"],
                    row["Unit2"],
                    row["HospAdmTime"],
                )
            )

        try:
            execute_values(self.cursor, sql, values, page_size=5000)

            self.connection.commit()

            print(f"✓ {len(values)} patients inserted.")

        except Exception as error:
            self.connection.rollback()

            print(error)

            raise

    def insert_measurements(self, measurements):

        sql = """

        INSERT INTO clinical_measurements

        (
            patient_id,
            iculos,
            hr,
            o2sat,
            temp,
            sbp,
            map,
            dbp,
            resp,
            etco2,
            baseexcess,
            hco3,
            fio2,
            ph,
            paco2,
            sao2,
            ast,
            bun,
            alkalinephos,
            calcium,
            chloride,
            creatinine,
            bilirubin_direct,
            glucose,
            lactate,
            magnesium,
            phosphate,
            potassium,
            bilirubin_total,
            troponini,
            hct,
            hgb,
            ptt,
            wbc,
            fibrinogen,
            platelets,
            sepsis_label
        )

        VALUES %s

        ON CONFLICT
        (
            patient_id,
            iculos
        )

        DO NOTHING;

        """

        values = []

        for _, row in measurements.iterrows():
            values.append(tuple(row))

        try:
            execute_values(self.cursor, sql, values, page_size=5000)

            self.connection.commit()

            print(f"✓ Inserted {len(values)} measurements.")

        except Exception as error:
            self.connection.rollback()

            print(error)

            raise

    def bulk_insert(self, sql, values):
        """
        Generic PostgreSQL bulk insert.

        Used for:
        - patient_risk_predictions
        - large inference outputs
        - batch writes
        """

        try:
            execute_values(self.cursor, sql, values, page_size=5000)

            self.connection.commit()

            print(f"✓ Bulk inserted {len(values)} rows.")

        except Exception as error:
            self.connection.rollback()

            print("\nBulk Insert Failed")
            print("----------------------------")
            print(error)

            raise

    def fetch_dataframe(self, query):
        """
        Execute SQL query and return pandas dataframe.
        """

        try:
            return pd.read_sql_query(query, self.connection)

        except Exception as error:
            print("\nQuery Failed")
            print("----------------------------")
            print(error)

            raise

    def execute_sql_file(self, sql_file):

        try:
            with open(sql_file, "r", encoding="utf-8") as file:
                sql = file.read()

            self.cursor.execute(sql)

            self.connection.commit()

            print(f"✓ Executed {sql_file}")

        except Exception as error:
            self.connection.rollback()

            print(error)

            raise

    def disconnect(self):

        if self.cursor:
            self.cursor.close()

        if self.connection:
            self.connection.close()

        print("✓ Database connection closed.")


if __name__ == "__main__":
    print("Database module loaded successfully.")
