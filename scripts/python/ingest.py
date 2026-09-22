"""
This script runs the complete data ingestion pipeline.
Moves PhysioNet patient data into PostgreSQL.

Workflow
--------
1. Discover patient files
2. Load one patient
3. Split demographics and hourly measurements
4. Connect to PostgreSQL
5. Create database tables (or reset if --reset flag is passed)
6. Insert patient
7. Insert measurements

Usage
-----
    python ingest.py              # APPEND mode: create tables if not exist
    python ingest.py --reset      # RESET mode: drop and recreate tables
"""

import argparse
from pathlib import Path

from data_loader import (
    discover_patient_files,
    load_patient,
    extract_patient_id,
    split_patient_data,
)

from database import Database
from config import SCHEMA_FILE

# Paths
SQL_DIR = Path(__file__).resolve().parents[1] / "sql"
RESET_SCHEMA_FILE = SQL_DIR / "reset_schema.sql"


def main(reset=False):

    print("=" * 60)
    print("EARLY SEPSIS PREDICTION")
    print("PhysioNet Data Ingestion")
    print("=" * 60)

    if reset:
        print("Mode : RESET")
    else:
        print("Mode : APPEND")

    
    # Discover patient files

    patient_files = discover_patient_files()

    print(f"\nPatients discovered : {len(patient_files):,} patients.")

    # Connect to PostgreSQL
    db = Database()

    try:

        db.connect()

        # Execute reset schema if --reset flag is set
        if reset:
            db.execute_sql_file(RESET_SCHEMA_FILE)

        # Execute normal schema (idempotent, uses CREATE TABLE IF NOT EXISTS)
        db.execute_sql_file(SCHEMA_FILE)

        # Load first 10 patients
        for patient_file in patient_files:

            try:

                print(f"\nLoading {patient_file.name}")

                # Read PSV
                df = load_patient(patient_file)

                
                # Extract patient identifier
                patient_id = extract_patient_id(patient_file)
                print(f"Patient ID : {patient_id}")

                
                # Split into two datasets
                patient, measurements = split_patient_data(df, patient_id)

                print("Patient demographics prepared.")

                print(f"{len(measurements)} hourly measurements prepared.")

                db.insert_patient(patient)

                db.insert_measurements(measurements)

                print(f"✓ Patient {patient_id} inserted.")
                print(f"✓ {len(measurements)} measurements inserted.")

            except Exception as e:
                print(f"✗ Error processing {patient_file.name}: {e}")
                continue

        print("\n✓ Ingestion completed successfully.")

    finally:

        db.disconnect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Ingest PhysioNet patient data into PostgreSQL"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate tables before ingestion (destructive)",
    )

    args = parser.parse_args()

    main(reset=args.reset)
