/*
  RESET SCHEMA
  ============
  
Destructively drops the schema tables and indexes.
Use with caution — this erases all data.

This script is only executed when the --reset flag is passed to ingest.py
*/

DROP TABLE IF EXISTS clinical_measurements CASCADE;
DROP TABLE IF EXISTS patients CASCADE;
