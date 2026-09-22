import psycopg2

from config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)


try:
    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )

    print("Connection successful")
    print("Database:", DB_NAME)

    connection.close()

except Exception as error:
    print("Connection failed")
    print(error)
