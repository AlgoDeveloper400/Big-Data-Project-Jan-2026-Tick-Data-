from trino.dbapi import connect
from minio import Minio
import time

# ----------------------------
# CONFIG
# ----------------------------
TRINO_HOST      = "localhost"
TRINO_PORT      = 8081
TRINO_USER      = "trino"
MINIO_ENDPOINT  = "localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password"
BUCKET_NAME     = "datalakehouse"
RAW_PREFIX      = "Training Batch/"

# ----------------------------
# CONNECTIONS
# ----------------------------
def get_trino_cursor():
    for i in range(10):
        try:
            conn = connect(host=TRINO_HOST, port=TRINO_PORT, user=TRINO_USER,
                           catalog="iceberg", schema="gold")
            print("[INFO] Connected to Trino")
            return conn, conn.cursor()
        except Exception as e:
            if i < 9:
                print(f"[INFO] Retrying connection... ({i+1}/10)")
                time.sleep(5)
            else:
                raise e

def get_minio_client():
    return Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY,
                 secret_key=MINIO_SECRET_KEY, secure=False)

# ----------------------------
# HELPERS
# ----------------------------
def clean_name(symbol):
    return (symbol.replace('.', '_').replace('-', '_')
                  .replace(' ', '_').replace('(', '').replace(')', ''))

def get_symbols():
    objects = get_minio_client().list_objects(BUCKET_NAME, prefix=RAW_PREFIX, recursive=False)
    symbols = [obj.object_name[len(RAW_PREFIX):].rstrip('/') for obj in objects if obj.is_dir]
    print(f"[INFO] Found {len(symbols)} symbols: {symbols}")
    return symbols

# ----------------------------
# SCHEMA + TABLE SETUP
# ----------------------------
def ensure_gold_schema(cursor):
    cursor.execute("SHOW SCHEMAS FROM iceberg LIKE 'gold'")
    if not cursor.fetchall():
        print("[INFO] Creating schema 'gold'...")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS iceberg.gold "
                       f"WITH (location = 's3://{BUCKET_NAME}/gold/')")
        print("[INFO] Schema 'gold' created")
    else:
        print("[INFO] Schema 'gold' already exists")

def create_table(cursor, symbol):
    name = clean_name(symbol)
    cursor.execute(f"""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'gold' AND LOWER(table_name) = LOWER('{name}')
    """)
    if cursor.fetchall():
        print(f"[INFO] Table '{name}' already exists, skipping")
        return
    cursor.execute(f"""
        CREATE TABLE gold."{name}" (
            datetime TIMESTAMP(3),
            bid      DOUBLE,
            ask      DOUBLE
        ) WITH (
            format       = 'PARQUET',
            partitioning = ARRAY['year(datetime)']
        )
    """)
    print(f"[SUCCESS] Created table '{name}'")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    conn, cursor = get_trino_cursor()

    ensure_gold_schema(cursor)

    for symbol in get_symbols():
        create_table(cursor, symbol)

    print("\n[INFO] Tables in gold schema:")
    cursor.execute("SHOW TABLES FROM iceberg.gold")
    for (t,) in cursor.fetchall():
        print(f"  - {t}")

    cursor.close()
    conn.close()
    print("\n[DONE] Metadata setup complete")