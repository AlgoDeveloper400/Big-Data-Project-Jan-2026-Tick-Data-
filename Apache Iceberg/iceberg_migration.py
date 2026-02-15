from trino.dbapi import connect
import time
from minio import Minio

# ----------------------------
# CONFIG
# ----------------------------
TRINO_HOST        = "localhost"
TRINO_PORT        = 8081
TRINO_USER        = "trino"
ICEBERG_CATALOG   = "iceberg"
HIVE_CATALOG      = "hive"

# MinIO config
MINIO_ENDPOINT    = "localhost:9000"
MINIO_ACCESS_KEY  = "admin"
MINIO_SECRET_KEY  = "password"
BUCKET_NAME       = "datalakehouse"

# Folder layout
RAW_PREFIX        = "Training Batch/"
GOLD_SCHEMA       = "gold"

# ----------------------------
# MINIO CLIENT
# ----------------------------
def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

# ----------------------------
# CONNECT TO TRINO
# ----------------------------
def get_trino_connection(catalog="iceberg", schema="gold"):
    max_retries = 10
    for i in range(max_retries):
        try:
            conn = connect(
                host=TRINO_HOST,
                port=TRINO_PORT,
                user=TRINO_USER,
                catalog=catalog,
                schema=schema
            )
            print(f"[INFO] Connected to Trino (catalog={catalog}, schema={schema})")
            return conn
        except Exception as e:
            if i < max_retries - 1:
                print(f"[INFO] Retrying connection... ({i+1}/{max_retries})")
                time.sleep(5)
            else:
                raise e

# ----------------------------
# AUTO DETECT SYMBOLS FROM Training Batch/
# Returns list of (symbol, raw_folder_name) tuples
# ----------------------------
def get_symbols_from_minio():
    """Scan Training Batch/ and return all symbol folder names"""
    client = get_minio_client()
    objects = client.list_objects(BUCKET_NAME, prefix=RAW_PREFIX, recursive=False)
    symbols = []

    for obj in objects:
        if obj.is_dir:
            folder_name = obj.object_name[len(RAW_PREFIX):].rstrip('/')
            symbols.append(folder_name)

    if not symbols:
        print("[WARNING] No symbols found in Training Batch/!")
    else:
        print(f"[INFO] Auto detected {len(symbols)} symbols: {symbols}")

    return symbols

# ----------------------------
# GET ALL GOLD FOLDERS FROM S3
# Returns dict of { symbol_prefix_lower: (exact_folder_name, full_s3_path) }
# e.g. { "eurusd": ("EURUSD-a1b2c3d4", "s3://datalakehouse/gold/EURUSD-a1b2c3d4/") }
# ----------------------------
def get_gold_folders():
    """
    Scan gold/ and return all subfolders.
    The folder name IS the table name — including the hash suffix Iceberg added.
    e.g.  gold/EURUSD-a1b2c3d4/  →  table name = EURUSD-a1b2c3d4
    """
    client = get_minio_client()
    gold_folders = {}

    try:
        objects = client.list_objects(BUCKET_NAME, prefix=f"{GOLD_SCHEMA}/", recursive=False)
        for obj in objects:
            if obj.is_dir:
                # e.g. "gold/EURUSD-a1b2c3d4/" → folder_name = "EURUSD-a1b2c3d4"
                folder_name = obj.object_name[len(GOLD_SCHEMA) + 1:].rstrip('/')
                full_path   = f"s3://{BUCKET_NAME}/{obj.object_name}"
                # key by lowercase symbol prefix (everything before the first dash)
                prefix_key  = folder_name.split('-')[0].lower()
                gold_folders[prefix_key] = (folder_name, full_path)
                print(f"[INFO] Found gold folder: {folder_name} -> {full_path}")
    except Exception as e:
        print(f"[ERROR] Failed to scan gold/ folder: {e}")

    return gold_folders

# ----------------------------
# SANITIZE SYMBOL FOR HIVE TABLE NAME
# ----------------------------
def clean_name(symbol):
    """Make symbol safe for Hive table names (raw_ prefix)"""
    return (symbol
            .replace('.', '_')
            .replace('-', '_')
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', ''))

# ----------------------------
# ENSURE HIVE DEFAULT SCHEMA EXISTS
# ----------------------------
def ensure_hive_schema_exists(hive_cursor):
    try:
        hive_cursor.execute("SHOW SCHEMAS FROM hive LIKE 'default'")
        if not hive_cursor.fetchall():
            print("[INFO] Creating Hive schema 'default'...")
            hive_cursor.execute(f"""
                CREATE SCHEMA IF NOT EXISTS hive.default
                WITH (location = 's3://{BUCKET_NAME}/')
            """)
            print("[INFO] Hive schema 'default' created")
        else:
            print("[INFO] Hive schema 'default' already exists")
    except Exception as e:
        print(f"[ERROR] Failed to ensure Hive schema: {e}")

# ----------------------------
# ENSURE GOLD SCHEMA EXISTS
# ----------------------------
def ensure_schema_exists(cursor):
    try:
        cursor.execute("SHOW SCHEMAS FROM iceberg LIKE 'gold'")
        if not cursor.fetchall():
            print("[INFO] Creating schema 'gold'...")
            cursor.execute(f"""
                CREATE SCHEMA IF NOT EXISTS iceberg.gold
                WITH (location = 's3://{BUCKET_NAME}/{GOLD_SCHEMA}/')
            """)
            print("[INFO] Schema 'gold' created")
        else:
            print("[INFO] Schema 'gold' already exists")
    except Exception as e:
        print(f"[ERROR] Failed to ensure schema: {e}")

# ----------------------------
# CHECK TABLE EXISTS
# ----------------------------
def table_exists(cursor, table_name):
    try:
        # Iceberg stores the table name as the symbol prefix before the hash
        # e.g. folder "btcusd-6746aac7..." is stored as table "btcusd"
        short_name = table_name.split('-')[0].lower()
        cursor.execute(f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'gold'
              AND LOWER(table_name) = '{short_name}'
        """)
        return len(cursor.fetchall()) > 0
    except Exception as e:
        print(f"[ERROR] Failed to check table existence: {e}")
        return False

# ----------------------------
# REGISTER RAW PARQUET IN HIVE
# ----------------------------
def ensure_hive_table_exists(hive_cursor, symbol, clean_symbol):
    """Register raw Parquet files from Training Batch/ into Hive catalog"""
    try:
        hive_cursor.execute(f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'default'
              AND LOWER(table_name) = LOWER('raw_{clean_symbol}')
        """)
        if hive_cursor.fetchall():
            print(f"[INFO] Hive table 'raw_{clean_symbol}' already registered")
            return True

        hive_cursor.execute(f"""
            CREATE TABLE hive.default.raw_{clean_symbol} (
                datetime TIMESTAMP(3),
                bid      DOUBLE,
                ask      DOUBLE
            )
            WITH (
                external_location = 's3://{BUCKET_NAME}/Training Batch/{symbol}/',
                format            = 'PARQUET'
            )
        """)
        print(f"[SUCCESS] Registered Hive table 'raw_{clean_symbol}'")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to register Hive table for '{symbol}': {e}")
        return False

# ----------------------------
# CREATE ICEBERG GOLD TABLE
# table_name  = exact folder name from gold/ e.g. "EURUSD-a1b2c3d4"
# gold_path   = full s3 path e.g. "s3://datalakehouse/gold/EURUSD-a1b2c3d4/"
# ----------------------------
def create_iceberg_table(cursor, table_name, gold_path):
    """
    Create Iceberg table using the exact folder name from gold/ as the table name.
    This keeps Trino in sync with what already exists in S3.
    """
    if table_exists(cursor, table_name):
        print(f"[INFO] Iceberg table '{table_name}' already exists, skipping creation")
        return True

    try:
        print(f"[INFO] Creating Iceberg table '{table_name}' at {gold_path}...")
        cursor.execute(f"""
            CREATE TABLE iceberg.gold."{table_name}" (
                datetime TIMESTAMP(3),
                bid      DOUBLE,
                ask      DOUBLE
            )
            WITH (
                location     = '{gold_path}',
                format       = 'PARQUET',
                partitioning = ARRAY['year(datetime)']
            )
        """)
        print(f"[SUCCESS] Created Iceberg table '{table_name}'")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create Iceberg table '{table_name}': {e}")
        return False

# ----------------------------
# LOAD RAW DATA INTO ICEBERG
# ----------------------------
def load_data(iceberg_cursor, table_name, clean_symbol):
    """Read from Hive raw table and insert into Iceberg gold table"""
    # Iceberg registered the table using the short name (prefix before hash)
    short_name = table_name.split('-')[0].lower()
    try:
        print(f"[INFO] Loading data into '{short_name}'...")

        # Skip if already has data
        iceberg_cursor.execute(f'SELECT COUNT(*) FROM iceberg.gold."{short_name}"')
        count = iceberg_cursor.fetchone()[0]
        if count > 0:
            print(f"[INFO] '{short_name}' already has {count:,} rows, skipping load")
            return

        iceberg_cursor.execute(f"""
            INSERT INTO iceberg.gold."{short_name}"
            SELECT datetime, bid, ask
            FROM hive.default.raw_{clean_symbol}
            WHERE datetime IS NOT NULL
        """)

        iceberg_cursor.execute(f'SELECT COUNT(*) FROM iceberg.gold."{short_name}"')
        final_count = iceberg_cursor.fetchone()[0]
        print(f"[SUCCESS] '{short_name}' loaded with {final_count:,} rows")

    except Exception as e:
        print(f"[ERROR] Failed to load data for '{short_name}': {e}")

# ----------------------------
# VERIFY PARTITIONS
# ----------------------------
def verify_partitions(cursor, table_name):
    """Show row count per year partition"""
    short_name = table_name.split('-')[0].lower()
    try:
        cursor.execute(f"""
            SELECT year(datetime) as year, COUNT(*) as rows
            FROM iceberg.gold."{short_name}"
            GROUP BY year(datetime)
            ORDER BY year
        """)
        rows = cursor.fetchall()
        if rows:
            print(f"[INFO] Partitions for '{short_name}':")
            for row in rows:
                print(f"       year={row[0]}  ->  {row[1]:,} rows")
        else:
            print(f"[WARNING] No data found in '{short_name}'")
    except Exception as e:
        print(f"[ERROR] Failed to verify partitions for '{short_name}': {e}")

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    try:
        # Step 1: Auto detect symbols from Training Batch/
        symbols = get_symbols_from_minio()
        if not symbols:
            print("[FATAL] No symbols found. Exiting.")
            exit(1)

        # Step 2: Get all existing hashed folders from gold/
        # { "eurusd": ("EURUSD-a1b2c3d4", "s3://datalakehouse/gold/EURUSD-a1b2c3d4/") }
        gold_folders = get_gold_folders()

        # Step 3: Connections
        iceberg_conn   = get_trino_connection(catalog="iceberg", schema="gold")
        iceberg_cursor = iceberg_conn.cursor()

        hive_conn      = get_trino_connection(catalog="hive", schema="default")
        hive_cursor    = hive_conn.cursor()

        # Step 4: Ensure both schemas exist
        ensure_hive_schema_exists(hive_cursor)
        ensure_schema_exists(iceberg_cursor)

        # Step 5: Process each symbol
        for symbol in symbols:
            clean_symbol = clean_name(symbol)
            prefix_key   = symbol.split('-')[0].lower()

            # Look up the exact hashed folder name for this symbol
            if prefix_key not in gold_folders:
                print(f"[SKIP] No gold folder found for '{symbol}' — run your original setup script first")
                continue

            table_name, gold_path = gold_folders[prefix_key]

            print(f"\n{'='*55}")
            print(f"  Symbol     : {symbol}")
            print(f"  Table name : {table_name}")
            print(f"  Gold path  : {gold_path}")
            print(f"{'='*55}")

            # Register raw parquet in Hive
            if not ensure_hive_table_exists(hive_cursor, symbol, clean_symbol):
                print(f"[SKIP] Could not register Hive table for '{symbol}'")
                continue

            # Create Iceberg table using exact folder name
            if not create_iceberg_table(iceberg_cursor, table_name, gold_path):
                print(f"[SKIP] Could not create Iceberg table for '{symbol}'")
                continue

            # Load data
            load_data(iceberg_cursor, table_name, clean_symbol)

            # Verify partitions
            verify_partitions(iceberg_cursor, table_name)

        # Step 6: Final summary
        print(f"\n{'='*55}")
        print("[INFO] All tables in gold schema:")
        iceberg_cursor.execute("SHOW TABLES FROM iceberg.gold")
        for table in iceberg_cursor.fetchall():
            print(f"  - {table[0]}")

        iceberg_cursor.close()
        iceberg_conn.close()
        hive_cursor.close()
        hive_conn.close()

        print("\n[DONE] All symbols processed successfully!")

    except Exception as e:
        print(f"[FATAL ERROR] Script failed: {e}")