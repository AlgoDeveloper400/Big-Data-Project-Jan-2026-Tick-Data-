import os
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from tqdm import tqdm
# --------------------------------------------------
# PATHS
# --------------------------------------------------
INPUT_BASE = r"path\to\your\data"
OUTPUT_BASE = r"output\save\path"

# --------------------------------------------------
# CHUNK RULE
# --------------------------------------------------
CSV_CHUNK_SIZE = 500 * 1024 * 1024   # 500MB
PARQUET_COMPRESSION = "snappy"       # safe, fast

# --------------------------------------------------
# SCHEMA (TICK SAFE)
# --------------------------------------------------
CSV_SCHEMA = pa.schema([
    ("DateTime", pa.string()),
    ("Bid", pa.float64()),
    ("Ask", pa.float64()),
    ("Volume", pa.float64()),
])

# --------------------------------------------------
# CHECK IF CSV ALREADY PROCESSED
# --------------------------------------------------
def already_processed(csv_path):
    relative_path = os.path.relpath(csv_path, INPUT_BASE)
    relative_dir = os.path.dirname(relative_path)
    output_dir = os.path.join(OUTPUT_BASE, relative_dir)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    if not os.path.exists(output_dir):
        return False

    return any(
        f.startswith(base_name) and f.endswith(".parquet")
        for f in os.listdir(output_dir)
    )

# --------------------------------------------------
# PROCESS SINGLE CSV (SAFE)
# --------------------------------------------------
def process_csv(csv_path):
    if already_processed(csv_path):
        print(f"⏭ Skipping already processed: {csv_path}")
        return

    relative_path = os.path.relpath(csv_path, INPUT_BASE)
    relative_dir = os.path.dirname(relative_path)
    output_dir = os.path.join(OUTPUT_BASE, relative_dir)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    print(f"▶ Processing CSV: {csv_path}")

    reader = pv.open_csv(
        csv_path,
        read_options=pv.ReadOptions(
            block_size=CSV_CHUNK_SIZE,
            autogenerate_column_names=False
        ),
        parse_options=pv.ParseOptions(delimiter=","),
        convert_options=pv.ConvertOptions(
            column_types=CSV_SCHEMA,
            include_columns=["DateTime", "Bid", "Ask", "Volume"]
        )
    )

    for chunk_id, record_batch in enumerate(reader):
        table = pa.Table.from_batches([record_batch])

        # Drop Volume
        table = table.remove_column(
            table.schema.get_field_index("Volume")
        )

        parquet_path = os.path.join(
            output_dir,
            f"{base_name}_chunk_{chunk_id:03d}.parquet"
        )

        pq.write_table(
            table,
            parquet_path,
            compression=PARQUET_COMPRESSION,
            use_dictionary=True,
            write_statistics=True
        )

        # Explicit cleanup (important on Windows)
        del table
        del record_batch

# --------------------------------------------------
# PROCESS FOLDER BY FOLDER (STRICT)
# --------------------------------------------------
def process_all():
    for root, _, files in os.walk(INPUT_BASE):
        files = sorted(f for f in files if f.lower().endswith(".csv"))

        if not files:
            continue

        print(f"\n📁 Processing folder: {root}")

        for f in files:
            csv_path = os.path.join(root, f)
            process_csv(csv_path)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    print("🚦 SAFE MODE ENABLED")
    print("• 1 folder at a time")
    print("• 1 CSV at a time")
    print("• 1 × 500MB chunk → 1 parquet")
    print("• No parallelism")

    process_all()

    print("\n✅ Completed safely without crashing")
