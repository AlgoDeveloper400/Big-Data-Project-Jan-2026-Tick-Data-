from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    hour, minute, to_timestamp, date_trunc, col, first, lit
)
from pyspark.sql.types import FloatType
from pathlib import Path
import shutil
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# ======================================================
# Spark Init
# ======================================================
spark = (
    SparkSession.builder
    .appName("Tick Data 1-Second Window Processor")
    .config("spark.driver.memory", "16g")
    .config("spark.sql.shuffle.partitions", "50")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    .getOrCreate()
)

# ======================================================
# Config
# ======================================================
BASE_INPUT_FOLDER = Path(
    r"output\path\of\spark_data_cleaner_script\script"
)
BASE_OUTPUT_FOLDER = Path(
    r"your/output/path"
)

TIMESTAMP_COL = "DateTime"
MAX_PARTITION_BYTES = 160 * 1024 * 1024  # 160 MB

# ======================================================
# Helper Functions (ORDER LOGIC)
# ======================================================
def parse_datetime_safe(v):
    return pd.to_datetime(v, format="%Y%m%d %H:%M:%S.%f", errors="coerce")

def get_first_last_datetime(file_path):
    pq_file = pq.ParquetFile(file_path)

    first_val = pq_file.read_row_group(
        0, columns=[TIMESTAMP_COL]
    ).to_pandas().iloc[0][TIMESTAMP_COL]

    last_group = pq_file.num_row_groups - 1
    last_idx = pq_file.metadata.row_group(last_group).num_rows - 1
    last_val = pq_file.read_row_group(
        last_group, columns=[TIMESTAMP_COL]
    ).to_pandas().iloc[last_idx][TIMESTAMP_COL]

    return parse_datetime_safe(first_val), parse_datetime_safe(last_val)

def determine_file_order(parquet_files):
    records = []
    for f in parquet_files:
        s, e = get_first_last_datetime(f)
        records.append((str(f), s, e))

    df = pd.DataFrame(records, columns=["file", "start", "end"])
    df["start"] = pd.to_datetime(df["start"]).fillna(pd.Timestamp.min)
    df["end"] = pd.to_datetime(df["end"]).fillna(pd.Timestamp.max)

    starts = df["start"].values.astype("datetime64[ns]")
    ends = df["end"].values.astype("datetime64[ns]")

    diff = np.abs(ends[:, None] - starts[None, :])
    np.fill_diagonal(diff, np.timedelta64(10000, "D"))

    next_idx = np.argmin(diff, axis=1)

    visited = set()
    order = []
    cur = np.argmin(starts)

    while cur not in visited:
        visited.add(cur)
        order.append(cur)
        cur = next_idx[cur]
        if cur in visited:
            break

    return df.iloc[order]["file"].tolist()

# ======================================================
# Main Processing Loop
# ======================================================
symbol_folders = [
    f for f in BASE_INPUT_FOLDER.iterdir()
    if f.is_dir()  # Removed: "and f.name not in IGNORE_DIRS"
]

for symbol_folder in symbol_folders:
    print(f"\n=== SYMBOL: {symbol_folder.name} ===")

    parquet_files = list(symbol_folder.rglob("*.parquet"))
    if not parquet_files:
        print("No parquet files found, skipping.")
        continue

    ordered_files = determine_file_order(parquet_files)
    file_order_map = {f: i for i, f in enumerate(ordered_files)}

    # --------------------------------------------------
    # Read files (PARALLEL, NO BOTTLENECK)
    # --------------------------------------------------
    dfs = []
    for f in ordered_files:
        df_tmp = (
            spark.read
            .option("maxPartitionBytes", MAX_PARTITION_BYTES)
            .parquet(f)
            .select("DateTime", "Bid", "Ask")
            .withColumn("file_order", lit(file_order_map[f]))
        )
        dfs.append(df_tmp)

    # SAFE UNION (NO *ARGS)
    df = dfs[0]
    for other in dfs[1:]:
        df = df.unionByName(other)

    print("Input partitions:", df.rdd.getNumPartitions())

    # --------------------------------------------------
    # Processing
    # --------------------------------------------------
    df = (
        df.withColumn("Bid", col("Bid").cast(FloatType()))
          .withColumn("Ask", col("Ask").cast(FloatType()))
          .withColumn("DateTime_ts", to_timestamp("DateTime", "yyyyMMdd HH:mm:ss.SSS"))
          .withColumn("hour", hour("DateTime_ts"))
          .withColumn("minute", minute("DateTime_ts"))
    )

    df_filtered = df.filter(
        ((col("hour") == 7) & (col("minute") >= 50)) |
        ((col("hour") == 13) & (col("minute") >= 50))
    )

    if df_filtered.count() == 0:
        print("No data after filtering.")
        continue

    df_1s = (
        df_filtered
        .withColumn("second_trunc", date_trunc("second", "DateTime_ts"))
        .groupBy("second_trunc", "file_order")
        .agg(
            first(col("DateTime_ts"), ignorenulls=True).alias("DateTime"),
            first(col("Bid"), ignorenulls=True).alias("Bid"),
            first(col("Ask"), ignorenulls=True).alias("Ask")
        )
        .orderBy("file_order", "second_trunc")
        .select("DateTime", "Bid", "Ask")
    )

    # --------------------------------------------------
    # Save (SINGLE FILE, CLEAN NAME)
    # --------------------------------------------------
    out_path = BASE_OUTPUT_FOLDER / symbol_folder.name
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path / "tmp"

    df_1s.coalesce(1).write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(str(tmp_path))

    parquet_file = list(tmp_path.glob("*.parquet"))[0]
    final_name = Path(ordered_files[0]).stem + ".parquet"

    shutil.move(str(parquet_file), str(out_path / final_name))
    shutil.rmtree(tmp_path)

    print(f"✅ Saved: {symbol_folder.name}/{final_name}")

# ======================================================
# Cleanup
# ======================================================
spark.stop()
print("\nAll symbols processed successfully.")