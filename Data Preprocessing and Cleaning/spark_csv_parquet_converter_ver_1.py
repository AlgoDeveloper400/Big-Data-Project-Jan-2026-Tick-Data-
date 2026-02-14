import os
from pathlib import Path
from pyspark.sql import SparkSession

# --------------------------------------------------
# PATHS
# --------------------------------------------------
INPUT_BASE = Path(r"path\to\your\data")
OUTPUT_BASE = Path(
    r"C:\your\output\path"
)

# --------------------------------------------------
# PARTITION / PARQUET RULES
# --------------------------------------------------
INPUT_SPLIT_SIZE = 500 * 1024 * 1024      # 500MB input splits
PARQUET_BLOCK_SIZE = 512 * 1024 * 1024    # ~512MB parquet blocks
PARQUET_COMPRESSION = "snappy"

# --------------------------------------------------
# SPARK SESSION
# --------------------------------------------------
spark = (
    SparkSession.builder
    .appName("TickCSV_to_Parquet_Optimized")
    .config("spark.sql.files.maxPartitionBytes", INPUT_SPLIT_SIZE)
    .config("spark.sql.parquet.block.size", PARQUET_BLOCK_SIZE)
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.executor.memory", "16g")
    .config("spark.driver.memory", "16g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# --------------------------------------------------
# PROCESS ONE DIRECTORY (BATCHED)
# --------------------------------------------------
def process_directory(input_dir: Path):
    csv_files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() == ".csv")
    if not csv_files:
        return

    relative_dir = input_dir.relative_to(INPUT_BASE)
    output_dir = OUTPUT_BASE / relative_dir

    # If output already exists, skip (idempotent)
    if output_dir.exists() and any(output_dir.glob("*.parquet")):
        print(f"⏭ Skipping already processed folder: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Processing folder: {input_dir}")
    print(f"   ➜ Writing to: {output_dir}")

    # --------------------------------------------------
    # READ ALL CSVs IN THIS DIRECTORY AT ONCE
    # --------------------------------------------------
    df = (
        spark.read
        .csv(
            [str(p) for p in csv_files],
            header=True,
            inferSchema=False,
            schema="DateTime STRING, Bid DOUBLE, Ask DOUBLE, Volume DOUBLE"
        )
        .select("DateTime", "Bid", "Ask")  # drop Volume early
    )

    # --------------------------------------------------
    # WRITE PARQUET (SPARK MANAGED)
    # --------------------------------------------------
    (
        df.write
        .mode("overwrite")
        .option("compression", PARQUET_COMPRESSION)
        .parquet(str(output_dir))
    )

# --------------------------------------------------
# WALK INPUT TREE
# --------------------------------------------------
def process_all():
    for root, _, _ in os.walk(INPUT_BASE):
        process_directory(Path(root))

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    print("🚦 SPARK MODE ENABLED (OPTIMIZED)")
    print("• Folder structure preserved")
    print("• Batched CSV ingestion")
    print("• ~500MB parquet blocks")
    print("• Spark-managed files (no driver IO bottlenecks)")
    print("• Safe to re-run (idempotent)")

    process_all()

    spark.stop()
    print("\n✅ Completed successfully")
