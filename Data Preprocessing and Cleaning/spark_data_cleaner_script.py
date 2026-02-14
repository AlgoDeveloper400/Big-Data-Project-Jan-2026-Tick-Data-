from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
import pandas as pd

# Config
#we have 2 scripts that can convert the csv files to parquet, one is a python script and another is a pyspark script, both yield similar results
INPUT_BASE = r"input\path\from csv_to_parquet_converter or spark_csv_parquet_converter_ver_1 \script"
REPORT_PATH = r"your\report\path"

# Create Spark session
spark = SparkSession.builder \
    .appName("Parquet Data Quality Scanner") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Simple scanner function
def scan_parquet_file(parquet_file):
    try:
        df = spark.read.parquet(parquet_file)
        total_rows = df.count()
        
        # Count nulls for each column
        datetime_nulls = df.filter(col("DateTime").isNull()).count() if "DateTime" in df.columns else 0
        bid_nulls = df.filter(col("Bid").isNull()).count() if "Bid" in df.columns else 0
        ask_nulls = df.filter(col("Ask").isNull()).count() if "Ask" in df.columns else 0
        
        return {
            "file": parquet_file,
            "total_rows": total_rows,
            "DateTime_nulls": datetime_nulls,
            "Bid_nulls": bid_nulls,
            "Ask_nulls": ask_nulls,
            "corrupted": False
        }
    except Exception as e:
        return {
            "file": parquet_file,
            "total_rows": 0,
            "DateTime_nulls": 0,
            "Bid_nulls": 0,
            "Ask_nulls": 0,
            "corrupted": True,
            "error": str(e)[:200]  # Truncate error message
        }

# Find all parquet files
parquet_files = []
for root, _, files in os.walk(INPUT_BASE):
    for f in files:
        if f.endswith(".parquet"):
            parquet_files.append(os.path.join(root, f))

print(f"Found {len(parquet_files)} files to scan")

# Scan files
report_rows = []
for i, parquet_file in enumerate(parquet_files):
    if i % 100 == 0:  # Progress update every 100 files
        print(f"Scanned {i}/{len(parquet_files)} files...")
    
    report = scan_parquet_file(parquet_file)
    report_rows.append(report)

# Save report
df_report = pd.DataFrame(report_rows)
df_report.to_csv(REPORT_PATH, index=False)

# Print summary
valid_files = sum(1 for r in report_rows if not r["corrupted"])
corrupted_files = len(report_rows) - valid_files
total_rows = sum(r["total_rows"] for r in report_rows if not r["corrupted"])

print(f"\nSummary:")
print(f"Total files: {len(report_rows)}")
print(f"Valid files: {valid_files}")
print(f"Corrupted files: {corrupted_files}")
print(f"Total rows in valid files: {total_rows:,}")

if corrupted_files > 0:
    print(f"\nCorrupted files found:")
    for r in report_rows:
        if r["corrupted"]:
            print(f"  {os.path.basename(r['file'])}: {r.get('error', 'Unknown error')}")

spark.stop()
print("\nDone!")