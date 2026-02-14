from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, to_date, count, hour, minute
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# ==============================
# Initialize Spark
# ==============================
spark = SparkSession.builder \
    .appName("Daily Tick Counts per Symbol") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ==============================
# Paths
# ==============================
base_path = r"output\path\of\spark_session_window_tick_cleaner\script"

output_path = r"your/own/output/path"
os.makedirs(output_path, exist_ok=True)

# ==============================
# Timestamp column and windows
# ==============================
timestamp_col = "DateTime"

window1_start, window1_end = (7, 50), (8, 0)
window2_start, window2_end = (13, 50), (14, 0)

# ==============================
# Helper function to filter by window
# ==============================
def filter_window(df, start, end):
    sh, sm = start
    eh, em = end
    return df.filter(
        ((col("hour") > sh) | ((col("hour") == sh) & (col("minute") >= sm))) &
        ((col("hour") < eh) | ((col("hour") == eh) & (col("minute") < em)))
    )

# ==============================
# Process each symbol folder
# ==============================
symbols = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
if not symbols:
    symbols = [""]

for symbol in symbols:
    symbol_path = os.path.join(base_path, symbol)
    parquet_files = glob.glob(os.path.join(symbol_path, "*.parquet"))

    if not parquet_files:
        print(f"No Parquet files found for symbol '{symbol}'. Skipping.")
        continue

    print(f"\nProcessing symbol: {symbol or 'BASE'}")

    df = spark.read.parquet(*parquet_files)

    df = (
        df.withColumn("ts", to_timestamp(col(timestamp_col)))
          .withColumn("date", to_date(col("ts")))
          .withColumn("hour", hour(col("ts")))
          .withColumn("minute", minute(col("ts")))
    )

    window1_df = filter_window(df, window1_start, window1_end)
    window2_df = filter_window(df, window2_start, window2_end)

    window1_counts = window1_df.groupBy("date").agg(count("*").alias("ticks_window1"))
    window2_counts = window2_df.groupBy("date").agg(count("*").alias("ticks_window2"))

    daily_counts = (
        window1_counts
        .join(window2_counts, on="date", how="outer")
        .fillna(0)
        .orderBy("date")
    )

    pdf = daily_counts.toPandas()

    symbol_clean = symbol.replace(" ", "_") or "BASE"
    csv_path = os.path.join(output_path, f"{symbol_clean}_daily_window_counts.csv")
    plot_path = os.path.join(output_path, f"{symbol_clean}_daily_window_counts.png")

    # Save CSV
    pdf.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(pdf["date"]), pdf["ticks_window1"], marker="o", label="Window 1 (07:50–08:00)")
    plt.plot(pd.to_datetime(pdf["date"]), pdf["ticks_window2"], marker="o", label="Window 2 (13:50–14:00)")
    plt.title(f"Daily Tick Counts per Window: {symbol_clean}")
    plt.xlabel("Date")
    plt.ylabel("Tick Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved plot: {plot_path}")

# ==============================
# Stop Spark
# ==============================
spark.stop()
print("\nProcessing complete!")
