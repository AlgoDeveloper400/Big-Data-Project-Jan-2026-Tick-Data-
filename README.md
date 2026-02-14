# 📈 Big Data Tick Data ML Trading Pipeline — Jan 2026  

![Gold Data Lake House Architecture](./Gold%20Data%20Lake%20House%20architecture%20Feb%202026_V1.png)

---

# 1️⃣ System Overview

This project implements a **full-stack quantitative ML trading architecture** built around:

- High-frequency tick ingestion  
- Distributed data processing  
- Lakehouse storage (Iceberg + S3 + Postgres catalog)  
- Model lifecycle management  
- Backtesting & walk-forward validation  
- MT5 deployment for live/demo execution  
- Experiment tracking  

The system is modular, scalable, and production-oriented.

---

# 2️⃣ Data Source Layer

## 2.1 Dukascopy Website

**Source:** Historical tick data  
**Format:** Raw CSV  
**Granularity:** Sub-second  

**Responsibilities:**

- Download immutable historical tick data  
- Maintain original dataset integrity  
- Provide reproducible base data  

---

# 3️⃣ Data Replication Layer

## 3.1 Local → S3 Transfer (NiFi)

Large finalized datasets are transferred:

- From local file system  
- To S3 object storage  
- Using Apache NiFi  

**NiFi handles:**

- Transfer orchestration  
- Backpressure control  
- Flow monitoring  
- Reliability  

This layer ensures stable movement into the lakehouse.

---

# 4️⃣ Raw CSV Storage

Raw tick data is stored unmodified in CSV format.

**Purpose:**

- Immutable recovery layer  
- Enables reprocessing  
- Protects against transformation errors  
- Maintains auditability  

No transformations occur at this stage.

---

# 5️⃣ Data Processing Layer

## 5.1 Python Conversion Stage

- Ingest raw CSV  
- Convert CSV to Parquet  
- Save as partitioned datasets  

**Benefits of partitioning:**

- Faster reads  
- Efficient Spark processing  
- Scalable storage  

---

## 5.2 Spark Processing

Apache Spark performs:

- Large-scale cleaning  
- Sorting  
- Missing value handling  
- Timestamp normalization  
- Feature preparation  

**Output:** ML-ready datasets

---

# 6️⃣ Data Version Control

Final datasets are versioned using:

- DVC (Data Version Control)

**Benefits:**

- Reproducibility  
- Dataset lineage  
- Controlled experimentation  
- Rollback capability  

---

# 7️⃣ Data Lakehouse Architecture

The lakehouse is the central storage and analytics layer.

## 7.1 Storage Components

- **S3** → Stores Parquet data files  
- **Postgres** → Stores Iceberg table metadata  
- **Apache Iceberg** → Table format layer  

**Iceberg provides:**

- ACID guarantees  
- Schema evolution  
- Snapshot isolation  
- Time travel  

---

## 7.2 Lakehouse Tables

Structured tables are created on top of S3 data using Iceberg.

These tables represent:

- Gold datasets (ML-ready)  

---

## 7.3 Query Engine

**Trino** is used as the query engine.

**Capabilities:**

- SQL analytics on Iceberg tables  
- Distributed query execution  
- Research queries for feature engineering  
- Dataset validation before ML usage  

---

# 8️⃣ Machine Learning Pipeline (FastAPI UI)

The ML system follows a strict lifecycle.

## 8.1 Model Core

- Anomaly / pattern detection models  
- Consumes gold datasets  
- Produces trading signals  

---

## 8.2 Training Phase

- Learns model parameters  
- Uses historical tick datasets  
- Logs metrics  

---

## 8.3 Validation Phase

- Evaluates on unseen data  
- Detects overfitting  
- Tunes hyperparameters  

---

## 8.4 Testing Phase

- Final performance validation  
- Simulates real-world behavior  
- Ensures production readiness  

Only fully validated models proceed to deployment.

---

## 8.5 Live Endpoint API

- Exposes trained model via FastAPI  
- Accepts live tick data  
- Returns structured trade signals  
- Used by MT5 and trading systems  

---

# 9️⃣ Symbol Backtest & Trading System (Optional)

This layer enables strategy evaluation before live deployment.

## 9.1 MT5 Historical Backtest

- Uses MT5 historical API  
- Tests model-generated signals  

## 9.2 Walk Forward Validation

- Rolling window evaluation  
- Mimics production retraining cycles  
- Reduces regime overfitting  

## 9.3 Deployment with Model

After passing walk-forward validation:

- Model integrated into trading execution layer  

---

# 🔟 Deployment Layer

## 10.1 MT5 Deployment

- Models deployed to MetaTrader 5  
- Signals streamed from Live Endpoint API  

## 10.2 Live / Demo Execution

- Automated trade execution  
- Demo or live mode  
- Execution logic separated from model logic  

---

# 1️⃣1️⃣ Model & Experiment Tracking

## MLflow

Tracks:

- Model versions  
- Hyperparameters  
- Training metrics  
- Artifacts  

**Provides:**

- Full audit trail  
- Experiment comparison  
- Reproducibility  

---

# 1️⃣2️⃣ Design Principles

- **Reproducibility:** Raw data preserved  
- **Scalability:** Distributed processing with Spark  
- **Separation of Concerns:** Data, ML, and execution isolated  
- **Observability:** Monitoring at all stages  
- **Flexibility:** Components can be upgraded independently  

---

# 1️⃣3️⃣ Future Enhancements

Planned improvements:

- Alternative data sources  
- Advanced feature engineering  
- Improved model architectures  
- Risk management layers  
- Enhanced monitoring dashboards  

This architecture represents a structured, evolving ML trading system built for continuous research and production deployment.

**YouTube Channel:** [https://www.youtube.com/@BDB5905](https://www.youtube.com/@BDB5905) 
