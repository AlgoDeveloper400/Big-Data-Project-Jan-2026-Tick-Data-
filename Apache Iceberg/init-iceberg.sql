-- init-iceberg.sql
-- Complete Iceberg JDBC catalog tables for Trino

-- 1. CRITICAL: Create namespace_properties table first (this was missing!)
CREATE TABLE IF NOT EXISTS iceberg_namespace_properties (
    catalog_name VARCHAR(255) NOT NULL,
    namespace VARCHAR(255) NOT NULL,
    property_key VARCHAR(255) NOT NULL,
    property_value TEXT,
    PRIMARY KEY (catalog_name, namespace, property_key)
);

-- 2. Main iceberg_tables table
CREATE TABLE IF NOT EXISTS iceberg_tables (
    catalog_name VARCHAR(255) NOT NULL,
    table_namespace VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    metadata_location TEXT,
    previous_metadata_location TEXT,
    PRIMARY KEY (catalog_name, table_namespace, table_name)
);

-- 3. Other Iceberg metadata tables
CREATE TABLE IF NOT EXISTS iceberg_columns (
    catalog_name VARCHAR(255) NOT NULL,
    table_namespace VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    column_type VARCHAR(255) NOT NULL,
    column_position INTEGER NOT NULL,
    PRIMARY KEY (catalog_name, table_namespace, table_name, column_name)
);

CREATE TABLE IF NOT EXISTS iceberg_partitions (
    catalog_name VARCHAR(255) NOT NULL,
    table_namespace VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    partition_spec_id INTEGER NOT NULL,
    partition_key VARCHAR(255) NOT NULL,
    transform VARCHAR(255) NOT NULL,
    column_name VARCHAR(255) NOT NULL,
    PRIMARY KEY (catalog_name, table_namespace, table_name, partition_spec_id, partition_key)
);

CREATE TABLE IF NOT EXISTS iceberg_snapshots (
    catalog_name VARCHAR(255) NOT NULL,
    table_namespace VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    snapshot_id BIGINT NOT NULL,
    parent_snapshot_id BIGINT,
    timestamp_ms BIGINT NOT NULL,
    manifest_list TEXT,
    summary TEXT,
    PRIMARY KEY (catalog_name, table_namespace, table_name, snapshot_id)
);

CREATE TABLE IF NOT EXISTS iceberg_manifest_files (
    catalog_name VARCHAR(255) NOT NULL,
    table_namespace VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    snapshot_id BIGINT NOT NULL,
    manifest_path TEXT NOT NULL,
    PRIMARY KEY (catalog_name, table_namespace, table_name, snapshot_id, manifest_path)
);

CREATE TABLE IF NOT EXISTS iceberg_data_files (
    catalog_name VARCHAR(255) NOT NULL,
    table_namespace VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    snapshot_id BIGINT NOT NULL,
    file_path TEXT NOT NULL,
    PRIMARY KEY (catalog_name, table_namespace, table_name, snapshot_id, file_path)
);

CREATE TABLE IF NOT EXISTS iceberg_refs (
    catalog_name VARCHAR(255) NOT NULL,
    table_namespace VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    ref_name VARCHAR(255) NOT NULL,
    snapshot_id BIGINT,
    type VARCHAR(255) NOT NULL,
    PRIMARY KEY (catalog_name, table_namespace, table_name, ref_name)
);

CREATE TABLE IF NOT EXISTS iceberg_properties (
    catalog_name VARCHAR(255) NOT NULL,
    table_namespace VARCHAR(255) NOT NULL,
    table_name VARCHAR(255) NOT NULL,
    property_key VARCHAR(255) NOT NULL,
    property_value TEXT,
    PRIMARY KEY (catalog_name, table_namespace, table_name, property_key)
);

-- 4. Create the gold schema
CREATE SCHEMA IF NOT EXISTS gold;

-- 5. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_iceberg_namespace_properties ON iceberg_namespace_properties(catalog_name, namespace);
CREATE INDEX IF NOT EXISTS idx_iceberg_tables ON iceberg_tables(catalog_name, table_namespace, table_name);
CREATE INDEX IF NOT EXISTS idx_iceberg_columns ON iceberg_columns(catalog_name, table_namespace, table_name);

-- 6. Insert initial namespace for the catalog
INSERT INTO iceberg_namespace_properties (catalog_name, namespace, property_key, property_value) 
VALUES ('catalog-dw', '', 'default', 'true') 
ON CONFLICT (catalog_name, namespace, property_key) DO NOTHING;
