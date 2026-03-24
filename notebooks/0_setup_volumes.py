# Databricks notebook source
"""
Setup Volumes for LLMOps Course

This notebook creates necessary volumes based on the configuration file.
It's designed to be run before other notebooks/jobs to ensure all volumes exist.

Parameters:
  config_path: Path to configuration file (arxiv_config.yml or learning_buddy_config.yml)
  env: Environment name (dev, acc, prd)
"""

from loguru import logger
from pyspark.sql import SparkSession

from commons.config import load_config

# COMMAND ----------

# Get parameters from Databricks widgets
dbutils.widgets.text("config_path", "arxiv_config.yml", "Config File Path")
dbutils.widgets.text("env", "dev", "Environment")

config_path = dbutils.widgets.get("config_path")
env = dbutils.widgets.get("env")

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Load configuration
logger.info(f"Loading configuration from: {config_path}")
cfg = load_config(config_path, env=env)

logger.info(f"Environment: {env}")
logger.info(f"Catalog: {cfg.catalog}")
logger.info(f"Schema: {cfg.db_schema}")
logger.info(f"Volume: {cfg.volume}")

# COMMAND ----------

# Create catalog if it doesn't exist
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {cfg.catalog}")
    logger.info(f"✓ Catalog '{cfg.catalog}' ready")
except Exception as e:
    logger.warning(f"Could not create catalog: {e}")

# COMMAND ----------

# Create schema if it doesn't exist
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{cfg.db_schema}")
    logger.info(f"✓ Schema '{cfg.catalog}.{cfg.db_schema}' ready")
except Exception as e:
    logger.warning(f"Could not create schema: {e}")

# COMMAND ----------

# Create volume if it doesn't exist
try:
    spark.sql(
        f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.db_schema}.{cfg.volume}"
    )
    logger.info(f"✓ Volume '{cfg.full_volume_path}' ready")
except Exception as e:
    logger.warning(f"Could not create volume: {e}")

# COMMAND ----------

logger.info("✓ Setup complete! All resources ready.")
logger.info(f"Full volume path: {cfg.full_volume_path}")
