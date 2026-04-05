# Databricks notebook source
# MAGIC %md
# MAGIC # Reset Learning Buddy Tables
# MAGIC
# MAGIC Drops the three learning buddy Delta tables so they are recreated with the
# MAGIC correct schema on the next pipeline run.
# MAGIC
# MAGIC **When to run:** After a schema migration (e.g. `processed` column type change
# MAGIC from LongType → BooleanType). Run this job, then re-run the data pipeline.
# MAGIC
# MAGIC Tables dropped:
# MAGIC - `learning_materials`       — `processed` must be BooleanType
# MAGIC - `learning_materials_parsed`
# MAGIC - `learning_materials_chunks`

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from commons.config import get_env
from learning_buddy.config import LearningBuddyProjectConfig

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = LearningBuddyProjectConfig.load(config_path="learning_buddy_config.yml", env=env)

catalog = cfg.catalog
schema = cfg.schema

logger.info(f"Resetting learning buddy tables in {catalog}.{schema} (env={env})")

# COMMAND ----------

tables = [
    f"{catalog}.{schema}.learning_materials",
    f"{catalog}.{schema}.learning_materials_parsed",
    f"{catalog}.{schema}.learning_materials_chunks",
]

for table in tables:
    spark.sql(f"DROP TABLE IF EXISTS {table}")
    logger.info(f"Dropped table: {table}")

logger.info("Reset complete. Re-run the learning-buddy-data-pipeline job to repopulate.")
