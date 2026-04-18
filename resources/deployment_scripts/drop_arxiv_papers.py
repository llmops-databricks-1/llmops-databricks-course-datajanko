# Databricks notebook source
from loguru import logger
from pyspark.sql import SparkSession

from commons.config import get_env, load_config

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("arxiv_config.yml", env=env)

table = f"{cfg.catalog}.{cfg.schema}.arxiv_papers"

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {table}")
logger.info(f"Dropped {table}")
