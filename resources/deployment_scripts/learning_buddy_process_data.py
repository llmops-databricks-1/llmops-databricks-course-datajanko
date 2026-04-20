# Databricks notebook source
# MAGIC %md
# MAGIC # Learning Buddy — Data Processing Pipeline
# MAGIC
# MAGIC This notebook processes learning materials and syncs the vector search index.
# MAGIC Runs on schedule to keep the homework buddy knowledge base up to date.
# MAGIC
# MAGIC Pipeline steps:
# MAGIC 1. Parse PDFs from volume with AI Parse Documents
# MAGIC 2. Extract and clean text chunks
# MAGIC 3. Sync vector search index

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from commons.config import get_env
from learning_buddy.config import LearningBuddyProjectConfig
from learning_buddy.data_processor import LearningBuddyDocumentProcessor
from learning_buddy.vector_search import LearningBuddyVectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = LearningBuddyProjectConfig.load(config_path="learning_buddy_config.yml", env=env)

logger.info("Configuration loaded:")
logger.info(f"  Environment: {env}")
logger.info(f"  Catalog: {cfg.catalog}")
logger.info(f"  Schema: {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Process Learning Materials

# COMMAND ----------

processor = LearningBuddyDocumentProcessor(config=cfg)
processor.run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Sync Vector Search Index

# COMMAND ----------

searcher = LearningBuddyVectorSearchManager(
    catalog=cfg.catalog,
    schema=cfg.schema,
    vector_search_endpoint=cfg.vector_search_endpoint,
    embedding_endpoint=cfg.embedding_endpoint,
    usage_policy_id=cfg.usage_policy_id,
)
searcher.sync()

logger.info("✓ Learning Buddy data processing pipeline complete!")
