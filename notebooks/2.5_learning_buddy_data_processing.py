# Databricks notebook source
"""
Learning Buddy — Interactive Data Processing Validation

Step-by-step validation of the full data pipeline:
1. Parse PDFs from volume with AI Parse Documents
2. Chunk and clean parsed content
3. Merge chunks into learning_materials_chunks
4. Create/sync vector search index
5. Spot-check search quality
"""

# COMMAND ----------

from pyspark.sql import SparkSession

from commons.config import get_env, load_config
from learning_buddy.data_processor import LearningBuddyDocumentProcessor
from learning_buddy.vector_search import LearningBuddyVectorSearchManager

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../learning_buddy_config.yml", env)

print(f"Environment: {env}")
print(f"Catalog: {cfg.catalog} | Schema: {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Parse Learning Materials

# COMMAND ----------

processor = LearningBuddyDocumentProcessor(cfg)
processor._parse_documents()

parsed_count = spark.table(f"{cfg.catalog}.{cfg.schema}.learning_materials_parsed").count()
print(f"Parsed documents: {parsed_count}")
display(spark.table(f"{cfg.catalog}.{cfg.schema}.learning_materials_parsed").limit(5))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Process Chunks

# COMMAND ----------

processor._process_chunks()

chunks_count = spark.table(f"{cfg.catalog}.{cfg.schema}.learning_materials_chunks").count()
print(f"Total chunks: {chunks_count}")
display(spark.table(f"{cfg.catalog}.{cfg.schema}.learning_materials_chunks").limit(10))  # noqa: F821

# COMMAND ----------

processor._mark_processed()

unprocessed = spark.sql(f"""
    SELECT COUNT(*) as unprocessed
    FROM {cfg.catalog}.{cfg.schema}.learning_materials
    WHERE processed IS NULL OR processed = false
""").collect()[0][0]
print(f"Remaining unprocessed documents: {unprocessed}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Vector Search Setup

# COMMAND ----------

searcher = LearningBuddyVectorSearchManager(
    catalog=cfg.catalog,
    schema=cfg.schema,
    vector_search_endpoint=cfg.vector_search_endpoint,
    embedding_endpoint=cfg.embedding_endpoint,
    usage_policy_id=cfg.usage_policy_id,
)
searcher.create_endpoint_if_not_exists()
searcher.create_or_get_index()
searcher.sync()

print("Vector search index synced.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Spot-check Search Quality

# COMMAND ----------

results = searcher.search("integration by parts", num_results=5)
print("Search results for 'integration by parts':")
print(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("Pipeline validation complete!")
print(f"  Parsed documents : {parsed_count}")
print(f"  Chunks created   : {chunks_count}")
print(f"  Unprocessed left : {unprocessed}")
