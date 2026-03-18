# Databricks notebook source
"""
Learning Buddy - Metadata Ingestion

This notebook ingests course metadata (lecture notes and homework assignments)
from two sources:
1. Analysis 1 (Bielefeld, German) - 1 lecture + 7 homework sets (0-6)
2. Real Analysis (MIT, English) - 1 lecture + 12 homework assignments (1-12)

v1 Focus: Metadata ingestion only (URLs and document references).
Future phases will add PDF text extraction, chunking, and embeddings.
"""

from datetime import datetime

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from arxiv_curator.config import get_env, load_config

# COMMAND ----------
# Create Spark session and load config
spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
TABLE_NAME = "learning_materials"

# Set ingestion timestamp at job start
ingestion_timestamp = datetime.now().isoformat()

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} ready")
logger.info(f"Ingestion timestamp: {ingestion_timestamp}")

# COMMAND ----------
# Define course metadata

# Bielefeld Analysis 1 (German)
bielefeld_materials = [
    {
        "material_id": "bielefeld_a1_lecture",
        "course": "bielefeld_a1",
        "document_type": "lecture",
        "language": "de",
        "title": "Analysis 1 - Lecture Script",
        "url": "https://www.math.uni-bielefeld.de/~grigor/a1lect.pdf",
        "homework_set_number": None,
        "authors": None,
        "description": "Complete lecture script for Analysis 1 (German)",
        "source_url": "https://www.math.uni-bielefeld.de/~grigor/a1ws2024-25.htm",
        "ingestion_timestamp": ingestion_timestamp,
        "processed": None,
        "volume_path": None,
        "metadata": None,
    }
]

# Add Bielefeld homework sets 0-6
for hw_num in range(7):
    bielefeld_materials.append({
        "material_id": f"bielefeld_a1_hw{hw_num}",
        "course": "bielefeld_a1",
        "document_type": "homework",
        "language": "de",
        "title": f"Analysis 1 - Homework Set {hw_num}",
        "url": f"https://www.math.uni-bielefeld.de/~grigor/a1b{hw_num}.pdf",
        "homework_set_number": hw_num,
        "authors": None,
        "description": f"Homework assignment set {hw_num} for Analysis 1",
        "source_url": "https://www.math.uni-bielefeld.de/~grigor/a1ws2024-25.htm",
        "ingestion_timestamp": ingestion_timestamp,
        "processed": None,
        "volume_path": None,
        "metadata": None,
    })

# MIT Real Analysis (English)
mit_materials = [
    {
        "material_id": "mit_18_100a_lecture",
        "course": "mit_18_100a",
        "document_type": "lecture",
        "language": "en",
        "title": "Real Analysis - Lecture Notes Full",
        "url": "https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/mit18_100af20_lec_full2.pdf",
        "homework_set_number": None,
        "authors": None,
        "description": "Complete lecture notes for 18.100A Real Analysis (Fall 2020)",
        "source_url": "https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/pages/lecture-notes-and-readings/",
        "ingestion_timestamp": ingestion_timestamp,
        "processed": None,
        "volume_path": None,
        "metadata": None,
    }
]

# Add MIT homework assignments 1-12
for hw_num in range(1, 13):
    mit_materials.append({
        "material_id": f"mit_18_100a_hw{hw_num}",
        "course": "mit_18_100a",
        "document_type": "homework",
        "language": "en",
        "title": f"Real Analysis - Homework {hw_num}",
        "url": f"https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/mit18_100af20_hw{hw_num}.pdf",
        "homework_set_number": hw_num,
        "authors": None,
        "description": f"Homework assignment {hw_num} for 18.100A Real Analysis",
        "source_url": "https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/pages/lecture-notes-and-readings/",
        "ingestion_timestamp": ingestion_timestamp,
        "processed": None,
        "volume_path": None,
        "metadata": None,
    })

# Combine all materials
all_materials = bielefeld_materials + mit_materials

logger.info(f"Total materials to ingest: {len(all_materials)}")
logger.info(f"  - Bielefeld: {len(bielefeld_materials)} documents")
logger.info(f"  - MIT: {len(mit_materials)} documents")

# COMMAND ----------
# Create Delta table schema
# Columns designed with future extensibility in mind for PDF extraction, chunking, embeddings

schema = StructType([
    StructField("material_id", StringType(), False),
    StructField("course", StringType(), False),
    StructField("document_type", StringType(), False),
    StructField("language", StringType(), False),
    StructField("title", StringType(), False),
    StructField("url", StringType(), False),
    StructField("homework_set_number", IntegerType(), True),
    StructField("authors", ArrayType(StringType()), True),
    StructField("description", StringType(), True),
    StructField("source_url", StringType(), True),
    StructField("ingestion_timestamp", StringType(), False),
    StructField("processed", LongType(), True),
    StructField("volume_path", StringType(), True),
    StructField("metadata", StringType(), True),
])

# COMMAND ----------
# Create DataFrame and write to Delta table

df = spark.createDataFrame(all_materials, schema=schema)

full_table_name = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .saveAsTable(full_table_name)

logger.info(f"Successfully wrote {len(all_materials)} materials to {full_table_name}")

# COMMAND ----------
# Verify ingestion and log statistics

result_df = spark.sql(f"SELECT COUNT(*) as total_count FROM {full_table_name}")
total_count = result_df.collect()[0][0]

by_course = spark.sql(f"""
    SELECT course, COUNT(*) as count
    FROM {full_table_name}
    GROUP BY course
    ORDER BY course
""")

by_type = spark.sql(f"""
    SELECT document_type, COUNT(*) as count
    FROM {full_table_name}
    GROUP BY document_type
    ORDER BY document_type
""")

logger.info(f"Ingestion complete!")
logger.info(f"Total records: {total_count}")
logger.info("By course:")
for row in by_course.collect():
    logger.info(f"  {row['course']}: {row['count']}")

logger.info("By document type:")
for row in by_type.collect():
    logger.info(f"  {row['document_type']}: {row['count']}")

# Display summary
display(by_course)
display(by_type)
