# Databricks notebook source
from typing import Any

from databricks.vector_search.reranker import DatabricksReranker
from loguru import logger
from pyspark.sql import SparkSession

from commons.config import get_env, load_config
from learning_buddy.vector_search import LearningBuddyVectorSearchManager

spark = SparkSession.builder.getOrCreate()

# Load configuration
env = get_env(spark)
cfg = load_config("learning_buddy_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema

# COMMAND ----------

vs_manager = LearningBuddyVectorSearchManager(config=cfg)

logger.info(f"Vector Search Endpoint: {vs_manager.endpoint_name}")
logger.info(f"Embedding Model: {vs_manager.embedding_model}")
logger.info(f"Index Name: {vs_manager.index_name}")

# COMMAND ----------

vs_manager.create_endpoint_if_not_exists()

# COMMAND ----------

vs_manager.client.list_endpoints()

# COMMAND ----------

index = vs_manager.create_or_get_index()

logger.info("\n✓ Vector search setup complete!")
logger.info(f"  Index: {vs_manager.index_name}")
logger.info(f"  Source: {vs_manager.source_table}")
logger.info(f"  Embedding Model: {vs_manager.embedding_model}")

# COMMAND ----------


def parse_vector_search_results(results: Any) -> list[dict[Any, Any]]:  # noqa ANN401
    """Parse vector search results from array format to dict format.

    Args:
        results: Raw results from similarity_search()

    Returns:
        List of dictionaries with column names as keys
    """
    columns = [col["name"] for col in results.get("manifest", {}).get("columns", [])]
    data_array = results.get("result", {}).get("data_array", [])

    return [dict(zip(columns, row_data)) for row_data in data_array]  # noqa B905


# COMMAND ----------

# MAGIC %md
# MAGIC # Similarity Search
# MAGIC Let's do a simple similarity search and check for the same query but in different languages

# COMMAND ----------

queries = ["Definition of a cauchy sequence", "Definition einer cauchy folge"]

for query in queries:
    results = index.similarity_search(
        query_text=query,
        columns=["text", "chunk_id", "course", "material_id", "title", "language", "document_type"],
        num_results=5,
    )

    logger.info(f"Query: {query}\n")
    logger.info("Top 5 Results:")
    logger.info("=" * 80)

    # Parse results using helper function
    for i, row in enumerate(parse_vector_search_results(results), 1):
        logger.info(f"\n{i}. text: {row.get('text', 'N/A')}")
        logger.info(f"   material ID: {row.get('material_id', 'N/A')}")
        logger.info(f"   Document Type: {row.get('document_type', 'N/A')}")
        logger.info(f"   Chunk ID: {row.get('chunk_id', 'N/A')}")
        logger.info(f"   Text preview: {row.get('text', '')[:200]}...")
        logger.info(f"   Course: {row.get('course', '')[:200]}...")
        logger.info(f"   Language: {row.get('language', '')[:200]}...")
        logger.info(f"   Title: {row.get('title', '')[:200]}...")
        logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")
        logger.info("=" * 80)
    logger.info(" " * 80)
    logger.info("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC # Filtered queries
# MAGIC For the definition of a cauchy sequence we only care about lectures

# COMMAND ----------

queries = ["Definition of a cauchy sequence", "Definition einer cauchy folge"]

for query in queries:
    results = index.similarity_search(
        query_text=query,
        columns=["text", "chunk_id", "course", "material_id", "title", "language", "document_type"],
        filters={"document_type": "lecture"},
        num_results=5,
    )

    logger.info(f"Query: {query}\n")
    logger.info("Top 5 Results:")
    logger.info("=" * 80)

    # Parse results using helper function
    for i, row in enumerate(parse_vector_search_results(results), 1):
        logger.info(f"\n{i}. text: {row.get('text', 'N/A')}")
        logger.info(f"   material ID: {row.get('material_id', 'N/A')}")
        logger.info(f"   Document Type: {row.get('document_type', 'N/A')}")
        logger.info(f"   Chunk ID: {row.get('chunk_id', 'N/A')}")
        logger.info(f"   Text preview: {row.get('text', '')[:200]}...")
        logger.info(f"   Course: {row.get('course', '')[:200]}...")
        logger.info(f"   Language: {row.get('language', '')[:200]}...")
        logger.info(f"   Title: {row.get('title', '')[:200]}...")
        logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")
        logger.info("=" * 80)
    logger.info(" " * 80)
    logger.info("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC # Hybrid search

# COMMAND ----------

queries = ["Definition of a cauchy sequence", "Definition einer cauchy folge"]

for query in queries:
    results = index.similarity_search(
        query_text=query,
        columns=["text", "chunk_id", "course", "material_id", "title", "language", "document_type"],
        query_type="hybrid",
        num_results=5,
    )

    logger.info(f"Query: {query}\n")
    logger.info("Top 5 Results:")
    logger.info("=" * 80)

    # Parse results using helper function
    for i, row in enumerate(parse_vector_search_results(results), 1):
        logger.info(f"\n{i}. text: {row.get('text', 'N/A')}")
        logger.info(f"   material ID: {row.get('material_id', 'N/A')}")
        logger.info(f"   Document Type: {row.get('document_type', 'N/A')}")
        logger.info(f"   Chunk ID: {row.get('chunk_id', 'N/A')}")
        logger.info(f"   Text preview: {row.get('text', '')[:200]}...")
        logger.info(f"   Course: {row.get('course', '')[:200]}...")
        logger.info(f"   Language: {row.get('language', '')[:200]}...")
        logger.info(f"   Title: {row.get('title', '')[:200]}...")
        logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")
        logger.info("=" * 80)
    logger.info(" " * 80)
    logger.info("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC # Hybrid Search with Reranking
# MAGIC
# MAGIC Super interesting to observe the effect of moving language around in the reranker.
# MAGIC Moving language to last spot, gives the definition in german instead of english

# COMMAND ----------

queries = ["Definition of a cauchy sequence", "Definition einer cauchy folge"]

for query in queries:
    results = index.similarity_search(
        query_text=query,
        columns=["text", "chunk_id", "course", "material_id", "title", "language", "document_type"],
        query_type="hybrid",
        reranker=DatabricksReranker(columns_to_rerank=["language", "document_type", "text", "course"]),
        num_results=5,
    )

    logger.info(f"Query: {query}\n")
    logger.info("Top 5 Results:")
    logger.info("=" * 80)

    # Parse results using helper function
    for i, row in enumerate(parse_vector_search_results(results), 1):
        logger.info(f"\n{i}. text: {row.get('text', 'N/A')}")
        logger.info(f"   material ID: {row.get('material_id', 'N/A')}")
        logger.info(f"   Document Type: {row.get('document_type', 'N/A')}")
        logger.info(f"   Chunk ID: {row.get('chunk_id', 'N/A')}")
        logger.info(f"   Text preview: {row.get('text', '')[:200]}...")
        logger.info(f"   Course: {row.get('course', '')[:200]}...")
        logger.info(f"   Language: {row.get('language', '')[:200]}...")
        logger.info(f"   Title: {row.get('title', '')[:200]}...")
        logger.info(f"   Score: {row.get('score', 'N/A'):.4f}")
        logger.info("=" * 80)
    logger.info(" " * 80)
    logger.info("-" * 80)

# COMMAND ----------
