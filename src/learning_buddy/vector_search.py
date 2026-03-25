"""Vector search management for the Learning Buddy use case."""

from typing import Any

from databricks.vector_search.client import VectorSearchClient
from loguru import logger

from commons.config import ProjectConfig


class LearningBuddyVectorSearchManager:
    """Manages vector search endpoint and index for learning materials.

    Implements the VectorSearchManager protocol:
      create_endpoint_if_not_exists() / create_or_get_index() / sync() / search()

    Index: {catalog}.{schema}.learning_buddy_index
    Source: learning_materials_chunks (primary key: chunk_id, embedding column: text)
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialize with project configuration.

        Args:
            config: ProjectConfig with catalog, schema, endpoint settings
        """
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.endpoint_name = config.vector_search_endpoint
        self.embedding_model = config.embedding_endpoint

        self.client = VectorSearchClient()
        self.index_name = f"{self.catalog}.{self.schema}.learning_buddy_index"
        self.source_table = f"{self.catalog}.{self.schema}.learning_materials_chunks"

    def create_endpoint_if_not_exists(self) -> None:
        """Create vector search endpoint if it doesn't exist."""
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name, endpoint_type="STANDARD"
            )
            logger.info(f"✓ Vector search endpoint created: {self.endpoint_name}")
        else:
            logger.info(f"✓ Vector search endpoint exists: {self.endpoint_name}")

    def create_or_get_index(self) -> Any:
        """Create or retrieve the vector search index.

        Returns:
            Vector search index object
        """
        self.create_endpoint_if_not_exists()

        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return index
        except Exception:
            logger.info(f"Index {self.index_name} not found, will create it")

        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=self.source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="chunk_id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
            )
            logger.info(f"✓ Vector search index created: {self.index_name}")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return self.client.get_index(index_name=self.index_name)

    def sync(self) -> None:
        """Sync the vector search index with the source table."""
        index = self.create_or_get_index()
        logger.info(f"Syncing vector search index: {self.index_name}")
        index.sync()
        logger.info("✓ Index sync triggered")

    def search(self, query_text: str, num_results: int = 5) -> list:
        """Search the learning buddy index.

        Args:
            query_text: Natural language query
            num_results: Number of results to return

        Returns:
            List of matching result rows
        """
        index = self.client.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query_text,
            columns=["chunk_id", "material_id", "text", "course", "title"],
            num_results=num_results,
        )
        return results
