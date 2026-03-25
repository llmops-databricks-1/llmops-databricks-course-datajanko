"""Document processor for the Learning Buddy use case."""

import re

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, udf
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from commons.config import ProjectConfig


class LearningBuddyDocumentProcessor:
    """Processes learning materials from volume into searchable chunks.

    Implements the DocumentProcessor protocol:
      run() → _parse_documents() → _process_chunks() → marks processed=true
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialize with project configuration.

        Args:
            config: ProjectConfig with catalog, schema, volume settings
        """
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.spark = SparkSession.getActiveSession()

        self.materials_table = f"{self.catalog}.{self.schema}.learning_materials"
        self.parsed_table = f"{self.catalog}.{self.schema}.learning_materials_parsed"
        self.chunks_table = f"{self.catalog}.{self.schema}.learning_materials_chunks"

    def run(self) -> None:
        """Run the full processing pipeline."""
        self._parse_documents()
        self._process_chunks()
        self._mark_processed()

    def _parse_documents(self) -> None:
        """Parse PDFs from volume using ai_parse_document.

        Reads unprocessed rows from learning_materials, calls ai_parse_document
        on each volume_path, and writes results to learning_materials_parsed.
        """
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table} (
                material_id STRING,
                volume_path STRING,
                parsed_content STRING,
                parse_ts TIMESTAMP
            )
        """)

        self.spark.sql(f"""
            INSERT INTO {self.parsed_table}
            SELECT
                material_id,
                volume_path,
                ai_parse_document(content) AS parsed_content,
                current_timestamp() AS parse_ts
            FROM (
                SELECT m.material_id, m.volume_path,
                       f.content
                FROM {self.materials_table} m
                JOIN (
                    SELECT path, content
                    FROM READ_FILES(
                        '/Volumes/{self.catalog}/{self.schema}/{self.cfg.volume}',
                        format => 'binaryFile'
                    )
                ) f ON f.path = m.volume_path
                WHERE m.processed IS NULL OR m.processed = false
            )
        """)

        logger.info(f"Parsed documents written to {self.parsed_table}")

    def _process_chunks(self) -> None:
        """Extract and clean chunks; MERGE into learning_materials_chunks.

        Applies UDFs to split parsed JSON into text chunks, then merges
        into the chunks table (with Change Data Feed enabled for delta sync).
        """
        chunk_schema = ArrayType(
            StructType([
                StructField("chunk_id", StringType(), True),
                StructField("content", StringType(), True),
            ])
        )
        extract_udf = udf(self._extract_chunks, chunk_schema)
        clean_udf = udf(self._clean_chunk, StringType())

        parsed_df = self.spark.table(self.parsed_table)
        materials_df = self.spark.table(self.materials_table).select(
            "material_id", "course", "document_type", "language", "title"
        )

        from pyspark.sql.functions import concat_ws, explode

        chunks_df = (
            parsed_df
            .withColumn("chunks", extract_udf(col("parsed_content")))
            .withColumn("chunk", explode(col("chunks")))
            .select(
                concat_ws("_", col("material_id"), col("chunk.chunk_id")).alias("chunk_id"),
                col("material_id"),
                clean_udf(col("chunk.content")).alias("text"),
                current_timestamp().alias("created_ts"),
            )
            .join(materials_df, "material_id", "left")
        )

        # Create table if needed (with CDF enabled)
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                chunk_id STRING,
                material_id STRING,
                text STRING,
                created_ts TIMESTAMP,
                course STRING,
                document_type STRING,
                language STRING,
                title STRING
            )
            TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

        chunks_df.createOrReplaceTempView("new_chunks")
        self.spark.sql(f"""
            MERGE INTO {self.chunks_table} target
            USING new_chunks source
            ON target.chunk_id = source.chunk_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)

        # Ensure CDF is enabled on existing tables
        self.spark.sql(f"""
            ALTER TABLE {self.chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

        logger.info(f"Chunks merged into {self.chunks_table}")

    def _mark_processed(self) -> None:
        """Mark ingested materials as processed in learning_materials."""
        self.spark.sql(f"""
            UPDATE {self.materials_table}
            SET processed = true
            WHERE processed IS NULL OR processed = false
        """)
        logger.info("Marked learning_materials rows as processed=true")

    @staticmethod
    def _extract_chunks(parsed_content_json: str) -> list[tuple[str, str]]:
        """Extract text chunks from ai_parse_document JSON output.

        Args:
            parsed_content_json: JSON string from ai_parse_document

        Returns:
            List of (chunk_id, content) tuples
        """
        import json

        if not parsed_content_json:
            return []
        try:
            parsed = json.loads(parsed_content_json)
        except (json.JSONDecodeError, TypeError):
            return []

        chunks = []
        for element in parsed.get("document", {}).get("elements", []):
            if element.get("type") == "text":
                chunk_id = element.get("id", "")
                content = element.get("content", "")
                if content.strip():
                    chunks.append((chunk_id, content))
        return chunks

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """Clean and normalize chunk text.

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        if not text:
            return ""
        # Fix hyphenation across line breaks: "docu-\nments" => "documents"
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)
        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)
        return t.strip()
