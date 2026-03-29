"""Document processor for the Learning Buddy use case."""

import os
import re
import time
from pathlib import Path

import requests
import yaml
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, udf
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from learning_buddy.config import LearningBuddyProjectConfig


class LearningBuddyDocumentProcessor:
    """Processes learning materials from volume into searchable chunks.

    Implements the DocumentProcessor protocol:
      run() → _sync_courses() → _download_materials() → _parse_documents()
             → _process_chunks() → _mark_processed()
    """

    def __init__(self, config: LearningBuddyProjectConfig) -> None:
        """Initialize with project configuration.

        Args:
            config: LearningBuddyProjectConfig with catalog, schema, volume, courses_path
        """
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.spark = SparkSession.getActiveSession()

        self.run_ts = time.strftime("%Y%m%d%H%M", time.gmtime())
        self.pdf_dir = f"/Volumes/{self.catalog}/{self.schema}/{self.cfg.volume}/{self.run_ts}"

        self.materials_table = f"{self.catalog}.{self.schema}.learning_materials"
        self.parsed_table = f"{self.catalog}.{self.schema}.learning_materials_parsed"
        self.chunks_table = f"{self.catalog}.{self.schema}.learning_materials_chunks"

    def run(self) -> None:
        """Run the full processing pipeline."""
        self._sync_courses()
        self._download_materials()
        self._parse_documents()
        self._process_chunks()
        self._mark_processed()

    def _resolve_courses_path(self) -> Path:
        """Resolve courses_path using 3-level parent search from cwd."""
        courses_path = self.cfg.courses_path
        p = Path(courses_path)
        if p.is_absolute():
            return p
        current = Path.cwd()
        for _ in range(3):
            candidate = current / courses_path
            if candidate.exists():
                return candidate
            current = current.parent
        raise FileNotFoundError(f"courses YAML not found: {courses_path} (searched from {Path.cwd()})")

    def _sync_courses(self) -> None:
        """Sync courses from YAML into learning_materials table.

        Reads courses_path YAML, builds rows from contents + exercises,
        creates the table if it doesn't exist, then MERGEs new rows in.
        Existing rows (including processed=true) are left untouched.
        """
        courses_file = self._resolve_courses_path()
        with open(courses_file) as f:
            data = yaml.safe_load(f)

        rows = []
        for course in data.get("courses", []):
            course_id = course["id"]
            language = course.get("language", "")
            source_url = course.get("source_url", "")

            for item in course.get("contents", []) + course.get("exercises", []):
                rows.append(
                    {
                        "material_id": item["material_id"],
                        "course": course_id,
                        "language": language,
                        "source_url": source_url,
                        "document_type": item.get("document_type", ""),
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "description": item.get("description", ""),
                    }
                )

        logger.info(f"Loaded {len(rows)} materials from {courses_file}")

        # Create table with BooleanType for processed
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.materials_table} (
                material_id STRING NOT NULL,
                course STRING,
                language STRING,
                source_url STRING,
                document_type STRING,
                title STRING,
                url STRING,
                description STRING,
                volume_path STRING,
                processed BOOLEAN
            )
        """)

        if not rows:
            logger.warning("No materials found in courses YAML — nothing to sync")
            return

        new_df = self.spark.createDataFrame(rows)
        new_df.createOrReplaceTempView("courses_source")

        self.spark.sql(f"""
            MERGE INTO {self.materials_table} target
            USING courses_source source
            ON target.material_id = source.material_id
            WHEN NOT MATCHED THEN INSERT (
                material_id, course, language, source_url,
                document_type, title, url, description,
                volume_path, processed
            ) VALUES (
                source.material_id, source.course, source.language, source.source_url,
                source.document_type, source.title, source.url, source.description,
                NULL, NULL
            )
        """)

        logger.info(f"Synced materials into {self.materials_table}")

    def _download_materials(self) -> None:
        """Download PDFs for unprocessed materials into a timestamped subfolder.

        Creates self.pdf_dir, downloads each unprocessed PDF there as
        {material_id}.pdf, and updates volume_path on success.
        Skips non-PDF responses and logs a warning on failure.
        """
        unprocessed = self.spark.sql(f"""
            SELECT material_id, url
            FROM {self.materials_table}
            WHERE (processed IS NULL OR processed = false)
              AND url IS NOT NULL AND url != ''
        """).collect()

        if not unprocessed:
            logger.info("No unprocessed materials to download")
            return

        os.makedirs(self.pdf_dir, exist_ok=True)
        logger.info(f"Downloading {len(unprocessed)} PDFs to {self.pdf_dir}")

        for row in unprocessed:
            material_id = row["material_id"]
            url = row["url"]
            dest_path = f"{self.pdf_dir}/{material_id}.pdf"

            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower():
                    logger.warning(f"Skipping {material_id}: expected PDF but got {content_type} from {url}")
                    continue
                with open(dest_path, "wb") as f:
                    f.write(response.content)
                self.spark.sql(f"""
                    UPDATE {self.materials_table}
                    SET volume_path = '{dest_path}'
                    WHERE material_id = '{material_id}'
                """)
                logger.info(f"Downloaded {material_id} → {dest_path}")
            except Exception as e:
                logger.warning(f"Failed to download {material_id} from {url}: {e}")

    def _parse_documents(self) -> None:
        """Parse PDFs from the timestamped run directory using ai_parse_document.

        Reads all files from self.pdf_dir (the folder populated by
        _download_materials), derives material_id from the filename,
        and inserts parsed results into learning_materials_parsed.
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
                regexp_extract(path, '([^/]+)\\.pdf$', 1) AS material_id,
                path AS volume_path,
                ai_parse_document(content) AS parsed_content,
                current_timestamp() AS parse_ts
            FROM READ_FILES(
                '{self.pdf_dir}',
                format => 'binaryFile'
            )
        """)

        logger.info(f"Parsed documents written to {self.parsed_table}")

    def _process_chunks(self) -> None:
        """Extract and clean chunks; MERGE into learning_materials_chunks.

        Applies UDFs to split parsed JSON into text chunks, then merges
        into the chunks table (with Change Data Feed enabled for delta sync).
        """
        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), True),
                    StructField("content", StringType(), True),
                ]
            )
        )
        extract_udf = udf(self._extract_chunks, chunk_schema)
        clean_udf = udf(self._clean_chunk, StringType())

        parsed_df = self.spark.table(self.parsed_table)
        materials_df = self.spark.table(self.materials_table).select("material_id", "course", "document_type", "language", "title")

        from pyspark.sql.functions import concat_ws, explode

        chunks_df = (
            parsed_df.withColumn("chunks", extract_udf(col("parsed_content")))
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
        """Mark ingested materials as processed in learning_materials.

        Only marks rows where volume_path is set, ensuring rows that failed
        to download are retried on the next run.
        """
        self.spark.sql(f"""
            UPDATE {self.materials_table}
            SET processed = true
            WHERE volume_path IS NOT NULL
              AND (processed IS NULL OR processed = false)
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
