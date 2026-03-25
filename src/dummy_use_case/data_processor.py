"""Dummy document processor for testing orchestration without volume access."""

from loguru import logger
from pyspark.sql import Row, SparkSession

from commons.config import ProjectConfig


class DummyDocumentProcessor:
    """Writes hardcoded fake chunks directly to learning_materials_chunks.

    Bypasses ai_parse_document and volume reads so orchestration can be
    tested against a real vector search endpoint without PDF access.
    Implements the DocumentProcessor protocol.
    """

    FAKE_CHUNKS = [
        {
            "chunk_id": "dummy_001",
            "material_id": "dummy_material_001",
            "text": "Integration by parts is a technique for evaluating integrals: ∫u dv = uv − ∫v du.",
            "course": "dummy_course",
            "document_type": "lecture",
            "language": "en",
            "title": "Dummy Lecture Notes",
        },
        {
            "chunk_id": "dummy_002",
            "material_id": "dummy_material_001",
            "text": "The fundamental theorem of calculus links differentiation and integration.",
            "course": "dummy_course",
            "document_type": "lecture",
            "language": "en",
            "title": "Dummy Lecture Notes",
        },
        {
            "chunk_id": "dummy_003",
            "material_id": "dummy_material_002",
            "text": "Homework 1: Use integration by parts to evaluate ∫x·e^x dx.",
            "course": "dummy_course",
            "document_type": "homework",
            "language": "en",
            "title": "Dummy Homework 1",
        },
    ]

    def __init__(self, config: ProjectConfig) -> None:
        """Initialize with project configuration.

        Args:
            config: ProjectConfig with catalog and schema settings
        """
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema
        self.spark = SparkSession.getActiveSession()
        self.chunks_table = f"{self.catalog}.{self.schema}.learning_materials_chunks"

    def run(self) -> None:
        """Write fake chunks to learning_materials_chunks."""
        rows = [Row(**chunk) for chunk in self.FAKE_CHUNKS]
        df = self.spark.createDataFrame(rows)

        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                chunk_id STRING,
                material_id STRING,
                text STRING,
                course STRING,
                document_type STRING,
                language STRING,
                title STRING
            )
            TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

        df.createOrReplaceTempView("dummy_chunks")
        self.spark.sql(f"""
            MERGE INTO {self.chunks_table} target
            USING dummy_chunks source
            ON target.chunk_id = source.chunk_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)

        self.spark.sql(f"""
            ALTER TABLE {self.chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)

        logger.info(f"Wrote {len(self.FAKE_CHUNKS)} dummy chunks to {self.chunks_table}")
