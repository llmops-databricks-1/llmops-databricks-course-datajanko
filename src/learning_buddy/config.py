"""Learning Buddy configuration (self-contained, no commons dependency)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(512)
    chunk_overlap: int = Field(50)
    separator: str = Field("\n\n")


class LearningBuddyProjectConfig(BaseModel):
    """Project configuration for the Learning Buddy use case."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    llm_endpoint: str = Field(..., description="LLM endpoint name")
    embedding_endpoint: str = Field(..., description="Embedding endpoint name")
    warehouse_id: str = Field(..., description="Warehouse ID")
    vector_search_endpoint: str = Field(..., description="Vector search endpoint name")
    genie_space_id: str | None = Field(None, description="Genie space ID")
    usage_policy_id: str = Field(..., description="Usage policy id")
    lakebase_project_id: str = Field(..., description="Lakebase project id")
    experiment_name: str = Field(..., description="MLflow experiment path")
    system_prompt: str = Field(
        default="You are a helpful AI assistant that helps users find and understand research papers.",
        description="System prompt for the agent",
    )
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    courses_path: str = Field(
        "learning_buddy_courses.yml",
        description="Path to courses YAML file (relative or absolute)",
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "LearningBuddyProjectConfig":
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")
        env_config = config_data[env]
        if "system_prompt" in config_data:
            env_config["system_prompt"] = config_data["system_prompt"]
        return cls(**env_config)

    @classmethod
    def load(cls, config_path: str = "project_config.yml", env: str = "dev") -> "LearningBuddyProjectConfig":
        if not Path(config_path).is_absolute():
            current = Path.cwd()
            for _ in range(3):
                candidate = current / config_path
                if candidate.exists():
                    config_path = str(candidate)
                    break
                current = current.parent
        return cls.from_yaml(config_path, env)

    @property
    def schema(self) -> str:
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.volume}"


def get_env(spark: SparkSession) -> str:
    """Get current environment from dbutils widget, falling back to 'dev'."""
    try:
        from pyspark.dbutils import DBUtils

        dbutils = DBUtils(spark)
        return dbutils.widgets.get("env")
    except Exception:
        return "dev"


__all__ = ["LearningBuddyProjectConfig", "get_env"]
