"""Configuration management for Arxiv Curator.

Re-exports common configuration from the commons module for backward compatibility.
"""

from commons.config import (
    ChunkingConfig,
    ModelConfig,
    ProjectConfig,
    VectorSearchConfig,
    get_env,
    load_config,
)

__all__ = [
    "ProjectConfig",
    "ModelConfig",
    "VectorSearchConfig",
    "ChunkingConfig",
    "load_config",
    "get_env",
]
