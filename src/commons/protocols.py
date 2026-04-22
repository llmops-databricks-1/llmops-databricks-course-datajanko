"""Shared protocols for all use cases."""

from typing import Protocol


class DocumentProcessor(Protocol):
    """Protocol for document processing pipelines."""

    def run(self) -> None: ...


class VectorSearchManager(Protocol):
    """Protocol for vector search management."""

    def create_endpoint_if_not_exists(self) -> None: ...

    def create_or_get_index(self) -> None: ...

    def sync(self) -> None: ...

    def search(self, query_text: str, num_results: int) -> list: ...
