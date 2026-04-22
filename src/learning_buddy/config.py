"""Learning Buddy configuration extending the shared ProjectConfig."""

from pydantic import Field

from commons.config import ProjectConfig


class LearningBuddyProjectConfig(ProjectConfig):
    """Extended project configuration for the Learning Buddy use case."""

    courses_path: str = Field(
        "learning_buddy_courses.yml",
        description="Path to courses YAML file (relative or absolute)",
    )


__all__ = ["LearningBuddyProjectConfig"]
