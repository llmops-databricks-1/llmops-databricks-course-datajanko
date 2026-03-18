"""Configuration models for learning buddy courses and materials."""

from typing import Optional

from pydantic import BaseModel, Field


class CourseConfig(BaseModel):
    """Configuration for a single course (lecture + homework assignments)."""

    name: str = Field(..., description="Course name (e.g., 'Analysis 1', 'Real Analysis')")
    language: str = Field(..., description="Language code (e.g., 'de', 'en')")
    course_id: str = Field(..., description="Unique course identifier (e.g., 'bielefeld_a1', 'mit_18_100a')")
    description: Optional[str] = Field(None, description="Course description")
    lecture_url: str = Field(..., description="URL to the full lecture script/notes PDF")
    homework_urls: dict[int, str] = Field(
        ...,
        description="Mapping of homework set number to URL. Example: {0: 'http://...a1b0.pdf', 1: 'http://...a1b1.pdf'}"
    )


class LearningMaterialsConfig(BaseModel):
    """Top-level configuration for learning buddy with multiple courses."""

    courses: list[CourseConfig] = Field(..., description="List of configured courses")

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
