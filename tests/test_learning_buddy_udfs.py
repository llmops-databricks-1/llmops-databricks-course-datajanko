"""Unit tests for LearningBuddyDocumentProcessor static UDFs.

These tests run without Spark — the UDFs are pure Python functions.
"""

import json

from learning_buddy.data_processor import LearningBuddyDocumentProcessor


class TestExtractChunks:
    """Tests for _extract_chunks."""

    def test_extracts_text_elements(self) -> None:
        doc = {
            "document": {
                "elements": [
                    {"type": "text", "id": "e1", "content": "Hello world"},
                    {"type": "image", "id": "e2", "content": "ignored"},
                    {"type": "text", "id": "e3", "content": "Second chunk"},
                ]
            }
        }
        result = LearningBuddyDocumentProcessor._extract_chunks(json.dumps(doc))
        assert len(result) == 2
        assert result[0] == ("e1", "Hello world")
        assert result[1] == ("e3", "Second chunk")

    def test_skips_blank_content(self) -> None:
        doc = {
            "document": {
                "elements": [
                    {"type": "text", "id": "e1", "content": "   "},
                    {"type": "text", "id": "e2", "content": "Real content"},
                ]
            }
        }
        result = LearningBuddyDocumentProcessor._extract_chunks(json.dumps(doc))
        assert len(result) == 1
        assert result[0][1] == "Real content"

    def test_returns_empty_for_empty_string(self) -> None:
        assert LearningBuddyDocumentProcessor._extract_chunks("") == []

    def test_returns_empty_for_invalid_json(self) -> None:
        assert LearningBuddyDocumentProcessor._extract_chunks("not json") == []

    def test_returns_empty_for_none(self) -> None:
        assert LearningBuddyDocumentProcessor._extract_chunks(None) == []  # type: ignore[arg-type]

    def test_empty_document(self) -> None:
        doc = {"document": {"elements": []}}
        assert LearningBuddyDocumentProcessor._extract_chunks(json.dumps(doc)) == []


class TestCleanChunk:
    """Tests for _clean_chunk."""

    def test_fixes_hyphenation_across_line_breaks(self) -> None:
        text = "docu-\nments"
        assert LearningBuddyDocumentProcessor._clean_chunk(text) == "documents"

    def test_collapses_internal_newlines(self) -> None:
        text = "line one\nline two"
        assert LearningBuddyDocumentProcessor._clean_chunk(text) == "line one line two"

    def test_collapses_repeated_whitespace(self) -> None:
        text = "too   many    spaces"
        assert LearningBuddyDocumentProcessor._clean_chunk(text) == "too many spaces"

    def test_strips_leading_trailing_whitespace(self) -> None:
        text = "  hello  "
        assert LearningBuddyDocumentProcessor._clean_chunk(text) == "hello"

    def test_returns_empty_string_for_empty_input(self) -> None:
        assert LearningBuddyDocumentProcessor._clean_chunk("") == ""

    def test_returns_empty_string_for_none(self) -> None:
        assert LearningBuddyDocumentProcessor._clean_chunk(None) == ""  # type: ignore[arg-type]

    def test_real_math_text(self) -> None:
        text = "Inte-\ngration by parts:\n∫u dv = uv −  ∫v du"
        result = LearningBuddyDocumentProcessor._clean_chunk(text)
        assert "Integration" in result
        assert "\n" not in result
