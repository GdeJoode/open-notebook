"""Tests for API request/response schemas."""

import pytest
from pydantic import ValidationError

from app_main.api.schemas import (
    ModelCreate,
    NotebookCreate,
    NoteCreate,
    SearchRequest,
    SettingsUpdate,
    SourceCreate,
    TransformationCreate,
)


class TestNotebookCreate:

    def test_valid_create(self):
        schema = NotebookCreate(name="My Notebook")
        assert schema.name == "My Notebook"
        assert schema.description == ""

    def test_name_required(self):
        with pytest.raises(ValidationError):
            NotebookCreate()


class TestNoteCreate:

    def test_valid_create(self):
        schema = NoteCreate(content="Some content")
        assert schema.content == "Some content"
        assert schema.note_type == "human"
        assert schema.title is None
        assert schema.notebook_id is None

    def test_content_required(self):
        with pytest.raises(ValidationError):
            NoteCreate()


class TestSourceCreate:

    def test_type_required(self):
        with pytest.raises(ValidationError):
            SourceCreate()

    def test_validate_notebook_fields_both_raises(self):
        with pytest.raises(ValidationError, match="Cannot specify both"):
            SourceCreate(
                type="text",
                notebook_id="notebook:1",
                notebooks=["notebook:2"],
            )

    def test_notebook_id_converted_to_notebooks(self):
        schema = SourceCreate(type="text", notebook_id="notebook:1")
        assert schema.notebooks == ["notebook:1"]

    def test_no_notebooks_defaults_to_empty(self):
        schema = SourceCreate(type="text")
        assert schema.notebooks == []


class TestSearchRequest:

    def test_defaults(self):
        schema = SearchRequest(query="hello")
        assert schema.type == "text"
        assert schema.limit == 100
        assert schema.search_sources is True
        assert schema.search_notes is True


class TestModelCreate:

    def test_all_fields_required(self):
        with pytest.raises(ValidationError):
            ModelCreate(name="gpt-4o")

    def test_valid_create(self):
        schema = ModelCreate(name="gpt-4o", provider="openai", type="language")
        assert schema.name == "gpt-4o"


class TestTransformationCreate:

    def test_all_fields_required(self):
        with pytest.raises(ValidationError):
            TransformationCreate(name="test")

    def test_apply_default_defaults_to_false(self):
        schema = TransformationCreate(
            name="sum",
            title="Summarize",
            description="Summarize text",
            prompt="Summarize: {text}",
        )
        assert schema.apply_default is False


class TestSettingsUpdate:

    def test_all_optional(self):
        schema = SettingsUpdate()
        assert schema.default_embedding_option is None
