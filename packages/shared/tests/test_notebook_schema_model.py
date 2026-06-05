"""Pydantic roundtrip tests for the Phase B.1b notebook_schema models.

Covers:

* Construction with required-only fields (defaults populate correctly).
* Construction with the full extension-rich payload (round-trips through
  ``model_dump`` + re-construction without loss).
* The ``ensure_metadata_dict`` validators coerce ``None`` to ``{}`` so
  legacy DB rows without the FLEXIBLE bag don't break deserialisation.
* ``table_name`` is set on both models — the repository layer relies on
  this ``ClassVar``.
"""

from typing import Any, Dict

from shared.models import NotebookSchema, Pass1Result


class TestNotebookSchema:
    """Tests for the per-notebook ontology state model."""

    def test_table_name(self) -> None:
        """``table_name`` matches the SCHEMAFULL table from migration 45."""
        assert NotebookSchema.table_name == "notebook_schema"

    def test_minimal_construction(self) -> None:
        """Only ``notebook`` and ``base_ontology`` are required."""
        schema = NotebookSchema(
            notebook="notebook:abc",
            base_ontology="scholarly",
        )
        assert schema.notebook == "notebook:abc"
        assert schema.base_ontology == "scholarly"
        assert schema.coverage_pct == 0.0
        assert schema.review_required is False
        assert schema.soft_nudge_dismissed is False
        assert schema.accepted_extensions == []
        assert schema.pending_extensions == []
        assert schema.metadata == {}
        assert schema.last_modified_by is None

    def test_full_roundtrip(self) -> None:
        """A populated schema survives model_dump → reconstruct unchanged."""
        original = NotebookSchema(
            notebook="notebook:abc",
            base_ontology="scholarly",
            accepted_extensions=[
                {
                    "extension_id": "ext-1",
                    "type_name": "Methodology",
                    "parent_type": "Concept",
                    "properties": ["name", "description"],
                }
            ],
            pending_extensions=[
                {
                    "extension_id": "ext-2",
                    "type_name": "Dataset",
                    "parent_type": "Resource",
                    "properties": ["name", "url"],
                }
            ],
            coverage_pct=0.85,
            review_required=True,
            soft_nudge_dismissed=True,
            last_modified_by="user:42",
            metadata={"freeform": {"nested": True}},
        )
        dumped = original.model_dump()
        rehydrated = NotebookSchema(**dumped)
        assert rehydrated == original

    def test_coverage_pct_bounds(self) -> None:
        """``coverage_pct`` clamps to the [0, 1] range via validator."""
        import pytest

        with pytest.raises(ValueError):
            NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                coverage_pct=1.5,
            )
        with pytest.raises(ValueError):
            NotebookSchema(
                notebook="notebook:abc",
                base_ontology="scholarly",
                coverage_pct=-0.1,
            )

    def test_metadata_none_coerced_to_empty_dict(self) -> None:
        """Legacy rows without the FLEXIBLE bag read back as None.

        The ``ensure_metadata_dict`` validator coerces that to ``{}`` so
        Pydantic deserialisation does not reject the row.
        """
        schema = NotebookSchema(
            notebook="notebook:abc",
            base_ontology="scholarly",
            metadata=None,  # type: ignore[arg-type]
        )
        assert schema.metadata == {}

    def test_metadata_invalid_type_coerced_to_empty_dict(self) -> None:
        """Non-dict metadata payloads coerce to ``{}`` defensively."""
        schema = NotebookSchema(
            notebook="notebook:abc",
            base_ontology="scholarly",
            metadata="not-a-dict",  # type: ignore[arg-type]
        )
        assert schema.metadata == {}


class TestPass1Result:
    """Tests for the append-only per-source pass-1 model."""

    def test_table_name(self) -> None:
        """``table_name`` matches the SCHEMAFULL table from migration 45."""
        assert Pass1Result.table_name == "pass1_results"

    def test_minimal_construction(self) -> None:
        """Required fields surface the smallest valid record."""
        result = Pass1Result(
            source="source:xyz",
            notebook="notebook:abc",
            schema_attempted="scholarly",
            detected_schema="scholarly",
        )
        assert result.source == "source:xyz"
        assert result.notebook == "notebook:abc"
        assert result.schema_attempted == "scholarly"
        assert result.detected_schema == "scholarly"
        assert result.confidence_in_choice == 0.0
        assert result.coverage_pct == 0.0
        assert result.uncovered_concepts == []
        assert result.proposed_extensions == []
        assert result.alternative_schemas == []
        assert result.pass1_metadata == {}

    def test_full_roundtrip(self) -> None:
        """A populated result survives model_dump → reconstruct unchanged."""
        uncovered: list[Dict[str, Any]] = [
            {"surface_form": "RAG", "suggested_type": "Methodology"}
        ]
        original = Pass1Result(
            source="source:xyz",
            notebook="notebook:abc",
            schema_attempted="scholarly",
            detected_schema="technical",
            confidence_in_choice=0.78,
            coverage_pct=0.62,
            uncovered_concepts=uncovered,
            proposed_extensions=[
                {
                    "extension_id": "ext-1",
                    "type_name": "Methodology",
                    "parent_type": "Concept",
                }
            ],
            alternative_schemas=[
                {"name": "scientific", "score": 0.55},
                {"name": "general", "score": 0.40},
            ],
            pass1_metadata={"llm_model": "gpt-4o", "duration_ms": 1234},
        )
        rehydrated = Pass1Result(**original.model_dump())
        assert rehydrated == original

    def test_confidence_bounds(self) -> None:
        """``confidence_in_choice`` is bounded to [0, 1]."""
        import pytest

        with pytest.raises(ValueError):
            Pass1Result(
                source="source:xyz",
                notebook="notebook:abc",
                schema_attempted="scholarly",
                detected_schema="scholarly",
                confidence_in_choice=1.01,
            )

    def test_pass1_metadata_none_coerced(self) -> None:
        """``pass1_metadata=None`` coerces to ``{}`` (legacy-row safety)."""
        result = Pass1Result(
            source="source:xyz",
            notebook="notebook:abc",
            schema_attempted="scholarly",
            detected_schema="scholarly",
            pass1_metadata=None,  # type: ignore[arg-type]
        )
        assert result.pass1_metadata == {}
