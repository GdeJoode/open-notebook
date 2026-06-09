"""
Per-notebook ontology evolution and per-source pass-1 result models.

Backs Phase B.1b of the KG-quality track. The two tables behind these
models (``notebook_schema`` and ``pass1_results``) are declared SCHEMAFULL
in ``migrations/45.surrealql``; this module is the Pydantic mirror.

Design notes
============

* ``NotebookSchema`` is a *singleton-per-notebook* — the SurrealDB
  ``idx_notebook_schema_notebook UNIQUE`` index enforces one row per
  notebook. The repository's ``upsert`` is responsible for rewrite
  semantics; the model itself just carries data.

* ``Pass1Result`` is *append-only*. Each row records a single pass-1
  attempt against a source, including the LLM's schema choice and the
  uncovered-concept list that drives B.1c's extension proposals.

* Both models keep a free-form ``metadata`` bag mirroring the
  ``Source.metadata`` pattern from migration 43 — additive keys do not
  require a schema bump. The corresponding DB field is
  ``FLEXIBLE option<object>``; legacy rows without the field deserialise
  via the Pydantic default plus the ``ensure_metadata_dict`` validator.

* Extension-shaped arrays (``accepted_extensions``,
  ``pending_extensions``, ``uncovered_concepts``, ``proposed_extensions``,
  ``alternative_schemas``) are typed as ``List[Dict[str, Any]]`` so the
  exact dict shape can evolve in code without forcing a migration. The
  DB columns are FLEXIBLE arrays for the same reason.
"""

from typing import Any, ClassVar, Dict, List, Optional

from pydantic import Field, field_validator

from shared.models.base import ObjectModel


class NotebookSchema(ObjectModel):
    """
    Per-notebook ontology state and review toggles.

    One row per notebook (UNIQUE index on ``notebook``). Tracks the
    chosen base ontology, the rolling coverage average across all
    processed sources, accepted and pending schema extensions, and the
    per-notebook review/soft-nudge toggles wired in B.3c.

    The ``notebook`` field stores the SurrealDB record ID of the parent
    notebook (e.g. ``"notebook:abc123"``). The repository layer is
    responsible for converting that string into a ``RecordID`` when
    writing.
    """

    table_name: ClassVar[str] = "notebook_schema"

    notebook: str = Field(
        description="Record ID of the owning notebook (e.g. 'notebook:abc123')"
    )
    base_ontology: str = Field(
        description="Name of the YAML ontology in use (e.g. 'scholarly')"
    )
    accepted_extensions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Extensions the user has approved into this notebook's "
            "ontology. Each dict is shaped roughly "
            "{type_name, parent_type, properties[]}; kept FLEXIBLE on "
            "the DB side so the exact shape can evolve."
        ),
    )
    pending_extensions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extensions proposed by pass-1 awaiting user decision.",
    )
    coverage_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Rolling-average coverage across all sources processed in "
            "this notebook (0.0–1.0). Drives the B.3c soft-nudge."
        ),
    )
    review_required: bool = Field(
        default=False,
        description=(
            "Per-notebook pause toggle — when true, B.1c routes every "
            "pass-1 result to the review queue regardless of confidence."
        ),
    )
    soft_nudge_dismissed: bool = Field(
        default=False,
        description=(
            "User has dismissed the low-coverage soft-nudge for this "
            "notebook. Re-arms when coverage drops below threshold "
            "again (logic lives in B.3c)."
        ),
    )
    excluded_types: List[str] = Field(
        default_factory=list,
        description=(
            "Soft-delete list of type names the user has hidden from "
            "the effective schema. The base YAML is never mutated; "
            "downstream consumers (TTL export, SchemaBrowser, Pass 2 "
            "extraction) filter against this list. Populated by the "
            "B.3b DELETE /schema/types/{type_name} endpoint."
        ),
    )
    last_modified_by: Optional[str] = Field(
        default=None,
        description="User id of the last editor; populated once edit-ops land in B.3b.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form extension bag for additive keys without further "
            "migrations. Mirrors the migration-43 pattern on Source."
        ),
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def ensure_metadata_dict(cls, v: Any) -> Dict[str, Any]:
        """Coerce a missing/null/non-dict metadata payload to ``{}``.

        Records written before this field existed read back as NONE
        from SurrealDB; coerce defensively so deserialisation never
        rejects an otherwise-valid row.
        """
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("excluded_types", mode="before")
    @classmethod
    def ensure_excluded_types_list(cls, v: Any) -> List[str]:
        """Coerce a missing/null excluded_types payload to ``[]``.

        The DB column is ``option<array<string>>`` so rows that pre-date
        migration 46 read back as NONE. Same defensive pattern as the
        metadata validator above.
        """
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item) for item in v if isinstance(item, (str, int, float))]
        return []


class Pass1Result(ObjectModel):
    """
    Append-only record of a single pass-1 attempt against a source.

    Captures the LLM's schema choice, its confidence, the resulting
    coverage and the list of uncovered concepts that downstream
    extension-proposal logic (B.1c) consumes. There is no unique-index
    on ``source`` — re-processing a source produces a new row.
    """

    table_name: ClassVar[str] = "pass1_results"

    source: str = Field(
        description="Record ID of the source under analysis (e.g. 'source:xyz')."
    )
    notebook: str = Field(
        description="Record ID of the owning notebook."
    )
    schema_attempted: str = Field(
        description="Base ontology the pass attempted to apply (e.g. 'scholarly')."
    )
    detected_schema: str = Field(
        description="Schema the LLM judged the source best matches."
    )
    confidence_in_choice: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM's confidence in ``detected_schema`` (0.0–1.0).",
    )
    coverage_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of source concepts that fit the attempted schema "
            "(0.0–1.0). Used to update the notebook's rolling average."
        ),
    )
    uncovered_concepts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Surface forms the LLM could not place under the attempted "
            "schema, with a suggested type. Each dict is roughly "
            "{surface_form, suggested_type}; FLEXIBLE on the DB side."
        ),
    )
    proposed_extensions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extension proposals derived from ``uncovered_concepts``.",
    )
    alternative_schemas: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top-3 fallback ontologies considered by the LLM.",
    )
    pass1_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form extension bag for additive keys without further "
            "migrations. Mirrors the migration-43 pattern."
        ),
    )

    @field_validator("pass1_metadata", mode="before")
    @classmethod
    def ensure_pass1_metadata_dict(cls, v: Any) -> Dict[str, Any]:
        """Coerce missing/null pass1_metadata to ``{}`` (see NotebookSchema)."""
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        return {}
