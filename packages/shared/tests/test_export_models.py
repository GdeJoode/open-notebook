"""Tests for ``shared.models.export`` -- Track D shared contracts.

Phase D.0 ships these models as the single source of truth for the
three export formats (D.1/D.2/D.3). The tests pin the published
defaults from the roadmap, the round-trip serializability the
front-end depends on, and the closed-set NetworkX format choices.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.models import (
    ExportFilter,
    ExportReport,
    JsonlExportRequest,
    NetworkxExportRequest,
    ObsidianExportRequest,
)
from shared.models.export import NetworkxFormat


# ---------------------------------------------------------------------------
# ExportFilter defaults
# ---------------------------------------------------------------------------


class TestExportFilterDefaults:
    """Lock in the roadmap-§344 default thresholds.

    These defaults are user-visible (they decide what gets exported in
    the common one-click path) so they need a regression net.
    """

    def test_min_connections_default_is_five(self):
        assert ExportFilter().min_connections == 5

    def test_min_confidence_default_is_zero_point_nine(self):
        assert ExportFilter().min_confidence == 0.9

    def test_min_relation_confidence_defaults_to_none(self):
        """``None`` is the signal to inherit from ``min_confidence``."""
        assert ExportFilter().min_relation_confidence is None

    def test_entity_types_default_is_none(self):
        """``None`` means "include all types"."""
        assert ExportFilter().entity_types is None

    def test_include_orphans_default_false(self):
        assert ExportFilter().include_orphans is False

    def test_include_archived_default_false(self):
        assert ExportFilter().include_archived is False


# ---------------------------------------------------------------------------
# ExportFilter validation guard-rails
# ---------------------------------------------------------------------------


class TestExportFilterValidation:
    """Pin the ``ge``/``le`` constraints declared on the model.

    The export pipeline trusts the filter to be pre-validated -- the
    repository's SurrealQL WHERE clauses don't re-check ranges.
    """

    def test_min_connections_rejects_negative(self):
        with pytest.raises(ValidationError):
            ExportFilter(min_connections=-1)

    def test_min_confidence_rejects_below_zero(self):
        with pytest.raises(ValidationError):
            ExportFilter(min_confidence=-0.01)

    def test_min_confidence_rejects_above_one(self):
        with pytest.raises(ValidationError):
            ExportFilter(min_confidence=1.01)

    def test_min_relation_confidence_rejects_out_of_range(self):
        with pytest.raises(ValidationError):
            ExportFilter(min_relation_confidence=1.5)

    def test_min_confidence_accepts_zero_and_one(self):
        """0.0 and 1.0 are valid corner cases."""
        ExportFilter(min_confidence=0.0)
        ExportFilter(min_confidence=1.0)


# ---------------------------------------------------------------------------
# ExportFilter round-trip
# ---------------------------------------------------------------------------


class TestExportFilterRoundtrip:
    """JSON serialization is round-trippable -- API contract.

    The frontend posts ``ExportFilter`` JSON; the API parses it back via
    ``ExportFilter.model_validate``. Drift between the two paths would
    break the export request flow silently.
    """

    def test_default_roundtrip(self):
        original = ExportFilter()
        dumped = original.model_dump()
        rehydrated = ExportFilter.model_validate(dumped)
        assert rehydrated == original

    def test_custom_filter_roundtrip(self):
        original = ExportFilter(
            min_connections=10,
            min_confidence=0.75,
            min_relation_confidence=0.85,
            entity_types=["Person", "Organization"],
            include_orphans=True,
            include_archived=True,
        )
        dumped = original.model_dump()
        rehydrated = ExportFilter.model_validate(dumped)
        assert rehydrated == original

    def test_json_string_roundtrip(self):
        original = ExportFilter(min_connections=3, entity_types=["Concept"])
        json_str = original.model_dump_json()
        rehydrated = ExportFilter.model_validate_json(json_str)
        assert rehydrated == original


# ---------------------------------------------------------------------------
# ObsidianExportRequest
# ---------------------------------------------------------------------------


class TestObsidianExportRequest:
    """Two-mode delivery and default-filter wiring."""

    def test_default_mode_is_zip(self):
        req = ObsidianExportRequest()
        assert req.mode == "zip"

    def test_default_filter_is_export_filter(self):
        req = ObsidianExportRequest()
        assert isinstance(req.filter, ExportFilter)
        assert req.filter.min_connections == 5

    def test_vault_path_mode_accepted(self):
        req = ObsidianExportRequest(mode="vault_path")
        assert req.mode == "vault_path"

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValidationError):
            ObsidianExportRequest(mode="zip-encrypted")

    def test_roundtrip(self):
        original = ObsidianExportRequest(
            mode="vault_path",
            filter=ExportFilter(min_connections=2),
        )
        dumped = original.model_dump()
        rehydrated = ObsidianExportRequest.model_validate(dumped)
        assert rehydrated == original


# ---------------------------------------------------------------------------
# JsonlExportRequest
# ---------------------------------------------------------------------------


class TestJsonlExportRequest:
    """Single-field model; lock the default filter contract."""

    def test_default_filter_is_export_filter(self):
        req = JsonlExportRequest()
        assert isinstance(req.filter, ExportFilter)

    def test_roundtrip(self):
        original = JsonlExportRequest(
            filter=ExportFilter(min_connections=0, min_confidence=0.0)
        )
        dumped = original.model_dump()
        rehydrated = JsonlExportRequest.model_validate(dumped)
        assert rehydrated == original


# ---------------------------------------------------------------------------
# NetworkxExportRequest
# ---------------------------------------------------------------------------


class TestNetworkxExportRequest:
    """Closed-set ``format`` choices + filter wiring."""

    def test_format_is_required(self):
        with pytest.raises(ValidationError):
            NetworkxExportRequest()  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "fmt",
        [
            "graphml",
            "gexf",
            "gml",
            "json-tree",
            "edge-list",
            "adjacency-list",
            "pickle",
        ],
    )
    def test_each_supported_format_accepted(self, fmt: NetworkxFormat):
        req = NetworkxExportRequest(format=fmt)
        assert req.format == fmt

    def test_rejects_unknown_format(self):
        with pytest.raises(ValidationError):
            NetworkxExportRequest(format="parquet")  # type: ignore[arg-type]

    def test_default_filter_is_export_filter(self):
        req = NetworkxExportRequest(format="graphml")
        assert isinstance(req.filter, ExportFilter)
        assert req.filter.min_confidence == 0.9

    def test_roundtrip(self):
        original = NetworkxExportRequest(
            format="json-tree",
            filter=ExportFilter(entity_types=["Person"]),
        )
        dumped = original.model_dump()
        rehydrated = NetworkxExportRequest.model_validate(dumped)
        assert rehydrated == original


# ---------------------------------------------------------------------------
# ExportReport
# ---------------------------------------------------------------------------


class TestExportReport:
    """Reports are the telemetry+UI return value; lock the shape."""

    def test_minimal_report(self):
        report = ExportReport(
            entities_written=10,
            relations_written=20,
            files_written=1,
            bytes_written=4096,
            duration_ms=150,
            filter_applied=ExportFilter(),
        )
        assert report.entities_written == 10
        assert report.metadata is None

    def test_metadata_is_optional_freeform_dict(self):
        report = ExportReport(
            entities_written=0,
            relations_written=0,
            files_written=0,
            bytes_written=0,
            duration_ms=1,
            filter_applied=ExportFilter(),
            metadata={"dropped_relations": 3, "files_skipped": 0},
        )
        assert report.metadata == {"dropped_relations": 3, "files_skipped": 0}

    def test_negative_counts_rejected(self):
        with pytest.raises(ValidationError):
            ExportReport(
                entities_written=-1,
                relations_written=0,
                files_written=0,
                bytes_written=0,
                duration_ms=0,
                filter_applied=ExportFilter(),
            )

    def test_filter_applied_roundtrips(self):
        """The filter snapshot must round-trip so audit logs can replay it."""
        applied = ExportFilter(
            min_connections=7,
            entity_types=["Person", "Event"],
            include_archived=True,
        )
        report = ExportReport(
            entities_written=5,
            relations_written=3,
            files_written=5,
            bytes_written=10000,
            duration_ms=200,
            filter_applied=applied,
        )
        rehydrated = ExportReport.model_validate(report.model_dump())
        assert rehydrated.filter_applied == applied


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """The five export contracts are exposed at ``shared.models``."""

    def test_export_filter_imported_from_models(self):
        from shared.models import ExportFilter as A
        from shared.models.export import ExportFilter as B

        assert A is B

    def test_all_contracts_re_exported(self):
        import shared.models as m

        for name in (
            "ExportFilter",
            "ExportReport",
            "JsonlExportRequest",
            "NetworkxExportRequest",
            "ObsidianExportRequest",
        ):
            assert hasattr(m, name), f"shared.models missing {name}"
            assert name in m.__all__, f"{name} missing from __all__"
