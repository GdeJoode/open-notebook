"""Regels van de bronkaart in het gedeelde model (F2-13, ADR 0019, 0020)."""

import pytest
from shared.models.source_map import (
    IntakeItem,
    effectieve_classificatie,
    pipeline_start_voor,
    regel_classificatie,
    standaard_classificatie,
)


@pytest.mark.parametrize("route_type,verwacht", [
    ("api", "extern"), ("mcp", "extern"), ("feed", "extern"), ("website", "extern"),
    ("bulkupload", "intern"), ("watchfolder", "intern"), ("upload", "intern"),
])
def test_standaard_per_route(route_type, verwacht) -> None:
    assert standaard_classificatie(route_type) == verwacht


def test_pipeline_start() -> None:
    assert pipeline_start_voor("feed") == "automatisch"
    assert pipeline_start_voor("watchfolder") == "wachtrij"
    assert pipeline_start_voor("upload") == "na_bevestiging"
    assert pipeline_start_voor("bulkupload") == "na_bevestiging"


@pytest.mark.parametrize("classification,door,active,verwacht", [
    ("extern", "Beheerder Test", True, "extern"),
    ("extern", None, True, "intern"),
    ("extern", " ", True, "intern"),
    ("extern", "Beheerder Test", False, "intern"),
    ("intern", "Beheerder Test", True, "intern"),
    (None, None, None, "intern"),
])
def test_extern_alleen_bevestigd_en_actief(classification, door, active, verwacht) -> None:
    assert effectieve_classificatie(classification, door, active=active) == verwacht


def test_wachtrij_item_zonder_besluit_is_intern() -> None:
    item = IntakeItem(origin="upload", path="/data/a.pdf", content_hash="sha256:x",
                      proposed_classification="extern", registered_by="langflow")
    assert item.effectieve_classificatie == "intern"
    besloten = item.model_copy(update={"status": "besloten", "decision": {
        "route": "r", "classification": "extern", "decided_by": "Beheerder Test"}})
    assert besloten.effectieve_classificatie == "extern"


@pytest.mark.parametrize("route_type,verwacht", [
    ("api", "extern"), ("mcp", "extern"), ("feed", "extern"), ("website", "extern"),
    ("watchfolder", "intern"), ("upload", "intern"), ("bulkupload", "intern"), (None, "intern"),
])
def test_regel_alleen_extern_op_openbare_route(route_type, verwacht) -> None:
    assert regel_classificatie("extern", "Beheerder Test", active=True,
                               route_type=route_type) == verwacht
