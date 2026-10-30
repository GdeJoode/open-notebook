"""Het script ``scripts/signaalsysteem/maak_sleutel.py`` (F2-01).

De beheerderssleutel staat één keer op het scherm en nergens anders; een
dienstsleutel gaat alleen naar een nieuw geheimbestand.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/signaalsysteem/maak_sleutel.py"
_spec = importlib.util.spec_from_file_location("maak_sleutel", _SCRIPT)
maak_sleutel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(maak_sleutel)

SLEUTEL = "ak_" + "s" * 43


def _service():
    svc = AsyncMock()
    svc.mint.return_value = ({"key_prefix": SLEUTEL[:10]}, SLEUTEL)
    return svc


async def test_beheerder_toont_de_sleutel_een_keer():
    svc = _service()
    tekst = await maak_sleutel.maak(
        "beheerder", recht="beheerder", agent="werkbak", persoon="Testpersoon",
        bestand=None, service=svc,
    )
    assert tekst.count(SLEUTEL) == 1
    kw = svc.mint.await_args.kwargs
    assert kw["permission"] == "beheerder" and kw["person"] == "Testpersoon"


async def test_dienst_schrijft_alleen_naar_een_nieuw_bestand(tmp_path):
    pad = tmp_path / "secrets" / "sleutel_langflow"
    tekst = await maak_sleutel.maak(
        "dienst", recht="pijplijn", agent="langflow", persoon=None,
        bestand=pad, service=_service(),
    )
    assert pad.read_text(encoding="utf-8") == SLEUTEL
    assert SLEUTEL not in tekst


async def test_bestaand_bestand_wordt_niet_overschreven(tmp_path):
    pad = tmp_path / "sleutel"
    pad.write_text("oud", encoding="utf-8")
    svc = _service()
    with pytest.raises(FileExistsError):
        await maak_sleutel.maak(
            "dienst", recht="pijplijn", agent="langflow", persoon=None,
            bestand=pad, service=svc,
        )
    svc.mint.assert_not_awaited()
    assert pad.read_text(encoding="utf-8") == "oud"


def test_beheerder_vraagt_een_persoon():
    with pytest.raises(SystemExit):
        maak_sleutel.main(["beheerder"])


def test_dienst_kent_geen_recht_beheerder():
    with pytest.raises(SystemExit):
        maak_sleutel.main(["dienst", "--recht", "beheerder", "--agent", "a",
                           "--bestand", "x"])


@pytest.mark.requires_docker
async def test_gemaakte_sleutel_authenticeert(live_surrealdb, tmp_path):
    from app_main.services.agents.key_service import AgentKeyService

    svc = AgentKeyService(config=live_surrealdb)
    pad = tmp_path / "sleutel_mcp"
    await maak_sleutel.maak(
        "dienst", recht="lezen_extern", agent="open-notebook-mcp", persoon=None,
        bestand=pad, service=svc,
    )
    row = await svc.authenticate(pad.read_text(encoding="utf-8"))
    assert row["permission"] == "lezen_extern"
