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
    svc.mint.return_value = ({"id": "agent_keys:n", "key_prefix": SLEUTEL[:10]}, SLEUTEL)
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
        bestand=pad, service=_service(), secrets_map=tmp_path / "secrets",
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
            bestand=pad, service=svc, secrets_map=tmp_path,
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
    pad = tmp_path / "sleutel_langflow"
    await maak_sleutel.maak(
        "dienst", recht="pijplijn", agent="langflow", persoon=None,
        bestand=pad, service=svc, secrets_map=tmp_path,
    )
    row = await svc.authenticate(pad.read_text(encoding="utf-8"))
    assert row["permission"] == "pijplijn"


@pytest.mark.requires_docker
async def test_mislukt_schrijven_trekt_de_sleutel_in(live_surrealdb, tmp_path, monkeypatch):
    from app_main.services.agents.key_service import AgentKeyService

    svc = AgentKeyService(config=live_surrealdb)
    gemaakt = {}
    echte_mint = svc.mint

    async def mint(**kw):
        gemaakt["public"], gemaakt["sleutel"] = await echte_mint(**kw)
        return gemaakt["public"], gemaakt["sleutel"]

    monkeypatch.setattr(svc, "mint", mint)

    def faalt(pad, sleutel):
        raise PermissionError("schijf vol")

    monkeypatch.setattr(maak_sleutel, "schrijf_geheim", faalt)
    with pytest.raises(PermissionError):
        await maak_sleutel.maak(
            "dienst", recht="pijplijn", agent="langflow", persoon=None,
            bestand=tmp_path / "sleutel", service=svc, secrets_map=tmp_path,
        )
    assert await svc.authenticate(gemaakt["sleutel"]) is None


async def test_lezen_extern_kan_sinds_f2_05(tmp_path):
    """De API filtert sinds F2-05 op classificatie (contracttest lezen_extern)."""
    svc = _service()
    pad = tmp_path / "sleutel_mcp"
    await maak_sleutel.maak(
        "dienst", recht="lezen_extern", agent="open-notebook-mcp", persoon=None,
        bestand=pad, service=svc, secrets_map=tmp_path,
    )
    assert svc.mint.await_args.kwargs["permission"] == "lezen_extern"
    assert pad.read_text(encoding="utf-8") == SLEUTEL


async def test_pad_buiten_secrets_wordt_geweigerd(tmp_path):
    svc = _service()
    with pytest.raises(ValueError, match="niet onder"):
        await maak_sleutel.maak(
            "dienst", recht="pijplijn", agent="langflow", persoon=None,
            bestand=tmp_path / "elders" / "sleutel", service=svc,
            secrets_map=tmp_path / "secrets",
        )
    with pytest.raises(ValueError, match="niet onder"):
        await maak_sleutel.maak(
            "dienst", recht="pijplijn", agent="langflow", persoon=None,
            bestand=tmp_path / "secrets" / ".." / "sleutel", service=svc,
            secrets_map=tmp_path / "secrets",
        )
    svc.mint.assert_not_awaited()


async def test_mislukt_schrijven_trekt_de_sleutel_in_zonder_database(tmp_path, monkeypatch):
    svc = _service()

    def faalt(pad, sleutel):
        raise OSError("schijf vol")

    monkeypatch.setattr(maak_sleutel, "schrijf_geheim", faalt)
    with pytest.raises(OSError):
        await maak_sleutel.maak(
            "dienst", recht="pijplijn", agent="langflow", persoon=None,
            bestand=tmp_path / "sleutel", service=svc, secrets_map=tmp_path,
        )
    svc.revoke.assert_awaited_once_with("agent_keys:n")


def test_standaard_secrets_map_is_secrets_in_de_repository():
    assert maak_sleutel.SECRETS_MAP == _SCRIPT.parents[2] / "secrets"
