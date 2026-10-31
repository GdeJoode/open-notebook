"""Woordenlijstbeheer op de nieuwe plek (F2-02, ADR 0012).

``vocabulary_manager`` stond in de extractiedienst en staat nu in
``shared.vocabulary``, naast ``tooi_provider``. Geen netwerk: de HTTP-aanroepen
zijn vervangen door ``respx``.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from shared.vocabulary import vocabulary_manager as vm


@pytest.fixture(autouse=True)
def _vocab_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "VOCAB_DIR", tmp_path / "vocabularies")
    return tmp_path / "vocabularies"


def test_opslaan_laden_en_tellen() -> None:
    entries = [{"name": "Gemeente Delft", "type": "Gemeente", "aliases": ["Delft"]}]
    vm._save_vocab("proef", entries)
    assert vm.load_vocab("proef") == entries
    assert vm.list_vocabs() == {"proef": 1}


def test_onbekende_lijst_is_leeg() -> None:
    assert vm.load_vocab("bestaat_niet") == []
    assert vm.list_vocabs() == {}


async def test_cbs_brede_welvaart_wordt_opgebouwd() -> None:
    aantal = await vm.refresh_cbs_brede_welvaart()
    lijst = vm.load_vocab("cbs_brede_welvaart")
    assert aantal == len(lijst) > 0
    assert {"name": "Gezondheid", "type": "BredeWelvaartThema"}.items() <= lijst[
        [e["name"] for e in lijst].index("Gezondheid")
    ].items()


@respx.mock
async def test_tooi_gemeenten_valt_terug_zonder_netwerk() -> None:
    respx.get(url__regex=r"https://(tardis|standaarden)\.overheid\.nl/.*").mock(
        side_effect=httpx.ConnectError("geen netwerk")
    )
    aantal = await vm.refresh_tooi_gemeenten()
    namen = {e["name"] for e in vm.load_vocab("tooi_gemeenten")}
    assert aantal == len(namen) and "Delft" in namen


@respx.mock
async def test_tooi_gemeenten_uit_de_waardelijst() -> None:
    respx.get(url__startswith="https://tardis.overheid.nl/").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "label": "gemeente Delft",
                    "uri": "https://identifier.overheid.nl/tooi/id/gemeente/gm0503",
                }
            ],
        )
    )
    assert await vm.refresh_tooi_gemeenten() == 1
    assert vm.load_vocab("tooi_gemeenten")[0]["uri"].endswith("gm0503")
