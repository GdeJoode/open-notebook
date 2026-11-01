"""Laadt de flows uit Git in Langflow, na toetsing aan de componentenlijst (F2-12).

ADR 0017 A (optie 2): automatisch aanmelden staat uit; dit script laadt de flows
via ``POST /api/v1/flows/upload/`` en zet hun schema-triggers aan via
``POST /api/v1/triggers/{id}/enable`` (P2). ADR 0022 D: vóór het laden toetst
het elke flow aan ``config/langflow-componenten.yaml``. Staat één knooppunttype
van één flow niet op de lijst, dan laadt het script niets, noemt het de flow en
het type, en eindigt het met exitcode 1.

De flows komen uit de minimale kloon van HEAD die ``scripts/signaalsysteem/start.sh``
maakt (``/app/flows-git``), dezelfde als die van het flowregister van de fork
(F2-04). Elke flow heeft een vaste ``id``: zo werkt het laden bij een herstart
de bestaande flow bij in plaats van een kopie te maken.

Na het toetsen zet het script ook het catalogusbeleid van Langflow: elke
component die niet op de lijst staat, is geblokkeerd, ook in de editor. Na het
laden zet het de triggers uit van flows die niet uit de kloon komen (review F2-12).

Gebruik (in de container, door ``start.sh``)::

    python flows_laden.py                    # toetsen, beleid, laden, triggers
    python flows_laden.py --alleen-toetsen   # alleen de componentenlijst

Exitcodes: 0 geslaagd; 1 een flow geweigerd (type niet op de lijst, ongeldige
flow); 2 lijst of map niet bruikbaar; 3 Langflow niet bereikbaar of een fout bij
het laden.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator

import yaml

STANDAARD_MAP = "/app/flows-git/services/langflow/flows"
STANDAARD_LIJST = "/app/flows-git/config/langflow-componenten.yaml"
SOORTEN = ("kern", "notities", "eigen")
GEEN_TYPE = "<geen type>"


class LijstFout(ValueError):
    """De componentenlijst ontbreekt of voldoet niet aan het schema."""


class FlowFout(ValueError):
    """Een flowbestand is geen bruikbare flow."""


def lees_lijst(pad: Path | str) -> frozenset[str]:
    """De toegestane typen uit de componentenlijst; streng: elke afwijking is een fout."""
    try:
        inhoud = yaml.safe_load(Path(pad).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise LijstFout(f"componentenlijst niet te lezen: {pad}") from e
    if not isinstance(inhoud, dict) or set(inhoud) != {"versie", *SOORTEN}:
        raise LijstFout(f"componentenlijst: sleutels moeten versie, {', '.join(SOORTEN)} zijn")
    typen: list[str] = []
    for soort in SOORTEN:
        items = inhoud[soort]
        if not isinstance(items, list):
            raise LijstFout(f"componentenlijst: {soort} is geen lijst")
        for item in items:
            if (not isinstance(item, dict) or set(item) != {"type", "reden"}
                    or not isinstance(item["type"], str) or not item["type"].strip()):
                raise LijstFout(f"componentenlijst: ongeldig item in {soort}: {item!r}")
            if soort == "eigen" and not (item["type"].startswith("ext:signaalsysteem:")
                                         and item["type"].endswith("@extra")):
                raise LijstFout(f"componentenlijst: eigen type {item['type']!r} heeft niet de "
                                "vorm ext:signaalsysteem:<Naam>@extra")
            typen.append(item["type"])
    dubbel = sorted({t for t in typen if typen.count(t) > 1})
    if dubbel:
        raise LijstFout(f"componentenlijst: dubbele typen {', '.join(dubbel)}")
    return frozenset(typen)


def _knooppunten(data: object) -> Iterator[dict]:
    """Alle knooppunten van een flow, ook die in een gegroepeerd knooppunt."""
    if not isinstance(data, dict):
        raise FlowFout("flow zonder data")
    knopen = data.get("nodes")
    if not isinstance(knopen, list):
        raise FlowFout("flow zonder lijst nodes")
    for knoop in knopen:
        if not isinstance(knoop, dict):
            raise FlowFout("knooppunt is geen object")
        yield knoop
        binnen = ((knoop.get("data") or {}).get("node") or {}).get("flow")
        if isinstance(binnen, dict) and "data" in binnen:
            yield from _knooppunten(binnen["data"])


def knooppunttypen(flow: dict) -> list[str]:
    """Het type (``data.type``) van elk knooppunt; zonder type: ``<geen type>``."""
    if not isinstance(flow, dict):
        raise FlowFout("flow is geen object")
    uit = []
    for knoop in _knooppunten(flow.get("data")):
        soort = (knoop.get("data") or {}).get("type")
        uit.append(soort if isinstance(soort, str) and soort else GEEN_TYPE)
    return uit


def lees_flows(map_: Path | str) -> dict[str, dict]:
    """De flows in de map (alleen ``*.json`` direct in de map), op bestandsnaam."""
    map_ = Path(map_)
    if not map_.is_dir():
        raise LijstFout(f"flowmap bestaat niet: {map_}")
    flows: dict[str, dict] = {}
    for pad in sorted(map_.glob("*.json")):
        if pad.is_symlink() or not pad.is_file():
            raise FlowFout(f"{pad.name}: geen gewoon bestand")
        try:
            flows[pad.name] = json.loads(pad.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise FlowFout(f"{pad.name}: geen geldige JSON") from e
    return flows


def toets(flows: dict[str, dict], toegestaan: Iterable[str]) -> list[tuple[str, str]]:
    """De weigeringen als (bestand, reden of type); leeg als alles op de lijst staat."""
    toegestaan = frozenset(toegestaan)
    weigeringen: list[tuple[str, str]] = []
    ids: dict[str, str] = {}
    for bestand, flow in flows.items():
        try:
            typen = knooppunttypen(flow)
        except FlowFout as e:
            weigeringen.append((bestand, str(e)))
            continue
        flow_id = flow.get("id")
        if not isinstance(flow_id, str) or not flow_id:
            weigeringen.append((bestand, "flow zonder vaste id"))
        elif flow_id in ids:
            weigeringen.append((bestand, f"id ook in {ids[flow_id]}"))
        else:
            ids[flow_id] = bestand
        # Alleen privé: een openbare flow is zonder aanmelding te draaien
        # (herreview F2-12, punt 1). Een export van Langflow laat het veld weg;
        # dan wordt de flow privé opgeslagen (zie laad()).
        if flow.get("access_type", PRIVE) != PRIVE:
            weigeringen.append((bestand, f"access_type is niet {PRIVE}"))
        for soort in sorted(set(typen) - toegestaan):
            weigeringen.append((bestand, f"type niet op de lijst: {soort}"))
    return weigeringen


#: De enige toegestane waarde van ``access_type`` (herreview F2-12, punt 1).
PRIVE = "PRIVATE"

#: Langflow 1.13.0 neemt hooguit zoveel sleutels in een blokkeerset.
MAX_BLOKKEERSET = 1000


def zet_catalogusbeleid(api, toegestaan: Iterable[str]) -> str:
    """Blokkeer in Langflow zelf elke component die niet op de lijst staat (review F2-12, punt 2).

    Het catalogusbeleid van Langflow 1.13.0 (``PUT /api/v1/catalog-policy/components``)
    geldt ook voor de editor: een geblokkeerde component is daar niet te kiezen en
    een flow met zo'n component draait niet. De set wordt daarna opnieuw gelezen.
    """
    # De hele catalogus, ook wat al geblokkeerd is, in de eigen schrijfwijze: een
    # sleutel in kleine letters blokkeert in 1.13.0 niets (vastgesteld in F2-12).
    namen = api.componentnamen(ook_geblokkeerd=True)
    laag = {t.lower() for t in toegestaan}
    blokkeren = {n for n in namen if n.lower() not in laag}
    # Geteld in kleine letters: zo geeft Langflow de set terug (herreview, punt 7).
    blokkeren_laag = {b.lower() for b in blokkeren}
    if not blokkeren or len(blokkeren) > MAX_BLOKKEERSET:
        raise ValueError(f"blokkeerset van {len(blokkeren)} componenten past niet")
    api.zet_blokkeerset(blokkeren)
    if {s.lower() for s in api.blokkeerset()} != blokkeren_laag:
        raise ValueError("catalogusbeleid staat na het zetten niet zoals gevraagd")
    # En het werkt: in de catalogus van de editor staat alleen nog wat op de lijst staat.
    zichtbaar = {n.lower() for n in api.componentnamen()}
    if not zichtbaar <= laag:
        raise ValueError(f"catalogusbeleid werkt niet: {len(zichtbaar - laag)} componenten nog zichtbaar")
    return (f"catalogusbeleid: {len(blokkeren_laag)} van {len({n.lower() for n in namen})} "
            "componenten geblokkeerd (sleutels in kleine letters)")


def zet_andere_triggers_uit(api, geladen: Iterable[str]) -> str:
    """Zet de triggers uit van flows die niet uit de kloon komen (review F2-12, punt 2)."""
    geladen = set(geladen)
    uit = 0
    for flow_id in api.flow_ids():
        if flow_id in geladen:
            continue
        for trigger in api.triggers(flow_id):
            if trigger.get("state") != "disabled":
                api.zet_uit(trigger["id"])
                uit += 1
    return f"triggers van flows buiten Git uitgezet: {uit}"


def laad(api, flows: dict[str, dict]) -> list[str]:
    """Laad de flows en zet hun triggers aan; geeft een regel per flow terug."""
    regels = []
    for bestand, flow in flows.items():
        opgeslagen = api.upload(flow, bestand)
        if any(f.get("access_type", PRIVE) != PRIVE for f in opgeslagen):
            raise ValueError(f"{bestand}: Langflow sloeg de flow niet privé op")
        aangezet = 0
        for trigger in api.triggers(flow["id"]):
            if trigger.get("state") != "active":
                api.zet_aan(trigger["id"])
            aangezet += 1
        regels.append(f"geladen: {bestand} (id {flow['id']}, triggers actief: {aangezet})")
    return regels


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--map", default=os.environ.get("SIGNAAL_FLOWS_MAP", STANDAARD_MAP))
    p.add_argument("--lijst", default=os.environ.get("SIGNAAL_COMPONENTENLIJST", STANDAARD_LIJST))
    p.add_argument("--alleen-toetsen", action="store_true")
    p.add_argument("--wacht", type=float, default=600.0, help="seconden wachten op Langflow")
    args = p.parse_args(argv)

    try:
        toegestaan = lees_lijst(args.lijst)
        flows = lees_flows(args.map)
    except LijstFout as e:
        print(f"flows_laden: {e}", file=sys.stderr)
        return 2
    except FlowFout as e:
        print(f"flows_laden: geweigerd: {e}", file=sys.stderr)
        return 1

    weigeringen = toets(flows, toegestaan)
    if weigeringen:
        for bestand, reden in weigeringen:
            print(f"flows_laden: geweigerd: {bestand}: {reden}", file=sys.stderr)
        print("flows_laden: geen enkele flow geladen", file=sys.stderr)
        return 1
    print(f"flows_laden: {len(flows)} flow(s) getoetst aan {args.lijst}")
    if args.alleen_toetsen:
        return 0

    from langflow_api import LangflowApi, LangflowFout

    api = LangflowApi()
    if not api.wacht(args.wacht):
        print("flows_laden: Langflow niet gezond", file=sys.stderr)
        return 3
    try:
        print(f"flows_laden: {zet_catalogusbeleid(api, toegestaan)}")
        for regel in laad(api, flows):
            print(f"flows_laden: {regel}")
        print(f"flows_laden: {zet_andere_triggers_uit(api, (f['id'] for f in flows.values()))}")
    except (LangflowFout, ValueError) as e:
        print(f"flows_laden: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
