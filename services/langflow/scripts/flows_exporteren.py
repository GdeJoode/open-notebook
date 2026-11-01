"""Exporteert flows uit Langflow naar het formaat in Git (F2-12, ADR 0017 A en B).

Na een wijziging in de editor haalt dit script de flows terug, zodat ze met een
commit in Git komen (ADR 0017 B, optie 1). Langflow normaliseert een export al
(``POST /api/v1/flows/download/``: zonder ``updated_at``, ``user_id``,
``folder_id`` en de stand van de editor, geheime velden leeg, code als regels).
Dit script haalt daarnaast de velden weg die per installatie verschillen
(:data:`PER_INSTALLATIE`) en schrijft met gesorteerde sleutels. Zo levert laden
(``flows_laden.py``) en daarna exporteren dezelfde flow-JSON op.

Drie vormen; de container ``langflow`` heeft geen schrijfbare koppeling naar de
repository, dus exporteren gaat in twee stappen::

    # 1. in de container: de flows uit Langflow als bundel op stdout
    docker compose -p signaal -f docker-compose.signaalsysteem.yml exec -T langflow \\
        python /app/signaal/flows_exporteren.py [--id <uuid> ...] > export.json
    # 2. op de host: de bundel naar de flowmap schrijven
    python services/langflow/scripts/flows_exporteren.py --schrijf services/langflow/flows < export.json

Zet ``export.json`` in een tijdelijke map buiten de repository (``.gitignore``
sluit ``export.json`` ook uit). ``--schrijf`` schrijft alleen als elke flow alleen
typen van de componentenlijst gebruikt (``--lijst``, standaard
``config/langflow-componenten.yaml``) en geen waarde bevat die op een sleutel lijkt
(:func:`geheimen`); anders schrijft het niets en eindigt het met exitcode 1.

    # controle: verschilt Langflow van de map? (exitcode 1 bij een verschil)
    ... flows_exporteren.py --vergelijk <map> < export.json

Zonder ``--id`` exporteert de eerste vorm de flows waarvan de ``id`` in de
flowmap staat (``--map``, standaard de kloon van HEAD). Een nieuwe flow uit de
editor krijgt een bestandsnaam uit haar naam.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

STANDAARD_MAP = "/app/flows-git/services/langflow/flows"
#: Op de host: de lijst in de werkmap (services/langflow/scripts -> root); in de
#: container is er geen werkmap en geldt SIGNAAL_COMPONENTENLIJST.
_HIER = Path(__file__).resolve()
STANDAARD_LIJST = os.environ.get("SIGNAAL_COMPONENTENLIJST") or (
    str(_HIER.parents[3] / "config" / "langflow-componenten.yaml") if len(_HIER.parents) > 3 else "")

#: Waarden die op een sleutel of token lijken (review F2-12, punt 6).
SLEUTELPATRONEN = tuple(re.compile(p) for p in (
    r"sk-[A-Za-z0-9_-]{16,}",                          # OpenAI, Anthropic
    r"\bak_[A-Za-z0-9_-]{16,}",                        # sleutels van de fork
    r"\bAKIA[0-9A-Z]{16}\b",                           # AWS
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",                   # GitHub
    r"\bxox[abpr]-[A-Za-z0-9-]{10,}",                  # Slack
    r"\bAIza[0-9A-Za-z_-]{35}",                        # Google
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.",  # JWT
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
))

#: Velden die Langflow 1.13.0 niet zelf weghaalt, maar die per installatie of per
#: opslag verschillen (vastgesteld in F2-12): de werkruimte, de gebruiker die het
#: laatst wijzigde en het versietoken voor gelijktijdige wijzigingen.
PER_INSTALLATIE = ("workspace_id", "last_modified_by", "version_token")


def normaliseer(flow: dict) -> dict:
    """De flow zoals hij in Git staat."""
    uit = copy.deepcopy(flow)
    for sleutel in PER_INSTALLATIE:
        uit.pop(sleutel, None)
    return uit


def als_tekst(flow: dict) -> str:
    return json.dumps(flow, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def bestandsnaam(flow: dict, bekend: dict[str, str]) -> str:
    """De bestaande bestandsnaam bij deze id, anders een naam uit de flownaam."""
    flow_id = str(flow.get("id", ""))
    if flow_id in bekend:
        return bekend[flow_id]
    stam = re.sub(r"[^a-z0-9]+", "-", str(flow.get("name", "")).lower()).strip("-")
    return f"{stam or flow_id or 'flow'}.json"


def ids_in_map(map_: Path | str) -> dict[str, str]:
    """id -> bestandsnaam voor de flows in de map."""
    uit: dict[str, str] = {}
    map_ = Path(map_)
    if not map_.is_dir():
        return uit
    for pad in sorted(map_.glob("*.json")):
        try:
            flow_id = json.loads(pad.read_text(encoding="utf-8")).get("id")
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(flow_id, str) and flow_id:
            uit[flow_id] = pad.name
    return uit


def exporteer(api, ids: list[str], bekend: dict[str, str]) -> dict[str, dict]:
    """bestandsnaam -> genormaliseerde flow, één download per flow."""
    bundel: dict[str, dict] = {}
    for flow_id in ids:
        flow = normaliseer(api.download(flow_id))
        naam = bestandsnaam(flow, bekend)
        if naam in bundel:
            raise ValueError(f"twee flows met bestandsnaam {naam}")
        bundel[naam] = flow
    return bundel


def geheimen(waarde, pad: str = "") -> list[str]:
    """Plaatsen in de flow met een waarde die op een sleutel lijkt.

    Ook een veld met ``password: true`` en een ingevulde waarde, tenzij die
    waarde de naam van een globale variabele is (``load_from_db``).
    """
    uit: list[str] = []
    if isinstance(waarde, dict):
        if (waarde.get("password") is True and isinstance(waarde.get("value"), str)
                and waarde["value"].strip() and not waarde.get("load_from_db")):
            uit.append(f"{pad}/value (wachtwoordveld ingevuld)")
        for sleutel, binnen in waarde.items():
            uit += geheimen(binnen, f"{pad}/{sleutel}")
    elif isinstance(waarde, list):
        for i, binnen in enumerate(waarde):
            uit += geheimen(binnen, f"{pad}[{i}]")
    elif isinstance(waarde, str) and any(p.search(waarde) for p in SLEUTELPATRONEN):
        uit.append(pad or "/")
    return uit


def toets_bundel(bundel: dict[str, dict], lijst: Path | str) -> list[str]:
    """Weigeringen voor ``--schrijf``: typen buiten de lijst en waarden die op een sleutel lijken."""
    from flows_laden import lees_lijst, toets

    fouten = [f"{b}: {r}" for b, r in toets(bundel, lees_lijst(lijst))]
    for naam, flow in sorted(bundel.items()):
        fouten += [f"{naam}: lijkt op een sleutel: {p}" for p in geheimen(flow)]
    return fouten


def schrijf(bundel: dict[str, dict], doel: Path | str) -> list[str]:
    doel = Path(doel)
    doel.mkdir(parents=True, exist_ok=True)
    geschreven = []
    for naam, flow in sorted(bundel.items()):
        if Path(naam).name != naam or not naam.endswith(".json"):
            raise ValueError(f"ongeldige bestandsnaam {naam!r}")
        # Zonder regeleindeconversie (services/langflow/flows/.gitattributes).
        (doel / naam).write_bytes(als_tekst(flow).encode("utf-8"))
        geschreven.append(naam)
    return geschreven


def vergelijk(bundel: dict[str, dict], map_: Path | str) -> list[str]:
    """Verschillen tussen de bundel en de bestanden in de map, op volgorde van sleutels na."""
    verschillen = []
    map_ = Path(map_)
    for naam, flow in sorted(bundel.items()):
        pad = map_ / naam
        if not pad.is_file():
            verschillen.append(f"{naam}: niet in {map_}")
            continue
        if normaliseer(json.loads(pad.read_text(encoding="utf-8"))) != flow:
            verschillen.append(f"{naam}: wijkt af")
    return verschillen


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    vorm = p.add_mutually_exclusive_group()
    vorm.add_argument("--schrijf", metavar="MAP", help="bundel van stdin naar MAP schrijven")
    vorm.add_argument("--vergelijk", metavar="MAP", help="bundel van stdin met MAP vergelijken")
    p.add_argument("--map", default=os.environ.get("SIGNAAL_FLOWS_MAP", STANDAARD_MAP))
    p.add_argument("--id", action="append", default=[], help="id van een flow (herhaalbaar)")
    p.add_argument("--lijst", default=STANDAARD_LIJST, help="componentenlijst voor --schrijf")
    args = p.parse_args(argv)

    if args.schrijf or args.vergelijk:
        bundel = json.load(sys.stdin)
        if not isinstance(bundel, dict) or not all(isinstance(f, dict) for f in bundel.values()):
            print("flows_exporteren: geen bundel op stdin", file=sys.stderr)
            return 2
        if args.schrijf:
            fouten = toets_bundel(bundel, args.lijst)
            for regel in fouten:
                print(f"flows_exporteren: geweigerd: {regel}", file=sys.stderr)
            if fouten:
                print("flows_exporteren: niets geschreven", file=sys.stderr)
                return 1
            for naam in schrijf(bundel, args.schrijf):
                print(f"flows_exporteren: geschreven: {naam}", file=sys.stderr)
            return 0
        verschillen = vergelijk(bundel, args.vergelijk)
        for regel in verschillen:
            print(f"flows_exporteren: {regel}", file=sys.stderr)
        return 1 if verschillen else 0

    from langflow_api import LangflowApi, LangflowFout

    bekend = ids_in_map(args.map)
    ids = args.id or list(bekend)
    try:
        bundel = exporteer(LangflowApi(), ids, bekend)
    except (LangflowFout, ValueError) as e:
        print(f"flows_exporteren: {e}", file=sys.stderr)
        return 3
    json.dump(bundel, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
