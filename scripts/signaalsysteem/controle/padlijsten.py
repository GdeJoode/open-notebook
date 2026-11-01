"""Startcontrole van de padlijsten tegen wat Compose koppelt (F2-10, ADR 0020).

De echte grens voor inname, export en vault is de koppeling in Compose:
``subst`` en junctions zijn vanuit de fork niet te zien ("Uitwerking in
F2-13"). Deze controle leest de opgeloste Compose-configuratie
(``docker compose config --format json``, op stdin) en faalt (exitcode 1) als:

* een koppeling van de vault niet in ``SIGNAAL_VAULT_PADEN`` van
  ``open-notebook`` staat, of de fork zelf de vault koppelt;
* een koppeling van een map boven de vault bestaat (die zou de vault meenemen);
* een koppeling voor inname niet in ``SIGNAAL_INNAME_PADEN`` staat of niet
  alleen-lezen is;
* een koppeling voor export niet in ``SIGNAAL_EXPORTMAP_PADEN`` staat;
* een container-map uit ``SIGNAAL_INNAME_PADEN`` niet alleen-lezen aan
  ``open-notebook`` is gekoppeld.

Een koppeling is van de vault, voor inname of voor export als de bron op de
host in of op de map uit ``SIGNAAL_VAULT``, ``SIGNAAL_INNAMEMAP``,
``SIGNAAL_WATCHFOLDER`` of ``SIGNAAL_EXPORTMAP`` ligt (de hostpaden uit
``.env``), of als het koppelpunt onder ``/vault``, ``/inname``,
``/watchfolder``, ``/inbox`` of ``/export`` ligt.

Gebruik: ``docker compose -f docker-compose.signaalsysteem.yml config --format
json | python padlijsten.py``. De hostpaden komen uit de omgeving.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import PurePosixPath
from typing import Iterable, Mapping

DIENST_FORK = "open-notebook"
_DOEL = {
    "vault": ("/vault",),
    "inname": ("/inname", "/watchfolder", "/inbox"),
    "export": ("/export",),
}
_HOST = {
    "vault": ("SIGNAAL_VAULT",),
    "inname": ("SIGNAAL_INNAMEMAP", "SIGNAAL_WATCHFOLDER"),
    "export": ("SIGNAAL_EXPORTMAP",),
}
_LIJST = {
    "vault": "SIGNAAL_VAULT_PADEN",
    "inname": "SIGNAAL_INNAME_PADEN",
    "export": "SIGNAAL_EXPORTMAP_PADEN",
}


def _norm_host(pad: str) -> str:
    """Hostpad vergelijkbaar maken: schuine strepen, kleine letters, geen slot-/."""
    return pad.replace("\\", "/").rstrip("/").lower()


def _onder(kind: str, ouder: str) -> bool:
    return kind == ouder or kind.startswith(ouder + "/")


def _lijst(waarde: object) -> list[str]:
    return [p.strip() for p in str(waarde or "").split(";") if p.strip()]


def _doel_onder(doel: str, wortels: Iterable[str]) -> bool:
    p = PurePosixPath(doel)
    return any(p == PurePosixPath(w) or PurePosixPath(w) in p.parents for w in wortels)


def soorten(bron: str, doel: str, host: Mapping[str, str]) -> set[str]:
    """De soorten (vault, inname, export) van een koppeling."""
    uit = set()
    b = _norm_host(bron)
    for soort, namen in _HOST.items():
        for naam in namen:
            h = _norm_host(host.get(naam) or "")
            if h and _onder(b, h):
                uit.add(soort)
    for soort, wortels in _DOEL.items():
        if _doel_onder(doel, wortels):
            uit.add(soort)
    return uit


def controleer(config: Mapping, host: Mapping[str, str]) -> list[str]:
    """Fouten in de koppelingen van een opgeloste Compose-configuratie."""
    fouten: list[str] = []
    diensten = config.get("services") or {}
    fork_env = (diensten.get(DIENST_FORK) or {}).get("environment") or {}
    lijsten = {soort: _lijst(fork_env.get(naam)) for soort, naam in _LIJST.items()}
    vault_host = _norm_host(host.get("SIGNAAL_VAULT") or "")
    fork_ro: set[str] = set()

    for naam, dienst in diensten.items():
        for kop in dienst.get("volumes") or []:
            if kop.get("type") != "bind":
                continue
            bron, doel = str(kop.get("source", "")), str(kop.get("target", ""))
            alleen_lezen = bool(kop.get("read_only"))
            if naam == DIENST_FORK and alleen_lezen:
                fork_ro.add(doel)
            if vault_host and _onder(vault_host, _norm_host(bron)) and _norm_host(bron) != vault_host:
                fouten.append(f"{naam}: {doel} koppelt een map boven de vault")
            for soort in sorted(soorten(bron, doel, host)):
                if doel not in lijsten[soort]:
                    fouten.append(f"{naam}: koppeling {doel} ({soort}) staat niet in {_LIJST[soort]}")
                if soort == "vault" and naam == DIENST_FORK:
                    fouten.append(f"{naam}: de fork koppelt de vault niet ({doel})")
                if soort == "inname" and not alleen_lezen:
                    fouten.append(f"{naam}: innamekoppeling {doel} is niet alleen-lezen")

    for pad in lijsten["inname"]:
        if pad not in fork_ro:
            fouten.append(f"{DIENST_FORK}: {pad} uit SIGNAAL_INNAME_PADEN is niet alleen-lezen gekoppeld")
    return fouten


def main() -> int:
    config = json.load(sys.stdin)
    fouten = controleer(config, os.environ)
    for fout in fouten:
        print(f"padlijsten: {fout}", file=sys.stderr)
    if not fouten:
        print("padlijsten: in orde")
    return 1 if fouten else 0


if __name__ == "__main__":
    sys.exit(main())
