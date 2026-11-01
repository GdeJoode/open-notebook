"""Startcontrole van de padlijsten tegen wat Compose koppelt (F2-10, ADR 0020).

De echte grens voor inname, export en vault is de koppeling in Compose:
``subst`` en junctions zijn vanuit de fork niet te zien ("Uitwerking in
F2-13"). Deze controle leest de opgeloste Compose-configuratie
(``docker compose config --format json``, op stdin) en faalt (exitcode 1) als:

* ``SIGNAAL_VAULT`` (de vault op de host) niet gezet is: zonder vault is er
  niets te toetsen, dus weigeren (fail-closed);
* een bind-pad op de host niet eenduidig is: een segment ``..`` of ``.``, een
  8.3-naam (``~<cijfer>``), een segment dat eindigt op een punt of spatie, een
  jokerteken, een UNC-pad, of een pad dat ``start.sh`` niet canoniek kon maken;
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
``SIGNAAL_WATCHFOLDER`` of ``SIGNAAL_EXPORTMAP`` ligt, of als het koppelpunt
onder ``/vault``, ``/inname``, ``/watchfolder``, ``/inbox`` of ``/export`` ligt.

Hostpaden worden vergeleken na canonicalisatie. ``start.sh`` lost elk pad op de
host op met PowerShell (``canoniek.ps1``: ``subst``, junctions en symbolische
koppelingen, 8.3-namen) en geeft de uitkomst als JSON in ``SIGNAAL_CANONIEK``
(``{pad: canoniek pad of null}``). Daarnaast zet deze controle de vormen
``\\\\?\\D:\\``, ``/run/desktop/mnt/host/d/`` en ``/mnt/d/`` om naar ``d:/``.

Gebruik: ``docker compose ... config --format json | python padlijsten.py``.
Met ``--bronnen`` schrijft het script alleen de bind-paden, één per regel.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import PurePosixPath
from typing import Iterable, Mapping, Optional

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
HOSTVARIABELEN = tuple(n for namen in _HOST.values() for n in namen)
_LIJST = {
    "vault": "SIGNAAL_VAULT_PADEN",
    "inname": "SIGNAAL_INNAME_PADEN",
    "export": "SIGNAAL_EXPORTMAP_PADEN",
}
_PREFIX = re.compile(r"^(?:/run/desktop/mnt/host|/mnt)/([a-z])(?=/|$)")


def _vorm(pad: str) -> str:
    """Schuine strepen, geen ``\\\\?\\``, kleine letters, Docker Desktop- en WSL-vorm."""
    p = pad.replace("\\", "/")
    for voor in ("//?/", "//./", "/??/"):
        if p.startswith(voor):
            p = p[len(voor):]
    p = p.lower()
    p = _PREFIX.sub(lambda m: m.group(1) + ":", p)
    return p


def onduidelijk(pad: str) -> Optional[str]:
    """Waarom een hostpad niet eenduidig is, of ``None``."""
    p = _vorm(pad)
    if p.startswith("//"):
        return "UNC-pad"
    for deel in p.split("/"):
        if deel in ("..", "."):
            return f"segment {deel!r}"
        if re.search(r"~\d", deel):
            return "8.3-naam"
        if deel and deel[-1] in ". " and not re.fullmatch(r"[a-z]:", deel):
            return "segment eindigt op punt of spatie"
        if any(t in deel for t in "*?\"<>|\t"):
            return "jokerteken of stuurteken"
    return None


def _norm(pad: str) -> str:
    p = re.sub(r"/+", "/", _vorm(pad))
    return p.rstrip("/") if len(p) > 3 else p


def _onder(kind: str, ouder: str) -> bool:
    return kind == ouder or kind.startswith(ouder.rstrip("/") + "/")


def _lijst(waarde: object) -> list[str]:
    return [p.strip() for p in str(waarde or "").split(";") if p.strip()]


def _doel_onder(doel: str, wortels: Iterable[str]) -> bool:
    p = PurePosixPath(doel)
    return any(p == PurePosixPath(w) or PurePosixPath(w) in p.parents for w in wortels)


def bind_bronnen(config: Mapping) -> list[str]:
    uit = []
    for dienst in (config.get("services") or {}).values():
        for kop in dienst.get("volumes") or []:
            if kop.get("type") == "bind" and kop.get("source"):
                uit.append(str(kop["source"]))
    return uit


def controleer(config: Mapping, host: Mapping[str, str],
               canoniek: Optional[Mapping[str, Optional[str]]] = None) -> list[str]:
    """Fouten in de koppelingen van een opgeloste Compose-configuratie."""
    fouten: list[str] = []
    canoniek = canoniek or {}

    def canon(pad: str) -> Optional[str]:
        if pad in canoniek:
            return _norm(canoniek[pad]) if canoniek[pad] else None
        return _norm(pad)

    hostpaden: dict[str, str] = {}
    for naam in HOSTVARIABELEN:
        waarde = (host.get(naam) or "").strip()
        if not waarde:
            continue
        reden = onduidelijk(waarde)
        c = canon(waarde)
        if reden or c is None:
            fouten.append(f"{naam} is niet eenduidig ({reden or 'niet op te lossen'})")
        else:
            hostpaden[naam] = c
    if not (host.get("SIGNAAL_VAULT") or "").strip():
        fouten.append("SIGNAAL_VAULT is niet gezet; zonder vault geen controle (fail-closed)")

    diensten = config.get("services") or {}
    fork_env = (diensten.get(DIENST_FORK) or {}).get("environment") or {}
    lijsten = {soort: _lijst(fork_env.get(naam)) for soort, naam in _LIJST.items()}
    vault = hostpaden.get("SIGNAAL_VAULT")
    fork_ro: set[str] = set()

    for naam, dienst in diensten.items():
        for kop in dienst.get("volumes") or []:
            if kop.get("type") != "bind":
                continue
            bron, doel = str(kop.get("source", "")), str(kop.get("target", ""))
            alleen_lezen = bool(kop.get("read_only"))
            if ".." in PurePosixPath(doel).parts:
                fouten.append(f"{naam}: koppelpunt {doel} bevat '..'")
            reden = onduidelijk(bron)
            b = canon(bron)
            if reden or b is None:
                fouten.append(f"{naam}: bron van {doel} is niet eenduidig "
                              f"({reden or 'niet op te lossen'})")
                continue
            if naam == DIENST_FORK and alleen_lezen:
                fork_ro.add(doel)
            if vault and _onder(vault, b) and b != vault:
                fouten.append(f"{naam}: {doel} koppelt een map boven de vault")
            soorten = {s for s, wortels in _DOEL.items() if _doel_onder(doel, wortels)}
            for soort, namen in _HOST.items():
                if any(n in hostpaden and _onder(b, hostpaden[n]) for n in namen):
                    soorten.add(soort)
            for soort in sorted(soorten):
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


def main(argv: list[str]) -> int:
    config = json.load(sys.stdin)
    if argv[:1] == ["--bronnen"]:
        print("\n".join(bind_bronnen(config)))
        return 0
    canoniek = json.loads(os.environ.get("SIGNAAL_CANONIEK") or "{}")
    fouten = controleer(config, os.environ, canoniek)
    for fout in fouten:
        print(f"padlijsten: {fout}", file=sys.stderr)
    if not fouten:
        print("padlijsten: in orde")
    return 1 if fouten else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
