"""Startscript van de container ``open-notebook`` (signaalsysteem, F2-10).

1. Leest de inloggegevens van SurrealDB uit de Docker secrets
   (``/run/secrets/surreal_gebruiker`` en ``surreal_wachtwoord``) in de omgeving
   van het proces, niet in die van de container (ADR 0002 F).
2. Startcontrole van de koppelingen (ADR 0010 en 0020, "Uitwerking in F2-14"):
   * elke container-map in ``SIGNAAL_INNAME_PADEN`` is een alleen-lezen
     koppeling (de fork hasht daar batchbestanden en watchfolder-items, B29);
   * ``SIGNAAL_BRONARCHIEF`` is alleen-lezen gekoppeld, ``SIGNAAL_FORK_ARCHIEF``
     schrijfbaar;
   * geen koppeling ligt in of onder een pad uit ``SIGNAAL_VAULT_PADEN``.
   Faalt de controle, dan start de API niet (exitcode 1).
3. Start supervisord (API en interface).

Met argumenten (``start.py <opdracht> ...``) worden alleen de geheimen gelezen
en wordt de opdracht uitgevoerd, zonder controle; zo draait een beheerscript
in de container met dezelfde verbinding als de API.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path, PurePosixPath

GEHEIMEN = Path("/run/secrets")
SUPERVISORD = ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]


def lees_geheimen(map_: Path = GEHEIMEN) -> dict[str, str]:
    """De omgevingsvariabelen van SurrealDB uit de geheimbestanden."""
    return {
        "SURREAL_USER": (map_ / "surreal_gebruiker").read_text(encoding="utf-8").strip(),
        "SURREAL_PASSWORD": (map_ / "surreal_wachtwoord").read_text(encoding="utf-8").strip(),
    }


def koppelingen(mountinfo: str) -> dict[str, set[str]]:
    """Koppelpunt -> opties, uit ``/proc/self/mountinfo``."""
    uit: dict[str, set[str]] = {}
    for regel in mountinfo.splitlines():
        velden = regel.split(" ")
        if len(velden) < 6:
            continue
        punt = velden[4].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")
        uit[punt] = set(velden[5].split(","))
    return uit


def _lijst(waarde: str | None) -> list[str]:
    return [p.strip() for p in (waarde or "").split(";") if p.strip()]


def _posix(pad: str) -> PurePosixPath | None:
    """Alleen absolute paden in de container tellen; een Windows-pad niet."""
    return PurePosixPath(pad) if pad.startswith("/") else None


def controleer_koppelingen(kop: dict[str, set[str]], env: dict[str, str]) -> list[str]:
    """Fouten in de koppelingen; leeg als alles klopt."""
    fouten: list[str] = []

    def eis(pad: str, optie: str, wat: str) -> None:
        if pad not in kop:
            fouten.append(f"{wat} {pad} is niet gekoppeld")
        elif optie not in kop[pad]:
            fouten.append(f"{wat} {pad} is niet {'alleen-lezen' if optie == 'ro' else 'schrijfbaar'}")

    for pad in _lijst(env.get("SIGNAAL_INNAME_PADEN")):
        if _posix(pad) is None:
            fouten.append(f"innamemap {pad} is geen pad in de container")
        else:
            eis(pad, "ro", "innamemap")
    eis(env.get("SIGNAAL_BRONARCHIEF") or "/bronarchief", "ro", "bronarchief")
    eis(env.get("SIGNAAL_FORK_ARCHIEF") or "/data/forkarchief", "rw", "forkarchief")
    for vault in filter(None, map(_posix, _lijst(env.get("SIGNAAL_VAULT_PADEN")))):
        for punt in kop:
            p = PurePosixPath(punt)
            if p == vault or vault in p.parents:
                fouten.append(f"koppeling {punt} ligt in de vault; de fork koppelt de vault niet")
    return fouten


def main(argv: list[str]) -> int:
    env = dict(os.environ)
    env.update(lees_geheimen())
    if argv:
        os.execvpe(argv[0], argv, env)
    fouten = controleer_koppelingen(
        koppelingen(Path("/proc/self/mountinfo").read_text(encoding="utf-8")), env)
    if fouten:
        for fout in fouten:
            print(f"startcontrole: {fout}", file=sys.stderr)
        return 1
    os.execvpe(SUPERVISORD[0], SUPERVISORD, env)
    return 0  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
