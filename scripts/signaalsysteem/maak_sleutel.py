"""Maak een sleutel voor de API van de fork (F2-01, ADR 0002 A).

Twee vormen:

  beheerder  de persoonlijke sleutel met recht ``beheerder``. De sleutel staat
             één keer op het scherm (stdout) en nergens anders: niet in een log,
             niet in een bestand.

      python scripts/signaalsysteem/maak_sleutel.py beheerder --persoon "Naam"

  dienst     een dienstsleutel (``pijplijn``, ``lezen_extern``, ``lezen_intern``
             of ``ophalen``) als geheimbestand voor Docker secrets. Op het scherm
             staat alleen het pad en het niet-geheime voorvoegsel.

      python scripts/signaalsysteem/maak_sleutel.py dienst --recht pijplijn \\
          --agent langflow --bestand secrets/sleutel_langflow

Het script praat met SurrealDB via de gewone instellingen (``SURREAL_URL`` en
verder). Alleen de SHA-256-hash van de sleutel komt in de database.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

DIENSTRECHTEN = ("pijplijn", "lezen_extern", "lezen_intern", "ophalen")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Maak een sleutel voor de API.")
    sub = p.add_subparsers(dest="vorm", required=True)

    b = sub.add_parser("beheerder", help="persoonlijke sleutel, één keer getoond")
    b.add_argument("--persoon", required=True, help="naam van de persoon")
    b.add_argument("--agent", default="werkbak", help="agent-id (standaard werkbak)")

    d = sub.add_parser("dienst", help="dienstsleutel als geheimbestand")
    d.add_argument("--recht", required=True, choices=DIENSTRECHTEN)
    d.add_argument("--agent", required=True, help="agent-id, bijvoorbeeld langflow")
    d.add_argument("--bestand", required=True, type=Path, help="pad onder secrets/")
    d.add_argument("--persoon", default=None, help="optioneel: verantwoordelijke")
    return p


def schrijf_geheim(pad: Path, sleutel: str) -> None:
    """Schrijf de sleutel naar een nieuw bestand, alleen leesbaar voor de eigenaar.

    Een bestaand bestand wordt niet overschreven: een tweede sleutel voor
    dezelfde dienst is een bewuste handeling (eerst de oude intrekken).
    """
    pad.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(pad, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(sleutel)


async def maak(
    vorm: str,
    *,
    recht: str,
    agent: str,
    persoon: Optional[str],
    bestand: Optional[Path],
    service=None,
) -> str:
    """Maak de sleutel en geef terug wat op het scherm mag."""
    if service is None:
        from app_main.services.agents.key_service import AgentKeyService

        service = AgentKeyService()
    if bestand is not None and bestand.exists():
        raise FileExistsError(f"{bestand} bestaat al")
    public, sleutel = await service.mint(
        agent_id=agent, permission=recht, person=persoon, label=f"maak_sleutel {vorm}"
    )
    if vorm == "beheerder":
        return (
            f"Sleutel voor {persoon} (recht beheerder). Bewaar hem nu; hij wordt "
            f"niet nog eens getoond:\n{sleutel}"
        )
    schrijf_geheim(bestand, sleutel)
    return f"Dienstsleutel {public['key_prefix']}… ({recht}) geschreven naar {bestand}"


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    # Geen log: loguru schrijft anders bij een fout de queryparameters weg.
    from loguru import logger

    logger.remove()
    recht = "beheerder" if args.vorm == "beheerder" else args.recht
    try:
        tekst = asyncio.run(
            maak(
                args.vorm,
                recht=recht,
                agent=args.agent,
                persoon=args.persoon,
                bestand=getattr(args, "bestand", None),
            )
        )
    except (ValueError, FileExistsError) as exc:
        print(f"Fout: {exc}", file=sys.stderr)
        return 1
    print(tekst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
