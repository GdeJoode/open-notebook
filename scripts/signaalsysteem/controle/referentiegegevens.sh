#!/usr/bin/env bash
# Controle van de referentiegegevens (F2-28; ADR 0021, 0013).
#
# Toetst de draaiende Compose-omgeving (project signaal); start en stopt niets.
# Bedoeld voor een nieuwe, lege database (bijv. SIGNAAL_SURREAL_DATABASE=toets_f228).
# Draait de beheerstap scripts/signaalsysteem/laad_referentiegegevens.sh twee keer.
#   R0  meldt of config/referentiegegevens.yaml afwijkt van HEAD (de container
#       leest de werkmap; telt niet als mislukt)
#   R1  alle migraties gedraaid (versie = hoogste migratie in migrations/)
#   R2  de registers staan op de host met de SHA-256 uit de inventaris (download
#       als ze ontbreken; niet in Git); de container heeft geen internet
#   R3  eerste lading: per set geladen = in de database = bron
#   R4  tweede lading: zelfde aantallen, zelfde record-id's en zelfde inhoud.
#       `last_validated` telt niet mee: elke lading zet dat veld opnieuw (het
#       moment van de laatste controle tegen de bron); de inhoud blijft gelijk.
#   R5  de ontologieën: SHA-256 klopt en de OntologyManager vindt ze allemaal,
#       zoals de extractie ze opzoekt (list_ontologies, get_ontology). Meldt welke
#       een actieve versie in ontology_version hebben: daarvoor gebruikt de
#       extractie die versie, niet het YAML-bestand, en zegt de hash niets.
#       R5 vervangt `scripts/n_extraction_gate.py` uit de stopconditie: dat script
#       laadt zelf geen ontologieën, maar vergelijkt de JSON van een gemeten run
#       met een baseline; zo'n run vraagt een model en documenten. Wat de gate
#       nodig heeft van de database, is dat de extractie de ontologieën vindt.
#
# Exitcodes: 0 alles geslaagd, 1 een criterium niet gehaald, 2 omgeving of bron
# ongeschikt, 3 scriptfout.
set -uo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/../../.." && { pwd -W 2>/dev/null || pwd; })"
PROJECT=signaal
BEHEER="$ROOT/scripts/signaalsysteem/laad_referentiegegevens.sh"
INVENTARIS=config/referentiegegevens.yaml

MISLUKT=0
criterium() { # <naam> <geslaagd|mislukt|let op> <toelichting>
  printf '%-3s %-8s %s\n' "$1" "$2" "$3"
  [[ "$2" == mislukt ]] && MISLUKT=1
}

docker info >/dev/null 2>&1 || { echo "Docker niet bereikbaar"; exit 2; }
C="$(docker ps -q --filter "label=com.docker.compose.project=$PROJECT" \
  --filter "label=com.docker.compose.service=open-notebook" | head -n1)"
[[ -n "$C" ]] || { echo "dienst open-notebook draait niet"; exit 2; }

# R0: de inventaris tegen HEAD.
if [[ -n "$(git -C "$ROOT" status --porcelain -- "$INVENTARIS")" ]]; then
  criterium R0 "let op" "$INVENTARIS wijkt af van HEAD; de toets gebruikt de werkmap"
else
  criterium R0 geslaagd "$INVENTARIS gelijk aan HEAD ($(git -C "$ROOT" rev-parse --short HEAD))"
fi

# R1: migraties.
hoogste="$(ls "$ROOT/migrations" | sed -n 's/^\([0-9]\+\)\.surrealql$/\1/p' | sort -n | tail -n1)"
versie="$(docker exec "$C" /app/.venv/bin/python /app/services/open-notebook/start.py \
  /app/.venv/bin/python -c 'import asyncio
from loguru import logger; logger.remove()
from surrealdb_service.migrations import AsyncMigrationManager
print(asyncio.run(AsyncMigrationManager().get_current_version()))' 2>/dev/null | tail -n1)" || true
[[ "$versie" == "$hoogste" ]] && criterium R1 geslaagd "migratieversie $versie" \
  || criterium R1 mislukt "migratieversie $versie, verwacht $hoogste"

# R2 tot en met R4: de beheerstap twee keer.
een="$(bash "$BEHEER" | tr -d '\r')"; rc1=$?
twee="$(bash "$BEHEER" | tr -d '\r')"; rc2=$?
echo "eerste lading (exitcode $rc1):"; printf '%s\n' "$een" | sed 's/^/  /'
echo "tweede lading (exitcode $rc2):"; printf '%s\n' "$twee" | sed 's/^/  /'
[[ $rc1 -eq 3 || $rc2 -eq 3 ]] && { echo "scriptfout in de beheerstap of het laadscript"; exit 3; }
[[ $rc1 -eq 2 || $rc2 -eq 2 ]] && { criterium R2 mislukt "bron ontbreekt of wijkt af"; exit 2; }
criterium R2 geslaagd "registers op de host en in de container met de SHA-256 uit de inventaris"

tooi="$(printf '%s\n' "$een" | grep '"naam": "tooi_overheidsorganisaties"' || true)"
[[ $rc1 -eq 0 && -n "$tooi" ]] && criterium R3 geslaagd "eerste lading: per set geladen = in de database = bron" \
  || criterium R3 mislukt "eerste lading: exitcode $rc1"
[[ $rc2 -eq 0 && -n "$een" && "$een" == "$twee" ]] \
  && criterium R4 geslaagd "tweede lading: zelfde aantallen, record-id's en inhoud (zonder last_validated)" \
  || criterium R4 mislukt "tweede lading wijkt af (exitcode $rc2)"
onto="$(printf '%s\n' "$twee" | grep '"naam": "ontologieen"' || true)"
if [[ "$onto" == *'"geslaagd": true'* ]]; then
  if [[ "$onto" == *'"actief_in_db": []'* ]]; then
    criterium R5 geslaagd "ontologieën gevonden, geen actieve versie in ontology_version: de extractie gebruikt de YAML"
  else
    criterium R5 geslaagd "ontologieën gevonden; met een actieve versie in ontology_version (die gebruikt de extractie, niet de YAML): $onto"
  fi
else
  criterium R5 mislukt "ontologieën: ${onto:-geen verslag}"
fi

exit "$MISLUKT"
