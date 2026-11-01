#!/usr/bin/env bash
# Controle van de referentiegegevens (F2-28; ADR 0021, 0013).
#
# Toetst de draaiende Compose-omgeving (project signaal); start en stopt niets.
# Bedoeld voor een nieuwe, lege database (bijv. SIGNAAL_SURREAL_DATABASE=toets_f228).
#   R1  alle migraties gedraaid (versie = hoogste migratie in migrations/)
#   R2  de registers uit config/referentiegegevens.yaml staan op de host, met de
#       SHA-256 uit de inventaris (download als ze ontbreken; niet in Git); de
#       container heeft geen internet
#   R3  eerste run van laad_referentiegegevens.py in de container open-notebook
#       (verbinding van de migraties): per set geladen = in de database = bron
#   R4  tweede run: dezelfde aantallen en dezelfde record-id's (geen nieuwe records)
#   R5  de ontologieën: SHA-256 klopt en de OntologyManager vindt ze allemaal,
#       zoals de extractie (en dus de harness van n_extraction_gate.py) ze opzoekt
#
# Exitcodes: 0 alles geslaagd, 1 een criterium niet gehaald, 2 omgeving of bron
# ongeschikt, 3 scriptfout.
set -uo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/../../.." && { pwd -W 2>/dev/null || pwd; })"
PROJECT=signaal
LAAD="$ROOT/scripts/signaalsysteem/laad_referentiegegevens.py"
CACHE="${SIGNAAL_STAAT:-$HOME/.cache/signaal}/referentiegegevens"
DOEL=/tmp/referentiegegevens

MISLUKT=0
criterium() { # <naam> <geslaagd|mislukt> <toelichting>
  printf '%-3s %-8s %s\n' "$1" "$2" "$3"
  [[ "$2" == geslaagd ]] || MISLUKT=1
}

docker info >/dev/null 2>&1 || { echo "Docker niet bereikbaar"; exit 2; }
C="$(docker ps -q --filter "label=com.docker.compose.project=$PROJECT" \
  --filter "label=com.docker.compose.service=open-notebook" | head -n1)"
[[ -n "$C" ]] || { echo "dienst open-notebook draait niet"; exit 2; }
# Het laadscript gaat via stdin; het draait met de geheimen uit start.py.
in_container() {
  docker exec -i "$C" /app/.venv/bin/python /app/services/open-notebook/start.py \
    /app/.venv/bin/python - "$@" <"$LAAD"
}
opruimen() { docker exec "$C" rm -rf "$DOEL" >/dev/null 2>&1 || true; }

# R1: migraties.
hoogste="$(ls "$ROOT/migrations" | sed -n 's/^\([0-9]\+\)\.surrealql$/\1/p' | sort -n | tail -n1)"
versie="$(docker exec "$C" /app/.venv/bin/python /app/services/open-notebook/start.py \
  /app/.venv/bin/python -c 'import asyncio
from loguru import logger; logger.remove()
from surrealdb_service.migrations import AsyncMigrationManager
print(asyncio.run(AsyncMigrationManager().get_current_version()))' 2>/dev/null | tail -n1)" || true
[[ "$versie" == "$hoogste" ]] && criterium R1 geslaagd "migratieversie $versie" \
  || criterium R1 mislukt "migratieversie $versie, verwacht $hoogste"

# R2: bronbestanden op de host, met SHA-256-controle.
lijst="$(in_container --downloadlijst | tr -d '\r')" || { echo "downloadlijst mislukt"; exit 3; }
mkdir -p "$CACHE"
CACHE_W="$(cd "$CACHE" && { pwd -W 2>/dev/null || pwd; })"
aantal=0
while IFS=$'\t' read -r bestand url sha; do
  [[ -n "$bestand" ]] || continue
  doel="$CACHE/$bestand"
  if [[ ! -f "$doel" || "$(sha256sum "$doel" | cut -d' ' -f1)" != "$sha" ]]; then
    curl -fsS --max-time 300 -o "$doel.deel" "$url" || { rm -f "$doel.deel"; echo "download mislukt: $url"; exit 2; }
    mv -f "$doel.deel" "$doel"
  fi
  werkelijk="$(sha256sum "$doel" | cut -d' ' -f1)"
  if [[ "$werkelijk" != "$sha" ]]; then
    rm -f "$doel"
    echo "SHA-256 van $bestand is $werkelijk, verwacht $sha"; exit 2
  fi
  aantal=$((aantal + 1))
done <<<"$lijst"
criterium R2 geslaagd "$aantal registers op de host met de SHA-256 uit de inventaris ($CACHE)"

# R3 en R4: twee runs in de container.
opruimen
docker cp "$CACHE_W" "$C:$DOEL" >/dev/null || { echo "kopiëren naar de container mislukt"; exit 2; }
een="$(in_container --bronmap "$DOEL" | tr -d '\r')"; rc1=$?
twee="$(in_container --bronmap "$DOEL" | tr -d '\r')"; rc2=$?
opruimen
echo "eerste run (exitcode $rc1):"; printf '%s\n' "$een" | sed 's/^/  /'
echo "tweede run (exitcode $rc2):"; printf '%s\n' "$twee" | sed 's/^/  /'
[[ $rc1 -eq 2 || $rc2 -eq 2 ]] && { echo "bron ongeschikt"; exit 2; }
[[ $rc1 -eq 3 || $rc2 -eq 3 ]] && { echo "scriptfout in het laadscript"; exit 3; }

tooi="$(printf '%s\n' "$een" | grep '"naam": "tooi_overheidsorganisaties"' || true)"
[[ $rc1 -eq 0 && -n "$tooi" ]] && criterium R3 geslaagd "eerste run: per set geladen = in de database = bron" \
  || criterium R3 mislukt "eerste run: exitcode $rc1"
[[ $rc2 -eq 0 && -n "$een" && "$een" == "$twee" ]] \
  && criterium R4 geslaagd "tweede run: zelfde aantallen en zelfde record-id's" \
  || criterium R4 mislukt "tweede run wijkt af (exitcode $rc2)"
onto="$(printf '%s\n' "$twee" | grep '"naam": "ontologieen"' || true)"
[[ "$onto" == *'"geslaagd": true'* ]] && criterium R5 geslaagd "ontologieën gevonden: $onto" \
  || criterium R5 mislukt "ontologieën: ${onto:-geen verslag}"

exit "$MISLUKT"
