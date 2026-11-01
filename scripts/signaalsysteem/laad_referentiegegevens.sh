#!/usr/bin/env bash
# Beheerstap: laadt de referentiegegevens in de draaiende omgeving (F2-28).
#
# 1. Downloadt op de host de registers uit config/referentiegegevens.yaml naar de
#    cache (${SIGNAAL_STAAT:-~/.cache/signaal}/referentiegegevens), met de SHA-256
#    uit de inventaris; niet in Git. De container heeft geen internet.
# 2. Kopieert de cache naar /app/data/referentiegegevens in de container
#    open-notebook (volume fork_data); daar leest ook POST /api/vocabulary/refresh.
# 3. Draait laad_referentiegegevens.py in de container, met de verbinding van de
#    migraties (start.py leest de geheimen). Het verslag (één JSON-regel per set)
#    komt op stdout; meldingen op stderr.
#
# Exitcodes: die van het laadscript (0 geslaagd, 1 een aantal klopt niet, 2 bron
# ontbreekt of wijkt af, 3 scriptfout); 2 ook als Docker of de dienst ontbreekt of
# een download mislukt.
set -uo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/../.." && { pwd -W 2>/dev/null || pwd; })"
PROJECT=signaal
LAAD="$ROOT/scripts/signaalsysteem/laad_referentiegegevens.py"
CACHE="${SIGNAAL_STAAT:-$HOME/.cache/signaal}/referentiegegevens"
DOEL=/app/data/referentiegegevens

docker info >/dev/null 2>&1 || { echo "Docker niet bereikbaar" >&2; exit 2; }
C="$(docker ps -q --filter "label=com.docker.compose.project=$PROJECT" \
  --filter "label=com.docker.compose.service=open-notebook" | head -n1)"
[[ -n "$C" ]] || { echo "dienst open-notebook draait niet" >&2; exit 2; }
in_container() {
  docker exec -i "$C" /app/.venv/bin/python /app/services/open-notebook/start.py \
    /app/.venv/bin/python - "$@" <"$LAAD"
}

# 1. Download met SHA-256-controle.
lijst="$(in_container --downloadlijst | tr -d '\r')" || { echo "downloadlijst mislukt" >&2; exit 3; }
mkdir -p "$CACHE"
# Windows-vorm: curl van Git Bash is een mingw-programma, en docker cp wil hem ook.
CACHE_W="$(cd "$CACHE" && { pwd -W 2>/dev/null || pwd; })"
aantal=0
while IFS=$'\t' read -r bestand url sha; do
  [[ -n "$bestand" ]] || continue
  doel="$CACHE_W/$bestand"
  if [[ ! -f "$doel" || "$(sha256sum "$doel" | cut -d' ' -f1)" != "$sha" ]]; then
    curl -fsS --max-time 300 -o "$doel.deel" "$url" \
      || { rm -f "$doel.deel"; echo "download mislukt: $url" >&2; exit 2; }
    mv -f "$doel.deel" "$doel"
  fi
  werkelijk="$(sha256sum "$doel" | cut -d' ' -f1)"
  if [[ "$werkelijk" != "$sha" ]]; then
    rm -f "$doel"
    echo "SHA-256 van $bestand is $werkelijk, verwacht $sha" >&2; exit 2
  fi
  aantal=$((aantal + 1))
done <<<"$lijst"
echo "$aantal registers in $CACHE met de SHA-256 uit de inventaris" >&2

# 2. Naar de container (vervangt een eerdere kopie).
docker exec "$C" rm -rf "$DOEL" || { echo "opruimen van $DOEL mislukt" >&2; exit 2; }
docker cp "$CACHE_W" "$C:$DOEL" >/dev/null || { echo "kopiëren naar de container mislukt" >&2; exit 2; }

# 3. Laden.
in_container --bronmap "$DOEL"
