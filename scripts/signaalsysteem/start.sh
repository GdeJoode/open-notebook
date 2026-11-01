#!/usr/bin/env bash
# Start de Compose-omgeving van het signaalsysteem (F2-10).
#
# Eerst de startcontrole van de padlijsten (controle/padlijsten.py) op de
# opgeloste configuratie; faalt die, dan start er niets (exitcode 1). Daarna
# `docker compose up -d`. Extra argumenten gaan naar `up`, bijvoorbeeld
# `--build` of een dienstnaam.
#
# Hostpaden (SIGNAAL_INNAMEMAP, SIGNAAL_WATCHFOLDER, SIGNAAL_VAULT,
# SIGNAAL_EXPORTMAP) komen uit de omgeving of uit .env (voorbeeld in .env.example).
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/../.." && { pwd -W 2>/dev/null || pwd; })"
COMPOSE=(docker compose -p signaal -f "$ROOT/docker-compose.signaalsysteem.yml")
# Zelfde basisimage als de fork; de controle draait zonder netwerk.
PY_IMG="python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337"

# Alleen de hostpaden uit .env overnemen, geen andere waarden.
for naam in SIGNAAL_INNAMEMAP SIGNAAL_WATCHFOLDER SIGNAAL_VAULT SIGNAAL_EXPORTMAP; do
  if [[ -z "${!naam:-}" && -f "$ROOT/.env" ]]; then
    waarde="$(grep -E "^${naam}=" "$ROOT/.env" | tail -n1 | cut -d= -f2- | tr -d '\r' || true)"
    [[ -n "$waarde" ]] && export "$naam=$waarde"
  fi
done

config="$("${COMPOSE[@]}" config --format json)"
printf '%s' "$config" | docker run --rm -i --network none \
  -e SIGNAAL_INNAMEMAP -e SIGNAAL_WATCHFOLDER -e SIGNAAL_VAULT -e SIGNAAL_EXPORTMAP \
  "$PY_IMG" python -c "$(cat "$ROOT/scripts/signaalsysteem/controle/padlijsten.py")"

exec "${COMPOSE[@]}" up -d "$@"
