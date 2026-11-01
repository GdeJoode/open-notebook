#!/usr/bin/env bash
# Stopt de Compose-omgeving van het signaalsysteem (F2-10), zonder volumes te
# verwijderen. Compose vraagt bij elke opdracht de verplichte hostpaden; voor
# `down` doen hun waarden er niet toe. Gebruik niet een kale
# `docker compose -p signaal down` in de root: die leest docker-compose.yml.
set -euo pipefail
export MSYS_NO_PATHCONV=1
ROOT="$(cd "$(dirname "$0")/../.." && { pwd -W 2>/dev/null || pwd; })"
for naam in SIGNAAL_INNAMEMAP SIGNAAL_WATCHFOLDER SIGNAAL_FLOWS_KLOON; do
  export "$naam=${!naam:-$ROOT}"   # bij down wordt niets gekoppeld
done
exec docker compose -p signaal -f "$ROOT/docker-compose.signaalsysteem.yml" down "$@"
