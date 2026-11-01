#!/usr/bin/env bash
# Controle van de netwerkafscherming van het signaalsysteem (F2-10; opzet van P8).
#
# Toetst de draaiende Compose-omgeving (project signaal); start en stopt niets.
#   N1  open-notebook en surrealdb hebben geen internet; voor werkbak-toegang en
#       editor-toegang (F2-12)
#       wordt het vastgelegd (proef masquerade, ADR 0002 F: onder Docker Desktop
#       heeft een proxy op `toegang` internet, daarom de geharde proxy, N7 en N8)
#   N2  geen bereikbare poort van surrealdb of open-notebook op de host
#   N3  surrealdb is alleen bereikbaar vanuit open-notebook, ook niet via zijn IP-adres
#   N4  de API geeft /health HTTP 200 via de werkbakproxy, met alle migraties gedraaid
#   N5  de werkbakproxy op 127.0.0.1 geeft de interface
#   N6  een API-aanroep zonder sleutel geeft 401
#   N7  de proxy is gehard (read_only, cap_drop ALL, no-new-privileges, geen root)
#   N8  de proxy heeft geen shell (exec van sh en busybox faalt)
#   N9  alleen met --met-classificatie: 409 en 422 bij de import (classificatie_e2e.py);
#       schrijft testgegevens, dus alleen tegen een toetsdatabase; het testbestand in
#       het bronarchief wordt daarna verwijderd
#   N10 containers op `toegang` bereiken elkaar niet (enable_icc=false)
#   N11 wat werkbak-toegang en editor-toegang op de host bereiken (host.docker.internal, 192.168.65.254):
#       vergeleken met de bekende uitkomst; een verandering telt als mislukt
#
# Exitcodes: 0 alles geslaagd, 1 een criterium niet gehaald, 2 omgeving ongeschikt.
set -uo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/../../.." && { pwd -W 2>/dev/null || pwd; })"
PROJECT=signaal
PY_IMG="python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337"
POORT="${SIGNAAL_WERKBAK_POORT:-8520}"
MET_CLASSIFICATIE=0
[[ "${1:-}" == "--met-classificatie" ]] && MET_CLASSIFICATIE=1

MISLUKT=0
criterium() { # <naam> <geslaagd|mislukt> <toelichting>
  printf '%-4s %-8s %s\n' "$1" "$2" "$3"
  [[ "$2" == geslaagd ]] || MISLUKT=1
}
container() { docker ps -q --filter "label=com.docker.compose.project=$PROJECT" \
  --filter "label=com.docker.compose.service=$1" | head -n1; }
# Python in de netwerknaamruimte van een container van het project.
in_netwerk_van() { local c=$1; shift; docker run --rm --network "container:$c" "$PY_IMG" python -c "$@"; }

docker info >/dev/null 2>&1 || { echo "Docker niet bereikbaar"; exit 2; }
declare -A C
for d in surrealdb open-notebook werkbak-toegang editor-toegang; do
  C[$d]="$(container "$d")"
  [[ -n "${C[$d]}" ]] || { echo "dienst $d draait niet"; exit 2; }
done
docker image inspect "$PY_IMG" >/dev/null 2>&1 || docker pull -q "$PY_IMG" >/dev/null

PY_INTERNET='
import socket, sys, urllib.error, urllib.request
uit = []
try:
    socket.getaddrinfo("example.org", 443); uit.append("dns=ok")
except Exception as e:
    uit.append("dns=fout:" + type(e).__name__)
try:
    urllib.request.urlopen("https://example.org", timeout=5); uit.append("https=ok"); rc = 0
except urllib.error.HTTPError as e:
    uit.append("https=%d" % e.code); rc = 0       # HTTP-fout: wel verbinding (proeven.md 8.8)
except Exception as e:
    uit.append("https=fout:" + type(e).__name__); rc = 1
try:
    socket.create_connection(("1.1.1.1", 443), timeout=5).close(); uit.append("ip=ok"); rc = 0
except Exception as e:
    uit.append("ip=fout:" + type(e).__name__)
print(" ".join(uit)); sys.exit(rc)
'
# N1: geen internet (rc 1 van het script = geen verbinding).
for d in open-notebook surrealdb werkbak-toegang editor-toegang; do
  uit="$(in_netwerk_van "${C[$d]}" "$PY_INTERNET" 2>&1 | tail -n1)"; rc=$?
  if [[ "$d" == *-toegang ]]; then
    printf '%-4s %-8s %s\n' N1 vastgelegd "$d (proef masquerade): $uit"
  elif [[ "$uit" == *"https=fout"* && "$uit" == *"ip=fout"* ]]; then
    criterium N1 geslaagd "$d heeft geen internet ($uit)"
  else
    criterium N1 mislukt "$d heeft internet ($uit)"
  fi
done

# N2: geen poort van surrealdb of open-notebook op de host.
for p in 8000 5055 8502; do
  rc=0; curl -s -o /dev/null --max-time 5 "http://127.0.0.1:$p/" || rc=$?
  case "$rc" in
    7|28|52|56) criterium N2 geslaagd "127.0.0.1:$p geeft geen antwoord (curl $rc)" ;;
    *) criterium N2 mislukt "127.0.0.1:$p antwoordt (curl $rc)" ;;
  esac
done

# N3: surrealdb alleen vanuit open-notebook.
PY_SURREAL='
import socket, sys
try:
    socket.create_connection(("surrealdb", 8000), timeout=5).close(); print("verbonden")
except Exception as e:
    print("fout:" + type(e).__name__); sys.exit(1)
'
uit="$(in_netwerk_van "${C[open-notebook]}" "$PY_SURREAL" 2>&1 | tail -n1)"
[[ "$uit" == verbonden ]] && criterium N3 geslaagd "open-notebook bereikt surrealdb" \
  || criterium N3 mislukt "open-notebook bereikt surrealdb niet ($uit)"
uit="$(in_netwerk_van "${C[werkbak-toegang]}" "$PY_SURREAL" 2>&1 | tail -n1)"
[[ "$uit" == fout:* ]] && criterium N3 geslaagd "werkbak-toegang bereikt surrealdb niet ($uit)" \
  || criterium N3 mislukt "werkbak-toegang bereikt surrealdb ($uit)"
for net in verwerking werkbak-intern; do
  uit="$(docker run --rm --network "${PROJECT}_$net" "$PY_IMG" python -c "$PY_SURREAL" 2>&1 | tail -n1)"
  [[ "$uit" == fout:* ]] && criterium N3 geslaagd "container op $net bereikt surrealdb niet ($uit)" \
    || criterium N3 mislukt "container op $net bereikt surrealdb ($uit)"
done
# Ook niet via het IP-adres (geen DNS nodig).
PY_TCP='
import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=5).close(); print("verbonden")
except Exception as e:
    print("fout:" + type(e).__name__); sys.exit(1)
'
tcp_van() { local c=$1; shift; docker run --rm --network "container:$c" "$PY_IMG" python -c "$PY_TCP" "$@" 2>&1 | tail -n1; }
surreal_ip="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${C[surrealdb]}")"
uit="$(tcp_van "${C[werkbak-toegang]}" "$surreal_ip" 8000)"
[[ "$uit" == fout:* ]] && criterium N3 geslaagd "werkbak-toegang bereikt surrealdb niet via $surreal_ip ($uit)" \
  || criterium N3 mislukt "werkbak-toegang bereikt surrealdb via $surreal_ip ($uit)"
uit="$(docker run --rm --network "${PROJECT}_verwerking" "$PY_IMG" python -c "$PY_TCP" "$surreal_ip" 8000 2>&1 | tail -n1)"
[[ "$uit" == fout:* ]] && criterium N3 geslaagd "container op verwerking bereikt surrealdb niet via $surreal_ip ($uit)" \
  || criterium N3 mislukt "container op verwerking bereikt surrealdb via $surreal_ip ($uit)"

# N4: /health en migraties.
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://127.0.0.1:$POORT/health")" || true
[[ "$code" == 200 ]] && criterium N4 geslaagd "/health via de proxy: HTTP 200" \
  || criterium N4 mislukt "/health via de proxy: HTTP $code"
hoogste="$(ls "$ROOT/migrations" | sed -n 's/^\([0-9]\+\)\.surrealql$/\1/p' | sort -n | tail -n1)"
versie="$(docker exec "${C[open-notebook]}" /app/.venv/bin/python /app/services/open-notebook/start.py \
  /app/.venv/bin/python -c 'import asyncio
from loguru import logger; logger.remove()
from surrealdb_service.migrations import AsyncMigrationManager
print(asyncio.run(AsyncMigrationManager().get_current_version()))' 2>/dev/null | tail -n1)" || true
[[ "$versie" == "$hoogste" ]] && criterium N4 geslaagd "migratieversie $versie (hoogste $hoogste)" \
  || criterium N4 mislukt "migratieversie $versie, verwacht $hoogste"

# N5: de interface via de proxy.
rc=0; html="$(curl -s -L --max-time 20 "http://127.0.0.1:$POORT/")" || rc=$?
code="$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 20 "http://127.0.0.1:$POORT/")" || true
if [[ "$code" == 200 && "$html" == *"<html"* ]]; then
  criterium N5 geslaagd "interface via 127.0.0.1:$POORT: HTTP 200, HTML"
else
  criterium N5 mislukt "interface via 127.0.0.1:$POORT: HTTP $code (curl $rc)"
fi

# N6: zonder sleutel 401.
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://127.0.0.1:$POORT/api/v1/auth/scope")" || true
[[ "$code" == 401 ]] && criterium N6 geslaagd "API zonder sleutel: HTTP 401" \
  || criterium N6 mislukt "API zonder sleutel: HTTP $code"

# N7: geharde proxy.
insp="$(docker inspect -f '{{.HostConfig.ReadonlyRootfs}}|{{json .HostConfig.CapDrop}}|{{json .HostConfig.SecurityOpt}}|{{.Config.User}}|{{len .Mounts}}' "${C[werkbak-toegang]}")"
IFS='|' read -r ro caps sec gebruiker mounts <<<"$insp"
if [[ "$ro" == true && "$caps" == '["ALL"]' && "$sec" == *no-new-privileges:true* \
      && -n "$gebruiker" && "$gebruiker" != 0* && "$gebruiker" != root* && "$mounts" == 0 ]]; then
  criterium N7 geslaagd "werkbak-toegang: read_only, cap_drop ALL, no-new-privileges, gebruiker $gebruiker, geen koppelingen"
else
  criterium N7 mislukt "werkbak-toegang: $insp"
fi

# N8: geen shell in de proxy.
shells=""
for sh in /bin/sh /bin/ash /bin/busybox /bin/bash; do
  docker exec "${C[werkbak-toegang]}" "$sh" -c true >/dev/null 2>&1 && shells+=" $sh"
done
[[ -z "$shells" ]] && criterium N8 geslaagd "werkbak-toegang heeft geen shell"   || criterium N8 mislukt "werkbak-toegang heeft een shell:$shells"

# N10: geen verkeer tussen containers op toegang.
proxy_ip="$(docker inspect -f "{{(index .NetworkSettings.Networks \"${PROJECT}_toegang\").IPAddress}}" "${C[werkbak-toegang]}")"
uit="$(docker run --rm --network "${PROJECT}_toegang" "$PY_IMG" python -c "$PY_TCP" "$proxy_ip" 8080 2>&1 | tail -n1)"
[[ "$uit" == fout:* ]] && criterium N10 geslaagd "andere container op toegang bereikt werkbak-toegang ($proxy_ip:8080) niet ($uit)" \
  || criterium N10 mislukt "andere container op toegang bereikt werkbak-toegang ($proxy_ip:8080)"

# N11: wat de proxy op de host bereikt; bekende uitkomst onder Docker Desktop
# (F2-10, 1 november 2026, met Ollama actief op de host): host.docker.internal is
# 192.168.65.254 en geeft de proxy toegang tot poorten op 127.0.0.1 van de host.
# Poort 8000 (SurrealDB van de oude stack) hoort dicht te zijn: start.sh weigert
# te starten als de oude stack draait.
BEKEND_HOST="${SIGNAAL_BEKEND_HOST:-hdi=192.168.65.254 host.docker.internal:${POORT}=open 192.168.65.254:${POORT}=open 192.168.65.254:11434=open 192.168.65.254:22=dicht 192.168.65.254:8000=dicht}"
PY_HOST='
import socket, sys
uit = []
try:
    uit.append("hdi=" + socket.gethostbyname("host.docker.internal"))
except Exception as e:
    uit.append("hdi=fout")
for h, p in (("host.docker.internal", int(sys.argv[1])), ("192.168.65.254", int(sys.argv[1])),
             ("192.168.65.254", 11434), ("192.168.65.254", 22), ("192.168.65.254", 8000)):
    try:
        socket.create_connection((h, p), timeout=3).close(); r = "open"
    except Exception:
        r = "dicht"
    uit.append("%s:%d=%s" % (h, p, r))
print(" ".join(uit))
'
for d in werkbak-toegang editor-toegang; do
  host_uit="$(in_netwerk_van "${C[$d]}" "$PY_HOST" "$POORT" 2>&1 | tail -n1)"
  if [[ "$host_uit" == "$BEKEND_HOST" ]]; then
    printf '%-4s %-8s %s\n' N11 vastgelegd "$d naar de host, zoals bekend: $host_uit"
  else
    criterium N11 mislukt "$d naar de host wijkt af van de bekende uitkomst: $host_uit (bekend: $BEKEND_HOST)"
  fi
done

# N9: classificatie end-to-end (schrijft testgegevens).
if [[ $MET_CLASSIFICATIE == 1 ]]; then
  naam="f2-10-e2e-$(date +%s).bin"
  printf '%%PDF-1.7 synthetisch testbestand F2-10 %s' "$naam" | docker run --rm -i --network none \
    -v "${PROJECT}_bronarchief:/b" "$PY_IMG" sh -c "cat > /b/$naam"
  uit="$(docker exec -i "${C[open-notebook]}" /app/.venv/bin/python /app/services/open-notebook/start.py \
    /app/.venv/bin/python - "$naam" <"$ROOT/scripts/signaalsysteem/controle/classificatie_e2e.py" 2>&1)"; rc=$?
  printf '%s\n' "$uit" | grep '"criterium"' || printf '%s\n' "$uit" | tail -n5
  docker run --rm --network none -v "${PROJECT}_bronarchief:/b" "$PY_IMG" rm -f "/b/$naam"
  [[ $rc == 0 ]] && criterium N9 geslaagd "import: 409 lagere_classificatie, 422 ongeldige waarde, geen 500" \
    || criterium N9 mislukt "classificatie_e2e.py exitcode $rc"
fi

exit $MISLUKT
