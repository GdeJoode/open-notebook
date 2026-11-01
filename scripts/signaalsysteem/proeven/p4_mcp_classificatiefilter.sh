#!/usr/bin/env bash
# P4. open-notebook-mcp met het classificatiefilter: geen intern fragment in een
# antwoord (docs/signaalsysteem/proeven.md, P4).
#
# Eerste uitvoering in fase 1 is een nulmeting: de fork heeft het filter nog niet,
# dus K1 hoort te falen (exitcode 1). Na de bouwtaken voor ADR 0001 en 0002 is dit
# script de acceptatietest.
#
# Het image van de fork wordt gebouwd uit de laatste commit die de code van de fork
# raakt (buiten docs/ en scripts/signaalsysteem/), via git archive, dus zonder
# ongetrackte mappen als notebook_data_live in de buildcontext. In die kopie
# worden alleen de zwevende basis-tags vervangen door vaste tags, zodat de lokale
# tags van de bestaande omgeving niet van versie wisselen.
source "$(dirname "$0")/lib/gemeenschappelijk.sh"

proef_start p4 p4
p0_voorwaarde

REPO="$(cd "$PROEF_SCRIPTS/../../.." && pwd)"
SP_COMMIT="$(git -C "$(pad_host "$REPO")" log -1 --format=%h -- . ':(exclude)docs' ':(exclude)scripts/signaalsysteem' ':(exclude).claude')"
export SP_COMMIT
meet fork_commit "$SP_COMMIT"
FORK_IMG="signaal-proef/open-notebook:$SP_COMMIT"

# --- Image van de fork bouwen (als het nog niet bestaat) ---
if ! docker image inspect "$FORK_IMG" >/dev/null 2>&1; then
  BRON="$SIGNAAL_PROEF_DIR/p4/bron-$SP_COMMIT"
  rm -rf "$BRON"; mkdir -p "$BRON"
  git -C "$(pad_host "$REPO")" archive --format=tar "$SP_COMMIT" | tar -x -C "$BRON"
  sed -i -e "s#^FROM python:3.12-slim-bookworm#FROM $IMG_PY#" \
         -e "s#ghcr.io/astral-sh/uv:latest#$IMG_UV#" "$BRON/Dockerfile"
  diff <(git -C "$(pad_host "$REPO")" show "$SP_COMMIT:Dockerfile") "$BRON/Dockerfile" >"$RUNMAP/dockerfile_aanpassing.diff" || true
  log "image $FORK_IMG bouwen uit $BRON (dit duurt lang)"
  rc=0
  bewaar build_fork docker build -t "$FORK_IMG" "$(pad_host "$BRON")" || rc=$?
  if [[ $rc -ne 0 ]]; then
    criterium bouwen mislukt "docker build van de fork faalde (exitcode $rc), zie build_fork.log"
    proef_einde_met "$EXIT_SCRIPT"
  fi
fi
meet fork_image "$FORK_IMG $(docker image inspect -f '{{.Id}}' "$FORK_IMG")"

bewaar build_mcp dc --profile p4 build mcp
drun --network none signaal-proef/p4-mcp:0.3.0 cat /proef-pip-freeze.txt >"$RUNMAP/pip_freeze_mcp.txt"
docker image inspect "$IMG_SURREAL" >/dev/null 2>&1 || bewaar pull_surreal docker pull "$IMG_SURREAL"
leg_images_vast "$IMG_SURREAL" "$IMG_PY" "$FORK_IMG" signaal-proef/p4-mcp:0.3.0

# Aanname A2, deel 1: werkt het officiële startpunt van open-notebook-mcp met mcp 1.30.0?
rc=0
officieel="$(drun --network none -e MCP_TRANSPORT=streamable-http -e HOST=0.0.0.0 \
  signaal-proef/p4-mcp:0.3.0 timeout 8 open-notebook-mcp 2>&1 | tail -n 3)" || rc=$?
meet mcp_officieel_startpunt "exitcode $rc: $(tr '\n' ' ' <<<"$officieel")"

# --- Geheimen per run ---
geheim_maken SP_SURREAL_PASS
geheim_maken SP_NOTEBOOK_PASSWORD

# --- Stap 1 en 2: database en API ---
bewaar up_db dc --profile p4 up -d surrealdb
bewaar up_api dc --profile p4 up -d --no-deps api
bewaar up_hulp dc --profile p4 up -d --no-deps p4-hulp
api_gezond() {
  dc --profile p4 exec -T p4-hulp python -c \
    'import urllib.request,sys; sys.exit(0 if urllib.request.urlopen("http://api:5055/health",timeout=5).status==200 else 1)'
}
if ! wacht_tot 300 api_gezond; then
  bewaar logs_api dc --profile p4 logs api || true
  criterium api mislukt "API van de fork niet gezond binnen 300 s, zie logs_api.log"
  proef_einde_met "$EXIT_SCRIPT"
fi
bewaar logs_api_start dc --profile p4 logs api || true

# --- Stap 3: vullen ---
rc=0
dc --profile p4 exec -T p4-hulp python /proef/vul_testdatabase.py vullen >"$RUNMAP/referentie.json" 2>"$RUNMAP/vullen.err" || rc=$?
if [[ $rc -ne 0 ]]; then
  criterium vullen mislukt "vullen van de testdatabase faalde, zie vullen.err"
  proef_einde_met "$EXIT_SCRIPT"
fi
meet referentie_meldingen "$(tr -d '\n' <"$RUNMAP/referentie.json" | grep -o '"meldingen": \[[^]]*\]' || true)"
meet classification_veld "$(grep -c '"classification_opgeslagen": "intern"' "$RUNMAP/referentie.json" || true) interne bronnen met classification opgeslagen"
dc --profile p4 cp "$(pad_host "$RUNMAP/referentie.json")" p4-hulp:/tmp/referentie.json >/dev/null

# --- Stap 4: open-notebook-mcp ---
bewaar up_mcp dc --profile p4 up -d --no-deps mcp
mcp_klaar() {
  dc --profile p4 exec -T p4-hulp python -c '
import urllib.request, urllib.error, sys
try:
    urllib.request.urlopen("http://mcp:8000/mcp", timeout=5)
except urllib.error.HTTPError:
    sys.exit(0)
except Exception:
    sys.exit(1)'
}
if ! wacht_tot 90 mcp_klaar; then
  bewaar logs_mcp dc --profile p4 logs mcp || true
  criterium mcp mislukt "open-notebook-mcp niet bereikbaar binnen 90 s, zie logs_mcp.log"
  proef_einde_met "$EXIT_SCRIPT"
fi

# --- Stap 5 en 6: lektest ---
rc=0
dc --profile p4 exec -T p4-hulp python /proef/mcp_lektest.py /tmp/referentie.json \
  >"$RUNMAP/lektest.json" 2>"$RUNMAP/lektest.ruw" || rc=$?
masker <"$RUNMAP/lektest.ruw" >"$RUNMAP/lektest.err"; rm -f "$RUNMAP/lektest.ruw"
meet lektest_exitcode "$rc"
bewaar logs_mcp dc --profile p4 logs mcp || true
bewaar logs_api dc --profile p4 logs api || true
if [[ ! -s "$RUNMAP/lektest.json" ]]; then
  criterium lektest mislukt "lektest gaf geen uitvoer (exitcode $rc), zie lektest.err"
  proef_einde_met "$EXIT_SCRIPT"
fi
meet_json lektest "$RUNMAP/lektest.json"
for k in K1 K2 K3 K4 K5 K6; do
  stand="$(grep -A1 "\"$k\": {" "$RUNMAP/lektest.json" | grep -o '"stand": "[a-z_]*"' | cut -d'"' -f4 || true)"
  toel="$(grep -A2 "\"$k\": {" "$RUNMAP/lektest.json" | grep -o '"toelichting": ".*"' | sed 's/^"toelichting": "//; s/"$//' || true)"
  criterium "$k" "${stand:-mislukt}" "${toel:-geen uitvoer}"
done

proef_einde
