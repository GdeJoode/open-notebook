#!/usr/bin/env bash
# P10. Migraties en testset van de fork op SurrealDB v3 (docs/signaalsysteem/proeven.md,
# P10; ADR 0021).
#
# Alle migraties en hun _down-varianten, en de databasetests van de fork, op een lege
# surrealdb/surrealdb:v3.3.5, met v2.7.0 als nulmeting; eerst zonder, daarna met alleen
# de mechanische vervangingen uit de migratiegids. Alles op een kopie (git archive van
# HEAD) in de runmap; de repository verandert niet.
#
# Python en pytest draaien in de WSL-.venv van de repository, met PYTHONPATH naar de
# src-mappen van de kopie, zodat de code uit de kopie komt en niet uit de repository.
# De databases zijn twee containers van het project signaal-proef in geheugenmodus,
# met een poort op 127.0.0.1 (18100 en 18101) voor de tests in WSL.
#
# Hulpscripts in p10/: patronen.py (10a), migraties.py (10b-10d, 10f), vervang.py
# (10d), harnas.py (testfixture naar de proefserver), vergelijk.py (10e, K3 en K5) en
# indeling.csv (oorzaken van de fouten op v3, na analyse door de bouwagent).
#
# Aanvullingen op het proefplan (proeven.md §8, punten 30-33): een nulcorrectie van
# migrations/5_down.surrealql in alle kopieën (ook op v2.7.0 een parsefout), en stap
# 10g ter informatie (ook FLEXIBLE TYPE x -> TYPE x FLEXIBLE), die niet telt voor K2-K4.
source "$(dirname "$0")/lib/gemeenschappelijk.sh"

proef_start p10 p10
p0_voorwaarde

REPO="$(cd "$PROEF_SCRIPTS/../../.." && pwd)"
P10="$PROEF_SCRIPTS/p10"
URL_V2="ws://127.0.0.1:18100/rpc"
URL_V3="ws://127.0.0.1:18101/rpc"

# Pad zoals WSL het ziet (E:/x -> /mnt/e/x).
wsl_pad() {
  local w d
  w="$(pad_host "$1")"; d="${w:0:1}"
  printf '/mnt/%s%s\n' "${d,,}" "${w:2}"
}
WSL_PY="$(wsl_pad "$REPO")/.venv/bin/python"
WSL_P10="$(wsl_pad "$P10")"

# Opdracht in WSL; het wachtwoord gaat via WSLENV mee, niet op de opdrachtregel.
wsl_run() { WSLENV="SP_SURREAL_PASS${WSLENV:+:$WSLENV}" wsl bash -lc "$1"; }

# PYTHONPATH met de src-mappen van een kopie, optioneel met een SDK-map vooraan.
pp_van() {
  local k=$1 w="${2:+$2:}" d
  for d in "$k"/apps/*/src "$k"/packages/*/src "$k"/pipelines/*/src; do
    [[ -d "$d" ]] && w+="$(wsl_pad "$d"):"
  done
  printf '%s' "${w%:}"
}

# --- Omgeving ---
rc=0
wsl_run "$WSL_PY -c 'import pytest, surrealdb, loguru'" >"$RUNMAP/omgeving_wsl.log" 2>&1 || rc=$?
[[ $rc -ne 0 ]] && omgeving_ongeschikt "WSL-.venv met pytest en surrealdb niet bruikbaar ($WSL_PY), zie omgeving_wsl.log"
meet wsl_python "$(wsl_run "$WSL_PY -c 'import sys,surrealdb,importlib.metadata as m; print(sys.version.split()[0], \"surrealdb\", m.version(\"surrealdb\"), \"pytest\", m.version(\"pytest\"))'" 2>&1 | tail -n1)"

for img in "$IMG_SURREAL" "$IMG_SURREAL3"; do
  docker image inspect "$img" >/dev/null 2>&1 || bewaar "pull_${img##*:}" docker pull "$img"
done
leg_images_vast "$IMG_SURREAL" "$IMG_SURREAL3"

# --- Werkkopie (stap 2) ---
COMMIT="$(git -C "$(pad_host "$REPO")" rev-parse --short HEAD)"
meet kopie_commit "$COMMIT"
KOPIE_V2="$RUNMAP/kopie_v2"; KOPIE_V3="$RUNMAP/kopie_v3"
mkdir -p "$KOPIE_V2"
git -C "$(pad_host "$REPO")" archive --format=tar HEAD \
  | tar -x -C "$KOPIE_V2"

# --- 10a, statische telling ---
wsl_run "$WSL_PY $WSL_P10/patronen.py $(wsl_pad "$KOPIE_V2") $(wsl_pad "$RUNMAP/p10_patronen.csv")" \
  >"$RUNMAP/patronen_samenvatting.json" 2>"$RUNMAP/patronen.err"
meet_json patronen "$RUNMAP/patronen_samenvatting.json"

# --- Databases ---
geheim_maken SP_SURREAL_PASS
bewaar up_db dc --profile p10 up -d surrealdb-v2 surrealdb-v3
gezond() { [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$1/health")" == "200" ]]; }
for poort in 18100 18101; do
  wacht_tot 60 gezond "$poort" || { criterium databases mislukt "SurrealDB op poort $poort niet gezond binnen 60 s"; proef_einde_met "$EXIT_SCRIPT"; }
done
wsl_run "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18101/health" >"$RUNMAP/wsl_bereik.log" 2>&1 || true
[[ "$(tail -c 3 "$RUNMAP/wsl_bereik.log")" == "200" ]] || omgeving_ongeschikt "proefdatabase niet bereikbaar vanuit WSL, zie wsl_bereik.log"
meet versie_v2 "$(curl -s http://127.0.0.1:18100/version)"
meet versie_v3 "$(curl -s http://127.0.0.1:18101/version)"

# migraties.py met de code van een kopie: <kopie> <sdk-map of leeg> <args...>
migr() {
  local k=$1 sdk=$2; shift 2
  wsl_run "cd /tmp && PYTHONPATH=$(pp_van "$k" "$sdk") $WSL_PY $WSL_P10/migraties.py $*"
}

# Migraties heen, terug, heen op een nieuwe lege namespace en database <naam>.
stap_migraties() {
  local naam=$1 k=$2 url=$3 sdk=$4 rc=0
  migr "$k" "$sdk" migraties "$url" "$(wsl_pad "$k")/migrations" "$naam" \
    >"$RUNMAP/migraties_$naam.json" 2>"$RUNMAP/migraties_$naam.ruw" || rc=$?
  masker <"$RUNMAP/migraties_$naam.ruw" >"$RUNMAP/migraties_$naam.err"; rm -f "$RUNMAP/migraties_$naam.ruw"
  if [[ $rc -ne 0 || ! -s "$RUNMAP/migraties_$naam.json" ]]; then
    criterium "$naam" mislukt "migratiescript faalde (exitcode $rc), zie migraties_$naam.err"
    proef_einde_met "$EXIT_SCRIPT"
  fi
  meet "migraties_$naam" "geslaagd $(telwaarde "$naam" geslaagd), al_aanwezig $(telwaarde "$naam" al_aanwezig), mislukt $(telwaarde "$naam" mislukt), geen_down $(telwaarde "$naam" geen_down); mislukt: $(fouten "$naam")"
}
telwaarde() { grep -o "\"$2\": [0-9]*" "$RUNMAP/migraties_$1.json" | head -n1 | grep -o '[0-9]*$'; }
fouten() { tr -d '\n' <"$RUNMAP/migraties_$1.json" | grep -o '"mislukte_bestanden": \[[^]]*\]' | sed 's/"mislukte_bestanden": //; s/  */ /g'; }
geen_fouten() { [[ "$(telwaarde "$1" mislukt)" == "0" && "$(telwaarde "$1" geen_down)" == "0" ]]; }

# --- 10b zonder nulcorrectie: de migraties zoals ze in de repository staan ---
stap_migraties p10b_v2_onaangepast "$KOPIE_V2" "$URL_V2" ""

# Nulcorrectie (in alle kopieën gelijk): migrations/5_down.surrealql bevat
# "REMOVE TABLE IF EXISTS transformation SCHEMAFULL;", ook op v2.7.0 een parsefout.
# Zonder correctie kan de nulmeting (K1) nooit slagen; de fout staat los van v3.
NUL_OUD='REMOVE TABLE IF EXISTS transformation SCHEMAFULL;'
NUL_NIEUW='REMOVE TABLE IF EXISTS transformation;'
if grep -qF "$NUL_OUD" "$KOPIE_V2/migrations/5_down.surrealql"; then
  sed -i "s/$NUL_OUD/$NUL_NIEUW/" "$KOPIE_V2/migrations/5_down.surrealql"
  meet nulcorrectie "migrations/5_down.surrealql: '$NUL_OUD' -> '$NUL_NIEUW' (alle kopieën)"
else
  meet nulcorrectie "niet nodig: 5_down.surrealql bevat de fout niet meer"
fi

# Testfixture naar de proefserver, in alle kopieën gelijk; daarna de v3-kopieën.
wsl_run "$WSL_PY $WSL_P10/harnas.py $(wsl_pad "$KOPIE_V2")" >>"$RUNMAP/harnas.log" 2>&1
git -C "$(pad_host "$REPO")" show HEAD:packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py \
  | diff - "$KOPIE_V2/packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py" >"$RUNMAP/harnas.diff" || true
cp -r "$KOPIE_V2" "$KOPIE_V3"

# --- 10d (deel 1): mechanische vervangingen in de v3-kopie ---
wsl_run "$WSL_PY $WSL_P10/vervang.py $(wsl_pad "$KOPIE_V3") $(wsl_pad "$RUNMAP/p10_vervangingen.csv")" \
  >"$RUNMAP/vervang.log" 2>&1
meet vervangingen "$(tail -n1 "$RUNMAP/vervang.log")"
(cd "$RUNMAP" && diff -ru kopie_v2 kopie_v3 >vervangingen.diff) || true

# --- 10g (ter informatie, telt niet voor K2-K4): ook FLEXIBLE TYPE x -> TYPE x FLEXIBLE ---
KOPIE_V3X="$RUNMAP/kopie_v3x"
cp -r "$KOPIE_V3" "$KOPIE_V3X"
wsl_run "$WSL_PY $WSL_P10/vervang.py $(wsl_pad "$KOPIE_V3X") $(wsl_pad "$RUNMAP/p10_vervangingen_10g.csv") --extra" \
  >"$RUNMAP/vervang_10g.log" 2>&1
meet vervangingen_10g "$(tail -n1 "$RUNMAP/vervang_10g.log") (inclusief die van 10d)"

# --- 10b, 10c, 10d, 10g: migraties heen, terug, heen ---
stap_migraties p10b_v2 "$KOPIE_V2" "$URL_V2" ""
stap_migraties p10c_v3 "$KOPIE_V2" "$URL_V3" ""
stap_migraties p10d_v3 "$KOPIE_V3" "$URL_V3" ""
stap_migraties p10g_v3 "$KOPIE_V3X" "$URL_V3" ""

# K1
if geen_fouten p10b_v2; then
  criterium K1 geslaagd "alle migraties en _down-varianten slagen op v2.7.0 (heen, terug, heen), na de nulcorrectie van 5_down.surrealql; zonder die correctie faalt: $(fouten p10b_v2_onaangepast)"
else
  criterium K1 ongeldig "nulmeting op v2.7.0 niet geslaagd: $(fouten p10b_v2)"
fi
# K2
n_verv="$(($(wc -l <"$RUNMAP/p10_vervangingen.csv") - 1))"
if geen_fouten p10d_v3; then
  criterium K2 geslaagd "alle migraties en _down-varianten slagen op v3.3.5 na $n_verv mechanische vervangingen"
else
  criterium K2 mislukt "na $n_verv mechanische vervangingen falen op v3.3.5: $(fouten p10d_v3); zonder vervangingen (10c): $(fouten p10c_v3); met ook de FLEXIBLE-volgorde (10g): $(fouten p10g_v3)"
fi

# --- 10f, schemacontrole (K6) ---
migr "$KOPIE_V2" "" schema "$URL_V2" p10f_v2 >"$RUNMAP/schema_v2.json" 2>"$RUNMAP/schema_v2.err" || true
migr "$KOPIE_V2" "" schema "$URL_V3" p10f_v3 >"$RUNMAP/schema_v3.json" 2>"$RUNMAP/schema_v3.err" || true
gedrag() { grep -o '"gedrag": "[a-z_]*"' "$1" 2>/dev/null | cut -d'"' -f4; }
g2="$(gedrag "$RUNMAP/schema_v2.json")"; g3="$(gedrag "$RUNMAP/schema_v3.json")"
meet_json schema_v2 "$RUNMAP/schema_v2.json"
meet_json schema_v3 "$RUNMAP/schema_v3.json"
if [[ -n "$g2" && -n "$g3" ]]; then
  criterium K6 vastgelegd "onbekend veld op SCHEMAFULL-tabel: v2.7.0 ${g2}, v3.3.5 ${g3}"
else
  criterium K6 mislukt "schemacontrole gaf geen uitkomst, zie schema_v2.err en schema_v3.err"
fi

# --- 10e, testset ---
mapfile -t TESTBESTANDEN < <(cd "$KOPIE_V2" && zoek -rlE 'live_surrealdb|requires_docker' --include='*.py' apps packages pipelines \
  | zoek '/tests/' | zoek -v '/conftest.py$' | sort)
mapfile -t LEDEN < <(printf '%s\n' "${TESTBESTANDEN[@]}" | cut -d/ -f1-2 | sort -u)
meet testbestanden "${#TESTBESTANDEN[@]} bestanden in ${#LEDEN[@]} workspace-leden: ${LEDEN[*]}"

# tests <kopie> <url> <uitmap> <sdk-map of leeg>
tests() {
  local k=$1 url=$2 uit=$3 sdk=$4 lid naam bestanden rc
  mkdir -p "$uit"
  for lid in "${LEDEN[@]}"; do
    naam="${lid//\//_}"
    bestanden="$(printf '%s\n' "${TESTBESTANDEN[@]}" | zoek "^$lid/" | sed "s#^$lid/##" | tr '\n' ' ')"
    rc=0
    wsl_run "cd $(wsl_pad "$k")/$lid && P10_SURREAL_URL=$url PYTHONPATH=$(pp_van "$k" "$sdk") \
      timeout 3600 $WSL_PY -m pytest -p no:cacheprovider -q -o junit_family=xunit2 \
      --junitxml=$(wsl_pad "$uit")/$naam.xml $bestanden" >"$uit/$naam.ruw" 2>&1 || rc=$?
    masker <"$uit/$naam.ruw" >"$uit/$naam.log"; rm -f "$uit/$naam.ruw"
    printf '%s\t%s\n' "$lid" "$rc" >>"$uit/exitcodes.tsv"
    log "pytest $lid tegen $url: exitcode $rc"
  done
}

tests "$KOPIE_V2" "$URL_V2" "$RUNMAP/tests_v2_sdk106" ""
SDK_K4="1.0.6"
migr "$KOPIE_V3" "" verbind "$URL_V3" >"$RUNMAP/verbind_v3_sdk106.json" 2>"$RUNMAP/verbind_v3_sdk106.err" || true
if grep -q '"verbinding": true' "$RUNMAP/verbind_v3_sdk106.json"; then
  meet sdk_106_verbinding_v3 "ja"
  tests "$KOPIE_V3" "$URL_V3" "$RUNMAP/tests_v3_sdk106" ""
  V3_TESTS="$RUNMAP/tests_v3_sdk106"
else
  # Geen verbinding met 1.0.6: nog een keer met de stabiele SDK 2.0.0 (geen bèta).
  meet sdk_106_verbinding_v3 "nee, zie verbind_v3_sdk106.json"
  SDK_K4="2.0.0"
  SDK_MAP="$RUNMAP/sdk200"
  bewaar sdk200 wsl_run "~/.local/bin/uv pip install --python $WSL_PY --target $(wsl_pad "$SDK_MAP") --no-deps surrealdb==2.0.0"
  stap_migraties p10d_v3_sdk200 "$KOPIE_V3" "$URL_V3" "$(wsl_pad "$SDK_MAP")"
  tests "$KOPIE_V3" "$URL_V3" "$RUNMAP/tests_v3_sdk200" "$(wsl_pad "$SDK_MAP")"
  V3_TESTS="$RUNMAP/tests_v3_sdk200"
fi
meet sdk_v3 "$SDK_K4"
meet pytest_exitcodes_v2 "$(tr '\t\n' '= ' <"$RUNMAP/tests_v2_sdk106/exitcodes.tsv")"
meet pytest_exitcodes_v3 "$(tr '\t\n' '= ' <"$V3_TESTS/exitcodes.tsv")"

# --- Vergelijking (K3, K4, K5) ---
rc=0
wsl_run "$WSL_PY $WSL_P10/vergelijk.py $(wsl_pad "$RUNMAP/tests_v2_sdk106") $(wsl_pad "$V3_TESTS") \
  $(wsl_pad "$RUNMAP/migraties_p10d_v3.json"),$(wsl_pad "$RUNMAP/migraties_p10g_v3.json") $WSL_P10/indeling.csv" \
  >"$RUNMAP/vergelijking.json" 2>"$RUNMAP/vergelijking.err" || rc=$?
if [[ $rc -ne 0 || ! -s "$RUNMAP/vergelijking.json" ]]; then
  criterium vergelijking mislukt "vergelijk.py faalde (exitcode $rc), zie vergelijking.err"
  proef_einde_met "$EXIT_SCRIPT"
fi
meet_json vergelijking "$RUNMAP/vergelijking.json"
json_getal() { tr -d '\n' <"$RUNMAP/vergelijking.json" | grep -o "\"$1\": {[^}]*}" | head -n1; }
n_reg="$(grep -o '"regressies_aantal": [0-9]*' "$RUNMAP/vergelijking.json" | grep -o '[0-9]*$')"
nul_telling="$(json_getal nulmeting)"; v3_telling="$(json_getal v3)"
nul_geslaagd="$(grep -o '"geslaagd": [0-9]*' <<<"$nul_telling" | grep -o '[0-9]*$')"
if [[ "${nul_geslaagd:-0}" -eq 0 ]]; then
  criterium K3 ongeldig "nulmeting van de testset op v2.7.0 heeft geen geslaagde tests: $nul_telling"
elif [[ "$n_reg" == "0" ]]; then
  criterium K3 geslaagd "geen test faalt op v3.3.5 die op v2.7.0 slaagt (v2.7.0 $nul_telling; v3.3.5 $v3_telling)"
else
  criterium K3 mislukt "$n_reg tests falen op v3.3.5 en slagen op v2.7.0 (v2.7.0 $nul_telling; v3.3.5 $v3_telling)"
fi
if [[ "$n_reg" == "0" ]]; then
  criterium K4 geslaagd "K3 gehaald met surrealdb==$SDK_K4"
else
  criterium K4 mislukt "K3 niet gehaald; gebruikte SDK surrealdb==$SDK_K4 (verbinding met v3.3.5: $(grep -o '"verbinding": [a-z]*' "$RUNMAP/verbind_v3_sdk106.json" | cut -d' ' -f2))"
fi
k5="$(tr -d '\n' <"$RUNMAP/vergelijking.json" | grep -o '"k5": {.*' )"
k5_stand="$(grep -o '"stand": "[a-z_]*"' <<<"$k5" | head -n1 | cut -d'"' -f4)"
k5_aantal="$(grep -o '"aantal_bestanden": [0-9]*' <<<"$k5" | head -n1 | grep -o '[0-9]*$')"
k5_bestanden="$(grep -o '"bestanden_niet_mechanisch": \[[^]]*\]' <<<"$k5" | sed 's/^[^[]*//; s/  */ /g')"
k5_herontwerp="$(grep -o '"herontwerp_tabel": \[[^]]*\]' <<<"$k5" | sed 's/^[^[]*//; s/  */ /g')"
case "$k5_stand" in
  geslaagd) criterium K5 geslaagd "niet-mechanisch: $k5_aantal bestanden $k5_bestanden; herontwerp van een tabel: $k5_herontwerp" ;;
  mislukt) criterium K5 mislukt "niet-mechanisch: $k5_aantal bestanden (grens 10) $k5_bestanden; herontwerp van een tabel: $k5_herontwerp" ;;
  *) criterium K5 mislukt "niet alle fouten op v3.3.5 zijn ingedeeld (p10/indeling.csv), zie vergelijking.json k5.niet_ingedeeld" ;;
esac

proef_einde
