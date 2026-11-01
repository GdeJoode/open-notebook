#!/usr/bin/env bash
# P2. Schema-trigger van Langflow 1.13.0, met APScheduler als terugval
# (docs/signaalsysteem/proeven.md, P2). Alles op het netwerk "verwerking" (geen internet).
#
# Duur: circa 30 minuten (drie starts voor 2b, 4 min looptijd, drie herstarts van 150 s + 120 s).
source "$(dirname "$0")/lib/gemeenschappelijk.sh"

proef_start p2 p2
p0_voorwaarde

STOP_S="${P2_STOP_S:-150}"
NA_HERSTART_S="${P2_NA_HERSTART_S:-120}"
LOOPTIJD_S="${P2_LOOPTIJD_S:-240}"

geheim_maken SP_LANGFLOW_PASSWORD
geheim_maken SP_LANGFLOW_SECRET_KEY fernet
export SP_LANGFLOW_API_KEY="nog-niet-gezet" SP_PLANNER_FLOW_ID="" SP_ENABLE_TRIGGERS="" SP_LANGFLOW_ENABLE_TRIGGERS=""

# Elke run begint met een lege Langflow-database en lege tickbestanden: het
# superuser-wachtwoord is per run nieuw. Alleen de drie P2-volumes van het project.
bewaar down_vooraf dc --profile p2 down --timeout 10
for v in langflow_data p2_ticks p2_planner; do
  if docker volume inspect "signaal-proef_$v" >/dev/null 2>&1; then bewaar "volrm_$v" docker volume rm "signaal-proef_$v"; fi
done

docker image inspect "$IMG_LANGFLOW" >/dev/null 2>&1 || bewaar pull_langflow docker pull "$IMG_LANGFLOW"
bewaar build dc --profile p2 build langflow p2-hulp
drun --network none signaal-proef/p2-hulp:local cat /proef-pip-freeze.txt >"$RUNMAP/pip_freeze_hulp.txt"
leg_images_vast "$IMG_LANGFLOW" "$IMG_PY" signaal-proef/p2-langflow:1.13.0 signaal-proef/p2-hulp:local

hulp() { dc --profile p2 exec -T p2-hulp "$@"; }
api() { hulp python /proef/langflow_api.py "$@"; }
nu() { hulp date -u +%Y-%m-%dT%H:%M:%S+00:00; }
# Eerste waarde van "sleutel": <waarde> in een JSON-bestand.
jwaarde() { grep -o "\"$2\": *[^,}]*" "$1" | head -n1 | sed -E "s/\"$2\": *//; s/^\"//; s/\"$//"; }
langflow_gezond() {
  hulp python -c 'import urllib.request,sys; sys.exit(0 if urllib.request.urlopen("http://langflow:7860/health_check",timeout=5).status==200 else 1)'
}
wacht_langflow() {
  if ! wacht_tot 420 langflow_gezond; then
    bewaar logs_langflow dc --profile p2 logs langflow || true
    criterium langflow mislukt "Langflow niet gezond binnen 420 s, zie logs_langflow.log"
    proef_einde_met "$EXIT_SCRIPT"
  fi
}

# --- 2b: ENABLE_TRIGGERS, drie starts ---
k1_samenvatting=""
for variant in ENABLE_TRIGGERS LANGFLOW_ENABLE_TRIGGERS zonder; do
  SP_ENABLE_TRIGGERS=""; SP_LANGFLOW_ENABLE_TRIGGERS=""
  [[ "$variant" == ENABLE_TRIGGERS ]] && SP_ENABLE_TRIGGERS=true
  [[ "$variant" == LANGFLOW_ENABLE_TRIGGERS ]] && SP_LANGFLOW_ENABLE_TRIGGERS=true
  export SP_ENABLE_TRIGGERS SP_LANGFLOW_ENABLE_TRIGGERS
  bewaar "up_2b_$variant" dc --profile p2 up -d --force-recreate langflow p2-hulp
  wacht_langflow
  api inventaris >"$RUNMAP/2b_$variant.json"
  bundel="$(dc --profile p2 exec -T langflow sh -c \
    'd="$(find /app/.venv -maxdepth 5 -type d -path "*site-packages/langflow/frontend" | head -n1)"; grep -rlo ENABLE_TRIGGERS "$d" | wc -l; grep -rho "ENABLE_TRIGGERS" "$d" | wc -l' | tr '\n' ' ')"
  meet "2b_${variant}_bundel_bestanden_en_treffers_ENABLE_TRIGGERS" "$bundel"
  meet_json "2b_$variant" "$RUNMAP/2b_$variant.json"
  k1_samenvatting+="$variant: schema in /all=$(jwaarde "$RUNMAP/2b_$variant.json" schedule_trigger_in_all), bundel=[$bundel]; "
done
criterium K1 vastgelegd "$k1_samenvatting"

# --- 2a: API-sleutel ---
SP_LANGFLOW_API_KEY="$(api apisleutel | tr -d '\r\n')"
PROEF_GEHEIMEN+=("$SP_LANGFLOW_API_KEY")
export SP_LANGFLOW_API_KEY
if [[ "$SP_LANGFLOW_API_KEY" == sk-* ]]; then meet 2a "inloggen en API-sleutel gelukt"
else meet 2a "API-sleutel niet verkregen"; fi

# --- 2c en 2f: flows en triggers ---
api opzet >"$RUNMAP/2c_opzet.json"
meet_json 2c_opzet "$RUNMAP/2c_opzet.json"
FLOW_SCHEMA="$(jwaarde "$RUNMAP/2c_opzet.json" flow_schema)"
FLOW_TZ="$(jwaarde "$RUNMAP/2c_opzet.json" flow_tijdzone)"
SP_PLANNER_FLOW_ID="$(jwaarde "$RUNMAP/2c_opzet.json" flow_planner)"; export SP_PLANNER_FLOW_ID
enable_status="$(tr -d '\n ' <"$RUNMAP/2c_opzet.json" | grep -o '"flow_schema_enable":\[{"enable_status":[0-9]*' | grep -o '[0-9]*$' || true)"
api triggers "$FLOW_SCHEMA" >"$RUNMAP/2c_triggers.json"
staat="$(jwaarde "$RUNMAP/2c_triggers.json" state)"
staat_na_opslaan="$(tr -d '\n ' <"$RUNMAP/2c_opzet.json" | grep -o '"flow_schema_triggers_na_opslaan":\[[^]]*' | grep -o '"state":"[a-z]*"' | head -n1 || true)"
meet 2c_status_na_opslaan "$staat_na_opslaan"
if [[ "$enable_status" =~ ^2 && "$staat" == "active" ]]; then
  criterium K2 geslaagd "trigger aangemaakt bij opslaan ($staat_na_opslaan); enable HTTP $enable_status; status $staat"
else
  criterium K2 mislukt "enable HTTP ${enable_status:-geen}; status ${staat:-geen trigger}"
fi

# 2f: tijdzone, volgend moment van 0 8 * * 1-5 in Europe/Amsterdam.
api triggers "$FLOW_TZ" >"$RUNMAP/2f_triggers.json"
volgend="$(jwaarde "$RUNMAP/2f_triggers.json" next_fire_at)"
if [[ "$volgend" =~ T07:00:00 ]]; then meet 2f "next_fire_at $volgend: 07:00 UTC, klopt voor wintertijd"
else meet 2f "next_fire_at ${volgend:-niet getoond}; verwacht 07:00 UTC"; fi

# --- 2d: vier minuten laten lopen, met de terugval (2g) tegelijk ---
bewaar up_planner dc --profile p2 up -d --no-deps planner
T0="$(nu)"
log "2d: $LOOPTIJD_S s wachten"
sleep "$LOOPTIJD_S"
T1="$(nu)"
api analyse /proef/ticks/ticks.jsonl schema "$T0" "$T1" >"$RUNMAP/2d_schema.json"
api analyse /proef/ticks/planner.jsonl planner "$T0" "$T1" >"$RUNMAP/2g_looptijd.json"
meet_json 2d_schema "$RUNMAP/2d_schema.json"
meet_json 2g_looptijd "$RUNMAP/2g_looptijd.json"
n="$(jwaarde "$RUNMAP/2d_schema.json" aantal)"; uniek="$(jwaarde "$RUNMAP/2d_schema.json" uniek_gepland)"
zonder="$(jwaarde "$RUNMAP/2d_schema.json" zonder_gepland)"; maxv="$(jwaarde "$RUNMAP/2d_schema.json" max_vertraging_s)"
if [[ "${n:-0}" -ge 3 && "$uniek" == "true" && "$zonder" == "0" && "$maxv" != "null" ]] \
   && awk -v v="$maxv" 'BEGIN {exit !(v <= 10)}'; then
  criterium K3 geslaagd "$n ticks, gepland moment uniek, max vertraging ${maxv} s"
else
  criterium K3 mislukt "$n ticks, uniek=$uniek, zonder gepland moment=$zonder, max vertraging ${maxv} s"
fi

# --- 2e: herstart met coalesce, daarna met skip ---
herstart_proef() { # <label> -> zet TS (stop), TH (herstart), TE (einde)
  TS="$(nu)"
  bewaar "stop_$1" dc --profile p2 stop langflow
  sleep "$STOP_S"
  bewaar "start_$1" dc --profile p2 start langflow
  TH="$(nu)"
  wacht_langflow
  sleep "$NA_HERSTART_S"
  TE="$(nu)"
}
herstart_proef coalesce
api analyse /proef/ticks/ticks.jsonl schema "$TS" "$TE" "$TH" >"$RUNMAP/2e_coalesce.json"
meet_json 2e_coalesce "$RUNMAP/2e_coalesce.json"
api triggers "$FLOW_SCHEMA" >"$RUNMAP/2e_triggers_coalesce.json"
inhaal_c="$(jwaarde "$RUNMAP/2e_coalesce.json" inhaalruns)"
gemist_c="$(tr -d '\n ' <"$RUNMAP/2e_coalesce.json" | grep -o '"inhaal_gemist":\[[0-9,]*\]' | grep -o '[0-9][0-9,]*' || true)"

api beleid "$FLOW_SCHEMA" skip >"$RUNMAP/2e_beleid_skip.json"
meet_json 2e_beleid_skip "$RUNMAP/2e_beleid_skip.json"
herstart_proef skip
api analyse /proef/ticks/ticks.jsonl schema "$TS" "$TE" "$TH" >"$RUNMAP/2e_skip.json"
meet_json 2e_skip "$RUNMAP/2e_skip.json"
inhaal_s="$(jwaarde "$RUNMAP/2e_skip.json" inhaalruns)"
if [[ "$inhaal_c" == "1" && "${gemist_c%%,*}" -ge 2 && "$inhaal_s" == "0" ]] 2>/dev/null; then
  criterium K4 geslaagd "coalesce: 1 inhaalrun met $gemist_c gemiste momenten; skip: 0 inhaalruns (replayvenster: trigger_replay_window_days, standaard 7)"
else
  criterium K4 mislukt "coalesce: ${inhaal_c:-?} inhaalruns, gemist [${gemist_c}]; skip: ${inhaal_s:-?} inhaalruns"
fi

# --- 2g: herstart van de terugval ---
TS="$(nu)"
bewaar stop_planner dc --profile p2 stop planner
sleep "$STOP_S"
# Herstartmoment vóór het startcommando: een run met een gepland moment daarvóór is
# een inhaalrun (APScheduler meldt bij coalesce het laatst gemiste moment).
TH="$(nu)"
bewaar start_planner dc --profile p2 start planner
sleep "$NA_HERSTART_S"
TE="$(nu)"
api analyse /proef/ticks/planner.jsonl planner "$TS" "$TE" "$TH" >"$RUNMAP/2g_herstart.json"
meet_json 2g_herstart "$RUNMAP/2g_herstart.json"
n200="$(jwaarde "$RUNMAP/2g_looptijd.json" status_200)"; pmaxv="$(jwaarde "$RUNMAP/2g_looptijd.json" max_vertraging_s)"
inhaal_p="$(jwaarde "$RUNMAP/2g_herstart.json" inhaalruns)"
if [[ "${n200:-0}" -ge 3 && "$pmaxv" != "null" && "$inhaal_p" == "1" ]] && awk -v v="$pmaxv" 'BEGIN {exit !(v <= 10)}'; then
  criterium K5 geslaagd "$n200 runs met HTTP 200, max vertraging $pmaxv s; na herstart $inhaal_p inhaalrun"
else
  criterium K5 mislukt "$n200 runs met HTTP 200, max vertraging $pmaxv s; na herstart $inhaal_p inhaalruns"
fi

bewaar logs_langflow dc --profile p2 logs langflow || true
bewaar logs_planner dc --profile p2 logs planner || true
# Uitgaande verbindingspogingen vanaf "verwerking" (geblokkeerd), voor ADR 0017.
meet langflow_uitgaande_pogingen "$(grep -ciE 'Name or service not known|ConnectError' "$RUNMAP/logs_langflow.log" || true) logregels; voorbeelden: $(grep -oiE 'fetching https?://[^ :]+' "$RUNMAP/logs_langflow.log" | sort -u | head -n 3 | tr '\n' ' ')"

# Beslisregel ADR 0017 C (vastlegging).
if [[ "$(grep -cE $'^K[234]	geslaagd' "$RUNMAP/.criteria")" == "3" ]]; then
  meet beslisregel_adr_0017_c "schema-trigger (via de API aangezet)"
elif grep -qE $'^K5	geslaagd' "$RUNMAP/.criteria"; then
  meet beslisregel_adr_0017_c "container planner met APScheduler"
else
  meet beslisregel_adr_0017_c "geen van beide bewezen"
fi

proef_einde
