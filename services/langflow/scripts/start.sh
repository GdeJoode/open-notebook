#!/bin/sh
# Startscript van de container langflow (F2-12; ADR 0017 A, ADR 0022 D en F).
#
# 1. Weigert te starten als een verboden variabele gezet is: een voorvoegsel van
#    een tracer van Langflow 1.13.0 (ADR 0022 F), of van een Claude-sleutel
#    (ANTHROPIC_, CLAUDE_; Langflow heeft in geen enkele fase een Claude-sleutel).
#    Hoofdletters tellen niet: pydantic-instellingen lezen ook kleine letters.
# 2. Weigert te starten als een afschermende instelling niet de vaste waarde
#    heeft (ADR 0017, 0022 D en F); zo kan een extra Compose-bestand ze niet
#    stilzwijgend omzetten.
#    Met --controle stopt het script hier (exitcode 0): voor de tests.
# 3. Het modelregister: SIGNAAL_MODELREGISTER is verplicht (het standaardpad van
#    llm_manager werkt alleen vanuit een checkout; review F2-07) en het register
#    moet geldig zijn.
# 4. Toetst de flows uit Git aan de componentenlijst, vóór Langflow start
#    (flows_laden.py --alleen-toetsen). Een flow met een type dat niet op de
#    lijst staat: Langflow start niet (exitcode 1).
# 5. Leest het superuserwachtwoord en LANGFLOW_SECRET_KEY uit /run/secrets in de
#    omgeving van het proces, niet in die van de container.
# 6. Start Langflow, laadt de flows (POST /api/v1/flows/upload/) en zet hun
#    triggers aan. Lukt dat niet, dan stopt Langflow (exitcode 1). Daarna
#    /tmp/signaal-flows-geladen, voor de healthcheck.
set -eu

SCRIPTS="${SIGNAAL_SCRIPTS:-/app/signaal}"
GEHEIMEN="${SIGNAAL_GEHEIMEN:-/run/secrets}"
VERBODEN="LANGCHAIN_ LANGSMITH_ LANGFUSE_ LANGWATCH_ ARIZE_ OPIK_ PHOENIX_ TRACELOOP_ OPENLAYER_ OTEL_EXPORTER_OTLP_ ANTHROPIC_ CLAUDE_"

fout() { echo "langflow-start: $*" >&2; }

# 1. Verboden variabelen.
gevonden=""
for naam in $(env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p'); do
  hoog="$(printf '%s' "$naam" | tr '[:lower:]' '[:upper:]')"
  for voorvoegsel in $VERBODEN; do
    case "$hoog" in
      "$voorvoegsel"*) gevonden="$gevonden $naam" ;;
    esac
  done
done
if [ -n "$gevonden" ]; then
  fout "weigert te starten: verboden variabele(n):$gevonden (ADR 0022 F)"
  exit 1
fi

# 2. Vaste instellingen.
vereist() { # <naam> <waarde, kleine letters>
  eval "waarde=\${$1-}"
  if [ "$(printf '%s' "$waarde" | tr '[:upper:]' '[:lower:]')" != "$2" ]; then
    fout "weigert te starten: $1 moet $2 zijn (ADR 0017, 0022)"
    exit 1
  fi
}
vereist LANGFLOW_AUTO_LOGIN false
vereist LANGFLOW_ALLOW_CUSTOM_COMPONENTS false
vereist LANGFLOW_SUBSTITUTE_OUTDATED_COMPONENT_CODE false
vereist LANGFLOW_BLOCK_CODE_INTERPRETER_COMPONENTS true
vereist LANGFLOW_AGENTIC_EXPERIENCE false
vereist LANGFLOW_ALLOW_COMPONENTS_PATHS_OVERRIDE true
vereist LANGFLOW_DEACTIVATE_TRACING true
vereist DO_NOT_TRACK true
vereist LANGFLOW_MCP_SERVER_ENABLED false
vereist LANGFLOW_ADD_PROJECTS_TO_MCP_SERVERS false
vereist LANGFLOW_MCP_COMPOSER_ENABLED false
vereist LITELLM_LOCAL_MODEL_COST_MAP true
vereist LANGFLOW_MODELS_DEV_REFRESH false

if [ "${1:-}" = "--controle" ]; then
  echo "langflow-start: controle geslaagd"
  exit 0
fi

# 3. Modelregister.
if [ -z "${SIGNAAL_MODELREGISTER:-}" ]; then
  fout "weigert te starten: SIGNAAL_MODELREGISTER is niet gezet"
  exit 1
fi
python -c 'from llm_manager.register import lees_register; lees_register()' || {
  fout "weigert te starten: modelregister $SIGNAAL_MODELREGISTER ongeldig of niet te lezen"
  exit 1
}

# 4. Componentenlijst, vóór Langflow start.
python "$SCRIPTS/flows_laden.py" --alleen-toetsen || {
  rc=$?
  fout "weigert te starten: flows niet goedgekeurd (exitcode $rc)"
  exit "$rc"
}

# 5. Geheimen in de omgeving van het proces.
for geheim in langflow_superuser langflow_secret_key; do
  [ -s "$GEHEIMEN/$geheim" ] || { fout "weigert te starten: geheim $geheim ontbreekt of is leeg"; exit 1; }
done
LANGFLOW_SUPERUSER_PASSWORD="$(cat "$GEHEIMEN/langflow_superuser")"
LANGFLOW_SECRET_KEY="$(cat "$GEHEIMEN/langflow_secret_key")"
export LANGFLOW_SUPERUSER_PASSWORD LANGFLOW_SECRET_KEY
if [ ! -s "$GEHEIMEN/sleutel_pijplijn" ]; then
  fout "let op: geen pijplijnsleutel in $GEHEIMEN/sleutel_pijplijn; de componenten bereiken de fork niet (docs/signaalsysteem/README.md, Beheer)"
fi

# 6. Langflow starten en de flows laden.
rm -f /tmp/signaal-flows-geladen
langflow run --host 0.0.0.0 --port "${LANGFLOW_PORT:-7860}" &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null || true' TERM INT
if ! python "$SCRIPTS/flows_laden.py"; then
  fout "flows niet geladen; Langflow stopt"
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  exit 1
fi
: > /tmp/signaal-flows-geladen
set +e
wait "$pid"
rc=$?
# Na een signaal keert wait terug voordat Langflow klaar is.
wait "$pid" 2>/dev/null
exit "$rc"
