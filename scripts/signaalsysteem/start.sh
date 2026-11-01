#!/usr/bin/env bash
# Start de Compose-omgeving van het signaalsysteem (F2-10). `up` gaat alleen via
# dit script; stoppen met scripts/signaalsysteem/stop.sh.
#
# 0. Weigert als de oude stack draait (containers van project open-notebook,
#    of iets op poort 8000 van de host).
# 1. Hostpaden (SIGNAAL_INNAMEMAP, SIGNAAL_WATCHFOLDER, SIGNAAL_VAULT,
#    SIGNAAL_EXPORTMAP) uit de omgeving of uit .env (voorbeeld in .env.example).
# 2. Een minimale kloon van de flowmap (HEAD, alleen services/langflow/flows)
#    voor het flowregister, alleen-lezen gekoppeld; niet de hele .git (K1).
# 3. SIGNAAL_FORK_COMMIT = HEAD, of HEAD-dirty bij een vuile werkmap, als
#    build-argument van het image (K2); zonder --build een waarschuwing als het
#    label van het bestaande image afwijkt (controle/forkcommit.sh).
# 4. Startcontrole: elk bind-pad op de host canoniek (controle/canoniek.ps1),
#    daarna controle/padlijsten.py. Faalt die, dan start er niets (exitcode 1).
#    Daarna modellen_laden.sh --controle: de hashes van de modellen (F2-11).
# 5. `docker compose up -d`; extra argumenten gaan naar `up` (bijv. --build).
# 6. De containers die llama-swap start (profiel door-llama-swap, F2-11) worden
#    alleen aangemaakt, niet gestart. Het geheim van docling-serve wordt
#    aangemaakt als het ontbreekt (secrets/, buiten Git).
#    Ook de geheimen van Langflow (F2-12): wachtwoord en Fernet-sleutel, en een
#    leeg bestand voor de pijplijnsleutel (die maakt de beheerder).
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/../.." && { pwd -W 2>/dev/null || pwd; })"
COMPOSE=(docker compose -p signaal -f "$ROOT/docker-compose.signaalsysteem.yml")
CONTROLE="$ROOT/scripts/signaalsysteem/controle"
# Zelfde basisimage als de fork; de controle draait zonder netwerk.
PY_IMG="python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337"
STAAT="${SIGNAAL_STAAT:-$HOME/.cache/signaal}"
FLOWS=services/langflow/flows

# 0. Niet naast de oude stack.
if [[ -n "$(docker ps -q --filter label=com.docker.compose.project=open-notebook)" ]]; then
  echo "weigert: containers van de oude stack (project open-notebook) draaien" >&2; exit 1
fi
rc=0; curl -s -o /dev/null --max-time 3 "http://127.0.0.1:8000/" || rc=$?
if [[ $rc -ne 7 ]]; then
  echo "weigert: poort 8000 op de host is in gebruik (curl $rc); de oude stack draait mogelijk" >&2
  exit 1
fi

# 1. Alleen de hostpaden uit .env overnemen, geen andere waarden.
for naam in SIGNAAL_INNAMEMAP SIGNAAL_WATCHFOLDER SIGNAAL_VAULT SIGNAAL_EXPORTMAP; do
  if [[ -z "${!naam:-}" && -f "$ROOT/.env" ]]; then
    waarde="$(grep -E "^${naam}=" "$ROOT/.env" | tail -n1 | cut -d= -f2- | tr -d '\r' || true)"
    if [[ -n "$waarde" ]]; then export "$naam=$waarde"; fi
  fi
done

# 2. Minimale kloon van de flowmap: één commit (HEAD), de bomen, en alleen de
#    bestanden onder de flowmap (partial clone, sparse checkout). Andere
#    bestandsinhoud zit er niet in. Een gewijzigde flowmap op de host telt niet
#    mee; een flow met een andere hash is dan niet vastgelegd (FlowRegister).
if [[ -n "$(git -C "$ROOT" status --porcelain -- "$FLOWS")" ]]; then
  echo "let op: $FLOWS heeft niet-vastgelegde wijzigingen; de kloon bevat HEAD" >&2
fi
mkdir -p "$STAAT"
kloon="$(cd "$STAAT" && { pwd -W 2>/dev/null || pwd; })/flows-git"
rm -rf "$kloon"
UPLOAD="git -c uploadpack.allowFilter=true upload-pack"
git -c protocol.file.allow=always clone -q --no-local --depth 1 --filter=tree:0 --sparse \
  --no-checkout -u "$UPLOAD" "$ROOT" "$kloon"
git -C "$kloon" config remote.origin.uploadpack "$UPLOAD"
git -C "$kloon" sparse-checkout set --no-cone "/$FLOWS/"
git -C "$kloon" -c protocol.file.allow=always -c advice.detachedHead=false checkout -q
git -C "$kloon" remote remove origin
export SIGNAAL_FLOWS_KLOON="$kloon"

# 3. De commit van de fork voor het image.
SIGNAAL_FORK_COMMIT="$(bash "$CONTROLE/forkcommit.sh" "$ROOT")"
export SIGNAAL_FORK_COMMIT
if [[ " $* " != *" --build "* ]]; then
  label="$(docker image inspect signaal/open-notebook:lokaal \
    -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' 2>/dev/null || true)"
  bash "$CONTROLE/forkcommit.sh" "$ROOT" --label "$label" || true
fi

# 4. Startcontrole.
# Met het profiel door-llama-swap: ook de binds van de diensten die llama-swap
# start (docling-serve; later kev en whisperx) gaan door de controle.
config="$("${COMPOSE[@]}" --profile door-llama-swap config --format json)"
bronnen="$(printf '%s' "$config" | docker run --rm -i --network none "$PY_IMG" \
  python -c "$(cat "$CONTROLE/padlijsten.py")" --bronnen)"
paden="$(printf '%s\n' "$bronnen"; for naam in SIGNAAL_INNAMEMAP SIGNAAL_WATCHFOLDER SIGNAAL_VAULT SIGNAAL_EXPORTMAP; do
  if [[ -n "${!naam:-}" ]]; then printf '%s\n' "${!naam}"; fi; done)"
SIGNAAL_CANONIEK="$(printf '%s' "$paden" | tr -d '\r' | powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$CONTROLE/canoniek.ps1" | tr -d '\r')"
export SIGNAAL_CANONIEK
printf '%s' "$config" | docker run --rm -i --network none \
  -e SIGNAAL_INNAMEMAP -e SIGNAAL_WATCHFOLDER -e SIGNAAL_VAULT -e SIGNAAL_EXPORTMAP -e SIGNAAL_CANONIEK \
  "$PY_IMG" python -c "$(cat "$CONTROLE/padlijsten.py")"

# 4b. De modellen: hashes zonder netwerk (F2-11). Wijkt er een af of ontbreekt
#     er een, dan start er niets; de melding zegt wat te doen.
if ! bash "$ROOT/scripts/signaalsysteem/modellen_laden.sh" --controle >/dev/null; then
  echo "weigert: de modellen in de volumes kloppen niet (scripts/signaalsysteem/modellen_laden.sh)" >&2
  exit 1
fi

# 5. Starten.
mkdir -p "$ROOT/secrets"
if [[ ! -s "$ROOT/secrets/docling_api_key" ]]; then
  (umask 077; head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$ROOT/secrets/docling_api_key")
fi
# Langflow (F2-12): wachtwoord van de superuser en LANGFLOW_SECRET_KEY, een
# Fernet-sleutel (32 bytes, url-veilig base64; proeven.md §8 punt 15). Alleen
# als ze ontbreken: de database van Langflow (langflow_data) hangt ervan af.
if [[ ! -s "$ROOT/secrets/langflow_superuser" ]]; then
  (umask 077; head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$ROOT/secrets/langflow_superuser")
fi
if [[ ! -s "$ROOT/secrets/langflow_secret_key" ]]; then
  (umask 077; head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '\n' > "$ROOT/secrets/langflow_secret_key")
fi
# De pijplijnsleutel maakt de beheerder met maak_sleutel.py (README, Beheer);
# tot die tijd een leeg bestand, zodat Compose kan starten. Het startscript van
# langflow meldt dat de sleutel ontbreekt.
if [[ ! -e "$ROOT/secrets/sleutel_pijplijn" ]]; then
  (umask 077; : > "$ROOT/secrets/sleutel_pijplijn")
  echo "let op: secrets/sleutel_pijplijn is leeg; maak de pijplijnsleutel (docs/signaalsysteem/README.md, Beheer)" >&2
fi
"${COMPOSE[@]}" up -d "$@"

# 6. Aanmaken, niet starten (ADR 0009).
"${COMPOSE[@]}" --profile door-llama-swap create docling-serve
