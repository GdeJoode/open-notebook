#!/usr/bin/env bash
# Gedeelde hulpfuncties voor de proeven van het signaalsysteem.
# Zie docs/signaalsysteem/proeven.md, paragraaf 1.
#
# Gebruik in een proefscript:
#   source "$(dirname "$0")/lib/gemeenschappelijk.sh"
#   proef_start p8 p8          # proefnaam en Compose-profiel
#   ... criterium K1 geslaagd "toelichting" ...
#   proef_einde
#
# Exitcodes (proeven.md 1.4): 0 alle criteria gehaald, 1 minstens één criterium
# niet gehaald, 2 omgeving ongeschikt, 3 scriptfout of wijziging buiten het project.

set -Eeuo pipefail

# Git Bash (MSYS) zet argumenten als "/data" of "naam:/pad" om naar Windows-paden.
# Dat breekt docker-argumenten. Paden naar de host zetten we zelf om met pad_host.
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'

readonly PROEF_PROJECT="signaal-proef"
PROEF_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PROEF_SCRIPTS
readonly PROEF_COMPOSE="$PROEF_SCRIPTS/compose.proef.yml"
SIGNAAL_PROEF_DIR="${SIGNAAL_PROEF_DIR:-$HOME/.cache/signaal-proef}"
export SIGNAAL_PROEF_DIR

readonly EXIT_GESLAAGD=0 EXIT_CRITERIUM=1 EXIT_OMGEVING=2 EXIT_SCRIPT=3

# Vaste versies (proeven.md 1.7). compose.proef.yml noemt dezelfde tags.
readonly IMG_LLAMA_SWAP="ghcr.io/mostlygeek/llama-swap:v268-cuda13-b11977"
readonly IMG_DOCLING="quay.io/docling-project/docling-serve-cu128:v1.38.2"
readonly IMG_WHISPERX_A="public.ecr.aws/deep-learning-containers/whisperx:3.8.6-cu128-amzn2023"
readonly IMG_LANGFLOW="langflowai/langflow:1.13.0"
readonly IMG_SOCKET_PROXY="tecnativa/docker-socket-proxy:v0.5.0"
readonly IMG_DOCKER_CLI="docker:29.9.2-cli"
readonly IMG_CURL="curlimages/curl:8.23.0"
readonly IMG_SURREAL="surrealdb/surrealdb:v2.7.0"
readonly IMG_PY="python:3.12.16-slim-bookworm"
readonly IMG_PY_KEV="python:3.13.18-slim-bookworm"
readonly IMG_UV="ghcr.io/astral-sh/uv:0.11.13"

# Images die de proeven mogen toevoegen. Een ander nieuw image, of een gewijzigd of
# verdwenen image, telt als wijziging buiten het project (exitcode 3).
readonly PROEF_IMAGES_TOEGESTAAN="$IMG_LLAMA_SWAP $IMG_DOCLING $IMG_WHISPERX_A $IMG_LANGFLOW $IMG_SOCKET_PROXY $IMG_DOCKER_CLI $IMG_CURL $IMG_SURREAL $IMG_PY $IMG_PY_KEV $IMG_UV"

# Proefgeheimen; worden in alle logs gemaskeerd.
PROEF_GEHEIMEN=()
PROEF_NAAM=""
PROEF_PROFIEL=""
RUNMAP=""
PROEF_LAAT_STAAN="${PROEF_LAAT_STAAN:-0}"
PROEF_AFGEROND=0

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | masker >&2; }

# Zet een hostpad om naar een vorm die docker.exe begrijpt (Git Bash) of laat het staan.
pad_host() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s\n' "$1"; fi
}

# Compose alleen met vast project en vast bestand (proeven.md 1.2).
dc() {
  docker compose -p "$PROEF_PROJECT" -f "$(pad_host "$PROEF_COMPOSE")" "$@"
}

# Losse container (docker run) met het projectlabel, zodat hij tot het project telt.
drun() {
  docker run --rm --label "com.docker.compose.project=$PROEF_PROJECT" "$@"
}

# Maak een geheim voor deze run; het staat alleen in het geheugen van het script.
# Met "fernet" als tweede argument: 32 bytes url-veilig base64 (Langflow-sleutel).
geheim_maken() {
  local naam=$1 waarde
  if [[ "${2:-}" == "fernet" ]]; then
    waarde="$(openssl rand -base64 32 | tr '+/' '-_' | tr -d '\r\n')"
  else
    waarde="$(openssl rand -hex 24)"
  fi
  PROEF_GEHEIMEN+=("$waarde")
  printf -v "$naam" '%s' "$waarde"
  export "${naam?}"
}

# Maskeer geheimen en regels met token, key of password.
masker() {
  local regel g
  # Geheimen letterlijk vervangen (geen reguliere expressie: base64 bevat - en _).
  while IFS= read -r regel || [[ -n "$regel" ]]; do
    for g in "${PROEF_GEHEIMEN[@]:-}"; do
      [[ -n "$g" ]] && regel=${regel//"$g"/***}
    done
    printf '%s\n' "$regel"
  done | sed -E 's/((token|api[_-]?key|key|password|passwd|secret)["'"'"']?[[:space:]]*[:=][[:space:]]*["'"'"']?)[^"'"'"' ,&}]+/\1***/Ig'
}

# Voer een opdracht uit en bewaar de gemaskeerde uitvoer in de runmap.
# Geeft de exitcode van de opdracht terug zonder het script te laten stoppen.
bewaar() {
  local naam=$1; shift
  local rc=0
  "$@" >"$RUNMAP/$naam.ruw" 2>&1 || rc=$?
  masker <"$RUNMAP/$naam.ruw" >"$RUNMAP/$naam.log"
  rm -f "$RUNMAP/$naam.ruw"
  return "$rc"
}

json_str() {
  local s=$1
  s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/\\n}; s=${s//$'\t'/\\t}; s=${s//$'\r'/}
  printf '"%s"' "$s"
}

# Leg een meetwaarde vast (tekst).
meet() { printf '%s\t%s\n' "$1" "$2" >>"$RUNMAP/.metingen"; }
# Leg een meetwaarde vast die al JSON is (inhoud van een bestand).
meet_json() { printf '%s\t%s\n' "$1" "$2" >>"$RUNMAP/.metingen_json"; }

# criterium <Kn> <geslaagd|mislukt|vastgelegd|niet_uitvoerbaar|ongeldig> <toelichting>
criterium() {
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$RUNMAP/.criteria"
  log "criterium $1: $2 ($3)"
}

# --- momentopnamen (proeven.md 1.2) ---

momentopname() {
  local doel=$1
  { docker ps -a --format '{{.ID}}|{{.Names}}|{{.Image}}|{{.State}}|{{.Label "com.docker.compose.project"}}' \
      | grep -v "|${PROEF_PROJECT}\$" || true; } | sort >"$doel.containers"
  { docker volume ls --format '{{.Name}}|{{.Label "com.docker.compose.project"}}' \
      | grep -v "|${PROEF_PROJECT}\$" || true; } | sort >"$doel.volumes"
  { docker network ls --format '{{.ID}}|{{.Name}}|{{.Label "com.docker.compose.project"}}' \
      | grep -v "|${PROEF_PROJECT}\$" || true; } | sort >"$doel.networks"
  docker image ls --no-trunc --format '{{.Repository}}:{{.Tag}}|{{.ID}}' | sort >"$doel.images"
}

# Vergelijk twee momentopnamen. Schrijft de afwijkingen naar stdout; rc 1 bij afwijking.
vergelijk_momentopnamen() {
  local voor=$1 na=$2 afwijking=0 soort regel naam
  for soort in containers volumes networks; do
    if ! diff -q "$voor.$soort" "$na.$soort" >/dev/null; then
      echo "afwijking in $soort:"; diff "$voor.$soort" "$na.$soort" || true
      afwijking=1
    fi
  done
  # Images: verdwenen of gewijzigd is altijd een afwijking; nieuw mag alleen voor
  # de vastgezette proefimages, lokaal gebouwde signaal-proef/*-images en <none>.
  while IFS= read -r regel; do
    [[ -z "$regel" ]] && continue
    case "${regel%%|*}" in signaal-proef/*|"<none>:<none>") continue ;; esac
    echo "image verdwenen of gewijzigd: $regel"; afwijking=1
  done < <(comm -23 "$voor.images" "$na.images")
  while IFS= read -r regel; do
    [[ -z "$regel" ]] && continue
    naam=${regel%%|*}
    case " $PROEF_IMAGES_TOEGESTAAN " in *" $naam "*) continue ;; esac
    case "$naam" in signaal-proef/*|"<none>:<none>") continue ;; esac
    echo "onverwacht nieuw image: $regel"; afwijking=1
  done < <(comm -13 "$voor.images" "$na.images")
  return "$afwijking"
}

# --- verloop van een proef ---

proef_start() {
  PROEF_NAAM=$1; PROEF_PROFIEL=${2:-}
  RUNMAP="$SIGNAAL_PROEF_DIR/runs/$PROEF_NAAM/$(date +%Y%m%dT%H%M%S)"
  mkdir -p "$RUNMAP"
  : >"$RUNMAP/.criteria"; : >"$RUNMAP/.metingen"; : >"$RUNMAP/.metingen_json"
  PROEF_START_TIJD="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  trap 'proef_scriptfout $LINENO "$BASH_COMMAND"' ERR
  trap 'proef_onderbroken' INT TERM
  if ! docker info >/dev/null 2>&1; then
    criterium omgeving niet_uitvoerbaar "Docker is niet bereikbaar"
    proef_schrijf_uitkomst "$EXIT_OMGEVING"
    PROEF_AFGEROND=1
    exit "$EXIT_OMGEVING"
  fi
  momentopname "$RUNMAP/voor"
  log "proef $PROEF_NAAM gestart; runmap $RUNMAP"
}

# Roep P0 aan als voorwaarde (proeven.md: "Elk ander script roept P0 eerst aan").
p0_voorwaarde() {
  local rc=0
  bash "$PROEF_SCRIPTS/p0_omgeving.sh" --als-voorwaarde "$@" >"$RUNMAP/p0.log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    criterium P0 niet_uitvoerbaar "voorwaarde P0 niet gehaald (exitcode $rc), zie p0.log"
    proef_einde_met "$EXIT_OMGEVING"
  fi
}

omgeving_ongeschikt() {
  criterium omgeving niet_uitvoerbaar "$1"
  proef_einde_met "$EXIT_OMGEVING"
}

proef_opruimen_profiel() {
  [[ -z "$PROEF_PROFIEL" ]] && return 0
  if [[ "$PROEF_LAAT_STAAN" == "1" ]]; then
    log "PROEF_LAAT_STAAN=1: containers van profiel $PROEF_PROFIEL blijven staan"
    return 0
  fi
  # Anonieme volumes (VOLUME in een image, zoals /data en /logs van SurrealDB) krijgen
  # geen projectlabel. Alleen die van de eigen containers verwijderen we na down.
  local anoniem=() c v
  while IFS= read -r c; do
    [[ -z "$c" ]] && continue
    while IFS= read -r v; do
      [[ -n "$v" ]] && anoniem+=("$v")
    done < <(docker inspect -f '{{range .Mounts}}{{if eq .Type "volume"}}{{.Name}}{{"\n"}}{{end}}{{end}}' "$c" \
             | while IFS= read -r v; do
                 [[ "$v" =~ ^[0-9a-f]{64}$ ]] && echo "$v"
               done)
  done < <(dc --profile "$PROEF_PROFIEL" ps -aq 2>/dev/null)
  bewaar compose_down dc --profile "$PROEF_PROFIEL" down --timeout 10 || log "compose down gaf een fout, zie compose_down.log"
  for v in "${anoniem[@]}"; do
    bewaar "anoniem_volume_rm" docker volume rm "$v" || log "anoniem volume $v niet verwijderd"
  done
  if docker volume inspect signaal-proef_geheimen >/dev/null 2>&1; then
    bewaar geheimen_volume_rm docker volume rm signaal-proef_geheimen || log "volume signaal-proef_geheimen niet verwijderd"
  fi
}

# Achtergrondprocessen (meting van het videogeheugen, docker events, wachtende
# subshells) stoppen, met al hun kindprocessen. Git Bash kent geen pkill; het
# MSYS-proces van nvidia-smi.exe of docker.exe stoppen stopt ook het Windows-proces.
kill_boom() {
  local p=$1 c
  for c in $(ps -ef | awk -v p="$p" '$3==p {print $2}'); do kill_boom "$c"; done
  kill "$p" 2>/dev/null || true
}
stop_kindprocessen() {
  local c
  for c in $(ps -ef | awk -v p="$$" '$3==p {print $2}'); do kill_boom "$c"; done
}

# Containers die llama-swap met "docker start" start (docling, kev), krijgen geen
# Compose-secrets. Hun sleutels gaan daarom in het volume signaal-proef_geheimen,
# dat Compose bij "create" aanmaakt; na de proef wordt het verwijderd.
geheimen_volume_vullen() {
  drun --network none -v signaal-proef_geheimen:/g -e SP_KEV_API_KEY -e SP_DOCLING_API_KEY "$IMG_PY"     sh -c 'printf %s "$SP_KEV_API_KEY" >/g/kev_api_sleutel && printf %s "$SP_DOCLING_API_KEY" >/g/docling_api_sleutel && chmod 444 /g/kev_api_sleutel /g/docling_api_sleutel'
}

# Bepaal de exitcode uit de criteria en sluit af.
proef_einde() {
  local code=$EXIT_GESLAAGD
  if grep -q $'\tongeldig\t' "$RUNMAP/.criteria"; then code=$EXIT_OMGEVING
  elif grep -q $'\tmislukt\t' "$RUNMAP/.criteria"; then code=$EXIT_CRITERIUM
  fi
  proef_einde_met "$code"
}

proef_einde_met() {
  local code=$1 afwijking
  trap - ERR
  PROEF_AFGEROND=1
  stop_kindprocessen
  proef_opruimen_profiel
  momentopname "$RUNMAP/na"
  if ! afwijking="$(vergelijk_momentopnamen "$RUNMAP/voor" "$RUNMAP/na")"; then
    printf '%s\n' "$afwijking" >"$RUNMAP/afwijking_buiten_project.txt"
    criterium momentopname mislukt "wijziging buiten het project signaal-proef, zie afwijking_buiten_project.txt"
    code=$EXIT_SCRIPT
  else
    criterium momentopname geslaagd "geen wijziging buiten het project signaal-proef"
  fi
  proef_schrijf_uitkomst "$code"
  log "proef $PROEF_NAAM klaar met exitcode $code; uitkomst in $RUNMAP/uitkomst.json"
  exit "$code"
}

proef_scriptfout() {
  local regel=$1 opdracht=$2
  # In een subshell alleen die subshell beëindigen; de hoofdshell handelt de fout af.
  [[ "$BASH_SUBSHELL" -gt 0 ]] && exit "$EXIT_SCRIPT"
  [[ "$PROEF_AFGEROND" == "1" ]] && exit "$EXIT_SCRIPT"
  trap - ERR
  log "scriptfout op regel $regel: $opdracht"
  criterium script mislukt "scriptfout op regel $regel: $opdracht"
  PROEF_AFGEROND=1
  stop_kindprocessen || true
  proef_opruimen_profiel || true
  proef_schrijf_uitkomst "$EXIT_SCRIPT" || true
  exit "$EXIT_SCRIPT"
}

proef_onderbroken() {
  log "onderbroken"
  criterium script mislukt "onderbroken"
  PROEF_AFGEROND=1
  stop_kindprocessen || true
  proef_opruimen_profiel || true
  proef_schrijf_uitkomst "$EXIT_SCRIPT" || true
  exit "$EXIT_SCRIPT"
}

image_digest() {
  local d
  if d="$(docker image inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' "$1" 2>/dev/null)"; then
    echo "$d"
  else
    echo "niet aanwezig"
  fi
}

# Leg de digests vast van de images die de proef gebruikt.
leg_images_vast() {
  local img
  for img in "$@"; do meet "image:$img" "$(image_digest "$img")"; done
}

proef_schrijf_uitkomst() {
  local code=$1 commit eerste k v s t
  commit="$(git -C "$(pad_host "$PROEF_SCRIPTS")" rev-parse --short HEAD 2>/dev/null || echo onbekend)"
  {
    printf '{\n  "proef": %s,\n  "start": %s,\n  "einde": %s,\n  "commit": %s,\n  "exitcode": %s,\n' \
      "$(json_str "$PROEF_NAAM")" "$(json_str "${PROEF_START_TIJD:-}")" \
      "$(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)")" "$(json_str "$commit")" "$code"
    printf '  "criteria": {'
    eerste=1
    while IFS=$'\t' read -r k s t; do
      [[ -z "$k" ]] && continue
      [[ $eerste -eq 0 ]] && printf ','
      printf '\n    %s: {"stand": %s, "toelichting": %s}' "$(json_str "$k")" "$(json_str "$s")" "$(json_str "$t")"
      eerste=0
    done <"$RUNMAP/.criteria"
    printf '\n  },\n  "metingen": {'
    eerste=1
    while IFS=$'\t' read -r k v; do
      [[ -z "$k" ]] && continue
      [[ $eerste -eq 0 ]] && printf ','
      printf '\n    %s: %s' "$(json_str "$k")" "$(json_str "$v")"
      eerste=0
    done <"$RUNMAP/.metingen"
    while IFS=$'\t' read -r k v; do
      [[ -z "$k" || ! -s "$v" ]] && continue
      [[ $eerste -eq 0 ]] && printf ','
      printf '\n    %s: ' "$(json_str "$k")"; masker <"$v"
      eerste=0
    done <"$RUNMAP/.metingen_json"
    printf '\n  }\n}\n'
  } >"$RUNMAP/uitkomst.json"
}

# Wacht tot een opdracht slaagt, hooguit <seconden>.
wacht_tot() {
  local max=$1; shift
  local t=0
  until "$@" >/dev/null 2>&1; do
    sleep 2; t=$((t + 2))
    [[ $t -ge $max ]] && return 1
  done
  return 0
}

# Download een openbaar testbestand naar de werkmap; SHA-256 vastleggen bij de eerste
# download en controleren bij elke herhaling (proeven.md 1.6).
testdata() {
  local id=$1 url=$2 doel map
  map="$SIGNAAL_PROEF_DIR/testdata"; mkdir -p "$map"
  doel="$map/$id.${3:-bin}"
  if [[ ! -s "$doel" ]]; then
    # curl in Git Bash is een Windows-programma; MSYS2_ARG_CONV_EXCL zet het pad niet om.
    curl -fsSL --retry 3 -o "$(pad_host "$doel")" "$url"
    sha256sum "$doel" | cut -d' ' -f1 >"$doel.sha256"
  elif [[ -s "$doel.sha256" ]] && [[ "$(sha256sum "$doel" | cut -d' ' -f1)" != "$(cat "$doel.sha256")" ]]; then
    log "SHA-256 van $id wijkt af van de eerste download"
    return 1
  fi
  printf '%s\n' "$doel"
}

# Is een container van het project op een bepaalde toestand?
container_toestand() {
  docker inspect -f '{{.State.Status}}' "$1" 2>/dev/null || echo "afwezig"
}
