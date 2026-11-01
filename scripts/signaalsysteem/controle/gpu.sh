#!/usr/bin/env bash
# Controle op de werkplek: llama-swap met socket-proxy, embeddingmodel, werkmodel
# en docling-serve (F2-11; ADR 0006, 0008, 0009, 0014). Uit te voeren op de werkplek.
#
# Toetst de draaiende Compose-omgeving (project signaal, gestart met start.sh);
# start of stopt zelf geen dienst, behalve via llama-swap en één externe
# `docker kill` van signaal-docling (G8). Aanroepen gaan door de echte modelpoort
# in de container open-notebook (gpu_poort.py), met een auditschrijver in het
# geheugen: er komt niets in de database. Invoer is synthetisch.
#
#   G0  omgeving: diensten draaien, signaal-docling is aangemaakt en gestopt;
#       nulmeting videogeheugen; andere GPU-gebruikers op de host (ter informatie)
#   G1  CUDA-omweg: driver, NVIDIA_DISABLE_REQUIRE en LD_LIBRARY_PATH in llama-swap
#   G2  de configuratie in het image is gelijk aan services/llama-swap/config.yaml in HEAD
#   G3  embedding via de poort: HTTP 200, 1 024 dimensies
#   G4  werkmodel via de poort: HTTP 200, geladen op de GPU (videogeheugen stijgt)
#   G5  docling-serve start via llama-swap (/upstream, zonder model-veld): de
#       container gaat van exited naar running; het werkmodel is dan ontladen (zwaar)
#   G6  na `docker start`: de sleutel is aanwezig (zonder sleutel 401/403, met
#       sleutel een synchrone conversie met HTTP 200), DOCLING_SERVE_MAX_SYNC_WAIT
#       staat in de omgeving van de gestarte container, de sleutel niet in de
#       configuratie (`docker inspect`)
#   G7  socket-proxy (K2 van P7): _ping 200; containerlijst, inspect, create, exec,
#       images, volumes, events en version 403
#   G8  herhaalpoging na een externe stop: de aanvraag via de poort staat klaar
#       (imports gedaan), `docker kill` van signaal-docling, en direct daarna de
#       aanvraag. Zolang llama-swap de stop nog niet heeft gezien (gezondheidslus
#       elke 5 s), geeft die 502; de poort probeert opnieuw. Criterium: in een
#       ronde ten minste 2 pogingen en daarna HTTP 200; hooguit 3 rondes, want
#       valt de gezondheidslus net tussen stop en aanvraag, dan komt direct 200.
#   G9  llama-swap en docling-serve bereiken geen internet
#   G10 docling-serve staat binnen ttl + 15 s na de laatste aanvraag op exited
#   G11 videogeheugen: piek en per fase vastgelegd; na de stop van docling-serve
#       terug op de rustwaarde van de lichte groep
#   G12 vrij werkgeheugen: laagste MemAvailable in de WSL-VM ≥ 2 048 MiB (R1);
#       vrij geheugen van Windows vastgelegd
#
# Exitcodes: 0 alles geslaagd, 1 een criterium niet gehaald, 2 omgeving
# ongeschikt, 3 scriptfout.
set -uo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/../../.." && { pwd -W 2>/dev/null || pwd; })"
PROJECT=signaal
DOCLING=signaal-docling
TTL=300
ONDERGRENS_MIB=2048
SLEUTEL="$ROOT/secrets/docling_api_key"
HULP="$(cat "$ROOT/scripts/signaalsysteem/controle/gpu_poort.py")" || exit 3
RUN="${SIGNAAL_STAAT:-$HOME/.cache/signaal}/gpu-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN" || exit 3

MISLUKT=0
criterium() { # <naam> <geslaagd|mislukt|let op> <toelichting>
  printf '%-4s %-8s %s\n' "$1" "$2" "$3" | tee -a "$RUN/uitkomst.txt"
  if [[ "$2" == mislukt ]]; then MISLUKT=1; fi
}
dienst() { # container-id van een dienst in het project
  docker ps -q --filter "label=com.docker.compose.project=$PROJECT" \
    --filter "label=com.docker.compose.service=$1" | head -n1
}
toestand() { docker inspect -f '{{.State.Status}}' "$DOCLING" 2>/dev/null | tr -d '\r'; }
vram() { nvidia-smi.exe --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' \r'; }
wsl_vrij() { docker exec "$FORK" awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo | tr -d '\r'; }
win_vrij() {
  powershell.exe -NoProfile -Command \
    '[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)' | tr -d '\r'
}
poort() { # <opdracht> [stdin]: gpu_poort.py in open-notebook
  docker exec -i "$FORK" /app/.venv/bin/python -c "$HULP" "$1" | tr -d '\r'
}
json() { # <json> <python-expressie op d>
  printf '%s' "$1" | python -c "import json,sys; d=json.load(sys.stdin); print($2)" 2>/dev/null \
    || printf '%s' "$1" | docker run --rm -i --network none python:3.12.16-slim-bookworm \
         python -c "import json,sys; d=json.load(sys.stdin); print($2)"
}
geheugen() { # <fase>: vrij werkgeheugen en videogeheugen vastleggen
  local w; w="$(wsl_vrij)"
  printf '%s,%s,%s,%s,%s\n' "$(date +%s)" "$1" "$(vram)" "$w" "$(win_vrij)" >>"$RUN/geheugen.csv"
  if [[ -n "$w" && "$w" -lt "$MIN_WSL" ]]; then MIN_WSL=$w; fi
}

# ------------------------------------------------------------------ G0
command -v nvidia-smi.exe >/dev/null || { echo "nvidia-smi.exe niet gevonden"; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker niet bereikbaar"; exit 2; }
LS="$(dienst llama-swap)"; SP="$(dienst socket-proxy)"; FORK="$(dienst open-notebook)"
[[ -n "$LS" && -n "$SP" && -n "$FORK" ]] || { echo "llama-swap, socket-proxy of open-notebook draait niet"; exit 2; }
[[ -s "$SLEUTEL" ]] || { echo "geen $SLEUTEL"; exit 2; }
t0="$(toestand)"
[[ "$t0" == exited || "$t0" == created ]] || { echo "$DOCLING is '$t0', verwacht created of exited"; exit 2; }
andere="$(docker ps -q | while read -r c; do
  docker inspect -f '{{.Name}}|{{index .Config.Labels "com.docker.compose.project"}}|{{json .HostConfig.DeviceRequests}}' "$c"
done | tr -d '\r' | awk -F'|' '$2!="signaal" && $3!="null" {print $1}' | tr '\n' ' ')"
nul="$(vram)"
echo "epoch,fase,vram_mib,wsl_vrij_mib,windows_vrij_mib" >"$RUN/geheugen.csv"
MIN_WSL=999999
geheugen nulmeting
# Zonder pijp, rechtstreeks naar een bestand: $! is dan nvidia-smi zelf, en het
# stoppen raakt alleen dit proces, geen andere nvidia-smi op de host.
nvidia-smi.exe --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 \
  >"$RUN/vram.txt" 2>/dev/null &
METER=$!
stop_meter() {
  kill "$METER" 2>/dev/null; wait "$METER" 2>/dev/null || true
}
trap stop_meter EXIT
criterium G0 geslaagd "diensten draaien; $DOCLING '$t0'; nulmeting ${nul} MiB; GPU-containers buiten $PROJECT: ${andere:-geen}"

# ------------------------------------------------------------------ G1
cuda="$(nvidia-smi.exe | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9.]*\).*/\2/p' | head -n1)"
env_ls="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$LS" | tr -d '\r')"
if grep -qx 'NVIDIA_DISABLE_REQUIRE=1' <<<"$env_ls" \
   && grep -qx 'LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64' <<<"$env_ls"; then
  criterium G1 geslaagd "driver CUDA $cuda; omweg NVIDIA_DISABLE_REQUIRE en LD_LIBRARY_PATH staat (tot F2-25)"
else
  criterium G1 mislukt "driver CUDA $cuda; omweg ontbreekt in llama-swap"
fi

# ------------------------------------------------------------------ G2
in_image="$(docker exec "$LS" sha256sum /app/config.yaml | cut -d' ' -f1 | tr -d '\r')"
in_repo="$(git -C "$ROOT" show HEAD:services/llama-swap/config.yaml | sha256sum | cut -d' ' -f1)"
[[ "$in_image" == "$in_repo" ]] && criterium G2 geslaagd "config.yaml in het image = HEAD (${in_repo:0:12})" \
  || criterium G2 mislukt "config.yaml in het image wijkt af; start.sh --build"

# ------------------------------------------------------------------ G3
e="$(poort embed)"; echo "$e" >>"$RUN/poort.jsonl"
geheugen embed
dim="$(json "$e" "d.get('dimensie')")"
[[ "$dim" == 1024 ]] && criterium G3 geslaagd "embedding via de poort: $dim dimensies ($(json "$e" "d['seconden']") s)" \
  || criterium G3 mislukt "embedding: $e"

# ------------------------------------------------------------------ G4
voor="$(vram)"
c="$(poort chat)"; echo "$c" >>"$RUN/poort.jsonl"
geheugen werkmodel
na="$(vram)"
if [[ "$(json "$c" "d.get('auditstatus')")" == "['geslaagd']" && $((na - voor)) -gt 4000 ]]; then
  criterium G4 geslaagd "werkmodel via de poort: geslaagd, videogeheugen $voor -> $na MiB"
else
  criterium G4 mislukt "werkmodel: $c; videogeheugen $voor -> $na MiB"
fi

# ------------------------------------------------------------------ G5
d="$(poort docling)"; echo "$d" >>"$RUN/poort.jsonl"
t1="$(toestand)"
running="$(docker exec "$FORK" curl -fsS -m 10 http://llama-swap:8080/running | tr -d '\r')"
echo "$running" >"$RUN/running.json"
zwaar="$(json "$running" "sorted(r['model'] for r in d['running'] if r['model'] in ('werkmodel','docling'))")"
if [[ "$(json "$d" "d.get('status')")" == 200 && "$t1" == running && "$zwaar" == "['docling']" ]]; then
  criterium G5 geslaagd "$DOCLING $t0 -> $t1 via /upstream/docling; zware diensten geladen: $zwaar"
else
  criterium G5 mislukt "docling: $d; $DOCLING '$t1'; zware diensten geladen: $zwaar"
fi

# ------------------------------------------------------------------ G6
k="$(poort docling-conversie <"$SLEUTEL")"; echo "$k" >>"$RUN/poort.jsonl"
geheugen docling-conversie
sleutel_aanwezig="$(docker exec "$DOCLING" sh -c 'test -s /run/secrets/docling_api_key && echo ja')"
wacht="$(docker exec "$DOCLING" printenv DOCLING_SERVE_MAX_SYNC_WAIT | tr -d '\r')"
in_inspect="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$DOCLING" | grep -c '^DOCLING_SERVE_API_KEY=' || true)"
zonder="$(json "$k" "d.get('zonder_sleutel')")"
if [[ "$sleutel_aanwezig" == ja* && ( "$zonder" == 401 || "$zonder" == 403 ) \
      && "$(json "$k" "d.get('status')")" == 200 && "$(json "$k" "d.get('tekst_gevonden')")" == True \
      && "$wacht" == 3600 && "$in_inspect" == 0 ]]; then
  criterium G6 geslaagd "sleutel na docker start aanwezig (zonder sleutel $zonder, met sleutel 200, tekst gevonden, $(json "$k" "d['seconden']") s); MAX_SYNC_WAIT=$wacht; sleutel niet in inspect"
else
  criterium G6 mislukt "conversie: $k; sleutelbestand '$sleutel_aanwezig'; MAX_SYNC_WAIT='$wacht'; sleutel in inspect: $in_inspect"
fi

# ------------------------------------------------------------------ G7
proxy() { # <methode> <pad>: HTTP-status van de socket-proxy, vanuit llama-swap
  docker exec "$LS" curl -s -o /dev/null -w '%{http_code}' -m 10 -X "$1" \
    -H 'Content-Type: application/json' -d '{"Image":"signaal/bestaat-niet","Cmd":["true"]}' \
    "http://socket-proxy:2375$2" | tr -d '\r'
}
p_ping="$(docker exec "$LS" curl -s -o /dev/null -w '%{http_code}' -m 10 http://socket-proxy:2375/_ping | tr -d '\r')"
uitslag="ping=$p_ping"
weigert=1
for regel in "GET /containers/json" "GET /containers/$DOCLING/json" "POST /containers/create" \
             "POST /containers/$DOCLING/exec" "GET /images/json" "GET /volumes" \
             "GET /events?until=0" "GET /version"; do
  s="$(proxy $regel)"; uitslag="$uitslag; $regel=$s"
  [[ "$s" == 403 ]] || weigert=0
done
[[ "$p_ping" == 200 && $weigert -eq 1 ]] && criterium G7 geslaagd "socket-proxy configuratie B: $uitslag" \
  || criterium G7 mislukt "socket-proxy: $uitslag"

# ------------------------------------------------------------------ G9 (docling draait nu)
# Een HTTP-foutstatus telt als verbinding (proeven.md §8 punt 8); een mislukte
# controle (geen curl of python) geeft geen 'dicht' (§8 punt 21).
s_ls="$(docker exec "$LS" curl -s -o /dev/null -w '%{http_code}' -m 8 https://example.org | tr -d '\r')"
n_ls="$([[ "$s_ls" == 000 ]] && echo dicht || echo "open ($s_ls)")"
n_dl="$(docker exec "$DOCLING" python3 -c 'import urllib.request, urllib.error
try:
    urllib.request.urlopen("https://example.org", timeout=8); print("open")
except urllib.error.HTTPError:
    print("open")
except Exception:
    print("dicht")' | tr -d '\r')"
[[ "$n_ls" == dicht && "$n_dl" == dicht ]] && criterium G9 geslaagd "geen internet: llama-swap $n_ls, docling-serve $n_dl" \
  || criterium G9 mislukt "internet: llama-swap $n_ls, docling-serve $n_dl"

# ------------------------------------------------------------------ G8
g8=mislukt; rondes=""
for ronde in 1 2 3; do
  [[ "$(toestand)" == running ]] || poort docling >>"$RUN/poort.jsonl"
  coproc AANVRAAG { docker exec -i "$FORK" /app/.venv/bin/python -c "$HULP" docling-na-signaal; }
  read -r klaar <&"${AANVRAAG[0]}"
  docker kill "$DOCLING" >/dev/null
  echo nu >&"${AANVRAAG[1]}"
  read -r r <&"${AANVRAAG[0]}"; r="$(printf '%s' "$r" | tr -d '\r')"
  wait "$AANVRAAG_PID" 2>/dev/null
  echo "$r" >>"$RUN/poort.jsonl"
  status="$(json "$r" "d.get('status')")"; pogingen="$(json "$r" "max(d.get('pogingen') or [0])")"
  rondes="$rondes ronde $ronde: status $status, pogingen $pogingen;"
  if [[ "$status" != 200 ]]; then break; fi
  if [[ "$pogingen" -ge 2 ]]; then g8=geslaagd; break; fi
done
laatste=$(date +%s)
if [[ "$g8" == geslaagd && "$(toestand)" == running ]]; then
  criterium G8 geslaagd "na docker kill: 502, herhaalpoging, daarna HTTP 200 via de poort;$rondes"
else
  criterium G8 mislukt "na docker kill:$rondes $DOCLING '$(toestand)' (klaar='$klaar')"
fi

# ------------------------------------------------------------------ G10
grens=$((TTL + 15))
while [[ "$(toestand)" == running && $(( $(date +%s) - laatste )) -le $((grens + 30)) ]]; do sleep 5; done
duur=$(( $(date +%s) - laatste ))
t2="$(toestand)"
[[ "$t2" == exited && $duur -le $grens ]] && criterium G10 geslaagd "$DOCLING exited na $duur s (ttl $TTL + 15 = $grens)" \
  || criterium G10 mislukt "$DOCLING '$t2' na $duur s (grens $grens)"
sleep 15
geheugen na-stop-docling

# ------------------------------------------------------------------ G11, G12
stop_meter; trap - EXIT
piek="$(tr -d ' \r' <"$RUN/vram.txt" | sort -n | tail -n1)"
rust="$(awk -F, '$2=="embed" {print $3}' "$RUN/geheugen.csv")"
eind="$(awk -F, '$2=="na-stop-docling" {print $3}' "$RUN/geheugen.csv")"
fasen="$(awk -F, 'NR>1 {printf "%s %s; ", $2, $3}' "$RUN/geheugen.csv")"
if [[ -n "$piek" && -n "$eind" && -n "$rust" && $((eind - rust)) -le 256 ]]; then
  criterium G11 geslaagd "piek ${piek} MiB van $(nvidia-smi.exe --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' \r'); per fase (MiB): $fasen"
else
  criterium G11 mislukt "piek ${piek:-?}; na de stop ${eind:-?} tegen rust ${rust:-?} MiB; $fasen"
fi
win="$(awk -F, 'NR>1 {print $5}' "$RUN/geheugen.csv" | sort -n | head -n1)"
[[ $MIN_WSL -ge $ONDERGRENS_MIB ]] && criterium G12 geslaagd "laagste MemAvailable WSL-VM $MIN_WSL MiB (grens $ONDERGRENS_MIB); laagste vrij Windows $win MiB" \
  || criterium G12 mislukt "laagste MemAvailable WSL-VM $MIN_WSL MiB < $ONDERGRENS_MIB; Windows $win MiB"

echo "verslag: $RUN"
exit "$MISLUKT"
