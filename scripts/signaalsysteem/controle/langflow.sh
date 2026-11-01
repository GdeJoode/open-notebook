#!/usr/bin/env bash
# Controle van de afscherming van Langflow (F2-12; ADR 0017, ADR 0022 A, D, F).
#
# Toetst de draaiende Compose-omgeving (project signaal, gestart met start.sh).
# Herstart alleen de container langflow, voor L7.
#   L1  aanmelden zonder wachtwoord faalt: auto_login geweigerd, leeg of fout
#       wachtwoord geen token, de API zonder token geweigerd
#   L2  geen MCP en geen andere agentinterface: de MCP-eindpunten, de MCP-server
#       van de assistent (POST /api/v1/agentic/mcp, initialize), A2A en de
#       OpenAI-Responses-API en de Store geven geen 2xx. Via editor-toegang
#       aangemeld 404; in de container zelf aangemeld hetzelfde antwoord als een
#       route die niet bestaat (GET 404; POST 405, want de interface bedient elk
#       ander pad met GET) (review F2-12, B1 en punt 10)
#   L3  langflow bereikt geen internet (opzet van P8)
#   L4  docker inspect: geen verboden variabele (tracers, ANTHROPIC_, CLAUDE_,
#       LANGFLOW_FEATURE_), entrypoint /app/signaal/start.sh,
#       geen geheim in de omgeving, alleen de secrets sleutel_pijplijn,
#       langflow_superuser en langflow_secret_key
#   L5  de editor alleen via editor-toegang op 127.0.0.1: langflow publiceert geen
#       poort, hangt alleen aan verwerking en editor-intern; de proxy alleen op
#       127.0.0.1 en alleen aan editor-intern en toegang, en geeft de editor;
#       een andere Host-header dan 127.0.0.1:<poort> of localhost:<poort> geeft 421
#   L6  editor-toegang is gehard (read_only, cap_drop ALL, no-new-privileges,
#       geen root, geen koppelingen) en heeft geen shell
#   L7  uitgaande verbindingspogingen bij het opstarten en daarna: langflow
#       herstarten, in zijn netwerknaamruimte de DNS-vragen naar namen buiten
#       Compose tellen, ook terwijl de editor- en store-eindpunten worden
#       aangeroepen, en de logregels zoals in P2. Doel 0 (ADR 0017)
#   L8  SIGNAAL_MODELREGISTER gezet, het register geldig, llm_manager.gate laadbaar
#   L9  koppelingen: geen vault; schrijfbaar alleen /app/data en /bronarchief; het
#       image alleen-lezen met /tmp als tmpfs (review F2-12, punt 8)
#
# Exitcodes: 0 alles geslaagd, 1 een criterium niet gehaald, 2 omgeving ongeschikt.
set -uo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/../../.." && { pwd -W 2>/dev/null || pwd; })"
PROJECT=signaal
PY_IMG="python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337"
POORT="${SIGNAAL_EDITOR_POORT:-8530}"
TELLEN_S="${SIGNAAL_TELLEN_S:-180}"
VERBODEN='^(LANGCHAIN_|LANGSMITH_|LANGFUSE_|LANGWATCH_|ARIZE_|OPIK_|PHOENIX_|TRACELOOP_|OPENLAYER_|OTEL_EXPORTER_OTLP_|ANTHROPIC_|CLAUDE_|LANGFLOW_FEATURE_)'

MISLUKT=0
criterium() { # <naam> <geslaagd|mislukt|vastgelegd> <toelichting>
  printf '%-4s %-10s %s\n' "$1" "$2" "$3"
  [[ "$2" == geslaagd || "$2" == vastgelegd ]] || MISLUKT=1
}
container() { docker ps -q --filter "label=com.docker.compose.project=$PROJECT" \
  --filter "label=com.docker.compose.service=$1" | head -n1; }
in_netwerk_van() { local c=$1; shift; docker run --rm --network "container:$c" "$PY_IMG" python -c "$@"; }
status() { # <pad> [timeout] [curl-opties...]
  curl -s -o /dev/null -w '%{http_code}' --max-time "${2:-10}" "${@:3}" "http://127.0.0.1:$POORT$1" || true; }
netten() { docker inspect -f '{{range $n, $_ := .NetworkSettings.Networks}}{{println $n}}{{end}}' "$1" \
  | grep -v '^$' | sort | tr '\n' ' '; }

docker info >/dev/null 2>&1 || { echo "Docker niet bereikbaar"; exit 2; }
LF="$(container langflow)"; ED="$(container editor-toegang)"
[[ -n "$LF" && -n "$ED" ]] || { echo "langflow of editor-toegang draait niet"; exit 2; }
[[ "$(docker inspect -f '{{.State.Health.Status}}' "$LF")" == healthy ]] || { echo "langflow is niet healthy"; exit 2; }
docker image inspect "$PY_IMG" >/dev/null 2>&1 || docker pull -q "$PY_IMG" >/dev/null
WACHTWOORD_BESTAND="$ROOT/secrets/langflow_superuser"
[[ -s "$WACHTWOORD_BESTAND" ]] || { echo "secrets/langflow_superuser ontbreekt"; exit 2; }

# Aanmelden in de container, met het wachtwoord uit het secret; drukt statussen af.
PY_IN_CONTAINER='
import json, sys, urllib.error, urllib.parse, urllib.request
URL = "http://127.0.0.1:7860"
def vraag(methode, pad, data=None, kop=None, timeout=5, lees=False):
    # Alleen de status; een stroom (SSE) wordt niet gelezen.
    v = urllib.request.Request(URL + pad, data=data, headers=kop or {}, method=methode)
    try:
        with urllib.request.urlopen(v, timeout=timeout) as a:
            return a.status, (a.read() if lees else b"")
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        return "fout:" + type(e).__name__, b""
# Aanmelden met de client uit het image: die probeert opnieuw bij 429 (Langflow
# staat 5 aanmeldingen per minuut toe).
sys.path.insert(0, "/app/signaal")
from langflow_api import LangflowApi
token = LangflowApi().token()
s, body = vraag("GET", "/api/v1/projects/", kop={"Authorization": "Bearer " + token}, lees=True)
project = json.loads(body)[0]["id"]
INIT = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "toets", "version": "0"}}}).encode()
def status(methode, pad, data, metaan):
    kop = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if metaan:
        kop["Authorization"] = "Bearer " + token
    return vraag(methode, pad, data=data, kop=kop, timeout=3)[0]
# Wat Langflow antwoordt op een route die niet bestaat: GET 404; POST 405, omdat de
# interface (statische bestanden) elk ander pad met GET bedient. Een uitgezette
# route moet hetzelfde antwoord geven.
referentie = {m: status(m, "/api/v1/f212-bestaat-niet", INIT if m == "POST" else None, True)
              for m in ("GET", "POST")}
uit, oordeel = [], "ok"
for methode, pad, data in (
        ("GET", "/api/v1/mcp/sse", None), ("GET", "/api/v1/mcp/streamable", None),
        ("GET", "/api/v1/mcp/project/%s/sse" % project, None),
        ("GET", "/api/v1/mcp/project/%s/streamable" % project, None), ("GET", "/api/v2/mcp/servers", None),
        ("POST", "/api/v1/agentic/mcp", INIT), ("POST", "/api/v1/agentic/mcp/", INIT),
        ("GET", "/api/v1/agentic/mcp", None),
        ("GET", "/api/v1/a2a/agents", None),
        ("POST", "/api/v1/responses", json.dumps({"model": project, "input": "x"}).encode()),
        ("GET", "/api/v1/store/check/", None), ("GET", "/api/v1/store/tags", None),
        ("GET", "/api/v1/voice/elevenlabs/voice_ids", None), ("POST", "/api/v1/agentic/assist/run", INIT)):
    for metaan in (False, True):
        s = status(methode, pad, data, metaan)
        if str(s).startswith("2") or (metaan and s != referentie[methode]):
            oordeel = "fout"
        uit.append("%s%s%s=%s" % ("POST " if methode == "POST" else "", pad, "(aangemeld)" if metaan else "", s))
print(" ".join(uit), "referentie(niet-bestaand): GET=%s POST=%s" % (referentie["GET"], referentie["POST"]),
      "oordeel=" + oordeel)
'

# ---------------------------------------------------------------- L1
c_auto="$(status /api/v1/auto_login)"
c_leeg="$(status /api/v1/login 10 -X POST --data-urlencode username=signaalbeheer --data-urlencode password=)"
c_fout="$(status /api/v1/login 10 -X POST --data-urlencode username=signaalbeheer --data-urlencode password=onjuist)"
c_flows="$(status /api/v1/flows/)"
if [[ "$c_auto" =~ ^4 && "$c_leeg" =~ ^4 && "$c_fout" =~ ^4 && "$c_flows" =~ ^40[13]$ ]]; then
  criterium L1 geslaagd "auto_login $c_auto, leeg wachtwoord $c_leeg, fout wachtwoord $c_fout, flows zonder token $c_flows"
else
  criterium L1 mislukt "auto_login $c_auto, leeg wachtwoord $c_leeg, fout wachtwoord $c_fout, flows zonder token $c_flows"
fi

# ---------------------------------------------------------------- L2
token_via_proxy() { # bij 429 (5 aanmeldingen per minuut) opnieuw, hooguit 6 keer
  local t="" i
  for i in 1 2 3 4 5 6; do
    t="$(curl -s --max-time 30 -X POST --data-urlencode username=signaalbeheer \
      --data-urlencode "password=$(tr -d '\r\n' < "$WACHTWOORD_BESTAND")" "http://127.0.0.1:$POORT/api/v1/login" \
      | sed -n 's/.*"access_token":"\([^"]*\)".*/\1/p')"
    [[ -n "$t" ]] && break
    sleep 15
  done
  printf '%s' "$t"
}
TOKEN="$(token_via_proxy)"
INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"toets","version":"0"}}}'
via_proxy=""
for pad in /api/v1/mcp/sse /api/v1/mcp/streamable /api/v1/mcp/project/x/sse /api/v2/mcp/servers /api/v1/a2a/agents \
           /api/v1/agentic/mcp /api/v1/store/check/ /api/v1/store/tags \
           /api/v1/flows/public_flow/x /api/v1/webhook-events/x /api/v2/workflows /api/v2/workflows/pending \
           /api/v1/voice/elevenlabs/voice_ids /api/v1/voice/ws/flow_tts/x /api/v1/build_public_tmp/x/events; do
  via_proxy+=" $pad=$(status "$pad" 3)"
  via_proxy+=" $pad(aangemeld)=$(status "$pad" 3 -H "Authorization: Bearer $TOKEN")"
done
# Flows draaien buiten de editor om (herreview F2-12, punt 1): alleen /api/v1/build mag.
for pad in /api/v1/agentic/mcp /api/v1/agentic/mcp/ /api/v1/responses /api/v1/agentic/assist/run \
           /api/v1/run/x /api/v1/run/advanced/x /api/v1/run/session/x /api/v1/webhook/x /api/v1/process/x \
           /api/v1/predict/x /api/v1/build_public_tmp/x/flow /api/v2/workflows; do
  for kop in "X-Geen: 1" "Authorization: Bearer $TOKEN"; do
    via_proxy+=" POST $pad$([[ "$kop" == Auth* ]] && echo '(aangemeld)')=$(status "$pad" 3 -X POST \
      -H "$kop" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -d "$INIT")"
  done
done
binnen="$(docker exec -i "$LF" python - <<<"$PY_IN_CONTAINER" 2>&1 | tail -n1)"
# Via de proxy: geen 2xx en aangemeld 404. In de container: geen 2xx en aangemeld
# hetzelfde antwoord als een route die niet bestaat (GET 404, POST 405).
geen_2xx() { [[ ! "$1" =~ =2[0-9][0-9]( |$) ]]; }
aangemeld_404() { local w; for w in $1; do [[ "$w" != *"(aangemeld)="* || "$w" == *"(aangemeld)=404" ]] || return 1; done; }
if [[ -n "$TOKEN" && "$binnen" == *"POST /api/v1/agentic/mcp(aangemeld)="* && "$binnen" == *"oordeel=ok" ]] \
    && geen_2xx "$via_proxy" && aangemeld_404 "$via_proxy"; then
  criterium L2 geslaagd "via editor-toegang:$via_proxy; in de container: $binnen"
else
  criterium L2 mislukt "via editor-toegang:$via_proxy; in de container: $binnen"
fi

# ---------------------------------------------------------------- L3
PY_INTERNET='
import socket, sys, urllib.error, urllib.request
uit = []
try:
    socket.getaddrinfo("example.org", 443); uit.append("dns=ok")
except Exception as e:
    uit.append("dns=fout:" + type(e).__name__)
try:
    urllib.request.urlopen("https://example.org", timeout=5); uit.append("https=ok")
except urllib.error.HTTPError as e:
    uit.append("https=%d" % e.code)
except Exception as e:
    uit.append("https=fout:" + type(e).__name__)
try:
    socket.create_connection(("1.1.1.1", 443), timeout=5).close(); uit.append("ip=ok")
except Exception as e:
    uit.append("ip=fout:" + type(e).__name__)
print(" ".join(uit))
'
uit="$(in_netwerk_van "$LF" "$PY_INTERNET" 2>&1 | tail -n1)"
if [[ "$uit" == *"dns=fout"* && "$uit" == *"https=fout"* && "$uit" == *"ip=fout"* ]]; then
  criterium L3 geslaagd "langflow heeft geen internet ($uit)"
else
  criterium L3 mislukt "langflow heeft internet ($uit)"
fi

# ---------------------------------------------------------------- L4
namen="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$LF" | sed -n 's/^\([^=]*\)=.*/\1/p')"
verboden="$(printf '%s\n' "$namen" | tr '[:lower:]' '[:upper:]' | grep -E "$VERBODEN" | tr '\n' ' ')"
inspect="$(docker inspect "$LF")"
gelekt=""
for g in langflow_superuser langflow_secret_key sleutel_pijplijn; do
  waarde="$(tr -d '\r\n' < "$ROOT/secrets/$g" 2>/dev/null || true)"
  if [[ -n "$waarde" && "$inspect" == *"$waarde"* ]]; then gelekt+=" $g"; fi
done
geheimen="$(docker inspect -f '{{range .Mounts}}{{println .Destination}}{{end}}' "$LF" | sed -n 's#^/run/secrets/##p' | sort | tr '\n' ' ')"
# Het startscript met de controles is het entrypoint (herreview F2-12, punt 6).
start="$(docker inspect -f '{{.Path}} {{json .Args}} {{json .Config.Entrypoint}}' "$LF")"
if [[ -z "$verboden" && -z "$gelekt" && "$geheimen" == "langflow_secret_key langflow_superuser sleutel_pijplijn " \
      && "$start" == '/app/signaal/start.sh [] ["/app/signaal/start.sh"]' ]]; then
  criterium L4 geslaagd "geen verboden variabele, geen geheim in de omgeving; secrets: $geheimen; entrypoint: $start"
else
  criterium L4 mislukt "verboden: [$verboden] geheim in inspect: [$gelekt] secrets: [$geheimen] entrypoint: [$start]"
fi

# ---------------------------------------------------------------- L5
lf_poorten="$(docker inspect -f '{{json .HostConfig.PortBindings}}' "$LF")"
lf_netten="$(netten "$LF")"
ed_netten="$(netten "$ED")"
ed_poorten="$(docker inspect -f '{{range $p, $b := .HostConfig.PortBindings}}{{range $b}}{{.HostIp}}:{{.HostPort}} {{end}}{{end}}' "$ED")"
rc=0; curl -s -o /dev/null --max-time 5 "http://127.0.0.1:7860/" || rc=$?
html="$(curl -s --max-time 20 "http://127.0.0.1:$POORT/")"
c_ui="$(status / 20)"
# Host-header (DNS-rebinding, herreview F2-12, punt 1): alleen 127.0.0.1 en localhost.
c_lh="$(status / 10 -H "Host: localhost:$POORT")"
c_vreemd="$(status / 10 -H "Host: aanvaller.example:$POORT")"
c_zonder_poort="$(status / 10 -H 'Host: 127.0.0.1')"
if [[ ( "$lf_poorten" == "{}" || "$lf_poorten" == "null" ) && "$lf_netten" == "${PROJECT}_editor-intern ${PROJECT}_verwerking " \
      && "$ed_netten" == "${PROJECT}_editor-intern ${PROJECT}_toegang " && "$ed_poorten" == "127.0.0.1:$POORT " \
      && $rc -ne 0 && "$c_ui" == 200 && "$html" == *"<html"* && "$c_lh" == 200 && "$c_vreemd" == 421 \
      && "$c_zonder_poort" == 421 ]]; then
  criterium L5 geslaagd "langflow zonder poort op [$lf_netten]; editor-toegang $ed_poorten op [$ed_netten] geeft de editor (HTTP $c_ui, Host localhost $c_lh, vreemde Host $c_vreemd, zonder poort $c_zonder_poort); 127.0.0.1:7860 dicht (curl $rc)"
else
  criterium L5 mislukt "langflow poorten $lf_poorten netten [$lf_netten]; editor-toegang [$ed_poorten] [$ed_netten] HTTP $c_ui; Host localhost $c_lh, vreemd $c_vreemd, zonder poort $c_zonder_poort; 7860 curl $rc"
fi

# ---------------------------------------------------------------- L6
insp="$(docker inspect -f '{{.HostConfig.ReadonlyRootfs}}|{{json .HostConfig.CapDrop}}|{{json .HostConfig.SecurityOpt}}|{{.Config.User}}|{{len .Mounts}}' "$ED")"
IFS='|' read -r ro caps sec gebruiker mounts <<<"$insp"
shells=""
for sh in /bin/sh /bin/ash /bin/busybox /bin/bash; do
  docker exec "$ED" "$sh" -c true >/dev/null 2>&1 && shells+=" $sh"
done
if [[ "$ro" == true && "$caps" == '["ALL"]' && "$sec" == *no-new-privileges:true* && -n "$gebruiker" \
      && "$gebruiker" != 0* && "$gebruiker" != root* && "$mounts" == 0 && -z "$shells" ]]; then
  criterium L6 geslaagd "editor-toegang: read_only, cap_drop ALL, no-new-privileges, gebruiker $gebruiker, geen koppelingen, geen shell"
else
  criterium L6 mislukt "editor-toegang: $insp; shells:$shells"
fi

# ---------------------------------------------------------------- L8
reg="$(docker exec "$LF" python -c 'import os; from llm_manager.register import lees_register; import llm_manager.gate, llm_manager.aanroep
print(os.environ["SIGNAAL_MODELREGISTER"], len(lees_register().modellen))' 2>&1 | tail -n1)"
if [[ "$reg" =~ ^/app/flows-git/config/modelregister\.yaml\ [1-9] ]]; then
  criterium L8 geslaagd "SIGNAAL_MODELREGISTER en register geldig ($reg modellen); llm_manager.gate laadbaar"
else
  criterium L8 mislukt "modelregister: $reg"
fi

# ---------------------------------------------------------------- L9
mounts="$(docker inspect -f '{{range .Mounts}}{{.Destination}}|{{.RW}}|{{.Source}}{{println}}{{end}}' "$LF")"
schrijfbaar="$(printf '%s\n' "$mounts" | awk -F'|' '$2=="true"{print $1}' | sort | tr '\n' ' ')"
vault="$(printf '%s\n' "$mounts" | grep -i -E '^/vault|obsidian|vault\|' || true)"
lf_ro="$(docker inspect -f '{{.HostConfig.ReadonlyRootfs}} {{json .HostConfig.Tmpfs}}' "$LF")"
if [[ "$schrijfbaar" == "/app/data /bronarchief " && -z "$vault" && "$lf_ro" == 'true {"/tmp":""}' ]]; then
  criterium L9 geslaagd "schrijfbaar alleen: $schrijfbaar en tmpfs /tmp; image alleen-lezen; geen vault"
else
  criterium L9 mislukt "schrijfbaar: [$schrijfbaar] alleen-lezen/tmpfs: [$lf_ro] vault: [$vault]"
fi

# ---------------------------------------------------------------- L7
# DNS-vragen tellen in de netwerknaamruimte van langflow, vanaf de herstart. De
# DNS van Docker zit op 127.0.0.11; op een intern netwerk beantwoordt hij geen
# naam buiten Compose. Een naam met een punt telt als uitgaande poging; een naam
# zonder punt is een dienst van Compose. Een verbinding met een IP-adres buiten
# het netwerk geeft geen pakket (geen route) en is hier niet te zien; die dekt
# de telling van de logregels (P2).
PY_TEL='
import json, signal, socket, struct, sys, time
duur = float(sys.argv[1])
s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3))
s.settimeout(1.0)
vragen, intern, syn = {}, {}, {}
def qnaam(d):
    i, delen = 12, []
    while i < len(d) and d[i]:
        n = d[i]; delen.append(d[i + 1:i + 1 + n].decode("ascii", "replace")); i += n + 1
    return ".".join(delen).lower()
einde = time.time() + duur
while time.time() < einde:
    try:
        pakket, adres = s.recvfrom(65535)
    except socket.timeout:
        continue
    if adres[2] != socket.PACKET_OUTGOING or len(pakket) < 34:
        continue
    if struct.unpack("!H", pakket[12:14])[0] != 0x0800:
        continue
    ip = pakket[14:]
    ihl = (ip[0] & 15) * 4
    proto, dst = ip[9], socket.inet_ntoa(ip[16:20])
    if proto == 17 and dst == "127.0.0.11":
        dns = ip[ihl + 8:]
        if len(dns) > 12 and not dns[2] & 0x80:
            naam = qnaam(dns)
            doel = vragen if "." in naam else intern
            doel[naam] = doel.get(naam, 0) + 1
    elif proto == 6 and ip[ihl + 13] & 0x02 and not ip[ihl + 13] & 0x10:
        a = ip[16]
        if not (a in (10, 127) or dst.startswith("192.168.") or (a == 172 and 16 <= ip[17] <= 31)):
            syn[dst] = syn.get(dst, 0) + 1
print(json.dumps({"extern": vragen, "intern": intern, "syn_buiten": syn}))
'
SNIF=signaal-toets-telling
docker rm -f "$SNIF" >/dev/null 2>&1 || true
herstart="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker restart "$LF" >/dev/null
docker run -d --name "$SNIF" --network "container:$LF" "$PY_IMG" python -c "$PY_TEL" "$TELLEN_S" >/dev/null
gestart_lf="$(docker inspect -f '{{.State.StartedAt}}' "$LF")"
gestart_sn="$(docker inspect -f '{{.State.StartedAt}}' "$SNIF")"
gezond=0
for _ in $(seq 1 90); do
  [[ "$(docker inspect -f '{{.State.Health.Status}}' "$LF")" == healthy ]] && { gezond=1; break; }
  sleep 2
done
# Binnen de telling: de editor en de eindpunten van de store en de modellen
# aanroepen, aangemeld via editor-toegang (review F2-12, punt 4).
aanroepen=""
if [[ $gezond == 1 ]]; then
  T2="$(token_via_proxy)"
  for pad in / /flows /api/v1/config /api/v1/version /api/v1/all /api/v1/flows/ /api/v1/projects/ \
             /api/v1/variables/ /api/v1/starter-projects/ /api/v1/store/check/ /api/v1/store/check/api_key \
             /api/v1/store/components/ /api/v1/store/tags /api/v1/store/users/likes /api/v1/models \
             /api/v1/models/providers /api/v1/models/enabled_providers /api/v1/models/enabled_models \
             /api/v1/models/default_model /api/v1/model_options/language /api/v1/model_options/embedding; do
    aanroepen+=" $pad=$(status "$pad" 30 -H "Authorization: Bearer $T2")"
  done
fi
docker wait "$SNIF" >/dev/null
telling="$(docker logs "$SNIF" 2>&1 | tail -n1)"
docker rm -f "$SNIF" >/dev/null 2>&1 || true
logregels="$(docker logs --since "$herstart" "$LF" 2>&1 | grep -ciE 'Name or service not known|Temporary failure in name resolution|ConnectError|Network is unreachable|Could not fetch|getaddrinfo failed' || true)"
voorbeelden="$(docker logs --since "$herstart" "$LF" 2>&1 | grep -oiE 'https?://[a-z0-9.-]+\.[a-z]{2,}[^ "]*' | grep -viE 'localhost|127\.0\.0\.1|github\.com/langflow-ai/langflow|discord\.com/invite|docs\.langflow\.org' | sort -u | head -n 5 | tr '\n' ' ')"
extern="$(docker run --rm -i --network none "$PY_IMG" python -c 'import json,sys
d=json.loads(sys.stdin.read()); print(sum(d["extern"].values()) + sum(d["syn_buiten"].values()), json.dumps(d, sort_keys=True))' <<<"$telling" 2>/dev/null)"
aantal="${extern%% *}"
toelichting="herstart $gestart_lf, telling vanaf $gestart_sn, ${TELLEN_S} s; healthy=$gezond; pakketten: ${extern#* }; logregels (P2): $logregels; adressen in de log: [${voorbeelden}]; aangeroepen:${aanroepen}"
if [[ $gezond == 1 && "$aantal" == 0 && "$logregels" == 0 && -n "$aanroepen" && -n "${T2:-}" ]]; then
  criterium L7 geslaagd "0 uitgaande pogingen; $toelichting"
else
  criterium L7 mislukt "${aantal:-?} DNS-vragen of SYN naar buiten; $toelichting"
fi

exit $MISLUKT
