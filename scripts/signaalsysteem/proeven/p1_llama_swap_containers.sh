#!/usr/bin/env bash
# P1. docling-serve, WhisperX en Kev als containers onder llama-swap
# (docs/signaalsysteem/proeven.md, P1). Uit te voeren op de werkplek.
#
#   p1_llama_swap_containers.sh --voorbereiden   images, Kev, modellen, testaudio (internet)
#   p1_llama_swap_containers.sh                  de proef zelf (netwerk "verwerking")
#
# Voorwaarden: akkoord voor de Docker-socket (SIGNAAL_PROEF_AKKOORD_SOCKET=ja), P7
# geslaagd, GPU vrij (P0 --gpu). Variant A van WhisperX vraagt geen token.
source "$(dirname "$0")/lib/gemeenschappelijk.sh"
source "$(dirname "$0")/lib/vram_meter.sh"

VOORBEREIDEN=0
[[ "${1:-}" == "--voorbereiden" ]] && VOORBEREIDEN=1

if [[ $VOORBEREIDEN -eq 1 ]]; then
  proef_start p1-voorbereiden p1-voorbereiden
  p0_voorwaarde --gpu
  for img in "$IMG_LLAMA_SWAP" "$IMG_DOCLING" "$IMG_WHISPERX_A" "$IMG_SOCKET_PROXY" "$IMG_DOCKER_CLI" "$IMG_CURL" "$IMG_PY" "$IMG_PY_KEV"; do
    # public.ecr.aws geeft anoniem soms "toomanyrequests"; drie pogingen met 60 s ertussen.
    for poging in 1 2 3; do
      docker image inspect "$img" >/dev/null 2>&1 && break
      bewaar "pull_${img//[\/:]/_}_$poging" docker pull "$img" || sleep 60
    done
    docker image inspect "$img" >/dev/null 2>&1 || omgeving_ongeschikt "image $img niet op te halen, zie pull-logs"
  done
  geheim_maken SP_KEV_API_KEY; geheim_maken SP_DOCLING_API_KEY
  bewaar build dc --profile p1 --profile p1-voorbereiden build kev llama-swap p1-modellen-voorbereiden
  rc=0; dc --profile p1-voorbereiden run --rm p1-modellen-voorbereiden >"$RUNMAP/modellen.json" 2>"$RUNMAP/modellen.err" || rc=$?
  meet_json modellen "$RUNMAP/modellen.json"
  [[ $rc -eq 0 ]] && criterium modellen geslaagd "GGUF met SHA-256-controle, Kev, basismodel, audio" \
                  || criterium modellen mislukt "downloaden of SHA-256-controle mislukt (exitcode $rc), zie modellen.err"
  rc=0; bewaar docling_modellen dc --profile p1-voorbereiden run --rm docling-voorbereiden || rc=$?
  [[ $rc -eq 0 ]] && criterium docling_modellen geslaagd "Docling-modellen in signaal-proef_docling_artifacts" \
                  || criterium docling_modellen mislukt "exitcode $rc"
  # Whisper large-v3 en de uitlijnmodellen voor nl (HF) en en (torchaudio) vooraf ophalen
  # door beide testfragmenten één keer met internet te transcriberen.
  for taal in nl en; do
    bestand=/audio/D7-nl.ogg; [[ $taal == en ]] && bestand=/audio/D8-en.ogg
    rc=0
    bewaar "whisperx_voorbereiden_$taal" dc --profile p1-voorbereiden run --rm --entrypoint whisperx whisperx-voorbereiden \
      "$bestand" --model large-v3 --language "$taal" --output_dir /tmp --output_format json || rc=$?
    [[ $rc -eq 0 ]] && criterium "uitlijnmodel_$taal" geslaagd "whisper large-v3 en uitlijnmodel $taal in het volume" \
                    || criterium "uitlijnmodel_$taal" mislukt "exitcode $rc"
  done
  bewaar create dc --profile p1 create docling whisperx kev
  leg_images_vast "$IMG_LLAMA_SWAP" "$IMG_DOCLING" "$IMG_WHISPERX_A" "$IMG_SOCKET_PROXY" "$IMG_DOCKER_CLI" "$IMG_CURL" \
    signaal-proef/kev:kev-1.0 signaal-proef/llama-swap:v268-cuda13-b11977-dockercli
  PROEF_LAAT_STAAN=1   # de aangemaakte containers blijven staan voor de proef
  proef_einde
fi

proef_start p1 p1
p0_voorwaarde --gpu
if [[ "${SIGNAAL_PROEF_AKKOORD_SOCKET:-}" != "ja" ]]; then
  omgeving_ongeschikt "geen akkoord voor de Docker-socket (SIGNAAL_PROEF_AKKOORD_SOCKET=ja ontbreekt)"
fi
for v in modellen kev_hf whisperx_hf docling_artifacts p1_audio; do
  docker volume inspect "signaal-proef_$v" >/dev/null 2>&1 || omgeving_ongeschikt "volume signaal-proef_$v ontbreekt: eerst --voorbereiden"
done
geheim_maken SP_KEV_API_KEY; geheim_maken SP_DOCLING_API_KEY
leg_images_vast "$IMG_LLAMA_SWAP" "$IMG_DOCLING" "$IMG_WHISPERX_A" "$IMG_SOCKET_PROXY" signaal-proef/kev:kev-1.0 \
  signaal-proef/llama-swap:v268-cuda13-b11977-dockercli

ZWAAR=(signaal-proef-docling-1 signaal-proef-whisperx-1 signaal-proef-kev-1)
bewaar create dc --profile p1 create --force-recreate docling whisperx kev
geheimen_volume_vullen
bewaar up dc --profile p1 up -d --no-deps socket-proxy llama-swap p1-client
client() { dc --profile p1 exec -T p1-client "$@"; }
# Voor metingen: een curl-fout (time-out, geen verbinding) is een uitkomst, geen scriptfout.
client_ok() { client "$@" || true; }
LS=http://llama-swap:8080
wacht_tot 120 client curl -sf "$LS/health" || omgeving_ongeschikt "llama-swap niet gezond"

# Alleen toestandswijzigingen: exec-events bevatten de opdrachtregel, met sleutel.
docker events --filter "label=com.docker.compose.project=$PROEF_PROJECT" --filter type=container \
  --filter event=start --filter event=stop --filter event=die --filter event=kill --filter event=destroy \
  --format '{{.Time}} {{.Action}} {{.Actor.Attributes.name}}' >"$RUNMAP/docker_events.txt" 2>&1 &
EVENTS_PID=$!
vram_start "$RUNMAP/vram.csv"
TOTAAL="$(vram_totaal)"; meet vram_totaal_mib "$TOTAAL"

kev_hdr() { echo "Authorization: Bearer $(client cat /run/secrets/kev_api_sleutel)"; }
doc_hdr() { echo "X-Api-Key: $(client cat /run/secrets/docling_api_sleutel)"; }

# --- 1a: docling-serve zonder model-veld ---
docling_taak() { # <pdf> <naam> -> status
  local pdf=$1 naam=$2 task status hdr
  hdr="$(doc_hdr)"
  task="$(client_ok curl -s -m 900 -H "$hdr" -F "files=@$pdf" -F to_formats=json -F ocr_lang=nl -F ocr_lang=en \
    "$LS/upstream/docling/v1/convert/file/async" | zoek -o '"task_id": *"[^"]*"' | cut -d'"' -f4)"
  [[ -z "$task" ]] && { echo "geen_task_id"; return 0; }
  for _ in $(seq 1 360); do
    status="$(client_ok curl -s -H "$hdr" "$LS/upstream/docling/v1/status/poll/$task" | zoek -o '"task_status": *"[^"]*"' | cut -d'"' -f4)"
    [[ "$status" == "success" || "$status" == "failure" ]] && break
    sleep 5
  done
  client_ok curl -s -H "$hdr" "$LS/upstream/docling/v1/result/$task" >"$RUNMAP/$naam.json"
  echo "$status"
}
D1="$(testdata D1 'https://ridderkerk.begrotingsapp.nl/Data/416de56b-57da-4d12-8484-3b5fc1ec689e/Public/jaarstukken-2025.pdf' pdf)"
D2="$(testdata D2 'https://wageningen.begrotingsapp.nl/Data/f9c877a9-64e0-4403-a24b-cc1825aa0bba/Public/jaarstukken-2025%20-%20paragraaf-1-weerstandsvermogen--risicobeheersing.pdf' pdf)"
dc --profile p1 cp "$(pad_host "$D1")" p1-client:/tmp/D1.pdf >/dev/null
dc --profile p1 cp "$(pad_host "$D2")" p1-client:/tmp/D2.pdf >/dev/null

k2_voor="docling=$(container_toestand signaal-proef-docling-1)"
( sleep 20; container_toestand signaal-proef-docling-1 >"$RUNMAP/docling_tijdens.txt" ) &
TIJDENS_PID=$!
s_d2="$(docling_taak /tmp/D2.pdf 1a_D2)"
wait "$TIJDENS_PID" 2>/dev/null || true
k2_tijdens="docling=$(cat "$RUNMAP/docling_tijdens.txt" 2>/dev/null || echo onbekend)"
s_d1="$(docling_taak /tmp/D1.pdf 1a_D1)"
tabellen="$(zoek -o '"tables": *\[ *{' "$RUNMAP/1a_D2.json" | wc -l | tr -d ' ')"
weerstand="$(grep -ci weerstandsvermogen "$RUNMAP/1a_D2.json" || true)"
meet 1a "D2=$s_d2 D1=$s_d1 tabellen_D2=$tabellen weerstandsvermogen_D2=$weerstand"
if [[ "$s_d2" == "success" && "$tabellen" -ge 1 && "$weerstand" -ge 1 ]]; then
  criterium K1 geslaagd "D2 success, tabellen en weerstandsvermogen aanwezig; D1 $s_d1"
else
  criterium K1 mislukt "D2=$s_d2 tabellen=$tabellen weerstandsvermogen=$weerstand; D1=$s_d1"
fi

# --- 1b: WhisperX met D4 ---
transcribeer() { # <bestand> <naam> [extra -F ...] -> http-status
  local b=$1 n=$2; shift 2
  client_ok curl -s -m 3600 -o "/tmp/$n.json" -w '%{http_code}' "$LS/v1/audio/transcriptions" \
    -F "file=@$b" -F model=whisperx -F response_format=verbose_json -F 'timestamp_granularities[]=word' "$@"
  client cat "/tmp/$n.json" >"$RUNMAP/$n.json" || true
}
k2_voor+=" whisperx=$(container_toestand signaal-proef-whisperx-1)"
( sleep 60; container_toestand signaal-proef-whisperx-1 >"$RUNMAP/whisperx_tijdens.txt" ) &
TIJDENS_PID=$!
h_1b="$(transcribeer /audio/D4-de-dag.mp3 1b_D4 -F language=nl -F diarize=true)"
wait "$TIJDENS_PID" 2>/dev/null || true
k2_tijdens+=" whisperx=$(cat "$RUNMAP/whisperx_tijdens.txt" 2>/dev/null || echo onbekend)"
sprekers="$(zoek -o '"speaker": *"[^"]*"' "$RUNMAP/1b_D4.json" | sort -u | wc -l | tr -d ' ')"
woorden="$(zoek -o '"words": *\[ *{' "$RUNMAP/1b_D4.json" | wc -l | tr -d ' ')"
segmenten="$(zoek -o '"segments": *\[ *{' "$RUNMAP/1b_D4.json" | wc -l | tr -d ' ')"
meet 1b "HTTP $h_1b, sprekers $sprekers, segmenten met woorden $woorden, segmentlijst $segmenten"
if [[ "$h_1b" == "200" && "$sprekers" -ge 2 && "$woorden" -ge 1 && "$segmenten" -ge 1 ]]; then
  criterium K3 geslaagd "HTTP 200, $sprekers sprekers, woordtijden aanwezig"
else
  criterium K3 mislukt "HTTP $h_1b, $sprekers sprekers, woorden $woorden, segmenten $segmenten"
fi

# --- 1h: Nederlands en Engels (aanvulling Geert 29-10), zonder language: detectie ---
for taal in nl en; do
  b=/audio/D7-nl.ogg; [[ $taal == en ]] && b=/audio/D8-en.ogg
  h="$(transcribeer "$b" "1h_$taal")"
  gedetecteerd="$(zoek -o '"language": *"[^"]*"' "$RUNMAP/1h_$taal.json" | head -n1 | cut -d'"' -f4)"
  tekst="$(zoek -o '"text": *"[^"]\{20,\}' "$RUNMAP/1h_$taal.json" | head -n1 | wc -l | tr -d ' ')"
  wt="$(zoek -o '"start": *[0-9.]*' "$RUNMAP/1h_$taal.json" | wc -l | tr -d ' ')"
  meet "1h_$taal" "HTTP $h, taal $gedetecteerd, tekst $tekst, tijden $wt"
  if [[ "$h" == "200" && "$tekst" == "1" && "$wt" -ge 1 && "${gedetecteerd,,}" =~ ^($taal|dutch|english)$ ]]; then
    criterium "K9_$taal" geslaagd "HTTP 200, niet-leeg transcript met woordtijden, taal $gedetecteerd"
  else
    criterium "K9_$taal" mislukt "HTTP $h, taal ${gedetecteerd:-onbekend}, tekst $tekst, tijden $wt"
  fi
done

# --- 1h: gebruiksmeldingen van variant A (OPT_OUT_TRACKING) ---
opt="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' signaal-proef-whisperx-1 | grep '^OPT_OUT_TRACKING=' || true)"
uitgaand="$(docker exec signaal-proef-whisperx-1 python -c '
import urllib.error, urllib.request
for url in ("https://example.org", "https://aws.amazon.com"):
    try:
        urllib.request.urlopen(url, timeout=5); print(url, "bereikbaar")
    except urllib.error.HTTPError:
        print(url, "bereikbaar (HTTP-fout)")
    except Exception as e:
        print(url, "geen verbinding:", type(e).__name__)
' 2>&1 | tr '\n' ' ')" || uitgaand="niet uit te voeren"
meldingen="$(docker logs signaal-proef-whisperx-1 2>&1 | grep -ciE 'track|telemetry|usage' || true)"
meet 1h_tracking "$opt; uitgaand: $uitgaand; logregels met track/telemetry/usage: $meldingen"
# Beide adressen moeten "geen verbinding" geven; een lege of mislukte controle telt niet.
n_geen="$(zoek -o 'geen verbinding' <<<"$uitgaand" | wc -l | tr -d ' ')"
if [[ "$opt" == "OPT_OUT_TRACKING=true" && "$n_geen" == "2" && "$uitgaand" != *bereikbaar* ]]; then
  criterium K10 geslaagd "OPT_OUT_TRACKING=true en geen uitgaande verbinding vanaf verwerking"
else
  criterium K10 mislukt "$opt; $uitgaand"
fi

# --- 1d: Kev zonder OpenAI-eindpunt ---
KEV_BODY='{"state":"De gemeente Proefstad legt nieuwe fietspaden aan en verhoogt daarvoor de begroting.","model":"kev-latest","questions":{"onderwerp":{"type":"choice","instructions":"Over welk beleidsterrein gaat dit?","criteria":{"verkeer":"Wegen, fietspaden en openbaar vervoer","zorg":"Zorg en welzijn","onderwijs":null}},"financieel":{"type":"noul","instructions":"Heeft dit financiele gevolgen?"}}}'
k2_voor+=" kev=$(container_toestand signaal-proef-kev-1)"
( sleep 30; container_toestand signaal-proef-kev-1 >"$RUNMAP/kev_tijdens.txt" ) &
TIJDENS_PID=$!
h_kev="$(client_ok curl -s -m 900 -o /tmp/1d.json -w '%{http_code}' -H "$(kev_hdr)" -H 'Content-Type: application/json' \
  -d "$KEV_BODY" "$LS/upstream/kev-4b/v1/systemone")"
client cat /tmp/1d.json >"$RUNMAP/1d_kev.json" || true
wait "$TIJDENS_PID" 2>/dev/null || true
k2_tijdens+=" kev=$(cat "$RUNMAP/kev_tijdens.txt" 2>/dev/null || echo onbekend)"
som="$(tr -d '\n ' <"$RUNMAP/1d_kev.json" | zoek -o '"probabilities":{[^}]*}' | head -n1 | zoek -o ':[0-9.eE-]*' | tr -d ':' | awk '{s+=$1} END {print s+0}')"
zonder_sleutel=""
for pad in / /docs /openapi.json /v1/models /health; do
  zonder_sleutel+="$pad=$(client_ok curl -s -o /dev/null -w '%{http_code}' -m 30 "http://kev:8009$pad") "
done
meet 1d "HTTP $h_kev, som kansen $som; zonder sleutel: $zonder_sleutel"
if [[ "$h_kev" == "200" ]] && awk -v s="$som" 'BEGIN {exit !(s >= 0.99 && s <= 1.01)}'; then
  criterium K4 geslaagd "HTTP 200, kansen tellen op tot $som"
else
  criterium K4 mislukt "HTTP $h_kev, som kansen $som"
fi
if [[ "$k2_voor" != *running* && "$k2_tijdens" != *exited* && "$k2_tijdens" == *running*running*running* ]]; then
  criterium K2 geslaagd "voor: $k2_voor; tijdens: $k2_tijdens"
else
  criterium K2 mislukt "voor: $k2_voor; tijdens: $k2_tijdens"
fi

# --- 1e: wisselen en wachtrij ---
T_1E="$(date +%s)"
werkmodel() {
  client_ok curl -s -m 900 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' "$LS/v1/chat/completions" \
    -d '{"model":"werkmodel","messages":[{"role":"user","content":"Noem drie voordelen van fietspaden."}],"max_tokens":64}'
}
reeks=""
for ronde in 1 2; do
  reeks+="docling=$(docling_taak /tmp/D1.pdf "1e_D1_$ronde") "
  reeks+="kev=$(client_ok curl -s -m 900 -o /dev/null -w '%{http_code}' -H "$(kev_hdr)" -H 'Content-Type: application/json' -d "$KEV_BODY" "$LS/upstream/kev-4b/v1/systemone") "
  reeks+="whisperx=$(transcribeer /audio/D7-nl.ogg "1e_wx_$ronde" -F language=nl) "
  reeks+="werkmodel=$(werkmodel) "
  # Twee gelijktijdige aanvragen voor verschillende zware diensten.
  ( client_ok curl -s -m 1800 -o /dev/null -w '%{http_code}' -H "$(kev_hdr)" -H 'Content-Type: application/json' -d "$KEV_BODY" "$LS/upstream/kev-4b/v1/systemone" >"$RUNMAP/gelijk_kev_$ronde.txt" ) &
  p1=$!
  ( werkmodel >"$RUNMAP/gelijk_werkmodel_$ronde.txt" ) &
  p2=$!
  wait "$p1" "$p2" || true
  reeks+="gelijktijdig=kev:$(cat "$RUNMAP/gelijk_kev_$ronde.txt"),werkmodel:$(cat "$RUNMAP/gelijk_werkmodel_$ronde.txt") "
done
T_1E_EIND="$(date +%s)"
meet 1e_reeks "$reeks"
# Overlap: tijdvakken waarin twee zware containers tegelijk draaien (uit docker events).
overlap="$(awk '
  $2=="start" {for (n in aan) if (aan[n] && n!=$3) o++; aan[$3]=1}
  $2=="die"   {aan[$3]=0}
  END {print o+0}' <(grep -E ' (start|die) signaal-proef-(docling|whisperx|kev)-1$' "$RUNMAP/docker_events.txt" || true))"
piek="$(vram_piek "$RUNMAP/vram.csv" "$T_1E" "$T_1E_EIND")"
oom="$(for c in "${ZWAAR[@]}" signaal-proef-llama-swap-1; do docker logs "$c" 2>&1; done | grep -ciE 'out of memory|CUDA error: out' || true)"
meet 1e "overlap $overlap, piek $piek MiB van $TOTAAL, geheugenfouten $oom"
if [[ "$overlap" == "0" && "$piek" -lt $((TOTAAL - 512)) && "$oom" == "0" ]]; then
  criterium K5 geslaagd "geen overlap, piek $piek MiB, geen geheugenfout"
else
  criterium K5 mislukt "overlap $overlap, piek $piek MiB (grens $((TOTAAL - 512))), geheugenfouten $oom"
fi

# --- 1f: /v1/rerank, en de embedding-dimensie (paragraaf 4) ---
h_rr="$(client_ok curl -s -m 300 -o /tmp/1f.json -w '%{http_code}' -H 'Content-Type: application/json' "$LS/v1/rerank" \
  -d '{"model":"rerank","query":"weerstandsvermogen gemeente","documents":["Het weerstandsvermogen is voldoende.","De fietspaden worden verbreed.","De begroting sluit met een overschot.","Risicobeheersing en weerstandsvermogen.","Afvalinzameling wordt anders."]}')"
client cat /tmp/1f.json >"$RUNMAP/1f_rerank.json" || true
n_rr="$(zoek -o '"relevance_score"' "$RUNMAP/1f_rerank.json" | wc -l | tr -d ' ')"
if [[ "$h_rr" == "200" && "$n_rr" == "5" ]]; then criterium K6 geslaagd "HTTP 200, 5 resultaten met score"
else criterium K6 mislukt "HTTP $h_rr, $n_rr resultaten"; fi
h_emb="$(client_ok curl -s -m 300 -o /tmp/emb.json -w '%{http_code}' -H 'Content-Type: application/json' "$LS/v1/embeddings" \
  -d '{"model":"embed","input":"weerstandsvermogen"}')"
client cat /tmp/emb.json >"$RUNMAP/embed.json" || true
dim="$(tr -d '\n ' <"$RUNMAP/embed.json" | zoek -o '"embedding":\[[^]]*\]' | head -n1 | tr ',' '\n' | wc -l | tr -d ' ')"
meet embedding_dimensie "HTTP $h_emb, $dim dimensies (verwacht 1024)"

# --- 1g: ontladen en herstel ---
sleep 75
na_ttl=""
for c in "${ZWAAR[@]}"; do na_ttl+="${c#signaal-proef-}=$(container_toestand "$c") "; done
client_ok curl -s -m 900 -o /dev/null -H "$(kev_hdr)" -H 'Content-Type: application/json' -d "$KEV_BODY" "$LS/upstream/kev-4b/v1/systemone" || true
bewaar externe_stop docker stop signaal-proef-kev-1
h_herstel="$(client_ok curl -s -m 900 -o /dev/null -w '%{http_code}' -H "$(kev_hdr)" -H 'Content-Type: application/json' -d "$KEV_BODY" "$LS/upstream/kev-4b/v1/systemone")"
# Aanvulling (P7): llama-swap merkt een externe stop pas na de gezondheidslus (5 s);
# alleen vastgelegd, K7 blijft de directe aanvraag.
sleep 20
h_herstel_20="$(client_ok curl -s -m 900 -o /dev/null -w '%{http_code}' -H "$(kev_hdr)" -H 'Content-Type: application/json' -d "$KEV_BODY" "$LS/upstream/kev-4b/v1/systemone")"
meet 1g "na ttl+15 s: $na_ttl; na externe stop: HTTP $h_herstel; 20 s later: HTTP $h_herstel_20"
if [[ "$na_ttl" != *running* && "$h_herstel" == "200" ]]; then criterium K7 geslaagd "$na_ttl; herstel HTTP 200"
else criterium K7 mislukt "$na_ttl; herstel HTTP $h_herstel (20 s later: $h_herstel_20)"; fi

# --- K8: geen internet vanaf verwerking (whisperx en kev) ---
k8=""
for c in signaal-proef-whisperx-1 signaal-proef-kev-1; do
  docker start "$c" >/dev/null 2>&1 || true
  k8+="${c#signaal-proef-}: $(docker exec "$c" python -c '
import urllib.request
try:
    urllib.request.urlopen("https://example.org", timeout=5); print("internet")
except urllib.error.HTTPError:
    print("internet")
except Exception as e:
    print("geen:" + type(e).__name__)' 2>&1 | tail -n1 || true) "
done
if [[ "$k8" != *internet* ]]; then criterium K8 geslaagd "$k8"; else criterium K8 mislukt "$k8"; fi

vram_stop
kill "$EVENTS_PID" 2>/dev/null || true
bewaar logs_llama_swap dc --profile p1 logs llama-swap || true
for c in "${ZWAAR[@]}"; do bewaar "logs_${c#signaal-proef-}" docker logs "$c" || true; done
criterium K_variant vastgelegd "WhisperX variant A; variant B alleen als A op een criterium van ADR 0014 afvalt (1c)"
proef_einde
