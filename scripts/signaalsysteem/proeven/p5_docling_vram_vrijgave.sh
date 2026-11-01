#!/usr/bin/env bash
# P5. docling-serve geeft het videogeheugen vrij (docs/signaalsysteem/proeven.md, P5).
# GPU exclusief: uit te voeren op de werkplek, na P1-voorbereiden (Docling-modellen
# in het volume signaal-proef_docling_artifacts).
source "$(dirname "$0")/lib/gemeenschappelijk.sh"
source "$(dirname "$0")/lib/vram_meter.sh"

proef_start p5 p5
p0_voorwaarde --gpu

D1="$(testdata D1 'https://ridderkerk.begrotingsapp.nl/Data/416de56b-57da-4d12-8484-3b5fc1ec689e/Public/jaarstukken-2025.pdf' pdf)"
D2="$(testdata D2 'https://wageningen.begrotingsapp.nl/Data/f9c877a9-64e0-4403-a24b-cc1825aa0bba/Public/jaarstukken-2025%20-%20paragraaf-1-weerstandsvermogen--risicobeheersing.pdf' pdf)"
leg_images_vast "$IMG_DOCLING" "$IMG_CURL"
geheim_maken SP_DOCLING_API_KEY

client() { dc --profile p5 exec -T p5-client "$@"; }
DOC=http://docling-p5:5001
sleutel_hdr() { echo "X-Api-Key: $(client cat /run/secrets/docling_api_sleutel)"; }

converteer() { # <pdf-in-client> <extra form-velden...> -> resultaat in $RUNMAP/<naam>.json
  local pdf=$1 naam=$2; shift 2
  local hdr task status
  hdr="$(sleutel_hdr)"
  task="$(client curl -s -H "$hdr" -F "files=@$pdf" -F to_formats=json -F 'ocr_lang=nl' -F 'ocr_lang=en' "$@" \
    "$DOC/v1/convert/file/async" | zoek -o '"task_id": *"[^"]*"' | cut -d'"' -f4)"
  [[ -z "$task" ]] && { echo "geen_task_id"; return 0; }
  for _ in $(seq 1 360); do
    status="$(client curl -s -H "$hdr" "$DOC/v1/status/poll/$task" | zoek -o '"task_status": *"[^"]*"' | cut -d'"' -f4)"
    [[ "$status" == "success" || "$status" == "failure" ]] && break
    sleep 5
  done
  client curl -s -H "$hdr" "$DOC/v1/result/$task" >"$RUNMAP/$naam.json"
  echo "$status"
}

ronde() { # <label>
  local l=$1 nul m1 m2 m3 m4 m5 m6 m7 t s1 s2
  bewaar "up_$l" dc --profile p5 up -d --force-recreate p5-client
  dc --profile p5 cp "$(pad_host "$D1")" p5-client:/tmp/D1.pdf >/dev/null
  dc --profile p5 cp "$(pad_host "$D2")" p5-client:/tmp/D2.pdf >/dev/null
  nul="$(vram_nulmeting 30)"
  vram_start "$RUNMAP/vram_$l.csv"
  bewaar "start_docling_$l" dc --profile p5 up -d --force-recreate docling-p5
  wacht_tot 600 client curl -sf "$DOC/health" || { vram_stop; criterium "health_$l" mislukt "docling-serve niet gezond"; return; }
  m1="$(vram_nu)"
  t="$(date +%s)"
  s1="$(converteer /tmp/D1.pdf "D1_$l")"
  m2="$(vram_piek "$RUNMAP/vram_$l.csv" "$t")"
  sleep 60; m3="$(vram_nu)"
  t="$(date +%s)"
  s2="$(converteer /tmp/D2.pdf "D2_$l" -F force_ocr=true)"
  m4="$(vram_piek "$RUNMAP/vram_$l.csv" "$t")"
  sleep 60; m5="$(vram_nu)"
  m6="geen eindpunt"
  if client curl -s "$DOC/openapi.json" | grep -o '"/v1/clear/[^"]*"' >"$RUNMAP/clear_$l.txt"; then
    while IFS= read -r pad; do
      pad="${pad//\"/}"
      client curl -s -X GET -H "$(sleutel_hdr)" "$DOC$pad" >/dev/null || true
    done <"$RUNMAP/clear_$l.txt"
    sleep 15; m6="$(vram_nu) (na $(tr '\n' ' ' <"$RUNMAP/clear_$l.txt"))"
  fi
  bewaar "stop_docling_$l" dc --profile p5 stop docling-p5
  sleep 15; m7="$(vram_nu)"
  vram_stop
  meet "$l" "nul=$nul M1=$m1 M2=$m2 M3=$m3 M4=$m4 M5=$m5 M6=$m6 M7=$m7 status_D1=$s1 status_D2=$s2"
  local weerstand
  weerstand="$(grep -ci 'weerstandsvermogen' "$RUNMAP/D1_$l.json" || true)"
  if [[ "$s1" == "success" && "$s2" == "success" && "$weerstand" -ge 1 ]]; then
    criterium "K1_$l" geslaagd "D1 en D2 success; D1 bevat weerstandsvermogen"
  else
    criterium "K1_$l" mislukt "D1=$s1 D2=$s2 weerstandsvermogen in D1: $weerstand"
  fi
  if [[ $((m7 - nul)) -le 256 ]]; then criterium "K2_$l" geslaagd "M7 - nul = $((m7 - nul)) MiB"
  else criterium "K2_$l" mislukt "M7 - nul = $((m7 - nul)) MiB"; fi
  criterium "K3_$l" vastgelegd "M3 - nul = $((m3 - nul)) MiB, M5 - nul = $((m5 - nul)) MiB, M6 = $m6"
}

export SP_DOCLING_LOAD_AT_BOOT=true SP_DOCLING_CACHE_SIZE=2
ronde standaard
export SP_DOCLING_LOAD_AT_BOOT=false SP_DOCLING_CACHE_SIZE=1
ronde zonder_laden_cache1

proef_einde
