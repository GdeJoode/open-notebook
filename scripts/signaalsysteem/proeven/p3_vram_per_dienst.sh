#!/usr/bin/env bash
# P3. Videogeheugen per dienst op de RTX 5070 Ti (docs/signaalsysteem/proeven.md, P3).
# GPU exclusief; gebruikt de voorbereiding en configuratie van P1 (profiel p1), maar
# start de diensten rechtstreeks, zonder het wisselgedrag van llama-swap.
# De llama-server-diensten draaien in een losse llama-swap-container met één model.
source "$(dirname "$0")/lib/gemeenschappelijk.sh"
source "$(dirname "$0")/lib/vram_meter.sh"

proef_start p3 p1
p0_voorwaarde --gpu
for v in modellen kev_hf whisperx_hf docling_artifacts p1_audio; do
  docker volume inspect "signaal-proef_$v" >/dev/null 2>&1 || omgeving_ongeschikt "volume signaal-proef_$v ontbreekt: eerst P1 --voorbereiden"
done
HERHALINGEN="${P3_HERHALINGEN:-3}"
geheim_maken SP_KEV_API_KEY; geheim_maken SP_DOCLING_API_KEY
TOTAAL="$(vram_totaal)"
CSV="$RUNMAP/p3_vram.csv"
echo "dienst,herhaling,nulmeting_mib,piek_mib,rust_mib,na_stop_mib,fout" >"$CSV"
bewaar up_client dc --profile p1 up -d --no-deps p1-client
client() { dc --profile p1 exec -T p1-client "$@"; }
D1="$(testdata D1 'https://ridderkerk.begrotingsapp.nl/Data/416de56b-57da-4d12-8484-3b5fc1ec689e/Public/jaarstukken-2025.pdf' pdf)"
D2="$(testdata D2 'https://wageningen.begrotingsapp.nl/Data/f9c877a9-64e0-4403-a24b-cc1825aa0bba/Public/jaarstukken-2025%20-%20paragraaf-1-weerstandsvermogen--risicobeheersing.pdf' pdf)"
dc --profile p1 cp "$(pad_host "$D1")" p1-client:/tmp/D1.pdf >/dev/null
dc --profile p1 cp "$(pad_host "$D2")" p1-client:/tmp/D2.pdf >/dev/null

# llama-server als losse container (zelfde image, model uit het volume).
LLAMA_PIDS=()
llama_start() { # <naam> <model> <extra args...>
  local naam=$1 model=$2; shift 2
  docker run -d --rm --label "com.docker.compose.project=$PROEF_PROJECT" --name "signaal-proef-p3-$naam" \
    --gpus all --network signaal-proef_verwerking -v signaal-proef_modellen:/models:ro \
    -e NVIDIA_DISABLE_REQUIRE=1 -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 \
    --entrypoint /app/llama-server "$IMG_LLAMA_SWAP" --host 0.0.0.0 --port 8080 -m "/models/$model" -ngl 99 "$@" >/dev/null
  wacht_tot 600 client curl -sf "http://signaal-proef-p3-$naam:8080/health"
}
llama_stop() { docker stop "signaal-proef-p3-$1" >/dev/null 2>&1 || true; }

docling_async() { # <pdf> [extra -F ...] -> takenstatus
  local pdf=$1 hdr task status=""; shift
  hdr="X-Api-Key: $(client cat /run/secrets/docling_api_sleutel)"
  task="$(client curl -s -m 300 -H "$hdr" -F "files=@$pdf" -F to_formats=json "$@"     http://docling:5001/v1/convert/file/async | zoek -o '"task_id": *"[^"]*"' | cut -d'"' -f4)"
  [[ -z "$task" ]] && { echo -n "geen_task_id"; return 0; }
  for _ in $(seq 1 360); do
    status="$(client curl -s -H "$hdr" "http://docling:5001/v1/status/poll/$task" | zoek -o '"task_status": *"[^"]*"' | cut -d'"' -f4)"
    [[ "$status" == "success" || "$status" == "failure" ]] && break
    sleep 5
  done
  echo -n "$status"
}

belasting() { # <dienst>
  case "$1" in
    embed-mxbai|embed-bge-m3)
      local tekst; tekst="$(printf 'weerstandsvermogen begroting fietspaden %.0s' $(seq 1 120))"
      for _ in $(seq 1 8); do
        local inv="" i; for i in $(seq 1 32); do inv+="\"$tekst $i\","; done
        client curl -s -o /dev/null -w '%{http_code} ' -H 'Content-Type: application/json' \
          "http://signaal-proef-p3-$1:8080/v1/embeddings" -d "{\"input\":[${inv%,}]}"
      done ;;
    werkmodel-8k|werkmodel-32k)
      local prompt; prompt="$(printf 'De gemeente Proefstad bespreekt de begroting en het weerstandsvermogen. %.0s' $(seq 1 150))"
      client curl -s -o /dev/null -w '%{http_code}' -m 900 -H 'Content-Type: application/json' \
        "http://signaal-proef-p3-$1:8080/v1/chat/completions" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$prompt Vat samen.\"}],\"max_tokens\":500}" ;;
    kev-4b)
      local i; for i in $(seq 1 50); do
        client curl -s -o /dev/null -w '%{http_code} ' -m 600 -H "Authorization: Bearer $(client cat /run/secrets/kev_api_sleutel)" \
          -H 'Content-Type: application/json' http://kev:8009/v1/systemone \
          -d "{\"state\":\"Proefstad vraag $i over fietspaden.\",\"questions\":{\"o\":{\"type\":\"noul\",\"instructions\":\"Gaat dit over verkeer?\"}}}"
      done ;;
    docling)
      # Asynchroon met pollen: het synchrone eindpunt kapt af na DOCLING_SERVE_MAX_SYNC_WAIT.
      docling_async /tmp/D1.pdf; echo -n ' '
      docling_async /tmp/D2.pdf -F force_ocr=true ;;
    whisperx)
      client curl -s -o /dev/null -w '%{http_code}' -m 3600 http://whisperx:8000/v1/audio/transcriptions \
        -F file=@/audio/D4-de-dag.mp3 -F language=nl -F diarize=true ;;
  esac
}

dienst_start() {
  case "$1" in
    embed-mxbai) llama_start "$1" embed-mxbai-q8_0.gguf --embeddings -c 512 -b 512 -ub 512 ;;
    embed-bge-m3) llama_start "$1" embed-bge-m3-q8_0.gguf --embeddings -c 8192 -b 8192 -ub 8192 ;;
    werkmodel-8k) llama_start "$1" werkmodel-qwen3.5-9b-q4_k_m.gguf -c 8192 ;;
    werkmodel-32k) llama_start "$1" werkmodel-qwen3.5-9b-q4_k_m.gguf -c 32768 ;;
    kev-4b) docker start signaal-proef-kev-1 >/dev/null; wacht_tot 900 client curl -sf -H "Authorization: Bearer $(client cat /run/secrets/kev_api_sleutel)" http://kev:8009/v1/models ;;
    docling) docker start signaal-proef-docling-1 >/dev/null; wacht_tot 600 client curl -sf http://docling:5001/health ;;
    whisperx) docker start signaal-proef-whisperx-1 >/dev/null; wacht_tot 900 client curl -sf http://whisperx:8000/ping ;;
  esac
}
dienst_stop() {
  case "$1" in
    kev-4b) docker stop signaal-proef-kev-1 >/dev/null ;;
    docling) docker stop signaal-proef-docling-1 >/dev/null ;;
    whisperx) docker stop signaal-proef-whisperx-1 >/dev/null ;;
    *) llama_stop "$1" ;;
  esac
}

bewaar create dc --profile p1 create docling whisperx kev
geheimen_volume_vullen
vram_start "$RUNMAP/vram_ruw.csv"
DIENSTEN=(embed-mxbai embed-bge-m3 werkmodel-8k werkmodel-32k kev-4b docling whisperx)
meet_dienst() { # <label> <diensten...>
  local label=$1; shift
  local h nul t piek rust na fout d
  for h in $(seq 1 "$HERHALINGEN"); do
    nul="$(vram_nulmeting 30)"
    t="$(date +%s)"; fout=""
    for d in "$@"; do dienst_start "$d" || fout+="start $d mislukt; "; done
    for d in "$@"; do belasting "$d" >>"$RUNMAP/belasting_$label.txt" || fout+="belasting $d; "; done
    piek="$(vram_piek "$RUNMAP/vram_ruw.csv" "$t")"
    sleep 30; rust="$(vram_nu)"
    for d in "$@"; do dienst_stop "$d"; done
    sleep 15; na="$(vram_nu)"
    for d in "$@"; do
      case "$d" in kev-4b|docling|whisperx) docker logs "signaal-proef-${d%-4b}-1" 2>&1 | grep -qiE 'out of memory' && fout+="OOM $d; " ;; esac
    done
    echo "$label,$h,$nul,$piek,$rust,$na,${fout%; }" >>"$CSV"
  done
}
for d in "${DIENSTEN[@]}"; do meet_dienst "$d" "$d"; done
for d in werkmodel-8k kev-4b docling whisperx; do meet_dienst "embed-mxbai+$d" embed-mxbai "$d"; done
meet_dienst "docling+whisperx" docling whisperx
vram_stop

# K1: drie herhalingen per dienst, spreiding piek <= 10 %.
k1=1; k1_t=""
for d in "${DIENSTEN[@]}"; do
  read -r n mn mx <<<"$(awk -F, -v d="$d" '$1==d {n++; if (mn==""||$4<mn) mn=$4; if ($4>mx) mx=$4} END {print n+0, mn+0, mx+0}' "$CSV")"
  spreiding="$(awk -v a="$mn" -v b="$mx" 'BEGIN {print (b>0 ? (b-a)/b*100 : 100)}')"
  k1_t+="$d: n=$n piek $mn-$mx (${spreiding}%); "
  [[ "$n" -ge 3 ]] && awk -v s="$spreiding" 'BEGIN {exit !(s <= 10)}' || k1=0
done
[[ $k1 -eq 1 ]] && criterium K1 geslaagd "$k1_t" || criterium K1 mislukt "$k1_t"
# K2: elke combinatie gemeten.
n_comb="$(grep -c '+' "$CSV" || true)"
[[ "$n_comb" -ge $((5 * HERHALINGEN)) ]] && criterium K2 geslaagd "$n_comb combinatiemetingen; fouten: $(awk -F, '$1 ~ /\+/ && $7!="" {print $1": "$7}' "$CSV" | tr '\n' ' ')" \
  || criterium K2 mislukt "$n_comb combinatiemetingen"
# K3: beslisregel nulmeting + som van de pieken <= totaal - 1024 MiB.
voorstel=""
for comb in werkmodel-8k kev-4b docling whisperx; do
  piek_e="$(awk -F, '$1=="embed-mxbai" {if ($4-$3>m) m=$4-$3} END {print m+0}' "$CSV")"
  piek_d="$(awk -F, -v d="$comb" '$1==d {if ($4-$3>m) m=$4-$3} END {print m+0}' "$CSV")"
  nul="$(awk -F, -v d="$comb" '$1==d {print $3; exit}' "$CSV")"
  som=$((nul + piek_e + piek_d))
  [[ $som -le $((TOTAAL - 1024)) ]] && voorstel+="embed+$comb samen ($som MiB); " || voorstel+="embed+$comb niet samen ($som MiB); "
done
pd="$(awk -F, '$1=="docling" {if ($4-$3>m) m=$4-$3} END {print m+0}' "$CSV")"
pw="$(awk -F, '$1=="whisperx" {if ($4-$3>m) m=$4-$3} END {print m+0}' "$CSV")"
nul="$(awk -F, '$1=="docling" {print $3; exit}' "$CSV")"
[[ $((nul + pd + pw)) -le $((TOTAAL - 1024)) ]] && voorstel+="docling+whisperx samen" || voorstel+="docling+whisperx niet samen"
criterium K3 vastgelegd "$voorstel"
meet p3_vram_tabel "$(tr '\n' ';' <"$CSV")"
cp "$CSV" "$RUNMAP/p3_vram_tabel.csv"
proef_einde
