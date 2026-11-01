#!/usr/bin/env bash
# Meting van het videogeheugen (proeven.md 1.5).
# Meet het totaal van de kaart op de host met nvidia-smi(.exe); valt dat weg, dan in
# een container met --gpus all. Geheugen per proces is onder WSL2 geen betrouwbare maat.
#
#   source lib/vram_meter.sh
#   vram_start "$RUNMAP/vram.csv"   # start de meting op de achtergrond (elke 500 ms)
#   vram_markeer "start docling"     # schrijft een markering in markeringen.txt
#   vram_stop
#   vram_piek  "$RUNMAP/vram.csv" [vanaf_epoch] [tot_epoch]   # piek memory.used (MiB)
#   vram_nu                          # één meting van memory.used (MiB)

VRAM_PID=""
VRAM_CSV=""

vram_bron() {
  if command -v nvidia-smi.exe >/dev/null 2>&1; then echo "nvidia-smi.exe"
  elif command -v nvidia-smi >/dev/null 2>&1; then echo "nvidia-smi"
  else echo "container"
  fi
}

_vram_query() {
  local bron
  bron="$(vram_bron)"
  if [[ "$bron" == "container" ]]; then
    drun --gpus all "${IMG_LLAMA_SWAP}" nvidia-smi "$@"
  else
    "$bron" "$@"
  fi
}

vram_start() {
  VRAM_CSV=$1
  echo "epoch,timestamp,memory_used_mib,memory_total_mib,utilization_gpu" >"$VRAM_CSV"
  (
    _vram_query --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits -lms 500 2>/dev/null \
      | while IFS= read -r regel; do
          printf '%s,%s\n' "$(date +%s)" "$(printf '%s' "$regel" | tr -d ' \r')"
        done >>"$VRAM_CSV"
  ) &
  VRAM_PID=$!
}

vram_markeer() {
  printf '%s,%s\n' "$(date +%s)" "$1" >>"$(dirname "$VRAM_CSV")/markeringen.txt"
}

vram_stop() {
  if [[ -n "$VRAM_PID" ]]; then
    # Git Bash kent geen pkill: de hele boom stoppen, ook nvidia-smi.exe.
    kill_boom "$VRAM_PID"
    wait "$VRAM_PID" 2>/dev/null || true
    VRAM_PID=""
  fi
}

vram_nu() {
  _vram_query --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' \r'
}

vram_totaal() {
  _vram_query --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' \r'
}

vram_piek() {
  local csv=$1 van=${2:-0} tot=${3:-99999999999}
  awk -F, -v van="$van" -v tot="$tot" 'NR>1 && $1>=van && $1<=tot && $3+0>m {m=$3+0} END {print m+0}' "$csv"
}

# Nulmeting: gemiddelde over <seconden> seconden.
vram_nulmeting() {
  local sec=${1:-30} som=0 n=0 w
  while [[ $n -lt $sec ]]; do
    w="$(vram_nu)"; som=$((som + w)); n=$((n + 1)); sleep 1
  done
  echo $((som / n))
}
