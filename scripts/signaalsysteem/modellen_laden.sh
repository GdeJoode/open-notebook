#!/usr/bin/env bash
# Beheerstap: de modellen voor llama-swap en docling-serve in de volumes van het
# signaalsysteem zetten en controleren (F2-11; proeven.md 1.7).
#
#   bash scripts/signaalsysteem/modellen_laden.sh [--van-proef] [--controle]
#
# Volumes (extern in docker-compose.signaalsysteem.yml; `stop.sh` en `down -v`
# raken ze niet):
#   signaal_modellen           GGUF-bestanden, alleen-lezen in llama-swap
#   signaal_docling_artifacts  Docling-modellen, alleen-lezen in docling-serve
#
# 1. GGUF: uit --van-proef (volume signaal-proef_modellen, alleen-lezen gekoppeld)
#    of downloaden van Hugging Face op de vaste revisie; altijd SHA-256-controle
#    (modellen_downloaden.py). Een bestaand bestand met een andere hash wordt niet
#    overschreven.
# 2. Docling: is het volume leeg, dan een kopie uit signaal-proef_docling_artifacts
#    (--van-proef, alleen-lezen) of `docling-tools models download` uit het vaste
#    docling-serve-image.
# 3. Controle zonder netwerk: GGUF-hashes en de hash over de Docling-boom.
# --controle doet alleen stap 3.
#
# Exitcodes: die van modellen_downloaden.py (0 in orde, 1 hash wijkt af,
# 2 ontbreekt of niet te halen, 3 scriptfout).
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT="$(cd "$(dirname "$0")/../.." && { pwd -W 2>/dev/null || pwd; })"
SCRIPT="$(cat "$ROOT/scripts/signaalsysteem/modellen_downloaden.py")"
PY_IMG="python:3.12.16-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337"
DOCLING_IMG="quay.io/docling-project/docling-serve-cu128:v1.38.2@sha256:f26a8dea114dfcfdebcd1617a9eb0a66f48351ff8b5140510d3bfc989c74c38d"
GGUF_VOL=signaal_modellen
DOCLING_VOL=signaal_docling_artifacts

van_proef=0; alleen_controle=0
for arg in "$@"; do
  case "$arg" in
    --van-proef) van_proef=1 ;;
    --controle) alleen_controle=1 ;;
    *) echo "onbekend argument: $arg" >&2; exit 3 ;;
  esac
done

if [[ $alleen_controle -eq 0 ]]; then
  docker volume create "$GGUF_VOL" >/dev/null
  docker volume create "$DOCLING_VOL" >/dev/null

  # 1. GGUF (met internet, alleen als een bestand ontbreekt en niet in de bron staat).
  bron=()
  if [[ $van_proef -eq 1 ]]; then bron=(-v signaal-proef_modellen:/bron:ro); fi
  docker run --rm "${bron[@]}" -v "$GGUF_VOL:/modellen" "$PY_IMG" \
    python -c "$SCRIPT" gguf /modellen ${bron:+--bron /bron}

  # 2. Docling-modellen.
  leeg="$(docker run --rm --network none -v "$DOCLING_VOL:/a:ro" "$PY_IMG" \
    sh -c 'find /a -type f | head -n1 | wc -l' | tr -d '\r')"
  if [[ "$leeg" == 0 && $van_proef -eq 1 ]]; then
    docker run --rm --network none -v signaal-proef_docling_artifacts:/bron:ro \
      -v "$DOCLING_VOL:/a" "$PY_IMG" cp -a /bron/. /a/
  elif [[ "$leeg" == 0 ]]; then
    # Het image draait als uid 1001; het nieuwe volume is van root (proeven.md §8 punt 26).
    docker run --rm --user 0:0 -v "$DOCLING_VOL:/a" --entrypoint docling-tools "$DOCLING_IMG" \
      models download --output-dir /a
  fi
fi

# 3. Controle zonder netwerk.
docker run --rm --network none -v "$GGUF_VOL:/modellen:ro" -v "$DOCLING_VOL:/artifacts:ro" \
  "$PY_IMG" python -c "$SCRIPT" controle /modellen /artifacts
