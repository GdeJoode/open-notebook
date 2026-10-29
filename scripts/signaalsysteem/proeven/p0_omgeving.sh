#!/usr/bin/env bash
# P0. Omgevingscontrole (docs/signaalsysteem/proeven.md, P0).
#
# Gebruik:
#   p0_omgeving.sh                       volledige controle (zelfstandig)
#   p0_omgeving.sh --gpu                 idem, en GPU-exclusiviteit is een voorwaarde
#   p0_omgeving.sh --als-voorwaarde      aangeroepen door een andere proef (geen GPU nodig)
#   p0_omgeving.sh --als-voorwaarde --gpu   aangeroepen door een GPU-proef
#   --voorbereiden                       haal het llama-swap-image op als het ontbreekt
#
# De GPU-controle in een container gebruikt het llama-swap-image als dat lokaal staat
# (of met --voorbereiden). Anders valt het script terug op python-slim met --gpus all,
# en daarna op nvidia-smi.exe op de host; de gebruikte bron staat in de uitkomst.
source "$(dirname "$0")/lib/gemeenschappelijk.sh"

GPU_MODUS=0; ALS_VOORWAARDE=0; VOORBEREIDEN=0
for arg in "$@"; do
  case "$arg" in
    --gpu) GPU_MODUS=1 ;;
    --als-voorwaarde) ALS_VOORWAARDE=1 ;;
    --voorbereiden) VOORBEREIDEN=1 ;;
    *) echo "onbekende optie: $arg" >&2; exit "$EXIT_SCRIPT" ;;
  esac
done

MIN_VRIJ_GB=60
proef_start p0 ""

# 1. Besturingssysteem en shell
os_naam="$(uname -s)"; meet os "$os_naam $(uname -r)"
if [[ "$os_naam" == MINGW* || "$os_naam" == MSYS* ]]; then shell_soort="Git Bash"
elif grep -qi microsoft /proc/version 2>/dev/null; then shell_soort="WSL2"
else shell_soort="anders"; fi
meet shell "$shell_soort"
if command -v cmd.exe >/dev/null 2>&1; then
  meet windows_versie "$(cmd.exe /c ver 2>/dev/null | tr -d '\r' | grep -v '^$' | head -n1)"
fi
if [[ "$shell_soort" == "anders" ]]; then
  criterium K_os mislukt "geen Git Bash of WSL2"
else
  criterium K_os geslaagd "$shell_soort"
fi

# 2. docker info
os_type="$(docker info --format '{{.OSType}}')"
meet docker_server "$(docker version --format '{{.Server.Version}}')"
meet docker_ostype "$os_type"
meet docker_os "$(docker info --format '{{.OperatingSystem}}')"
bewaar docker_system_df docker system df || true
if [[ "$os_type" == "linux" ]]; then criterium K_linux geslaagd "Linux-containers"
else criterium K_linux mislukt "OSType=$os_type"; fi

# Vrije ruimte: in de Docker-VM (schijf van Docker Desktop) en op de host-schijf
# waar die op staat. De kleinste telt.
vrij_vm_kb="$(drun --network none "$IMG_PY" df -Pk / 2>/dev/null | awk 'NR==2 {print $4}')" || vrij_vm_kb=""
docker_root="$(docker info --format '{{.DockerRootDir}}')"
meet docker_root "$docker_root"
vrij_vm_gb=$(( ${vrij_vm_kb:-0} / 1024 / 1024 ))
meet vrij_docker_vm_gb "$vrij_vm_gb"
vrij_host_gb=""
if [[ -n "${SIGNAAL_PROEF_DOCKER_SCHIJF:-}" ]]; then
  vrij_host_gb="$(df -Pk "$SIGNAAL_PROEF_DOCKER_SCHIJF" | awk 'NR==2 {print int($4/1024/1024)}')"
  meet vrij_host_schijf_gb "$vrij_host_gb ($SIGNAAL_PROEF_DOCKER_SCHIJF)"
fi
vrij_gb=$vrij_vm_gb
if [[ -n "$vrij_host_gb" && "$vrij_host_gb" -lt "$vrij_gb" ]]; then vrij_gb=$vrij_host_gb; fi
if [[ -z "$vrij_vm_kb" ]]; then
  criterium K_ruimte mislukt "vrije ruimte niet te meten"
elif [[ $vrij_gb -ge $MIN_VRIJ_GB ]]; then
  criterium K_ruimte geslaagd "$vrij_gb GB vrij (minimaal $MIN_VRIJ_GB)"
else
  criterium K_ruimte mislukt "$vrij_gb GB vrij, minder dan $MIN_VRIJ_GB"
fi

# 5. Momentopname van de beginstand (de functie proef_start heeft die al gemaakt).
for s in containers volumes networks images; do cp "$RUNMAP/voor.$s" "$RUNMAP/beginstand.$s"; done
meet aantal_containers "$(wc -l <"$RUNMAP/voor.containers" | tr -d ' ')"
meet aantal_volumes "$(wc -l <"$RUNMAP/voor.volumes" | tr -d ' ')"
meet aantal_netwerken "$(wc -l <"$RUNMAP/voor.networks" | tr -d ' ')"
meet aantal_images "$(wc -l <"$RUNMAP/voor.images" | tr -d ' ')"

if [[ $ALS_VOORWAARDE -eq 1 && $GPU_MODUS -eq 0 ]]; then
  # Een niet-GPU-proef heeft alleen Docker, Linux-containers en ruimte nodig.
  proef_einde
fi

# 3. GPU in een container
gpu_bron=""; gpu_regel=""; cuda_regel=""
if ! docker image inspect "$IMG_LLAMA_SWAP" >/dev/null 2>&1 && [[ $VOORBEREIDEN -eq 1 ]]; then
  bewaar pull_llama_swap docker pull "$IMG_LLAMA_SWAP" || true
fi
query=(nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader)
if docker image inspect "$IMG_LLAMA_SWAP" >/dev/null 2>&1; then
  gpu_regel="$(drun --gpus all --network none --entrypoint "" "$IMG_LLAMA_SWAP" "${query[@]}" 2>/dev/null | head -n1)" || gpu_regel=""
  cuda_regel="$(drun --gpus all --network none --entrypoint "" "$IMG_LLAMA_SWAP" nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: *[0-9.]+')" || cuda_regel=""
  [[ -n "$gpu_regel" ]] && gpu_bron="container $IMG_LLAMA_SWAP"
fi
if [[ -z "$gpu_regel" ]]; then
  gpu_regel="$(drun --gpus all --network none "$IMG_PY" "${query[@]}" 2>/dev/null | head -n1)" || gpu_regel=""
  cuda_regel="$(drun --gpus all --network none "$IMG_PY" nvidia-smi 2>/dev/null | grep -oE 'CUDA (UMD )?Version: *[0-9.]+')" || cuda_regel=""
  [[ -n "$gpu_regel" ]] && gpu_bron="container $IMG_PY (llama-swap-image niet lokaal)"
fi
if [[ -z "$gpu_regel" ]] && command -v nvidia-smi.exe >/dev/null 2>&1; then
  gpu_regel="$(nvidia-smi.exe "${query[@]:1}" 2>/dev/null | tr -d '\r' | head -n1)" || gpu_regel=""
  cuda_regel="$(nvidia-smi.exe 2>/dev/null | tr -d '\r' | grep -oE 'CUDA (UMD )?Version: *[0-9.]+')" || cuda_regel=""
  [[ -n "$gpu_regel" ]] && gpu_bron="host nvidia-smi.exe (geen GPU in een container gezien)"
fi
meet gpu_bron "${gpu_bron:-geen}"
meet gpu "$gpu_regel"
meet cuda "$cuda_regel"

IFS=',' read -r gpu_naam gpu_mem gpu_driver gpu_cc <<<"$gpu_regel"
gpu_naam="$(echo "$gpu_naam" | xargs)"; gpu_mem="$(echo "${gpu_mem:-0}" | tr -dc '0-9')"; gpu_cc="$(echo "${gpu_cc:-}" | xargs)"
cuda_versie="$(echo "$cuda_regel" | grep -o '[0-9][0-9.]*' || true)"

if [[ -z "$gpu_regel" ]]; then
  criterium K_gpu mislukt "geen GPU gevonden"
elif [[ "$gpu_naam" == *"RTX 5070 Ti"* && ${gpu_mem:-0} -ge 15900 && "$gpu_cc" == "12.0" ]]; then
  criterium K_gpu geslaagd "$gpu_naam, $gpu_mem MiB, compute_cap $gpu_cc, driver $(echo "$gpu_driver" | xargs)"
else
  criterium K_gpu mislukt "$gpu_naam, $gpu_mem MiB, compute_cap $gpu_cc"
fi
if [[ "$gpu_bron" != container* ]]; then
  criterium K_gpu_container mislukt "GPU niet zichtbaar in een container (bron: ${gpu_bron:-geen})"
else
  criterium K_gpu_container geslaagd "$gpu_bron"
fi
if [[ -n "$cuda_versie" ]] && awk -v v="$cuda_versie" 'BEGIN {split(v,a,"."); exit !(a[1] >= 13)}'; then
  criterium K_cuda geslaagd "CUDA Version $cuda_versie"
else
  criterium K_cuda mislukt "CUDA Version ${cuda_versie:-onbekend}, nodig >= 13.0"
fi

# 4. GPU-containers buiten het project en Ollama (alleen lezen)
gpu_containers=""
while IFS= read -r id; do
  [[ -z "$id" ]] && continue
  regel="$(docker inspect -f '{{.Name}}|{{index .Config.Labels "com.docker.compose.project"}}|{{json .HostConfig.DeviceRequests}}' "$id")"
  project="$(cut -d'|' -f2 <<<"$regel")"; dev="$(cut -d'|' -f3- <<<"$regel")"
  if [[ "$project" != "$PROEF_PROJECT" && "$dev" != "null" && "$dev" != "[]" ]]; then
    gpu_containers+="$(cut -d'|' -f1 <<<"$regel" | sed 's#^/##') "
  fi
done < <(docker ps -q)
meet gpu_containers_buiten_project "${gpu_containers:-geen}"

ollama_modellen="niet gemeten"
ollama_cmd=""
if command -v ollama >/dev/null 2>&1; then ollama_cmd=ollama
elif command -v ollama.exe >/dev/null 2>&1; then ollama_cmd=ollama.exe; fi
if [[ -n "$ollama_cmd" ]]; then
  if uit="$("$ollama_cmd" ps 2>/dev/null | tr -d '\r')"; then
    ollama_modellen="$(tail -n +2 <<<"$uit" | awk 'NF {print $1}' | xargs)"
    ollama_modellen="${ollama_modellen:-geen}"
  else
    ollama_modellen="geen (Ollama draait niet)"
  fi
fi
meet ollama_geladen "$ollama_modellen"

exclusief=1
[[ -n "$gpu_containers" ]] && exclusief=0
[[ "$ollama_modellen" != geen* ]] && exclusief=0
if [[ $exclusief -eq 1 ]]; then
  criterium K_gpu_exclusief geslaagd "geen GPU-container buiten het project; ollama ps leeg"
elif [[ $GPU_MODUS -eq 1 ]]; then
  criterium K_gpu_exclusief ongeldig "GPU niet vrij: containers [${gpu_containers:-geen}], ollama [$ollama_modellen]"
else
  criterium K_gpu_exclusief vastgelegd "GPU niet vrij voor GPU-proeven: containers [${gpu_containers:-geen}], ollama [$ollama_modellen]; alleen het niet-GPU-deel geldt als geslaagd"
fi

leg_images_vast "$IMG_LLAMA_SWAP" "$IMG_PY"
proef_einde
