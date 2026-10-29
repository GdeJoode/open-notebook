#!/usr/bin/env bash
# P9. Opruimen van het project signaal-proef (docs/signaalsysteem/proeven.md, P9).
#
#   p9_opruimen.sh              containers en netwerken van het project
#   p9_opruimen.sh --volumes    ook de volumes met label com.docker.compose.project=signaal-proef
#   p9_opruimen.sh --images     ook de lokaal gebouwde images signaal-proef/*
source "$(dirname "$0")/lib/gemeenschappelijk.sh"

MET_VOLUMES=0; MET_IMAGES=0
for arg in "$@"; do
  case "$arg" in
    --volumes) MET_VOLUMES=1 ;;
    --images) MET_IMAGES=1 ;;
    *) echo "onbekende optie: $arg" >&2; exit "$EXIT_SCRIPT" ;;
  esac
done

proef_start p9 ""

# Alle profielen, zodat elke dienst van het project wordt meegenomen.
profielen=()
for p in p1 p1-voorbereiden p2 p4 p5 p6 p7 p8; do profielen+=(--profile "$p"); done
if [[ $MET_VOLUMES -eq 1 ]]; then
  bewaar down dc "${profielen[@]}" down --timeout 10 --remove-orphans --volumes
else
  bewaar down dc "${profielen[@]}" down --timeout 10 --remove-orphans
fi

# Losse containers met het projectlabel (drun gebruikt --rm, maar na een afbreking
# kan er een blijven staan).
while IFS= read -r id; do
  [[ -n "$id" ]] && bewaar "rm_$id" docker rm -f "$id"
done < <(docker ps -aq --filter "label=com.docker.compose.project=$PROEF_PROJECT")

while IFS= read -r net; do
  [[ -n "$net" ]] && bewaar "netrm_$net" docker network rm "$net"
done < <(docker network ls -q --filter "label=com.docker.compose.project=$PROEF_PROJECT")

if [[ $MET_VOLUMES -eq 1 ]]; then
  while IFS= read -r vol; do
    [[ -n "$vol" ]] && bewaar "volrm_$vol" docker volume rm "$vol"
  done < <(docker volume ls -q --filter "label=com.docker.compose.project=$PROEF_PROJECT")
fi

if [[ $MET_IMAGES -eq 1 ]]; then
  while IFS= read -r img; do
    [[ -n "$img" ]] && bewaar "imgrm_${img//[\/:]/_}" docker image rm "$img"
  done < <(docker image ls --format '{{.Repository}}:{{.Tag}}' | grep '^signaal-proef/' || true)
fi

rest_c="$(docker ps -aq --filter "label=com.docker.compose.project=$PROEF_PROJECT" | wc -l | tr -d ' ')"
rest_n="$(docker network ls -q --filter "label=com.docker.compose.project=$PROEF_PROJECT" | wc -l | tr -d ' ')"
rest_v="$(docker volume ls -q --filter "label=com.docker.compose.project=$PROEF_PROJECT" | wc -l | tr -d ' ')"
meet resterende_containers "$rest_c"
meet resterende_netwerken "$rest_n"
meet resterende_volumes "$rest_v"
if [[ "$rest_c" == "0" && "$rest_n" == "0" ]]; then
  criterium K_opgeruimd geslaagd "geen containers en netwerken van het project meer; volumes: $rest_v"
else
  criterium K_opgeruimd mislukt "nog $rest_c containers en $rest_n netwerken van het project"
fi

proef_einde
