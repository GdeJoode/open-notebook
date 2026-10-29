#!/usr/bin/env bash
# P7. Docker-toegang voor llama-swap: socket-proxy en Docker-client
# (docs/signaalsysteem/proeven.md, P7). Geen GPU.
#
# Koppelt de Docker-socket alleen-lezen in de container socket-proxy. Alleen uitvoeren
# na een uitdrukkelijk akkoord van Geert: zet daarvoor SIGNAAL_PROEF_AKKOORD_SOCKET=ja.
source "$(dirname "$0")/lib/gemeenschappelijk.sh"

proef_start p7 p7
p0_voorwaarde
if [[ "${SIGNAAL_PROEF_AKKOORD_SOCKET:-}" != "ja" ]]; then
  omgeving_ongeschikt "geen akkoord voor de Docker-socket (SIGNAAL_PROEF_AKKOORD_SOCKET=ja ontbreekt)"
fi

for img in "$IMG_LLAMA_SWAP" "$IMG_SOCKET_PROXY" "$IMG_DOCKER_CLI" "$IMG_CURL" "$IMG_PY"; do
  docker image inspect "$img" >/dev/null 2>&1 || bewaar "pull_${img//[\/:]/_}" docker pull "$img"
done
leg_images_vast "$IMG_LLAMA_SWAP" "$IMG_SOCKET_PROXY" "$IMG_DOCKER_CLI" "$IMG_CURL" "$IMG_PY"

# --- 7a: Docker-client in het llama-swap-image ---
basis="$(drun --network none --entrypoint sh "$IMG_LLAMA_SWAP" -c \
  'for c in docker curl wget sh; do printf "%s=%s " "$c" "$(command -v $c || echo ontbreekt)"; done')"
meet 7a_basisimage "$basis"
bewaar build_llama_swap_p7 dc --profile p7 build llama-swap-p7
rc=0
versie="$(drun --network none --entrypoint docker signaal-proef/llama-swap-p7:v268-cuda13-b11977-dockercli \
  version --format '{{.Client.Version}}' 2>/dev/null)" || rc=$?
meet 7a_afgeleid_image "docker version exitcode $rc, client $versie"
if [[ $rc -eq 0 && -n "$versie" ]]; then
  criterium K1 geslaagd "werkende docker-client $versie; basisimage: $basis"
else
  criterium K1 mislukt "geen werkende docker-client in het afgeleide image (exitcode $rc)"
fi

# --- opzet 7b ---
bewaar create_dummies dc --profile p7 create dummy dummy2
bewaar up_clients dc --profile p7 up -d p7-curl p7-docker-client

status() { # <methode> <pad> [body]
  local m=$1 p=$2 b=${3:-}
  if [[ -n "$b" ]]; then
    dc --profile p7 exec -T p7-curl curl -s -o /dev/null -w '%{http_code}' -X "$m" \
      -H 'Content-Type: application/json' -d "$b" "http://socket-proxy:2375$p" || echo "fout"
  else
    dc --profile p7 exec -T p7-curl curl -s -o /dev/null -w '%{http_code}' -X "$m" "http://socket-proxy:2375$p" || echo "fout"
  fi
}

matrix() { # <configuratie>
  local c=$1 d=signaal-proef-dummy-1 d2=signaal-proef-dummy2-1 uit=""
  uit+="ping=$(status GET /_ping) "
  uit+="start=$(status POST /containers/$d/start) "
  uit+="stop=$(status POST /containers/$d/stop) "
  uit+="list=$(status GET /containers/json) "
  uit+="inspect=$(status GET /containers/$d/json) "
  uit+="create=$(status POST /containers/create '{"Image":"python:3.12.16-slim-bookworm","Cmd":["true"],"Labels":{"com.docker.compose.project":"signaal-proef"}}') "
  uit+="exec=$(status POST /containers/$d/exec '{"Cmd":["true"]}') "
  uit+="images=$(status GET /images/json) "
  uit+="volumes=$(status GET /volumes) "
  uit+="start_dummy2=$(status POST /containers/$d2/start) "
  status POST /containers/$d2/stop >/dev/null
  status POST /containers/$d/stop >/dev/null
  meet "7b_$c" "$uit"
  echo "$uit"
}

zet_proxy() { # <POST> <ALLOW_START> <ALLOW_STOP> <CONTAINERS>
  export SP_PROXY_POST=$1 SP_PROXY_ALLOW_START=$2 SP_PROXY_ALLOW_STOP=$3 SP_PROXY_CONTAINERS=$4
  bewaar "proxy_$1$2$3$4" dc --profile p7 up -d --force-recreate socket-proxy
  sleep 3
}

zet_proxy 0 1 1 0; uit_a="$(matrix A)"
zet_proxy 1 1 1 1; uit_c="$(matrix C)"
zet_proxy 1 1 1 0; uit_b="$(matrix B)"
# Ruim een container op die configuratie C eventueel heeft aangemaakt (alleen met projectlabel).
while read -r id naam; do
  [[ -n "$id" ]] && bewaar "rm_create_$id" docker rm -f "$id"
done < <(docker ps -a --filter "label=com.docker.compose.project=$PROEF_PROJECT" --filter status=created \
          --format '{{.ID}} {{.Names}}' | grep -v -e 'signaal-proef-dummy-1$' -e 'signaal-proef-dummy2-1$' || true)

waarde() { grep -o "$2=[0-9a-z]*" <<<"$1" | cut -d= -f2; }
k2=1
[[ "$(waarde "$uit_b" start)" =~ ^(204|304)$ && "$(waarde "$uit_b" stop)" =~ ^(204|304)$ ]] || k2=0
for k in create exec list inspect images volumes; do [[ "$(waarde "$uit_b" "$k")" == "403" ]] || k2=0; done
if [[ $k2 -eq 1 ]]; then criterium K2 geslaagd "configuratie B: $uit_b"
else criterium K2 mislukt "configuratie B: $uit_b"; fi
criterium K3 vastgelegd "A: $uit_a; C: $uit_c (verwacht: A start 403, C create toegestaan)"

# --- 7c: llama-swap volgt een container ---
bewaar up_llama_swap dc --profile p7 up -d llama-swap-p7
sleep 5
ls_status() { dc --profile p7 exec -T p7-curl curl -s -o /dev/null -w '%{http_code}' -m 120 "http://llama-swap-p7:8080$1" || echo fout; }
voor="$(container_toestand signaal-proef-dummy-1)"
eerste="$(ls_status /upstream/dummy/)"
tijdens="$(container_toestand signaal-proef-dummy-1)"
running="$(dc --profile p7 exec -T p7-curl curl -s -m 10 http://llama-swap-p7:8080/running || true)"
sleep 45
na_ttl="$(container_toestand signaal-proef-dummy-1)"
# Externe stop en een nieuwe aanvraag.
ls_status /upstream/dummy/ >/dev/null
bewaar externe_stop docker stop signaal-proef-dummy-1
tweede="$(ls_status /upstream/dummy/)"
meet 7c "voor=$voor eerste=$eerste tijdens=$tijdens na_ttl+15s=$na_ttl na_externe_stop=$tweede"
meet 7c_running "$(tr -d '\n' <<<"$running")"
bewaar logs_llama_swap dc --profile p7 logs llama-swap-p7 || true
if [[ "$eerste" == "200" && "$tijdens" == "running" && "$na_ttl" == "exited" && "$tweede" == "200" ]]; then
  criterium K4 geslaagd "eerste aanvraag 200 en running; na ttl+15 s exited; na externe stop weer 200"
else
  criterium K4 mislukt "eerste=$eerste tijdens=$tijdens na_ttl=$na_ttl na_externe_stop=$tweede"
fi

# K5 is de momentopname in proef_einde.
proef_einde
