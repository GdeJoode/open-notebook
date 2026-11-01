#!/usr/bin/env bash
# P8. Netwerkafscherming met internal: true (docs/signaalsysteem/proeven.md, P8).
source "$(dirname "$0")/lib/gemeenschappelijk.sh"

proef_start p8 p8
p0_voorwaarde

leg_images_vast "$IMG_PY"
docker image inspect "$IMG_PY" >/dev/null 2>&1 || bewaar pull_python docker pull "$IMG_PY"

# Stap 4: waarschuwingen van compose up vastleggen.
up_rc=0
bewaar compose_up dc --profile p8 up -d --wait --wait-timeout 60 || up_rc=$?
meet compose_up_exitcode "$up_rc"
for d in alleen_intern dubbel controle; do
  meet "toestand_$d" "$(container_toestand "signaal-proef-$d-1")"
done
meet compose_up_meldingen "$(grep -iE 'warn|error' "$RUNMAP/compose_up.log" | tr '\n' ' ' || true)"

# Stap 1: vanaf de host.
host_curl() { # <poort> -> "exitcode http_code"
  local rc=0 code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$1/")" || rc=$?
  echo "$rc $code"
}
read -r k1_rc k1_code <<<"$(host_curl 18081)"
read -r k3a_rc k3a_code <<<"$(host_curl 18082)"
read -r k3b_rc k3b_code <<<"$(host_curl 18083)"
meet host_naar_alleen_intern "curl-exitcode $k1_rc, http $k1_code"
meet host_naar_dubbel "curl-exitcode $k3a_rc, http $k3a_code"
meet host_naar_controle "curl-exitcode $k3b_rc, http $k3b_code"

# Ook vanuit WSL2 als dat beschikbaar is (alleen vastleggen).
if command -v wsl.exe >/dev/null 2>&1; then
  for p in 18081 18082 18083; do
    uit="$(wsl.exe -e sh -c "curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$p/; echo \" rc=\$?\"" 2>/dev/null | tr -d '\r\0')" || uit="wsl niet uitvoerbaar"
    meet "wsl_naar_$p" "$uit"
  done
fi

case "$k1_rc" in
  7|28|52) criterium K1 geslaagd "host naar alleen_intern: geen HTTP-antwoord (curl-exitcode $k1_rc)" ;;
  *) criterium K1 mislukt "host naar alleen_intern: curl-exitcode $k1_rc, http $k1_code" ;;
esac
if [[ "$k3a_code" == "200" && "$k3b_code" == "200" ]]; then
  criterium K3 geslaagd "host naar dubbel en controle: HTTP 200"
else
  criterium K3 mislukt "host naar dubbel: $k3a_code, naar controle: $k3b_code"
fi

# Stap 2: vanuit elke container internet en DNS.
PY_NET='
import socket, sys, urllib.error, urllib.request
uit = []
try:
    socket.getaddrinfo("example.org", 443)
    uit.append("dns=ok")
except Exception as e:
    uit.append("dns=fout:" + type(e).__name__)
try:
    r = urllib.request.urlopen("https://example.org", timeout=5)
    uit.append("https=%d" % r.status)
    rc = 0
except urllib.error.HTTPError as e:
    uit.append("https=%d (HTTP-fout, wel verbinding)" % e.code)
    rc = 0
except Exception as e:
    uit.append("https=fout:" + type(e).__name__)
    rc = 1
print(" ".join(uit))
sys.exit(rc)
'
net_test() { # <dienst> -> "rc uitvoer"
  local rc=0 uit
  uit="$(dc --profile p8 exec -T "$1" python -c "$PY_NET" 2>&1 | tail -n1)" || rc=$?
  echo "$rc $uit"
}
read -r k2_rc k2_uit <<<"$(net_test alleen_intern)"
read -r k5_rc k5_uit <<<"$(net_test dubbel)"
read -r c_rc c_uit <<<"$(net_test controle)"
meet alleen_intern_naar_internet "rc $k2_rc: $k2_uit"
meet dubbel_naar_internet "rc $k5_rc: $k5_uit"
meet controle_naar_internet "rc $c_rc: $c_uit"
if [[ "$k2_rc" != "0" ]]; then
  criterium K2 geslaagd "alleen_intern naar internet mislukt ($k2_uit)"
else
  criterium K2 mislukt "alleen_intern heeft internet ($k2_uit)"
fi
criterium K5 vastgelegd "dubbel naar internet: rc $k5_rc ($k5_uit); controle: rc $c_rc ($c_uit)"

# Stap 3: alleen_intern naar dubbel.
k4_code="$(dc --profile p8 exec -T alleen_intern python -c '
import urllib.request
try:
    print(urllib.request.urlopen("http://dubbel:8000/", timeout=5).status)
except Exception as e:
    print("fout:" + type(e).__name__)
' 2>&1 | tail -n1)" || true
meet alleen_intern_naar_dubbel "$k4_code"
if [[ "$k4_code" == "200" ]]; then
  criterium K4 geslaagd "alleen_intern naar dubbel: HTTP 200"
else
  criterium K4 mislukt "alleen_intern naar dubbel: $k4_code"
fi

proef_einde
