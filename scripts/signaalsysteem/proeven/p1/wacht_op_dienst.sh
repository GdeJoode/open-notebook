#!/bin/sh
# cmd voor llama-swap (docs/signaalsysteem/proeven.md, P1 stap 2 en P7 stap 7c).
#
#   wacht_op_dienst.sh <container> <gezondheids-URL> [sleutelbestand] [max_seconden]
#
# Start de container via de socket-proxy (docker start), wacht tot de URL HTTP 200
# geeft (hooguit max_seconden, standaard 600) en blijft daarna lopen zolang dat zo
# blijft. Nodig omdat "docker start -a" en "docker wait" paden onder /containers
# gebruiken die de proxy dichthoudt (P7). Bij SIGTERM van llama-swap stopt het
# script; het stoppen van de container zelf doet cmdStop.
naam=$1
url=$2
sleutel=${3:-}
max=${4:-600}

gezond() {
  if [ -n "$sleutel" ] && [ "$sleutel" != "-" ]; then
    h="Authorization: Bearer $(cat "$sleutel")"
    if command -v curl >/dev/null 2>&1; then curl -fsS -o /dev/null -m 5 -H "$h" "$url"
    else wget -q -O /dev/null -T 5 --header "$h" "$url"; fi
  else
    if command -v curl >/dev/null 2>&1; then curl -fsS -o /dev/null -m 5 "$url"
    else wget -q -O /dev/null -T 5 "$url"; fi
  fi
}

docker start "$naam" >/dev/null || exit 1
t=0
until gezond; do
  sleep 1
  t=$((t + 1))
  [ "$t" -ge "$max" ] && { echo "wacht_op_dienst: $naam niet gezond na $max s" >&2; exit 1; }
done
echo "wacht_op_dienst: $naam gezond na $t s" >&2
trap 'exit 0' TERM INT
while gezond; do
  sleep 5 &
  wait $!
done
echo "wacht_op_dienst: $naam niet meer gezond" >&2
exit 1
