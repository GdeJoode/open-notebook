#!/usr/bin/env bash
# P6. Feedcontrole voor een YouTube-kanaal en een Spotify-podcast
# (docs/signaalsysteem/proeven.md, P6). Netwerk "uitgaand" (internet).
source "$(dirname "$0")/lib/gemeenschappelijk.sh"

proef_start p6 p6
p0_voorwaarde

bewaar build dc --profile p6 build feedcontrole
leg_images_vast "$IMG_PY" signaal-proef/p6-feedcontrole:local
meet sleutels "podcastindex=$([[ -n "${PODCASTINDEX_KEY:-}" ]] && echo gezet || echo 'niet gezet, 6c-deel overgeslagen'); youtube=$([[ -n "${YOUTUBE_API_KEY:-}" ]] && echo gezet || echo 'niet gezet, 6d overgeslagen')"

rc=0
dc --profile p6 run --rm --no-deps feedcontrole >"$RUNMAP/p6_ruw.json" 2>"$RUNMAP/p6_stderr.ruw" || rc=$?
masker <"$RUNMAP/p6_stderr.ruw" >"$RUNMAP/p6_stderr.log"; rm -f "$RUNMAP/p6_stderr.ruw"
meet feedcontrole_exitcode "$rc"
meet_json feedcontrole "$RUNMAP/p6_ruw.json"

# Criteria uit de uitvoer van feedcontrole.py overnemen.
if [[ -s "$RUNMAP/p6_ruw.json" ]]; then
  for k in K1 K2 K3 K4 K5; do
    stand="$(grep -A1 "\"$k\": {" "$RUNMAP/p6_ruw.json" | grep -o '"stand": "[a-z_]*"' | cut -d'"' -f4 || true)"
    toel="$(grep -A2 "\"$k\": {" "$RUNMAP/p6_ruw.json" | grep -o '"toelichting": ".*"' | sed 's/^"toelichting": "//; s/"$//' || true)"
    criterium "$k" "${stand:-mislukt}" "${toel:-geen uitvoer}"
  done
else
  criterium feedcontrole mislukt "geen uitvoer (exitcode $rc), zie p6_stderr.log"
fi

proef_einde
