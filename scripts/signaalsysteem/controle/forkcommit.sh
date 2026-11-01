#!/usr/bin/env bash
# De commit van de fork voor het image (F2-10, K2; herreview).
#
#   forkcommit.sh <repo>                 schrijft HEAD, of HEAD-dirty als de werkmap
#                                        niet schoon is
#   forkcommit.sh <repo> --label <label> exitcode 0 als het label van het image
#                                        gelijk is aan die waarde; anders een
#                                        waarschuwing op stderr en exitcode 1
#
# Schoon volgens `git status --porcelain` (genegeerde paden tellen niet), zonder
# de mappen die .dockerignore buiten het image houdt en die alleen werkbestanden
# bevatten: .claude, claudedocs en docs.
set -euo pipefail
export MSYS_NO_PATHCONV=1

repo=$1
head="$(git -C "$repo" rev-parse HEAD)"
if [[ -n "$(git -C "$repo" status --porcelain -- . ':(exclude).claude' ':(exclude)claudedocs' ':(exclude)docs')" ]]; then
  waarde="$head-dirty"
else
  waarde="$head"
fi

if [[ "${2:-}" == --label ]]; then
  label="${3:-}"
  if [[ "$label" == "$waarde" ]]; then exit 0; fi
  echo "let op: het image is gebouwd uit ${label:-onbekend}, de werkmap is $waarde;" \
       "bouw opnieuw met start.sh --build" >&2
  exit 1
fi
echo "$waarde"
