#!/usr/bin/env bash
# Fetch AMI Mix-Headset audio for the meetings that have decision annotations.
# Plain GET only: this host answers HEAD and Range requests with 503, so
# --continue-at / -z resume tricks fail. Each file is re-fetched whole or not
# at all, and a short file is deleted so a rerun retries it.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/ami/audio"
BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
mkdir -p "$OUT"
MEETINGS=$(ls "$ROOT/data/ami/annotations/extracted/decision/manual" \
           | sed 's/\.decision\.xml//')
total=$(echo "$MEETINGS" | wc -l); i=0
for m in $MEETINGS; do
  i=$((i+1))
  f="$OUT/$m.Mix-Headset.wav"
  if [ -s "$f" ] && [ "$(stat -c%s "$f")" -gt 1000000 ]; then
    echo "[$i/$total] $m already present ($(stat -c%s "$f") bytes)"; continue
  fi
  echo "[$i/$total] fetching $m ..."
  if curl -sfL --retry 4 --retry-delay 5 --retry-all-errors --max-time 1800 -o "$f" "$BASE/$m/audio/$m.Mix-Headset.wav"; then
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -lt 1000000 ]; then
      echo "    SHORT ($sz bytes) - removing"; rm -f "$f"
    else
      echo "    ok $sz bytes"
    fi
  else
    echo "    FAILED $m"; rm -f "$f"
  fi
done
echo "=== AUDIO FETCH DONE: $(ls "$OUT" | wc -l) files, $(du -sh "$OUT" | cut -f1) ==="
