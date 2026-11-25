#!/usr/bin/env bash
# Pass 2 of the contention measurement. Closes two gaps in contend.sh:
#   1. Kev solo baselines at t=4 and t=20, so the 4+4 and 20+20 contention
#      figures are not conflated with having fewer/more threads.
#   2. A much longer measurement window, so several whisper runs complete and
#      the whisper-side figure has a real distribution instead of n<=1.
# Also measures whisper solo at t=20, absent from the first baseline table.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LS="$ROOT/vendor/llama.cpp/build/bin/llama-server"
WC="$ROOT/vendor/whisper.cpp/build/bin/whisper-cli"
AUDIO="$ROOT/audio/long.wav"; AUDIO_SEC=330
PORT=9933
cd "$ROOT"; mkdir -p results
exec > >(tee -a results/contend2.log) 2>&1

LOCK="results/.contend2.lock"
mkdir "$LOCK" 2>/dev/null || { echo "FATAL: $LOCK exists"; exit 1; }
KPID=""; WPID=""
cleanup() { [ -n "$WPID" ] && kill "$WPID" 2>/dev/null
            [ -s results/.wpid2 ] && kill "$(cat results/.wpid2)" 2>/dev/null
            [ -n "$KPID" ] && kill "$KPID" 2>/dev/null
            rm -f results/.wpid2; rmdir "$LOCK" 2>/dev/null; }
trap cleanup EXIT
curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && {
  echo "FATAL: port $PORT already serving"; exit 1; }

echo "### contention pass 2   started: $(date -Is)"

whisper_loop() {
  while :; do
    local t0 t1 cpid
    t0=$(date +%s.%N)
    "$WC" -m "$1" -t "$2" -f "$AUDIO" -np -nt >/dev/null 2>&1 &
    cpid=$!; echo "$cpid" > results/.wpid2
    wait "$cpid" || break
    t1=$(date +%s.%N)
    python3 -c "print(round($t1-$t0,2))" >> "$3"
  done
}
start_kev() {
  "$LS" -m "$1" --port $PORT -c 8192 -ub 4096 -b 4096 -np 1 -t "$2" \
      --no-warmup > "results/c2server_$3.log" 2>&1 &
  KPID=$!
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$KPID" 2>/dev/null || return 1
    sleep 1
  done; return 1
}
stop_kev() { [ -n "$KPID" ] && { kill "$KPID" 2>/dev/null; wait "$KPID" 2>/dev/null; }
             KPID=""
             for _ in $(seq 1 60); do
               curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break
               sleep 1
             done; sleep 2; }
wstats() { python3 - "$1" "$AUDIO_SEC" <<'PY'
import sys,statistics
ts=[float(x) for x in open(sys.argv[1]) if x.strip()]
sec=float(sys.argv[2])
if not ts: print("    whisper n=0"); raise SystemExit
m=statistics.median(ts)
print(f"    whisper wall p50 {m:.2f}s  rtf {m/sec:.4f}  n={len(ts)}  "
      f"min {min(ts):.2f} max {max(ts):.2f}")
PY
}

KEV="models/Kev-0.8B-Q8_0.gguf"
BASE="models/ggml-base.en-q5_1.bin"
SMALL="models/ggml-small.en-q5_1.bin"

# --- whisper solo at t=20 (missing from pass 1) -----------------------------
echo; echo "=== whisper solo t=20 ==="
for m in "$BASE" "$SMALL"; do
  s=$( { /usr/bin/time -f "%e" "$WC" -m "$m" -t 20 -f "$AUDIO" -np -nt >/dev/null; } 2>&1 | tail -1 )
  echo "    $(basename "$m") t=20 wall=${s}s rtf=$(python3 -c "print(round($s/$AUDIO_SEC,4))")"
done

# --- Kev solo at t=4 and t=20 ----------------------------------------------
for kt in 4 20; do
  echo; echo "=== kev solo t=$kt ==="
  start_kev "$KEV" "$kt" "solo_t$kt" && \
    python3 -u bench/bench.py --port $PORT --label "kev_solo_t$kt" \
        --out "results/c2_kev_solo_t$kt.json" \
        --cold --state-words 90 --questions 1 --reps 20 --warmup 2
  stop_kev
done

# --- long-window contention, for the whisper side --------------------------
# reps raised so multiple whisper passes finish inside the window.
CFG=(
  "8  4  $BASE  150 w_base_t4__k_t8"
  "20 20 $BASE  150 w_base_t20__k_t20"
  "8  4  $SMALL 200 w_small_t4__k_t8"
  "20 20 $SMALL 200 w_small_t20__k_t20"
)
for c in "${CFG[@]}"; do
  set -- $c; kt="$1"; wt="$2"; wm="$3"; reps="$4"; tag="$5"
  echo; echo "=== long $tag (kev t=$kt, whisper t=$wt, $(basename "$wm"), reps=$reps) ==="
  WT="results/c2_wtimes_$tag.txt"; : > "$WT"
  if ! start_kev "$KEV" "$kt" "$tag"; then echo "  kev failed"; stop_kev; continue; fi
  whisper_loop "$wm" "$wt" "$WT" & WPID=$!
  sleep 3
  python3 -u bench/bench.py --port $PORT --label "$tag" \
      --out "results/c2_$tag.json" \
      --cold --state-words 90 --questions 1 --reps "$reps" --warmup 2
  kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null; WPID=""
  [ -s results/.wpid2 ] && kill "$(cat results/.wpid2)" 2>/dev/null
  rm -f results/.wpid2; sleep 2
  echo "    whisper runs completed: $(wc -l < "$WT")"
  wstats "$WT"
  stop_kev
done
echo; echo "=== PASS 2 COMPLETE ==="; date -Is
