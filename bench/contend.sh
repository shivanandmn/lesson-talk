#!/usr/bin/env bash
# Measures CPU contention between whisper.cpp (continuous STT) and Kev
# (occasional decisions) on the same cores.
#
# Design: whisper runs in a loop for the whole measurement window, exactly as a
# streaming pipeline would keep it busy. Kev decisions are issued concurrently.
# Both sides' degradation vs. their solo baselines is reported.
#
# Same guards as final.sh: lockfile, refuse a busy port, kill only by PID.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LS="$ROOT/vendor/llama.cpp/build/bin/llama-server"
WC="$ROOT/vendor/whisper.cpp/build/bin/whisper-cli"
AUDIO="$ROOT/audio/long.wav"
AUDIO_SEC=330
PORT=9932
cd "$ROOT"; mkdir -p results
exec > >(tee -a results/contend.log) 2>&1

LOCK="results/.contend.lock"
mkdir "$LOCK" 2>/dev/null || { echo "FATAL: $LOCK exists"; exit 1; }
KPID=""; WPID=""
cleanup() { [ -n "$WPID" ] && kill "$WPID" 2>/dev/null
            [ -s results/.wpid ] && kill "$(cat results/.wpid)" 2>/dev/null
            [ -n "$KPID" ] && kill "$KPID" 2>/dev/null
            rm -f results/.wpid; rmdir "$LOCK" 2>/dev/null; }
trap cleanup EXIT
curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && {
  echo "FATAL: port $PORT already serving"; exit 1; }

echo "### whisper/Kev CPU contention"
echo "### host: $(uname -srm)  cpu: $(awk -F: '/model name/{print $2;exit}' /proc/cpuinfo|xargs)"
echo "### llama.cpp $(cd vendor/llama.cpp && git rev-parse --short HEAD)  whisper.cpp $(cd vendor/whisper.cpp && git rev-parse --short HEAD)"
echo "### audio: ${AUDIO_SEC}s 16k mono   started: $(date -Is)"

# whisper loop: transcribe repeatedly, append each wall time to $1
# Records its current whisper-cli pid to results/.wpid so the parent can kill
# that process explicitly -- killing the loop's subshell alone would orphan it.
whisper_loop() {  # whisper_loop <model> <threads> <outfile>
  while :; do
    local t0 t1 cpid
    t0=$(date +%s.%N)
    "$WC" -m "$1" -t "$2" -f "$AUDIO" -np -nt >/dev/null 2>&1 &
    cpid=$!; echo "$cpid" > results/.wpid
    wait "$cpid" || break
    t1=$(date +%s.%N)
    echo "$(python3 -c "print(round($t1-$t0,2))")" >> "$3"
  done
}

start_kev() {  # start_kev <model> <threads> <tag>
  "$LS" -m "$1" --port $PORT -c 8192 -ub 4096 -b 4096 -np 1 -t "$2" \
      --no-warmup > "results/cserver_$3.log" 2>&1 &
  KPID=$!
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$KPID" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
stop_kev() { [ -n "$KPID" ] && { kill "$KPID" 2>/dev/null; wait "$KPID" 2>/dev/null; }
             KPID=""
             for _ in $(seq 1 60); do
               curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break
               sleep 1
             done; sleep 2; }

KEV="models/Kev-0.8B-Q8_0.gguf"
WMOD="models/ggml-base.en-q5_1.bin"
BENCH=(--cold --state-words 90 --questions 1 --reps 20 --warmup 2)

# --- A. Kev solo, for a same-session baseline -------------------------------
echo; echo "=== A kev_solo_t8 ==="
start_kev "$KEV" 8 kev_solo && \
  python3 -u bench/bench.py --port $PORT --label kev_solo_t8 \
      --out results/c_kev_solo_t8.json "${BENCH[@]}"
stop_kev

# --- B. contention matrix ---------------------------------------------------
# (kev_threads whisper_threads whisper_model tag)
CFG=(
  "8 4 $WMOD w_base_t4__k_t8"
  "8 8 $WMOD w_base_t8__k_t8"
  "4 4 $WMOD w_base_t4__k_t4"
  "8 4 models/ggml-small.en-q5_1.bin w_small_t4__k_t8"
  "20 20 $WMOD w_base_t20__k_t20"
)
for c in "${CFG[@]}"; do
  set -- $c; kt="$1"; wt="$2"; wm="$3"; tag="$4"
  echo; echo "=== B $tag (kev t=$kt, whisper t=$wt, $(basename "$wm")) ==="
  WT="results/c_wtimes_$tag.txt"; : > "$WT"
  if ! start_kev "$KEV" "$kt" "$tag"; then echo "  kev failed to start"; stop_kev; continue; fi
  whisper_loop "$wm" "$wt" "$WT" & WPID=$!
  sleep 3   # let whisper reach steady state before the first decision
  python3 -u bench/bench.py --port $PORT --label "$tag" \
      --out "results/c_$tag.json" "${BENCH[@]}"
  kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null; WPID=""
  [ -s results/.wpid ] && kill "$(cat results/.wpid)" 2>/dev/null
  rm -f results/.wpid
  sleep 2
  echo "    whisper runs completed during window: $(wc -l < "$WT")"
  [ -s "$WT" ] && python3 - "$WT" "$AUDIO_SEC" <<'PY'
import sys,statistics
ts=[float(x) for x in open(sys.argv[1]) if x.strip()]
sec=float(sys.argv[2])
print(f"    whisper wall p50 {statistics.median(ts):.2f}s  "
      f"rtf {statistics.median(ts)/sec:.4f}  n={len(ts)}  "
      f"min {min(ts):.2f} max {max(ts):.2f}")
PY
  stop_kev
done

echo; echo "=== CONTENTION COMPLETE ==="; date -Is
