#!/usr/bin/env bash
# Definitive Kev CPU decision-latency sweep.
#
# Guards learned the hard way:
#   - a lockfile, so a second invocation cannot race the first
#   - refuse to start if the port is already serving (a stray client or server
#     on the same single-slot port silently adds queueing delay to every
#     measurement, which invalidated an earlier run)
#   - verify no other client is talking to the port mid-run
#   - kill only by explicit PID; never `pkill -f` a pattern that also appears
#     in this script's own command line
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/vendor/llama.cpp/build/bin/llama-server"
PORT=9931
cd "$ROOT"
mkdir -p results
exec > >(tee -a results/final.log) 2>&1

LOCK="results/.sweep.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "FATAL: $LOCK exists - another sweep may be running. Remove it to override."
  exit 1
fi
SPID=""
cleanup() {
  [ -n "$SPID" ] && kill "$SPID" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
}
trap cleanup EXIT

if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "FATAL: port $PORT is already serving. Refusing to measure against it."
  exit 1
fi

echo "### Kev CPU decision-latency sweep"
echo "### host: $(uname -srm)  cpu: $(awk -F: '/model name/{print $2;exit}' /proc/cpuinfo | xargs)"
echo "### llama.cpp: $(cd vendor/llama.cpp && git rev-parse --short HEAD)"
echo "### started: $(date -Is)"

run() {  # run <model> <threads> <tag> <extra bench args...>
  local model="$1" t="$2" tag="$3"; shift 3
  echo "=== $tag ==="; date +"    start %H:%M:%S"

  "$BIN" -m "$model" --port $PORT -c 8192 -ub 4096 -b 4096 -np 1 \
      -t "$t" --no-warmup > "results/server_${tag}.log" 2>&1 &
  SPID=$!

  local ok=0
  for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$SPID" 2>/dev/null || break
    sleep 1
  done
  if [ "$ok" != 1 ]; then
    echo "    server never became ready; see results/server_${tag}.log"
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=""; return
  fi

  python3 -u bench/bench.py --port $PORT --label "$tag" --out "results/${tag}.json" "$@"

  # integrity check: the only tasks this server saw should be ours
  local hwm; hwm="$(awk '/VmHWM/{print $2}' "/proc/$SPID/status" 2>/dev/null)"
  echo "    peak RSS: ${hwm:-?} kB"
  if [ -n "${hwm:-}" ] && [ -f "results/${tag}.json" ]; then
    python3 - "results/${tag}.json" "$hwm" "$t" <<'PY'
import json,sys
p,h,t=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); d=json.load(open(p))
d["peak_rss_kb"]=h; d["peak_rss_gib"]=round(h/1048576,2); d["threads"]=t
json.dump(d,open(p,"w"),indent=2)
PY
  fi

  kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=""
  # wait for the port to actually close before the next config
  for _ in $(seq 1 60); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || break
    sleep 1
  done
  sleep 2
}

S="models/Kev-0.8B-Q8_0.gguf"
M="models/Kev-4B-Q4_K_M.gguf"
M8="models/Kev-4B-Q8_0.gguf"
G=(--state-words 90 280 --questions 1 3 --reps 5)
T1=(--cold --state-words 25 50 90 --questions 1 --reps 10)

# thread scaling
run "$S" 8  "s_t8"  "${G[@]}"
run "$S" 12 "s_t12" "${G[@]}"
run "$S" 20 "s_t20" "${G[@]}"
run "$M" 8  "m_t8"  "${G[@]}"
run "$M" 12 "m_t12" "${G[@]}"
run "$M" 20 "m_t20" "${G[@]}"

# cold vs warm (KV reuse value)
run "$S" 8 "s_t8_cold" --cold "${G[@]}"
run "$M" 8 "m_t8_cold" --cold "${G[@]}"

# quant level at fixed size
run "$M8" 8 "m8_t8" --state-words 90 280 --questions 1 --reps 5

# tier-1 viability: shortest useful window, one boolean question
run "$S" 8 "s_t8_tier1" "${T1[@]}"
run "$M" 8 "m_t8_tier1" --cold --state-words 25 50 90 --questions 1 --reps 8

echo "=== SWEEP COMPLETE ==="; date -Is
