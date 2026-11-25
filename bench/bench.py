#!/usr/bin/env python3
"""
Kev CPU latency benchmark against llama.cpp's /v1/systemone endpoint.

Measures single-forward-pass decision latency on CPU across:
  state length x question count x thread count x model/quant

Stdlib only. No torch, no HF deps.

Usage:
  python3 bench.py --port 9931 --label "Kev-4B-Q4_K_M t=8" --out results/x.json
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- state corpus
# Transcript-style utterances. Realistic shape matters: Kev was trained on
# short states (384 tok), and real input is disfluent speech, not clean prose.
UTTERANCES = [
    "So I called last Tuesday about the duplicate charge and nobody got back to me.",
    "Right, I can see two transactions here on the fourteenth, both for the same amount.",
    "Yeah that's the problem, I only ordered once, I've got the confirmation email.",
    "Let me just pull up the account, bear with me a second.",
    "I've been a customer for six years and this is the third time this has happened.",
    "I understand, that's frustrating, I'm going to get that refunded today.",
    "The thing is I need it back before the end of the month or my rent bounces.",
    "Okay, so refunds normally take three to five working days to clear.",
    "That's not good enough honestly, can you not just reverse it straight away?",
    "Let me check whether we can expedite that for you, one moment.",
    "And while you're there, can you cancel the auto-renewal as well please.",
    "Sure, I can switch off the automatic renewal on the subscription.",
    "I'd also like some kind of compensation for the time I've wasted on this.",
    "I can log that as a complaint and someone from the team will review it.",
    "How long is that going to take, because last time nobody ever called back.",
    "The complaint process is usually about ten working days for a full response.",
    "Fine. Can I get a reference number so I've got something in writing.",
    "Absolutely, I'll email you a confirmation with the case reference now.",
    "And you're definitely sure the second charge is gone, it won't come back?",
    "Yes, the duplicate has been voided, you'll see it drop off within the hour.",
]


def make_state(target_words: int) -> str:
    """Build a transcript-shaped state of roughly target_words words."""
    out, n, i = [], 0, 0
    while n < target_words:
        u = UTTERANCES[i % len(UTTERANCES)]
        speaker = "Customer" if i % 2 == 0 else "Agent"
        line = f"{speaker}: {u}"
        out.append(line)
        n += len(line.split())
        i += 1
    return "\n".join(out)


# ------------------------------------------------------------------- questions
# Ordered so that QUESTIONS[:n] is always a sensible triage set.
QUESTIONS = [
    ("route", {
        "type": "choice",
        "instructions": "Which team should handle this call?",
        "criteria": {"billing": "Charges, refunds, invoices",
                     "retention": "Cancellation or churn risk",
                     "complaints": "Formal complaint or escalation",
                     "technical": "Product or service fault"},
    }),
    ("escalate", {
        "type": "noul",
        "instructions": "Does this call need supervisor escalation?",
    }),
    ("urgency", {
        "type": "score",
        "instructions": "How urgent is the caller's need?",
        "criteria": ["can wait", "this week", "today", "right now"],
    }),
    ("commitment", {
        "type": "noul",
        "instructions": "Did the agent commit to a specific action with a deadline?",
    }),
    ("sentiment", {
        "type": "score",
        "instructions": "How dissatisfied is the customer?",
        "criteria": ["satisfied", "mildly annoyed", "frustrated", "angry"],
    }),
    ("compliance", {
        "type": "noul",
        "instructions": "Did the agent state the refund timeframe accurately?",
    }),
]


def build_questions(n: int) -> dict:
    return {k: v for k, v in QUESTIONS[:n]}


# ----------------------------------------------------------------------- http
def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(base: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 - server not up yet
            last = e
        time.sleep(1.0)
    raise RuntimeError(f"server not ready after {timeout}s (last: {last})")


# ----------------------------------------------------------------------- bench
def bench_one(base: str, state: str, nq: int, reps: int, warmup: int,
              unique: bool = False) -> dict:
    """unique=True prefixes each rep with a distinct token so the server's
    longest-common-prefix KV reuse cannot fire -> cold-cache numbers."""
    url = f"{base}/v1/systemone"

    def mk(i):
        st = f"[call {i:04d}]\n{state}" if unique else state
        return {"state": st, "questions": build_questions(nq)}

    for i in range(warmup):
        post(url, mk(-i - 1))

    lats, tokens = [], None
    for i in range(reps):
        t0 = time.perf_counter()
        resp = post(url, mk(i))
        lats.append((time.perf_counter() - t0) * 1000.0)
        tokens = resp.get("usage", {}).get("input_tokens")

    lats.sort()
    return {
        "n_questions": nq,
        "input_tokens": tokens,
        "cold": unique,
        "reps": reps,
        "ms_p50": round(statistics.median(lats), 1),
        "ms_p90": round(lats[max(0, int(len(lats) * 0.9) - 1)], 1),
        "ms_min": round(lats[0], 1),
        "ms_max": round(lats[-1], 1),
        "ms_mean": round(statistics.fmean(lats), 1),
        "ms_per_question": round(statistics.median(lats) / nq, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9931)
    ap.add_argument("--label", required=True, help="e.g. 'Kev-4B-Q4_K_M t=8'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-words", type=int, nargs="+", default=[90, 280, 760],
                    help="approx word counts -> ~128/384/1024 tokens")
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 3, 6])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--cold", action="store_true",
                    help="unique state per rep; defeats LCP KV reuse")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    wait_ready(base)

    try:
        with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as r:
            models = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        models = {"error": str(e)}

    results = []
    for words in args.state_words:
        state = make_state(words)
        for nq in args.questions:
            tag = "cold" if args.cold else "warm"
            print(f"  [{args.label}] {tag} words~{words} q={nq} ...", flush=True)
            try:
                row = bench_one(base, state, nq, args.reps, args.warmup,
                                unique=args.cold)
            except Exception as e:  # noqa: BLE001
                row = {"n_questions": nq, "error": repr(e)}
            row["state_words"] = words
            results.append(row)
            if "error" in row:
                print(f"      ERROR: {row['error']}", flush=True)
            else:
                print(f"      {row['input_tokens']} tok -> "
                      f"p50 {row['ms_p50']} ms "
                      f"({row['ms_per_question']} ms/question)", flush=True)

    out = {
        "label": args.label,
        "models_endpoint": models,
        "reps": args.reps,
        "warmup": args.warmup,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
