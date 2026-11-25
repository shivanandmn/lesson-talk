#!/usr/bin/env python3
"""
Accuracy of Kev's System One decisions against AMI's manual decision
annotations.

The latency profile in PROFILE.md says nothing about whether answers are
correct. This closes that gap: every window in a split is posted to
/v1/systemone as a single `noul` question, and the returned probability is
scored against the gold label built from AMI's decision/manual XML.

Two things make this honest rather than flattering:

  * `noul` returns a PROBABILITY, so accuracy depends on a threshold. A
    threshold tuned on the data it is reported on is leakage. So the
    threshold-free AUC is the headline, and any thresholded number is
    reported with the split the threshold came from (--threshold, written
    out by a dev run, applied to test).

  * 32% of windows are positive, so a model that answers "no" to everything
    scores 68%. The majority-class baseline is printed next to every
    accuracy figure, and balanced accuracy is reported as well.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

# A sliding window has no notion of "the meeting decided X" -- it holds ~35s
# of speech. The question has to be about what is happening in the excerpt.
PROMPTS = {
    "commit": "Did the group commit to a decision in this excerpt?",
    "agreed": "Do the speakers agree on a specific choice here?",
    "settled": "Is a choice about the product being settled in this excerpt, "
               "rather than merely discussed?",
    "decided": "Does this excerpt contain a decision?",
}


def post(url: str, payload: dict, timeout: float = 300.0,
         retries: int = 2) -> dict:
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 - transient server hiccup
            last = e
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"request failed after {retries + 1} tries: {last!r}")


def auc(scored: list[tuple[float, int]]) -> float:
    """ROC AUC by rank (Mann-Whitney U), ties averaged.

    Written out rather than imported: no sklearn in this environment, and a
    wrong AUC is worse than none.
    """
    pos = [s for s, y in scored if y == 1]
    neg = [s for s, y in scored if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(scored, key=lambda sy: sy[0])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][0] == order[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0          # average rank of the tied block
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rank_sum = sum(r for k, r in ranks.items() if order[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc_ci(scored: list[tuple[float, int]], reps: int = 2000,
           seed: int = 12345) -> tuple[float, float]:
    """Percentile bootstrap CI for AUC.

    At a few hundred windows an AUC near 0.6 can easily be noise, and the
    whole point of this run is to tell a weak-but-real signal apart from
    none. Resampling is stratified (positives and negatives drawn
    separately) so every replicate keeps both classes.
    """
    import random
    rng = random.Random(seed)
    pos = [s for s, y in scored if y == 1]
    neg = [s for s, y in scored if y == 0]
    if not pos or not neg:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(reps):
        rp = [rng.choice(pos) for _ in pos]
        rn = [rng.choice(neg) for _ in neg]
        vals.append(auc([(v, 1) for v in rp] + [(v, 0) for v in rn]))
    vals.sort()
    return (vals[int(0.025 * reps)], vals[min(reps - 1, int(0.975 * reps))])


def metrics_at(scored: list[tuple[float, int]], thr: float) -> dict:
    tp = sum(1 for s, y in scored if s >= thr and y == 1)
    fp = sum(1 for s, y in scored if s >= thr and y == 0)
    fn = sum(1 for s, y in scored if s < thr and y == 1)
    tn = sum(1 for s, y in scored if s < thr and y == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": round(thr, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / max(1, len(scored)),
            "precision": prec, "recall": rec, "specificity": spec,
            "f1": f1, "balanced_accuracy": (rec + spec) / 2}


def run(base: str, rows: list[dict], prompt: str, qtype: str,
        progress_every: int = 50) -> tuple[list[dict], list[float]]:
    url = f"{base}/v1/systemone"
    q = {"decision": {"type": qtype, "instructions": prompt}}
    out, lats = [], []
    t_start = time.time()
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        resp = post(url, {"state": r["text"], "questions": q})
        lats.append((time.perf_counter() - t0) * 1000.0)
        ans = resp.get("answers", {}).get("decision", {})
        score = ans.get("noul")
        if score is None:                       # shape changed -> stop, loudly
            raise RuntimeError(f"no 'noul' in answer: {json.dumps(resp)[:400]}")
        out.append({"meeting": r["meeting"], "series": r["series"],
                    "idx": r["idx"], "n_words": r["n_words"],
                    "overlap_words": r["overlap_words"],
                    "gold": r["decision"], "score": score,
                    "input_tokens": resp.get("usage", {}).get("input_tokens")})
        if progress_every and (i + 1) % progress_every == 0:
            el = time.time() - t_start
            print(f"    {i + 1}/{len(rows)}  {el:.0f}s elapsed, "
                  f"~{el / (i + 1) * (len(rows) - i - 1):.0f}s left",
                  file=sys.stderr, flush=True)
    return out, lats


def report(preds: list[dict], lats: list[float], label: str,
           fixed_thr: float | None) -> dict:
    scored = [(p["score"], p["gold"]) for p in preds]
    pos_rate = sum(y for _, y in scored) / max(1, len(scored))
    baseline = max(pos_rate, 1 - pos_rate)
    a = auc(scored)

    # best-F1 threshold ON THIS SPLIT: a tuning artefact, not a result.
    cands = sorted({round(s, 3) for s, _ in scored})
    best = max((metrics_at(scored, t) for t in cands),
               key=lambda m: m["f1"]) if cands else {}

    print(f"\n=== {label} ===")
    print(f"windows {len(scored)}   positive {pos_rate:.1%}   "
          f"majority-class accuracy baseline {baseline:.1%}")
    print(f"latency p50 {statistics.median(lats):.0f} ms   "
          f"total {sum(lats) / 1000:.0f} s")
    lo, hi = auc_ci(scored)
    print(f"\nAUC (threshold-free) {a:.4f}  95% CI [{lo:.3f}, {hi:.3f}]"
          f"   [0.5 = coin flip, 1.0 = perfect ranking]")
    if lo <= 0.5:
        print("  CI includes 0.5 -- NOT distinguishable from chance at this n")
    sp = [p["score"] for p in preds if p["gold"] == 1]
    sn = [p["score"] for p in preds if p["gold"] == 0]
    if sp and sn:
        print(f"mean score  positive {statistics.fmean(sp):.3f}   "
              f"negative {statistics.fmean(sn):.3f}")

    rows_out = []
    print(f"\n{'threshold':>10} {'acc':>7} {'prec':>7} {'rec':>7} "
          f"{'f1':>7} {'bal-acc':>8}")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        m = metrics_at(scored, t)
        rows_out.append(m)
        print(f"{t:>10.2f} {m['accuracy']:>7.1%} {m['precision']:>7.1%} "
              f"{m['recall']:>7.1%} {m['f1']:>7.3f} "
              f"{m['balanced_accuracy']:>8.1%}")

    if best:
        print(f"\nbest-F1 threshold on THIS split: {best['threshold']} -> "
              f"f1 {best['f1']:.3f}, acc {best['accuracy']:.1%}, "
              f"bal-acc {best['balanced_accuracy']:.1%}")
        print("  (tuned on the split it is reported on -- carry it to the "
              "other split with --threshold before believing it)")

    held = metrics_at(scored, fixed_thr) if fixed_thr is not None else None
    if held:
        print(f"\nHELD-OUT threshold {fixed_thr} (chosen elsewhere): "
              f"acc {held['accuracy']:.1%} (baseline {baseline:.1%}), "
              f"f1 {held['f1']:.3f}, bal-acc {held['balanced_accuracy']:.1%}")

    verdict = ("not distinguishable from chance (CI spans 0.5)" if lo <= 0.5
               else "ranks no better than chance" if not (a == a) or a <= 0.55 else
               "weak but real signal" if a < 0.65 else
               "usable signal" if a < 0.8 else "strong signal")
    print(f"\nverdict: AUC {a:.3f} -- {verdict}")

    return {"label": label, "n": len(scored), "positive_rate": pos_rate,
            "baseline_accuracy": baseline, "auc": a,
            "auc_ci95": [lo, hi],
            "ms_p50": round(statistics.median(lats), 1),
            "sweep": rows_out, "best_f1_this_split": best,
            "held_out_threshold": held}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="windows jsonl")
    ap.add_argument("--port", type=int, default=9932)
    ap.add_argument("--prompt", default="commit",
                    help=f"one of {sorted(PROMPTS)} or a literal question")
    ap.add_argument("--qtype", default="noul")
    ap.add_argument("--limit", type=int, help="first N windows (smoke test)")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth window -- a subsample that still "
                         "spans every meeting, unlike --limit")
    ap.add_argument("--threshold", type=float,
                    help="threshold chosen on ANOTHER split, applied here")
    ap.add_argument("--out", help="write per-window predictions jsonl")
    ap.add_argument("--summary", help="write summary json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.split) if l.strip()]
    if args.stride > 1:
        rows = rows[::args.stride]
    if args.limit:
        rows = rows[:args.limit]

    prompt = PROMPTS.get(args.prompt, args.prompt)
    base = f"http://127.0.0.1:{args.port}"
    label = f"{args.split.rsplit('/', 1)[-1]}  prompt={args.prompt}"
    print(f"{len(rows)} windows   question: {prompt!r}", flush=True)

    preds, lats = run(base, rows, prompt, args.qtype)

    if args.out:
        with open(args.out, "w") as fh:
            for p in preds:
                fh.write(json.dumps(p) + "\n")
    summary = report(preds, lats, label, args.threshold)
    summary["prompt"] = prompt
    summary["prompt_key"] = args.prompt
    summary["split"] = args.split
    if args.summary:
        json.dump(summary, open(args.summary, "w"), indent=2)
        print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()
