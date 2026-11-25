"""
Split labelled windows into train/dev/test and report the class balance.

The split is on MEETING SERIES, never on windows and never on meetings:
meetings ES2002a-d share the same four participants working on the same
product design project, so a per-meeting split leaks participants, vocabulary
and subject matter across the boundary and inflates every score. Sliding
windows overlap each other outright, so a per-window split is worse still.

The split is deterministic and stratified. With only ~17 series, hashing each
series name independently gives badly lumpy splits (it produced 14/2/1, i.e. a
test set of one meeting). Instead, series are grouped by family (ES/IS/TS),
sorted within family, and dealt out in a repeating pattern, so every split gets
series from every family in roughly the target proportions.

The assignment depends on the series list, so it is written to disk once and
reused; it must not be silently regenerated against a different meeting set.
"""

from __future__ import annotations

import os
import sys

# -I (isolated) implies -P, so this script's own directory is not on
# sys.path. Re-add only src/ -- never the data directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib
import json
from collections import Counter, defaultdict


def _family(series: str) -> str:
    """'ES2002' -> 'ES'. AMI families differ in participants and room."""
    return series[:2]


def assign_splits(series: list[str], dev: float = 0.2,
                  test: float = 0.2) -> dict[str, str]:
    """Deal series out to train/dev/test, stratified by family.

    Deterministic given the same series list. The repeating pattern is built
    from the requested proportions, then applied within each family so no
    split misses a family entirely.
    """
    n = max(1, round(1 / max(test, dev, 1e-9)))  # pattern length, e.g. 5
    n_test = max(1, round(test * n))
    n_dev = max(1, round(dev * n))
    pattern = (["test"] * n_test + ["dev"] * n_dev
               + ["train"] * max(0, n - n_test - n_dev))

    by_fam: dict[str, list[str]] = {}
    for s in sorted(set(series)):
        by_fam.setdefault(_family(s), []).append(s)

    # Interleave families round-robin, THEN deal the pattern over the pooled
    # list. Dealing within each family instead would either over-sample the
    # small families into dev/test or (with an offset) drop a family out of
    # them entirely -- AMI's TS family has only 3 series, and leaving it out
    # of test means no measurement of generalisation to it.
    pooled: list[str] = []
    fams = sorted(by_fam)
    for i in range(max(len(v) for v in by_fam.values())):
        for fam in fams:
            if i < len(by_fam[fam]):
                pooled.append(by_fam[fam][i])

    return {s: pattern[i % len(pattern)] for i, s in enumerate(pooled)}


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-prefix", help="write <prefix>.{train,dev,test}.jsonl")
    ap.add_argument("--dev", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.2)
    args = ap.parse_args()

    rows = load(args.windows)
    splits = assign_splits([r["series"] for r in rows], args.dev, args.test)

    by_split = defaultdict(list)
    for r in rows:
        r["split"] = splits[r["series"]]
        by_split[r["split"]].append(r)

    print(f"{len(rows)} windows, {len(set(splits))} series\n")
    print(f"{'split':6} {'series':>6} {'meetings':>9} {'windows':>8} "
          f"{'pos':>6} {'rate':>7} {'words p50':>10}")
    for sp in ("train", "dev", "test"):
        rs = by_split[sp]
        if not rs:
            print(f"{sp:6} {'-':>6}"); continue
        pos = sum(r["decision"] for r in rs)
        ws = sorted(r["n_words"] for r in rs)
        print(f"{sp:6} {len({r['series'] for r in rs}):>6} "
              f"{len({r['meeting'] for r in rs}):>9} {len(rs):>8} "
              f"{pos:>6} {pos/len(rs):>6.1%} {ws[len(ws)//2]:>10}")

    print("\nseries -> split")
    for s, sp in sorted(splits.items(), key=lambda kv: (kv[1], kv[0])):
        n = sum(1 for r in rows if r["series"] == s)
        print(f"  {s:8} {sp:6} ({n} windows)")

    # a label that is nearly all one class makes accuracy meaningless
    overall = sum(r["decision"] for r in rows) / max(1, len(rows))
    print(f"\noverall positive rate: {overall:.1%}")
    if overall < 0.05 or overall > 0.95:
        print("  WARNING: severe class imbalance -- majority-class baseline "
              "will beat the model on accuracy; report balanced metrics.")
    print("majority-class accuracy baseline: "
          f"{max(overall, 1 - overall):.1%}")

    if args.out_prefix:
        for sp in ("train", "dev", "test"):
            p = f"{args.out_prefix}.{sp}.jsonl"
            with open(p, "w") as fh:
                for r in by_split[sp]:
                    fh.write(json.dumps(r) + "\n")
            print(f"wrote {p} ({len(by_split[sp])} rows)")


if __name__ == "__main__":
    main()
