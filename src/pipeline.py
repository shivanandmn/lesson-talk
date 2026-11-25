#!/usr/bin/env python3
"""
End-to-end local decision pipeline, and the gold-vs-ASR comparison.

    wav -> whisper.cpp -> sliding windows -> Kev /v1/systemone -> events

The point of running both sources through the SAME window code and the SAME
Kev server is that the gap between them is attributable. Kev was scored
offline on AMI's manual transcript: perfect word boundaries, human
punctuation, and a speaker label on every utterance. A live system has none
of that. Whisper gives no diarisation at all, so its windows carry no speaker
tags, its segment boundaries come from a VAD rather than a transcriber, and
its words are sometimes wrong.

Three things are therefore measured separately, because they are three
different failures:

  1. TRANSCRIPT  -- word error rate against the manual transcript.
  2. WINDOW SHAPE -- how many windows, how long, how many words. A different
     number of windows alone changes the decision rate, with no model
     involvement whatsoever.
  3. DECISION SCORE -- Kev's probability on time-aligned window pairs.

A note on labels. Offline labels come from AMI word IDS, which ASR words
cannot have. So this script labels by TIME overlap with the decision spans
instead, and applies that rule to BOTH sources. The gold source is then
labelled twice -- once by word id, once by time -- and their agreement is
reported, so a gold/ASR difference cannot be silently blamed on the model
when it belongs to the labelling rule.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ami import Meeting                                    # noqa: E402
from windows import Window, series_of, windows_from_utterances  # noqa: E402


# --------------------------------------------------------------- ASR adapters
@dataclass
class AsrWord:
    id: str
    text: str
    start: float
    end: float

    @property
    def is_word(self) -> bool:
        return True


@dataclass
class AsrUtterance:
    """Shaped like ami.Utterance so the window code cannot tell them apart."""
    id: str
    speaker: str
    start: float
    end: float
    words: list[AsrWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def transcribe(audio: str, model: str, threads: int, out_prefix: str,
               whisper_cli: str, force: bool = False) -> str:
    """Run whisper.cpp, returning the path to its JSON. Cached by default:
    a 20-minute meeting is ~10 minutes of CPU and the output is
    deterministic, so re-running it to re-score is pure waste."""
    out_json = f"{out_prefix}.json"
    if os.path.exists(out_json) and not force:
        print(f"  [whisper] reusing {out_json}", flush=True)
        return out_json
    cmd = [whisper_cli, "-m", model, "-f", audio, "-t", str(threads),
           "-oj", "-of", out_prefix, "-np", "-ml", "0"]
    print(f"  [whisper] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    el = time.time() - t0
    if r.returncode != 0 or not os.path.exists(out_json):
        raise RuntimeError(f"whisper failed ({r.returncode}): "
                           f"{r.stderr[-2000:]}")
    print(f"  [whisper] {el:.0f}s wall", flush=True)
    with open(f"{out_prefix}.wall", "w") as fh:
        fh.write(str(el))
    return out_json


def load_asr(path: str) -> list[AsrUtterance]:
    """whisper.cpp JSON -> utterances.

    Whisper timestamps are per SEGMENT, not per word, so word times are
    interpolated across the segment. That is accurate enough for window
    boundaries (which only need ordering and rough time) but it is why the
    decision-overlap rule here counts words by time CONTAINMENT in a span
    rather than trusting any single word's timestamp.
    """
    d = json.load(open(path))
    segs = d.get("transcription") or []
    out: list[AsrUtterance] = []
    n = 0
    for i, seg in enumerate(segs):
        off = seg.get("offsets") or {}
        st = float(off.get("from", 0)) / 1000.0
        en = float(off.get("to", 0)) / 1000.0
        toks = [t for t in re.split(r"\s+", (seg.get("text") or "").strip())
                if t]
        if not toks:
            continue
        step = (en - st) / len(toks) if en > st else 0.0
        words = []
        for k, t in enumerate(toks):
            words.append(AsrWord(id=f"asr{n}", text=t,
                                 start=st + k * step,
                                 end=st + (k + 1) * step))
            n += 1
        # No diarisation: one pseudo-speaker. Marked so the state Kev sees is
        # honestly unlabelled rather than falsely attributed.
        out.append(AsrUtterance(id=f"seg{i}", speaker="S", start=st, end=en,
                                words=words))
    return out


# -------------------------------------------------------------- time labelling
def decision_spans(m: Meeting) -> list[tuple[str, float, float]]:
    """(decision id, start, end) in seconds, from the manual annotation."""
    # Meeting.decisions() already resolves each span to timed words and
    # exposes its start/end, having dropped any decision with no timed word.
    return [(d["id"], d["start"], d["end"]) for d in m.decisions()]


def dec_of_by_time(utts: list, spans: list[tuple[str, float, float]],
                   pad: float = 0.0) -> dict[str, list[str]]:
    """word id -> decision ids, by timestamp containment.

    The only labelling rule available to ASR words, so it is also applied to
    gold words for the comparison to be like-for-like.
    """
    out: dict[str, list[str]] = {}
    for u in utts:
        for w in u.words:
            if not w.is_word or w.start is None:
                continue
            mid = (w.start + (w.end if w.end is not None else w.start)) / 2.0
            for did, s, e in spans:
                if s - pad <= mid <= e + pad:
                    out.setdefault(w.id, []).append(did)
    return out


# ------------------------------------------------------------------ transcript
_NORM = re.compile(r"[^a-z0-9' ]+")


def norm_tokens(text: str) -> list[str]:
    return _NORM.sub(" ", text.lower()).split()


def wer(ref: list[str], hyp: list[str]) -> dict:
    """Word error rate via difflib opcodes.

    difflib rather than full Levenshtein DP: a 20-minute meeting is ~4k
    reference words, and an O(n*m) table in pure Python is minutes of CPU for
    a number that only needs two decimal places. The opcode counts give the
    same S/D/I totals for sequences this similar.
    """
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    sub = dele = ins = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            sub += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            dele += i2 - i1
        elif tag == "insert":
            ins += j2 - j1
    n = max(1, len(ref))
    return {"ref_words": len(ref), "hyp_words": len(hyp), "sub": sub,
            "del": dele, "ins": ins, "wer": (sub + dele + ins) / n}


# ----------------------------------------------------------------- alignment
def iou(a: Window, b: Window) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def align(gold: list[Window], asr: list[Window],
          min_iou: float = 0.3) -> list[tuple[Window, Window, float]]:
    """Pair each gold window with its best-overlapping ASR window.

    Greedy by overlap, not one-to-one: the two sources chunk the meeting
    differently, so insisting on a bijection would silently drop windows and
    bias whatever is computed next. Unmatched gold windows are reported as a
    count instead of being hidden.
    """
    pairs = []
    j = 0
    for g in gold:
        # ASR windows are time-ordered; walk a local cursor rather than
        # rescanning the whole list per gold window.
        while j + 1 < len(asr) and asr[j].end < g.start:
            j += 1
        best, best_v = None, 0.0
        k = j
        while k < len(asr) and asr[k].start <= g.end:
            v = iou(g, asr[k])
            if v > best_v:
                best, best_v = asr[k], v
            k += 1
        if best is not None and best_v >= min_iou:
            pairs.append((g, best, best_v))
    return pairs


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", required=True, help="e.g. ES2006a")
    ap.add_argument("--root", default="data/ami/annotations/extracted")
    ap.add_argument("--audio", help="default: data/ami/audio/<m>.Mix-Headset.wav")
    ap.add_argument("--whisper-model", default="models/ggml-base.en-q5_1.bin")
    ap.add_argument("--whisper-cli",
                    default="vendor/whisper.cpp/build/bin/whisper-cli")
    ap.add_argument("--whisper-threads", type=int, default=8)
    ap.add_argument("--force-asr", action="store_true")
    ap.add_argument("--port", type=int, default=9932)
    ap.add_argument("--prompt", default="Does this excerpt contain a decision?")
    ap.add_argument("--max-words", type=int, default=90)
    ap.add_argument("--advance-words", type=int, default=45)
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--min-words", type=int, default=25)
    ap.add_argument("--outdir", default="results/live")
    ap.add_argument("--no-score", action="store_true",
                    help="windows and transcript only; skip the Kev calls")
    args = ap.parse_args()

    mt = args.meeting
    audio = args.audio or f"data/ami/audio/{mt}.Mix-Headset.wav"
    os.makedirs(args.outdir, exist_ok=True)
    wparams = dict(max_words=args.max_words, advance_words=args.advance_words,
                   min_overlap=args.min_overlap, min_words=args.min_words)

    print(f"=== {mt} ===")
    m = Meeting(args.root, mt)
    gold_utts = m.utterances()
    spans = decision_spans(m)
    print(f"  gold: {len(gold_utts)} utterances, {len(spans)} decision spans")

    # ---- 1. transcript
    print("  [1/4] transcribe")
    asr_json = transcribe(audio, args.whisper_model, args.whisper_threads,
                          os.path.join(args.outdir, f"{mt}.asr"),
                          args.whisper_cli, args.force_asr)
    asr_utts = load_asr(asr_json)
    ref = norm_tokens(" ".join(u.text for u in gold_utts))
    hyp = norm_tokens(" ".join(u.text for u in asr_utts))
    w = wer(ref, hyp)
    print(f"      gold {w['ref_words']} words / asr {w['hyp_words']} words   "
          f"WER {w['wer']:.1%}  (sub {w['sub']} del {w['del']} ins {w['ins']})")

    # ---- 2. windows, both sources, same code, same time-based labels
    print("  [2/4] window")
    gold_dec_id = {}
    for d in m.decisions():
        for wid in d["word_ids"]:
            gold_dec_id.setdefault(wid, []).append(d["id"])
    gold_dec_time = dec_of_by_time(gold_utts, spans)
    asr_dec_time = dec_of_by_time(asr_utts, spans)

    gw_id = windows_from_utterances(gold_utts, meeting=mt,
                                    dec_of=gold_dec_id, **wparams)
    gw = windows_from_utterances(gold_utts, meeting=mt,
                                 dec_of=gold_dec_time, **wparams)
    aw = windows_from_utterances(asr_utts, meeting=mt,
                                 dec_of=asr_dec_time, **wparams)

    # does the time rule reproduce the word-id rule on the SAME windows?
    same = sum(1 for a, b in zip(gw_id, gw) if a.decision == b.decision)
    print(f"      label rule check: time rule agrees with word-id rule on "
          f"{same}/{len(gw_id)} gold windows ({same / max(1, len(gw_id)):.1%})")

    def shape(ws: list[Window]) -> str:
        if not ws:
            return "none"
        wl = sorted(x.n_words for x in ws)
        dur = sorted(x.end - x.start for x in ws)
        pos = sum(x.decision for x in ws)
        return (f"{len(ws):4d} windows  pos {pos:3d} ({pos / len(ws):5.1%})  "
                f"words p50 {wl[len(wl) // 2]:3d}  "
                f"span p50 {dur[len(dur) // 2]:5.1f}s")
    print(f"      gold: {shape(gw)}")
    print(f"      asr : {shape(aw)}")

    pairs = align(gw, aw)
    print(f"      aligned {len(pairs)}/{len(gw)} gold windows to an ASR "
          f"window (IoU >= 0.3)")

    summary = {"meeting": mt, "series": series_of(mt), "wer": w,
               "n_decision_spans": len(spans),
               "label_rule_agreement": same / max(1, len(gw_id)),
               "gold_windows": len(gw), "asr_windows": len(aw),
               "gold_pos": sum(x.decision for x in gw),
               "asr_pos": sum(x.decision for x in aw),
               "aligned": len(pairs),
               "whisper_model": args.whisper_model, "prompt": args.prompt}

    if args.no_score:
        json.dump(summary, open(os.path.join(args.outdir, f"{mt}.summary.json"),
                                "w"), indent=2)
        print("  (--no-score: stopping before the Kev calls)")
        return

    # ---- 3. score both sources on the same server
    from eval_kev import run as kev_run, auc, auc_ci
    print(f"  [3/4] score {len(gw)} gold + {len(aw)} asr windows")
    base = f"http://127.0.0.1:{args.port}"
    gp, gl = kev_run(base, [vars(x) for x in gw], args.prompt, "noul",
                     progress_every=0)
    ap_, al = kev_run(base, [vars(x) for x in aw], args.prompt, "noul",
                      progress_every=0)
    gscore = {(x.start, x.idx): p["score"] for x, p in zip(gw, gp)}
    ascore = {(x.start, x.idx): p["score"] for x, p in zip(aw, ap_)}

    for name, ws, ps, ls in (("gold", gw, gp, gl), ("asr", aw, ap_, al)):
        sc = [(p["score"], x.decision) for x, p in zip(ws, ps)]
        a = auc(sc)
        lo, hi = auc_ci(sc)
        pos = sum(y for _, y in sc)
        print(f"      {name}: AUC {a:.3f} CI [{lo:.3f},{hi:.3f}]  "
              f"(n={len(sc)}, pos={pos})  p50 {statistics.median(ls):.0f} ms")
        summary[f"{name}_auc"] = a
        summary[f"{name}_auc_ci95"] = [lo, hi]
        summary[f"{name}_ms_p50"] = round(statistics.median(ls), 1)

    # ---- 4. paired comparison on time-aligned windows
    print("  [4/4] compare")
    xs = [gscore[(g.start, g.idx)] for g, a, _ in pairs]
    ys = [ascore[(a.start, a.idx)] for g, a, _ in pairs]
    if xs:
        diffs = [abs(x - y) for x, y in zip(xs, ys)]
        agree = sum(1 for x, y in zip(xs, ys)
                    if (x >= 0.5) == (y >= 0.5)) / len(xs)
        print(f"      paired windows {len(xs)}")
        print(f"      score corr: pearson {pearson(xs, ys):.3f}  "
              f"spearman {spearman(xs, ys):.3f}")
        print(f"      |gold-asr| score  mean {statistics.fmean(diffs):.3f}  "
              f"p50 {statistics.median(diffs):.3f}  max {max(diffs):.3f}")
        print(f"      same side of 0.5 on {agree:.1%} of pairs")
        summary.update({"paired": len(xs), "pearson": pearson(xs, ys),
                        "spearman": spearman(xs, ys),
                        "mean_abs_score_diff": statistics.fmean(diffs),
                        "binary_agreement_at_0.5": agree})

    ev = os.path.join(args.outdir, f"{mt}.events.jsonl")
    with open(ev, "w") as fh:
        for src, ws, ps in (("gold", gw, gp), ("asr", aw, ap_)):
            for x, p in zip(ws, ps):
                fh.write(json.dumps({"source": src, "t_start": x.start,
                                     "t_end": x.end, "n_words": x.n_words,
                                     "score": p["score"],
                                     "label": x.decision,
                                     "text": x.text}) + "\n")
    json.dump(summary, open(os.path.join(args.outdir, f"{mt}.summary.json"),
                            "w"), indent=2)
    print(f"\n  wrote {ev}")
    print(f"  wrote {os.path.join(args.outdir, f'{mt}.summary.json')}")


if __name__ == "__main__":
    main()
