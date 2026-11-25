#!/usr/bin/env python3
"""
Talk-time analytics from voice activity, and its validation against AMI's
manual segments.

Why this layer exists at all: `src/pipeline.py` showed that at 28-34% WER the
word-level signal is badly distorted -- whisper recovers ~75% of words, drops
speech 18x more often than it invents it, and Kev's judgement flips on ~30%
of windows between the manual transcript and the ASR of the same audio. So a
product metric that depends on WHICH WORDS were said inherits that.

Talk-time metrics do not. "Who held the floor, for how long, how often, and
how long was the silence after a question" is computed from voice activity
and turn boundaries. A word can be mistranscribed without moving any of
those numbers.

The claim that they are robust is tested here rather than asserted: the same
metrics are computed from (a) AMI's manual segments and (b) Silero VAD over
the mixed audio, and compared.

Attribution. One mic cannot say who is speaking. The product design avoids
diarisation by giving the teacher a close mic and the room a second mic, so
attribution becomes a per-frame energy comparison between two channels.
`--near` / `--far` tests exactly that against AMI's per-speaker ground truth.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import re
import statistics
import subprocess
import sys
import wave
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ami import Meeting                                  # noqa: E402

SEG_RE = re.compile(r"start\s*=\s*([\d.]+),\s*end\s*=\s*([\d.]+)")


@dataclass
class Seg:
    start: float
    end: float
    who: str = "?"

    @property
    def dur(self) -> float:
        return self.end - self.start


# ----------------------------------------------------------------------- VAD
# Measured, not guessed: a min-silence sweep (100/250/500/800/1200 ms) over
# two AMI meetings. At the tool default of 100 ms every breath splits a turn,
# giving 436 speech runs where the manual transcript has 182, and inventing
# ~190 s of silence. 800 ms lands turn counts within 10% on both meetings
# (165 vs 182, 121 vs 115) and raises speech F1 from 0.928 to 0.957.
DEFAULT_MIN_SILENCE_MS = 800


def run_vad(audio: str, model: str, binary: str, threads: int,
            out_txt: str, min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
            threshold: float = 0.5, force: bool = False) -> list[Seg]:
    """Silero VAD via whisper.cpp. Output is CENTISECONDS, not seconds --
    the tool says so in its usage text and it is an easy 100x error."""
    if not os.path.exists(out_txt) or force:
        cmd = [binary, "-vm", model, "-f", audio, "-t", str(threads), "-np",
               "-vsd", str(min_silence_ms), "-vt", str(threshold)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"vad failed: {r.stderr[-1500:]}")
        with open(out_txt, "w") as fh:
            fh.write(r.stdout)
    segs = []
    for line in open(out_txt):
        m = SEG_RE.search(line)
        if m:
            segs.append(Seg(float(m.group(1)) / 100.0,
                            float(m.group(2)) / 100.0))
    return segs


# ------------------------------------------------------------------ metrics
def metrics(segs: list[Seg], span: float,
            by_speaker: bool = False) -> dict:
    """The numbers a teacher would actually be shown.

    Deliberately not "engagement" or any other composite: each of these is a
    quantity someone can verify by listening to the recording, which is what
    makes the product defensible when a teacher disputes it.
    """
    segs = sorted(segs, key=lambda s: s.start)
    # Merge first. Summing raw segment durations double-counts overlapping
    # speech: with four simultaneous speakers it produced a gold "speech
    # ratio" of 1.097 -- more speech than wall clock, which is impossible and
    # was silently inflating the manual baseline this product is judged
    # against. `speaker_seconds` keeps the overlapping total, which is a
    # different and also useful quantity.
    merged: list[Seg] = []
    for x in segs:
        if merged and x.start <= merged[-1].end:
            merged[-1] = Seg(merged[-1].start, max(merged[-1].end, x.end))
        else:
            merged.append(Seg(x.start, x.end))
    talk = sum(s.dur for s in merged)
    out = {
        "speaker_seconds": round(sum(s.dur for s in segs), 1),
        "span_s": round(span, 1),
        "speech_s": round(talk, 1),
        "silence_s": round(max(0.0, span - talk), 1),
        "speech_ratio": round(talk / span, 4) if span else 0.0,
        "n_turns": len(segs),
        "turn_s_mean": round(statistics.fmean([s.dur for s in segs]), 2)
        if segs else 0.0,
        "turn_s_p50": round(statistics.median([s.dur for s in segs]), 2)
        if segs else 0.0,
        "turn_s_max": round(max((s.dur for s in segs), default=0.0), 2),
        "turns_per_min": round(len(segs) / (span / 60.0), 2) if span else 0.0,
    }

    # Gaps are measured on the same MERGED speech, for the same reason.
    out["n_speech_runs"] = len(merged)
    gaps = [b.start - a.end for a, b in zip(merged, merged[1:])
            if b.start > a.end]
    out["n_gaps"] = len(gaps)
    out["gap_s_p50"] = round(statistics.median(gaps), 2) if gaps else 0.0
    out["gaps_over_3s"] = sum(1 for g in gaps if g >= 3.0)

    if by_speaker:
        who: dict[str, list[Seg]] = {}
        for s in segs:
            who.setdefault(s.who, []).append(s)
        tot = sum(s.dur for s in segs) or 1.0
        out["speakers"] = {
            w: {"talk_s": round(sum(x.dur for x in v), 1),
                "share": round(sum(x.dur for x in v) / tot, 4),
                "turns": len(v)}
            for w, v in sorted(who.items())
        }
        # the headline coaching number
        dom = max(who, key=lambda w: sum(x.dur for x in who[w]))
        out["dominant_speaker"] = dom
        out["dominant_share"] = round(
            sum(x.dur for x in who[dom]) / tot, 4)
    return out


# --------------------------------------------------- frame-level agreement
def frame_mask(segs: list[Seg], span: float, hop: float = 0.1) -> list[bool]:
    n = int(span / hop) + 1
    mask = [False] * n
    for s in segs:
        i0 = max(0, int(s.start / hop))
        i1 = min(n - 1, int(s.end / hop))
        for i in range(i0, i1 + 1):
            mask[i] = True
    return mask


def agreement(a: list[bool], b: list[bool]) -> dict:
    n = min(len(a), len(b))
    tp = sum(1 for i in range(n) if a[i] and b[i])
    fp = sum(1 for i in range(n) if not a[i] and b[i])
    fn = sum(1 for i in range(n) if a[i] and not b[i])
    tn = sum(1 for i in range(n) if not a[i] and not b[i])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"frames": n, "accuracy": (tp + tn) / max(1, n),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "iou": tp / max(1, tp + fp + fn)}


# --------------------------------------------- two-channel attribution test
def read_wav_rms(path: str, hop: float = 0.1) -> tuple[list[float], float]:
    """Per-hop RMS of a 16-bit PCM wav, in dB. Stdlib `wave` + `array`:
    no numpy in this environment and a meeting is only a few MB of frames
    once reduced to one value per 100 ms."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        if sw != 2:
            raise RuntimeError(f"{path}: expected 16-bit PCM, got {sw * 8}-bit")
        n = w.getnframes()
        step = int(sr * hop)
        out = []
        while True:
            raw = w.readframes(step)
            if not raw:
                break
            a = array.array("h")
            a.frombytes(raw[:len(raw) - (len(raw) % 2)])
            if ch > 1:
                a = a[::ch]
            if not a:
                break
            s = sum(float(v) * v for v in a) / len(a)
            out.append(10.0 * math.log10(s + 1e-9))
        return out, n / sr


def channel_delta(near: str, far: str, segs: list[Seg],
                  hop: float = 0.1,
                  normalize: bool = True) -> list[float]:
    """Per-segment near-minus-far energy, one value per VAD segment.

    Raw dB is NOT comparable across channels: a lapel and a room mic have
    different gains and different preamps, so a fixed 0 dB boundary simply
    picks whichever channel is hotter overall (measured: it attributed 1023
    of 1024 seconds to one side). Each channel is therefore referenced to its
    OWN median frame before subtracting, which removes the gain offset and
    leaves the thing that actually carries the information -- how far this
    moment deviates from that channel's own baseline.
    """
    rn, _ = read_wav_rms(near, hop)
    rf, _ = read_wav_rms(far, hop)
    bn = statistics.median(rn) if normalize and rn else 0.0
    bf = statistics.median(rf) if normalize and rf else 0.0
    out = []
    for s in segs:
        i0, i1 = int(s.start / hop), int(s.end / hop)
        n = [rn[i] for i in range(i0, min(i1 + 1, len(rn)))]
        f = [rf[i] for i in range(i0, min(i1 + 1, len(rf)))]
        if not n or not f:
            out.append(float("nan"))
            continue
        out.append((statistics.fmean(n) - bn) - (statistics.fmean(f) - bf))
    return out


def attribute(segs: list[Seg], deltas: list[float],
              margin_db: float = 0.0) -> list[Seg]:
    """Label each VAD segment near/far from its channel delta.

    The diarisation shortcut: a lapel mic on the teacher rises far above its
    own baseline when the teacher speaks, and barely moves when someone else
    does. No speaker model, no clustering, no training.
    """
    return [Seg(s.start, s.end,
                "?" if d != d else ("near" if d > margin_db else "far"))
            for s, d in zip(segs, deltas)]


def attribute_frames(near: str, far: str, speech: list[bool],
                     hop: float = 0.1, margin_db: float = 0.0,
                     normalize: bool = True) -> list[bool]:
    """Per-FRAME near/far decision over the frames VAD calls speech.

    Attribution was originally done per VAD segment, which quietly made the
    headline metric depend on the segmentation parameter: lengthening segments
    to fix turn counts (section 4.1 of the spec) let a single segment span a
    speaker change, and talk-share error grew from 3.3 to 9.2 points. The two
    requirements are in direct conflict only because both were reading the
    same segments.

    They need not be. Which channel is louder is a property of a 100 ms frame,
    not of a turn. Deciding per frame makes talk share independent of
    `-vsd` altogether, so segmentation can be tuned for turn counts and
    attribution for accuracy, with no trade.
    """
    rn, _ = read_wav_rms(near, hop)
    rf, _ = read_wav_rms(far, hop)
    bn = statistics.median(rn) if normalize and rn else 0.0
    bf = statistics.median(rf) if normalize and rf else 0.0
    n = min(len(speech), len(rn), len(rf))
    return [bool(speech[i]) and ((rn[i] - bn) - (rf[i] - bf)) > margin_db
            for i in range(n)]


def seg_gold_near(segs: list[Seg], gold: list[Seg], target: str,
                  hop: float = 0.1, frac: float = 0.5) -> list[int]:
    """1 if most of this VAD segment's speech is the target speaker's.

    Needed because attribution must be scored per segment against a
    per-speaker truth, and AMI speakers overlap in time.
    """
    out = []
    for s in segs:
        tot = ov = 0.0
        for g in gold:
            inter = max(0.0, min(s.end, g.end) - max(s.start, g.start))
            if inter <= 0:
                continue
            tot += inter
            if g.who == target:
                ov += inter
        out.append(1 if tot > 0 and ov / tot >= frac else 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--meeting", help="validate against this AMI meeting")
    ap.add_argument("--root", default="data/ami/annotations/extracted")
    ap.add_argument("--vad-model", default="models/ggml-silero-v5.1.2.bin")
    ap.add_argument("--vad-bin",
                    default="vendor/whisper.cpp/build/bin/"
                            "whisper-vad-speech-segments")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-silence-ms", type=int,
                    default=DEFAULT_MIN_SILENCE_MS)
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--near", help="close mic, for the attribution test")
    ap.add_argument("--far", help="room mic, for the attribution test")
    ap.add_argument("--near-speaker", default="A",
                    help="which AMI speaker the near mic belongs to")
    ap.add_argument("--margin-db", type=float, default=0.0)
    ap.add_argument("--fixed-margin", type=float,
                    help="apply a margin chosen on ANOTHER meeting instead "
                         "of tuning one here -- the transfer test")
    ap.add_argument("--raw-db", action="store_true",
                    help="compare raw dB instead of referencing each channel "
                         "to its own median -- kept only to reproduce the "
                         "gain-offset failure described in channel_delta()")
    ap.add_argument("--outdir", default="results/vad")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.basename(args.audio).replace(".wav", "")
    segs = run_vad(args.audio, args.vad_model, args.vad_bin, args.threads,
                   os.path.join(args.outdir, f"{tag}.segments.txt"),
                   args.min_silence_ms, args.vad_threshold, args.force)
    span = max((s.end for s in segs), default=0.0)

    gold_segs: list[Seg] = []
    if args.meeting:
        m = Meeting(args.root, args.meeting)
        gold_segs = [Seg(u.start, u.end, u.speaker) for u in m.utterances()]
        span = max(span, max((s.end for s in gold_segs), default=0.0))

    print(f"=== {tag} ===")
    print(f"span {span:.0f}s  vad segments {len(segs)}")
    vm = metrics(segs, span)
    print("\nVAD-derived metrics")
    for k, v in vm.items():
        print(f"  {k:16} {v}")

    summary = {"audio": args.audio, "vad": vm}

    if gold_segs:
        gm = metrics(gold_segs, span, by_speaker=True)
        print("\nGold (manual segments) metrics")
        for k, v in gm.items():
            if k != "speakers":
                print(f"  {k:16} {v}")
        print("  speakers:")
        for w, d in gm["speakers"].items():
            print(f"    {w}  talk {d['talk_s']:7.1f}s  share {d['share']:6.1%}"
                  f"  turns {d['turns']:4d}")

        # Gold segments overlap (people talk at once); VAD on a mixed channel
        # cannot, so compare the UNION of gold speech against VAD speech.
        ag = agreement(frame_mask(gold_segs, span), frame_mask(segs, span))
        print("\nFrame agreement, VAD vs union of gold speech (100 ms frames)")
        for k, v in ag.items():
            print(f"  {k:16} "
                  f"{v if isinstance(v, int) else format(v, '.4f')}")
        gold_union = sum(
            1 for x in frame_mask(gold_segs, span) if x) * 0.1
        print(f"\n  gold speech (union of speakers) {gold_union:.0f}s"
              f"   vad speech {vm['speech_s']:.0f}s"
              f"   ratio {vm['speech_s'] / max(1e-9, gold_union):.3f}")
        summary["gold"] = gm
        summary["frame_agreement"] = ag

    if args.near and args.far:
        print(f"\n=== two-mic attribution: near={os.path.basename(args.near)}"
              f" far={os.path.basename(args.far)} ===")
        deltas = channel_delta(args.near, args.far, segs,
                               normalize=not args.raw_db)
        ok = [d for d in deltas if d == d]
        print(f"  channel delta (near-far, baseline-referenced): "
              f"p10 {sorted(ok)[len(ok) // 10]:.1f} dB  "
              f"p50 {statistics.median(ok):.1f} dB  "
              f"p90 {sorted(ok)[9 * len(ok) // 10]:.1f} dB")

        if gold_segs:
            tgt = args.near_speaker
            y = seg_gold_near(segs, gold_segs, tgt)
            scored = [(d, g) for d, g in zip(deltas, y) if d == d]
            # Separability FIRST, threshold second. An AUC near 0.5 means no
            # threshold exists and tuning one would be self-deception.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from eval_kev import auc, auc_ci
            a = auc(scored)
            lo, hi = auc_ci(scored)
            print(f"  separability of speaker {tgt} by channel delta: "
                  f"AUC {a:.3f} CI [{lo:.3f},{hi:.3f}]  "
                  f"({sum(y)}/{len(y)} segments are {tgt})")

            # Which speaker does the near mic belong to? The convention is
            # Lapel-N <-> the Nth speaker, so this is a check on the
            # CONVENTION, not a free choice of whichever scores best.
            per = []
            for w in sorted({g.who for g in gold_segs}):
                yy = seg_gold_near(segs, gold_segs, w)
                sc = [(d, g) for d, g in zip(deltas, yy) if d == d]
                per.append((w, auc(sc)))
            print("  per-speaker separability (mic<->speaker check): "
                  + "  ".join(f"{w} {v:.3f}" for w, v in per))
            top = max(per, key=lambda kv: kv[1])[0]
            if top != tgt:
                print(f"    NOTE: speaker {top} separates better than the "
                      f"assumed {tgt} -- near-mic mapping may be wrong")

            # best margin on this meeting, reported as the tuning artefact
            # it is -- one meeting, one mic placement.
            best = None
            for t in sorted({round(d, 1) for d, _ in scored}):
                lab = attribute(segs, deltas, t)
                am = agreement(
                    frame_mask([s for s, g in zip(segs, y) if g], span),
                    frame_mask([s for s in lab if s.who == "near"], span))
                if best is None or am["f1"] > best[1]["f1"]:
                    best = (t, am)
            if best:
                t, am = best
                print(f"  best margin {t:+.1f} dB -> frame f1 {am['f1']:.3f} "
                      f"precision {am['precision']:.3f} "
                      f"recall {am['recall']:.3f} iou {am['iou']:.3f}")
                lab = attribute(segs, deltas, t)
                near_s = sum(s.dur for s in lab if s.who == "near")
                far_s = sum(s.dur for s in lab if s.who == "far")
                gshare = sum(s.dur for s in gold_segs if s.who == tgt) / max(
                    1e-9, sum(s.dur for s in gold_segs))
                print(f"  talk share for {tgt}: measured "
                      f"{near_s / max(1e-9, near_s + far_s):.1%}  "
                      f"gold {gshare:.1%}")
                print(f"  ACCEPTANCE (spec section 5): f1 >= 0.85 -> "
                      f"{'PASS' if am['f1'] >= 0.85 else 'FAIL'}")

                if args.fixed_margin is not None:
                    lab2 = attribute(segs, deltas, args.fixed_margin)
                    am2 = agreement(
                        frame_mask([x for x, g in zip(segs, y) if g], span),
                        frame_mask([x for x in lab2 if x.who == "near"], span))
                    n2 = sum(x.dur for x in lab2 if x.who == "near")
                    f2 = sum(x.dur for x in lab2 if x.who == "far")
                    print(f"\n  TRANSFER TEST -- margin {args.fixed_margin:+.1f} "
                          f"dB carried from another meeting:")
                    print(f"    frame f1 {am2['f1']:.3f} "
                          f"precision {am2['precision']:.3f} "
                          f"recall {am2['recall']:.3f}")
                    print(f"    talk share for {tgt}: measured "
                          f"{n2 / max(1e-9, n2 + f2):.1%}  gold {gshare:.1%}")
                    print(f"    ACCEPTANCE: f1 >= 0.85 -> "
                          f"{'PASS' if am2['f1'] >= 0.85 else 'FAIL'}")
                    summary["transfer"] = {
                        "margin_db": args.fixed_margin, "frame": am2,
                        "measured_share": n2 / max(1e-9, n2 + f2),
                        "gold_share": gshare}
                summary["attribution"] = {
                    "auc": a, "auc_ci95": [lo, hi], "best_margin_db": t,
                    "frame": am, "gold_share": gshare,
                    "measured_share": near_s / max(1e-9, near_s + far_s),
                    "near": args.near, "far": args.far}

    p = os.path.join(args.outdir, f"{tag}.talk.json")
    json.dump(summary, open(p, "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
