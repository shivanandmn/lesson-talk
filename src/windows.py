"""
Turn AMI meetings into labelled decision windows.

Window shape is dictated by the CPU profile (PROFILE.md), not by convenience:

  * Cost is linear in TOTAL tokens, and the state is re-encoded once per
    question, so per-decision cost is `questions x state_length`. An
    append-only window that grows with the meeting therefore has unbounded
    cost and is not viable. We use a SLIDING window over the last N words.
  * Prefix KV reuse is worth nothing (every question re-encodes the state),
    so there is no cache argument for append-only either.

So: a fixed-size sliding window of the last `max_words` words, advanced at
utterance boundaries -- which is also what a streaming pipeline can actually
produce, since it never has to re-read the whole meeting.

Labels come from AMI's manual `decision` annotation. A window is positive when
it overlaps a decision span by at least `min_overlap` words. Overlap is
computed on WORD IDS, not timestamps, because the annotation itself is defined
over word ids and speakers overlap in time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import os
import sys

# -I (isolated) implies -P, so this script's own directory is not on
# sys.path. Re-add only src/ -- never the data directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ami import Meeting


@dataclass
class Window:
    meeting: str
    series: str
    idx: int
    start: float
    end: float
    n_words: int
    n_utterances: int
    speakers: str
    text: str
    decision: int           # the label: 1 if a decision is being made here
    overlap_words: int
    decision_ids: str


def series_of(meeting: str) -> str:
    """'ES2002a' -> 'ES2002'. Meetings a-d share participants and project, so
    the train/test split must be made on the series, not the meeting."""
    return meeting[:-1] if meeting and meeting[-1].isalpha() else meeting


def build_windows(root: str, meeting: str, max_words: int = 90,
                  advance_words: int = 45, min_overlap: int = 5,
                  min_words: int = 25,
                  stats: dict | None = None) -> list[Window]:
    """Windows for one AMI meeting, labelled from the manual annotation."""
    m = Meeting(root, meeting)
    utts = m.utterances()
    decisions = m.decisions()

    # word-id -> decision ids covering it
    dec_of: dict[str, list[str]] = {}
    for d in decisions:
        for wid in d["word_ids"]:
            dec_of.setdefault(wid, []).append(d["id"])

    return windows_from_utterances(
        utts, meeting=meeting, dec_of=dec_of, max_words=max_words,
        advance_words=advance_words, min_overlap=min_overlap,
        min_words=min_words, n_decisions=len(decisions), stats=stats)


def windows_from_utterances(utts: list, *, meeting: str,
                            dec_of: dict[str, list[str]] | None = None,
                            max_words: int = 90, advance_words: int = 45,
                            min_overlap: int = 5, min_words: int = 25,
                            n_decisions: int = 0,
                            stats: dict | None = None) -> list[Window]:
    """The chunking rule itself, over any utterance-like sequence.

    Separated from AMI parsing so the live pipeline chunks ASR output with
    exactly this code rather than a re-implementation that drifts. Anything
    with `.speaker`, `.start`, `.end`, `.text` and `.words` (each word having
    `.id` and `.is_word`) will do.

    `dec_of` maps word id -> decision ids. ASR words are in no annotation, so
    the caller passes a mapping it built by TIME overlap, or none at all for
    an unlabelled live run.
    """
    dec_of = dec_of or {}

    # Sequential scan with a coverage invariant.
    #
    # Earlier attempts emitted as a side effect of buffer state (gating on
    # word count broke coverage; flushing on every overflow tripled the
    # window count), and a fixed start grid every `advance_words` words left
    # ~2-3% of words uncovered: an utterance too long to fit in the previous
    # window is skipped, and if it contains no grid point it never becomes a
    # start either.
    #
    # Here each window takes whole utterances from its start until it reaches
    # `max_words`, and the next start is the first utterance that is at least
    # `advance_words` words further on, CAPPED at the first utterance this
    # window did not take. That cap is the invariant: the next window always
    # begins at or before the first uncovered utterance, so every word lands
    # in at least one window by construction, whatever the utterance lengths.
    counts = [sum(1 for w in u.words if w.is_word) for u in utts]
    keep = [i for i, c in enumerate(counts) if c > 0]
    utts = [utts[i] for i in keep]
    counts = [counts[i] for i in keep]

    # cum[i] = words before utterance i
    cum = [0]
    for c in counts:
        cum.append(cum[-1] + c)

    out: list[Window] = []
    emitted_ids: set[str] = set()
    idx = 0

    def emit(buf: list) -> None:
        nonlocal idx
        words = [w for u in buf for w in u.words if w.is_word]
        if not words:
            return
        wids = {w.id for w in words}
        emitted_ids.update(wids)
        hits = sorted({did for wid in wids for did in dec_of.get(wid, [])})
        overlap = sum(1 for wid in wids if wid in dec_of)
        text = "\n".join(f"{u.speaker}: {u.text}" for u in buf if u.text)
        out.append(Window(
            meeting=meeting,
            series=series_of(meeting),
            idx=idx,
            start=round(min(u.start for u in buf), 2),
            end=round(max(u.end for u in buf), 2),
            n_words=len(words),
            n_utterances=len(buf),
            speakers="".join(sorted({u.speaker for u in buf})),
            text=text,
            decision=1 if overlap >= min_overlap else 0,
            overlap_words=overlap,
            decision_ids=",".join(hits),
        ))
        idx += 1

    s = 0
    while s < len(utts):
        # take whole utterances until max_words; always at least one, so an
        # utterance longer than max_words becomes a window on its own.
        k, n = s, 0
        while k < len(utts) and (n == 0 or n + counts[k] <= max_words):
            n += counts[k]
            k += 1
        cand = utts[s:k]
        cand_ids = {w.id for u in cand for w in u.words if w.is_word}
        # Drop a short window only when its words are ALREADY covered -- never
        # on the assumption that they are. A window can fall short of
        # min_words mid-meeting when the next utterance is too long to fit,
        # not only at the tail.
        if n >= min_words or not cand_ids <= emitted_ids:
            emit(cand)
        if k >= len(utts):
            break
        # next start: >= advance_words further on, but never past k, the first
        # utterance this window did not take.
        s2 = s + 1
        while s2 < k and cum[s2] - cum[s] < advance_words:
            s2 += 1
        s = s2

    if stats is not None:
        # Coverage guards against the emission bug above: every word that is
        # part of a decision must appear in at least one emitted window, or it
        # is unlabelable and silently counted as negative.
        all_utt_ids = {w.id for u in utts for w in u.words if w.is_word}
        dec_word_ids = {wid for wid in dec_of if wid in all_utt_ids}
        stats["n_utt_words"] = len(all_utt_ids)
        stats["word_coverage"] = (
            len(emitted_ids & all_utt_ids) / len(all_utt_ids)
            if all_utt_ids else 1.0)
        stats["decision_word_coverage"] = (
            len(emitted_ids & dec_word_ids) / len(dec_word_ids)
            if dec_word_ids else 1.0)
        stats["n_decisions"] = n_decisions
        stats["n_decision_words"] = len(dec_word_ids)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="annotations/extracted dir")
    ap.add_argument("--meetings", nargs="*", help="default: all with decisions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-words", type=int, default=90)
    ap.add_argument("--advance-words", type=int, default=45)
    ap.add_argument("--min-overlap", type=int, default=5)
    ap.add_argument("--min-words", type=int, default=25)
    args = ap.parse_args()

    import os
    if args.meetings:
        meetings = args.meetings
    else:
        d = os.path.join(args.root, "decision", "manual")
        meetings = sorted(f.split(".")[0] for f in os.listdir(d)
                          if f.endswith(".decision.xml"))

    n_pos = n_tot = 0
    worst_cov = worst_dcov = 1.0
    with open(args.out, "w") as fh:
        for mt in meetings:
            st: dict = {}
            ws = build_windows(args.root, mt, args.max_words,
                               args.advance_words, args.min_overlap,
                               args.min_words, stats=st)
            pos = sum(w.decision for w in ws)
            n_pos += pos; n_tot += len(ws)
            cov = st.get("word_coverage", 1.0)
            dcov = st.get("decision_word_coverage", 1.0)
            flag = "" if cov > 0.999 and dcov > 0.999 else "   <-- GAP"
            worst_cov = min(worst_cov, cov)
            worst_dcov = min(worst_dcov, dcov)
            print(f"{mt}: {len(ws):4d} windows, {pos:3d} positive "
                  f"({pos / max(1, len(ws)):5.1%})  "
                  f"cov {cov:.3f} deccov {dcov:.3f} "
                  f"({st.get('n_decisions',0)} dec){flag}", flush=True)
            for w in ws:
                fh.write(json.dumps(asdict(w)) + "\n")
    print(f"\nTOTAL: {n_tot} windows, {n_pos} positive "
          f"({n_pos / max(1, n_tot):.1%}) -> {args.out}")
    print(f"worst word coverage: {worst_cov:.4f}   "
          f"worst decision-word coverage: {worst_dcov:.4f}")
    if worst_cov < 0.999 or worst_dcov < 0.999:
        print("  ERROR: windows do not cover every word. Labels derived from "
              "uncovered words are silently lost.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
