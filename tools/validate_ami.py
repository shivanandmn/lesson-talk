#!/usr/bin/env python3
"""End-to-end check of the product engine against AMI's hand-coded segments.

Runs the code path the app runs (`lessontalk.engine.prepare` + `compute`),
feeding the lapel (speaker A, playing the teacher) and the far-field array as
two separate "recorders", and compares each number on the card with the same
number computed from AMI's manual per-speaker segments.

AMI meetings are ~47% "teacher". Lessons are usually far more teacher-heavy,
so each meeting is also re-run with most of the other speakers' stretches
cut out, raising the true teacher share to ~75% and ~85%. That is the case
the original median baseline failed on.

    python3 tools/validate_ami.py [IS1001c ES2006c]
"""

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ami import Meeting                      # noqa: E402
from lessontalk import engine                # noqa: E402

HOP = 0.1


def gold_labels(meeting: str, n: int) -> list[int]:
    m = Meeting(os.path.join(ROOT, "data/ami/annotations/extracted"), meeting)
    a = [False] * n
    o = [False] * n
    for u in m.utterances():
        tgt = a if u.speaker == "A" else o
        for i in range(max(0, int(u.start / HOP)), min(n, int(u.end / HOP) + 1)):
            tgt[i] = True
    # overlap counts as the teacher, as a listener would hear it
    return [1 if x else (2 if y else 0) for x, y in zip(a, o)]


def gold_share(g: list[int]) -> float:
    return g.count(1) / max(1, sum(1 for v in g if v))


def segs_from_mask(m: list[bool]) -> list[list[float]]:
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            out.append([i * HOP, (j - 1) * HOP])
            i = j
        else:
            i += 1
    return out


def teacher_heavy(frames: dict, g: list[int], keep: float, seed: int = 0):
    """Drop whole runs of other-speaker frames, keeping a fraction `keep`."""
    rnd = random.Random(seed)
    n = len(g)
    mask = [True] * n
    i = 0
    while i < n:
        if g[i] == 2:
            j = i
            while j < n and g[j] == 2:
                j += 1
            if rnd.random() > keep:
                mask[i:j] = [False] * (j - i)
            i = j
        else:
            i += 1
    idx = [k for k in range(n) if mask[k]]
    vs = engine.speech_mask(frames["vad_segments"], n)
    sub = {"hop": HOP,
           "near_db": [frames["near_db"][k] for k in idx],
           "far_db": [frames["far_db"][k] for k in idx],
           "vad_segments": segs_from_mask([vs[k] for k in idx]),
           "lag_frames": 0}
    return sub, [g[k] for k in idx]


def load(mt: str) -> dict:
    wd = os.path.join(ROOT, "results/validate", mt)
    if os.path.exists(os.path.join(wd, "frames.json")):
        return engine.load_frames(wd)
    return engine.prepare(
        wd, near=os.path.join(ROOT, f"data/ami/audio/{mt}.Lapel-0.wav"),
        far=os.path.join(ROOT, f"data/ami/audio/{mt}.Array1-01.wav"))


def main(meetings):
    M = {}
    for mt in meetings:
        f = load(mt)
        g = gold_labels(mt, len(f["near_db"]))
        # Calibrate as a school would: on a typical, teacher-heavy lesson
        # (~75%). Calibrating on the balanced 47% meeting and extrapolating
        # to 89% was measured to be off by up to 12 points.
        cf, cg = teacher_heavy(f, g, 0.3)
        M[mt] = {"frames": f, "gold": g,
                 "recall": engine.calibrate(cf, gold_share(cg))}

    out = {}
    for mt, d in M.items():
        f, g = d["frames"], d["gold"]
        r = engine.compute(f, engine.Settings(d["recall"], True))["tier1"]
        g_stretch = max((b - a for a, b in engine.teacher_stretches(g)),
                        default=0) * HOP
        g_turns = len(engine.pupil_turns(g))
        print(f"\n=== {mt} ===  recorders lined up at "
              f"{f['lag_frames'] * HOP:+.1f} s (corr {f['align_corr']})")
        print(f"  teacher recall {d['recall']:.3f}, calibrated on the ~75% "
              "version")
        rows = []
        print(f"  {'true share':>10} {'uncalibrated':>13} {'calibrated':>11}"
              + "".join(f" {'cal. on ' + o:>16}" for o in M if o != mt))
        for keep in (1.0, 0.3, 0.12):
            sub, gs = (f, g) if keep == 1.0 else teacher_heavy(f, g, keep)
            gt = gold_share(gs)
            row = {"keep": keep, "gold": gt,
                   "uncalibrated": engine.compute(sub)["tier1"]["teacher_share"],
                   "calibrated": engine.compute(
                       sub, engine.Settings(d["recall"], True)
                   )["tier1"]["teacher_share"]}
            for o in M:
                if o != mt:
                    row["cal_on_" + o] = engine.compute(
                        sub, engine.Settings(M[o]["recall"], True)
                    )["tier1"]["teacher_share"]
            rows.append(row)
            print(f"  {gt:>10.1%} {row['uncalibrated']:>13.1%} "
                  f"{row['calibrated']:>11.1%}"
                  + "".join(f" {row['cal_on_' + o]:>16.1%}"
                            for o in M if o != mt))
        print(f"  longest teacher stretch  gold {g_stretch:6.1f} s  measured "
              f"{r['longest_teacher_stretch_s']:6.1f} s  "
              f"({(r['longest_teacher_stretch_s'] - g_stretch) / g_stretch:+.0%})")
        print(f"  pupil turns >= 2 s       gold {g_turns:6d}    measured "
              f"{r['pupil_turns']:6d}    "
              f"({(r['pupil_turns'] - g_turns) / g_turns:+.0%})")
        out[mt] = {"recall": d["recall"], "share": rows,
                   "stretch": [g_stretch, r["longest_teacher_stretch_s"]],
                   "pupil_turns": [g_turns, r["pupil_turns"]]}

    errs = [abs(r[k] - r["gold"]) for v in out.values() for r in v["share"]
            for k in r if k not in ("keep", "gold")]
    print(f"\nworst talk-share error across every row: {max(errs) * 100:.1f} "
          "points")
    os.makedirs(os.path.join(ROOT, "results/validate"), exist_ok=True)
    with open(os.path.join(ROOT, "results/validate/summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["IS1001c", "ES2006c"])
