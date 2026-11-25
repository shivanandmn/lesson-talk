"""Lesson analysis: voice activity + two-mic attribution + arithmetic.

The method and every default here come from measurements, documented in
SPEC-talk.md:

* Silero VAD (whisper.cpp) with 800 ms minimum silence: speech F1 0.957 /
  0.960 on two AMI meetings.
* Teacher vs pupil by per-FRAME energy comparison between a lapel mic and a
  room mic, each referenced to its own noise floor.
* The method misses about a quarter of teacher frames, which pulls every
  share towards the middle; compute() corrects for it, and one hand-coded
  lesson per mic setup calibrates the correction.

No language model, no speaker model, nothing that identifies a pupil.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
from dataclasses import dataclass

from . import audio
from .audio import HOP, AudioError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAD_BIN = os.environ.get(
    "LESSONTALK_VAD_BIN",
    os.path.join(ROOT, "vendor/whisper.cpp/build/bin/"
                       "whisper-vad-speech-segments"))
VAD_MODEL = os.environ.get(
    "LESSONTALK_VAD_MODEL", os.path.join(ROOT, "models/ggml-silero-v5.1.2.bin"))

MIN_SILENCE_MS = 800      # measured; the tool default of 100 is wrong here
BASELINE_PCT = 10         # each channel is referenced to its own noise floor.
                          # The median was used first and breaks when the
                          # teacher talks most of the lesson: the lapel's
                          # median frame is then teacher speech, and at a true
                          # 89% share it read 54% (tools/validate_ami.py).
MARGIN_DB = 4.0           # lapel-minus-room above this = teacher. Chosen for
                          # balanced accuracy on AMI (+3.8 / +4.2 dB).

# The method misses teacher speech far more often than it invents it: on AMI
# 76% of teacher frames are caught, and 3-8% of pupil frames are wrongly
# given to the teacher. Uncorrected, that pulls every share towards the
# middle (true 84% read as 62%). Measured share m relates to true share t by
#     m = q + (p - q) * t
# so t = (m - q) / (p - q). p is what calibration estimates per mic setup.
DEFAULT_TEACHER_RECALL = 0.76   # p
PUPIL_LEAK = 0.055              # q, mean of 0.031 and 0.079
SMOOTH_FRAMES = 5         # +/- 0.5 s majority vote before counting runs
STRETCH_BREAK_S = 3.0     # silence this long ends a teacher stretch
MIN_TURN_S = 2.0          # pupil speech shorter than this is not a turn,
                          # and does not interrupt a teacher stretch.
                          # Chosen on AMI IS1001c/ES2006c (tools/validate_ami.py):
                          # at 1 s, backchannels and misattributed frames cut
                          # the longest stretch by 30-55%; at 2 s it lands
                          # within 1% and 12%, and pupil turns within 4%/8%.

SEG_RE = re.compile(r"start\s*=\s*([\d.]+),\s*end\s*=\s*([\d.]+)")


# --------------------------------------------------------------- prepare
def run_vad(wav: str, threads: int | None = None) -> list[tuple[float, float]]:
    """Speech segments in seconds. The tool prints centiseconds."""
    for p, what in ((VAD_BIN, "VAD program"), (VAD_MODEL, "VAD model")):
        if not os.path.exists(p):
            raise RuntimeError(f"{what} not found at {p}")
    threads = threads or max(1, (os.cpu_count() or 2) // 2)
    r = subprocess.run(
        [VAD_BIN, "-vm", VAD_MODEL, "-f", wav, "-t", str(threads), "-np",
         "-vsd", str(MIN_SILENCE_MS), "-vt", "0.5"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"speech detection failed: {r.stderr[-800:]}")
    return [(float(m.group(1)) / 100.0, float(m.group(2)) / 100.0)
            for m in SEG_RE.finditer(r.stdout)]


def prepare(workdir: str, *, stereo: str | None = None,
            teacher_channel: int = 0, near: str | None = None,
            far: str | None = None) -> dict:
    """The expensive, margin-independent part: energy envelopes, alignment,
    VAD. Cached to `workdir/frames.json`, so calibrating or swapping channels
    later is instant.

    Either `stereo` (one recorder, teacher lapel on one channel, room mic on
    the other) or `near` + `far` (two recorders).
    """
    os.makedirs(workdir, exist_ok=True)
    warnings: list[str] = []
    if stereo:
        wi = audio.info(stereo)
        if wi.channels < 2:
            raise AudioError(
                "this recording has one channel. The tool needs two: the "
                "teacher's lapel mic and a room mic, either as one stereo "
                "file or as two separate files.")
        env = audio.envelopes(stereo)
        rn, rf = env[teacher_channel], env[1 - teacher_channel]
        lag, corr, vad_wav = 0, None, stereo
    elif near and far:
        en, ef = audio.envelopes(near), audio.envelopes(far)
        rn, rf = en[0], ef[0]
        lag, corr = audio.align(rn, rf)
        if corr < 0.3:
            warnings.append(
                f"the two recordings could not be lined up confidently "
                f"(correlation {corr:.2f}). Check they are from the same "
                "lesson.")
        if abs(lag) * HOP >= 29.9:
            warnings.append("the recorders were started 30 s or more apart; "
                            "alignment may be wrong. Start them together.")
        if lag >= 0:
            rf = rf[lag:]
        else:
            rn = rn[-lag:]
        vad_wav = os.path.join(workdir, "mix.wav")
        audio.write_stereo(near, far, vad_wav, lag)
    else:
        raise ValueError("give either a stereo file or near + far files")

    n = min(len(rn), len(rf))
    rn, rf = rn[:n], rf[:n]
    segs = run_vad(vad_wav)
    frames = {
        "version": 1,
        "hop": HOP,
        "near_db": [round(v, 2) for v in rn],
        "far_db": [round(v, 2) for v in rf],
        "vad_segments": [[round(a, 2), round(b, 2)] for a, b in segs],
        "lag_frames": lag,
        "align_corr": None if corr is None else round(corr, 3),
        "teacher_channel": teacher_channel if stereo else None,
        "playback": os.path.basename(vad_wav),
        "warnings": warnings,
    }
    with open(os.path.join(workdir, "frames.json"), "w") as fh:
        json.dump(frames, fh)
    return frames


def load_frames(workdir: str) -> dict:
    with open(os.path.join(workdir, "frames.json")) as fh:
        return json.load(fh)


def swap_channels(frames: dict) -> dict:
    """For a stereo file recorded with the lapel on the other side."""
    f = dict(frames)
    f["near_db"], f["far_db"] = frames["far_db"], frames["near_db"]
    if f.get("teacher_channel") is not None:
        f["teacher_channel"] = 1 - f["teacher_channel"]
    return f


# --------------------------------------------------------------- frames
def speech_mask(segs, n: int, hop: float = HOP) -> list[bool]:
    m = [False] * n
    for a, b in segs:
        for i in range(max(0, int(a / hop)), min(n, int(b / hop) + 1)):
            m[i] = True
    return m


def _pct(x: list[float], pct: float) -> float:
    s = sorted(x)
    return s[int(pct / 100 * (len(s) - 1))] if s else 0.0


DEAD_DB = -90.0  # a real mic never sits this low; this is a cut-out


def alive(frames: dict) -> list[bool]:
    """False where either mic is digitally silent. AMI ES2002a's lapel cuts
    out for 7% of the meeting; read as "lapel quiet, so a pupil", those
    frames wrecked its talk share."""
    return [a > DEAD_DB and b > DEAD_DB
            for a, b in zip(frames["near_db"], frames["far_db"])]


def usable_speech(frames: dict, hop: float = HOP) -> list[bool]:
    al = alive(frames)
    sp = speech_mask(frames["vad_segments"], len(al), hop)
    return [x and y for x, y in zip(sp, al)]


def deltas(frames: dict) -> list[float]:
    rn, rf = frames["near_db"], frames["far_db"]
    al = alive(frames)
    bn = _pct([v for v, ok in zip(rn, al) if ok] or rn, BASELINE_PCT)
    bf = _pct([v for v, ok in zip(rf, al) if ok] or rf, BASELINE_PCT)
    return [(a - bn) - (b - bf) for a, b in zip(rn, rf)]


def labels(speech: list[bool], delta: list[float], margin_db: float,
           smooth: int = SMOOTH_FRAMES) -> tuple[list[int], list[int]]:
    """Per-frame 0 = silence, 1 = teacher, 2 = pupil(s).

    Returns (raw, smoothed). Talk share is computed on raw frame decisions,
    which is what was validated. Turn and stretch counting uses a majority
    vote over +/- `smooth` frames, so a single 100 ms flicker is not counted
    as an interruption.
    """
    raw = [0 if not s else (1 if d > margin_db else 2)
           for s, d in zip(speech, delta)]
    n = len(raw)
    pre_t = [0] * (n + 1)
    pre_p = [0] * (n + 1)
    for i, v in enumerate(raw):
        pre_t[i + 1] = pre_t[i] + (v == 1)
        pre_p[i + 1] = pre_p[i] + (v == 2)
    sm = []
    for i, v in enumerate(raw):
        if v == 0:
            sm.append(0)
            continue
        lo, hi = max(0, i - smooth), min(n, i + smooth + 1)
        t, p = pre_t[hi] - pre_t[lo], pre_p[hi] - pre_p[lo]
        sm.append(1 if t > p else 2 if p > t else v)
    return raw, sm


def runs(lab: list[int], who: int, bridge_frames: int,
         min_frames: int = 1) -> list[tuple[int, int]]:
    """[start, end) runs of `who`, bridging silences up to `bridge_frames`.
    Any frame of the other speaker ends a run."""
    out = []
    start = last = None
    gap = 0
    for i, v in enumerate(lab):
        if v == who:
            if start is None:
                start = i
            last, gap = i, 0
        elif start is not None:
            gap += 1
            if v != 0 or gap > bridge_frames:
                out.append((start, last + 1))
                start = None
    if start is not None:
        out.append((start, last + 1))
    return [r for r in out if r[1] - r[0] >= min_frames]


def drop_short(lab: list[int], who: int, min_frames: int) -> list[int]:
    """Relabel runs of `who` shorter than `min_frames` as silence, so a
    backchannel does not count as an interruption."""
    out = list(lab)
    i, n = 0, len(lab)
    while i < n:
        if lab[i] != who:
            i += 1
            continue
        j = i
        while j < n and lab[j] == who:
            j += 1
        if j - i < min_frames:
            out[i:j] = [0] * (j - i)
        i = j
    return out


def teacher_stretches(sm: list[int], hop: float = HOP) -> list[tuple[int, int]]:
    return runs(drop_short(sm, 2, int(MIN_TURN_S / hop)), 1,
                int(STRETCH_BREAK_S / hop))


def pupil_turns(sm: list[int], hop: float = HOP) -> list[tuple[int, int]]:
    return runs(sm, 2, int(MIN_SILENCE_MS / 1000 / hop), int(MIN_TURN_S / hop))


def share_at(speech, delta, margin_db: float) -> float:
    t = sum(1 for s, d in zip(speech, delta) if s and d > margin_db)
    sp = sum(speech)
    return t / sp if sp else 0.0


def correct(measured: float, recall: float, leak: float = PUPIL_LEAK) -> float:
    """Undo the pull towards the middle: true share from measured share."""
    if recall - leak <= 0.05:
        return measured
    return min(1.0, max(0.0, (measured - leak) / (recall - leak)))


def rescale(part: float, raw_total: float, corrected_total: float) -> float:
    """Correct a short window using the whole lesson's correction.

    Applying `correct()` to a ten-minute window on its own pushed teacher-heavy
    windows past 100% and clamped them there, while the timeline plainly
    showed pupil speech in them. Instead the lesson-wide correction is applied
    as an odds ratio: odds(window) * odds(corrected) / odds(raw). That stays
    strictly between 0 and 100%, keeps the windows in the same order, and
    leaves a window that looks like the whole lesson equal to the headline.
    """
    def odds(x):
        x = min(max(x, 1e-4), 1 - 1e-4)
        return x / (1 - x)
    if part <= 0.0 or part >= 1.0:
        return part
    o = odds(part) * odds(corrected_total) / odds(raw_total)
    return o / (1 + o)


def calibrate(frames: dict, teacher_share: float) -> float:
    """The teacher recall p for this mic setup, from one lesson whose teacher
    share was hand-coded: m = q + (p - q) t  =>  p = q + (m - q) / t.

    One number from the user fixes one unknown. q (pupil speech wrongly
    given to the teacher) is small and stays at its measured default."""
    if not 0.0 < teacher_share < 1.0:
        raise ValueError("teacher share must be between 0 and 100%")
    d = deltas(frames)
    sp = usable_speech(frames)
    m = share_at(sp, d, MARGIN_DB)
    p = PUPIL_LEAK + (m - PUPIL_LEAK) / teacher_share
    if not PUPIL_LEAK + 0.1 < p <= 1.05:
        raise ValueError(
            f"that does not fit this recording: the tool measured "
            f"{m:.0%} teacher talk before correction, which cannot come from "
            f"{teacher_share:.0%}. Check the channels are the right way round "
            "and the hand-coded figure is the teacher's share of all talk.")
    return round(min(p, 1.0), 4)


# --------------------------------------------------------------- metrics
@dataclass
class Settings:
    teacher_recall: float = DEFAULT_TEACHER_RECALL
    calibrated: bool = False


def compute(frames: dict, settings: Settings | None = None) -> dict:
    s = settings or Settings()
    hop = frames["hop"]
    d = deltas(frames)
    n = len(d)
    speech = usable_speech(frames, hop)
    dead = n - sum(alive(frames))
    raw, sm = labels(speech, d, MARGIN_DB)

    dur = n * hop
    teacher_s = raw.count(1) * hop
    pupil_s = raw.count(2) * hop
    speech_s = teacher_s + pupil_s
    raw_share = teacher_s / speech_s if speech_s else 0.0
    tshare = correct(raw_share, s.teacher_recall)
    teacher_s, pupil_s = tshare * speech_s, (1 - tshare) * speech_s

    stretches = teacher_stretches(sm, hop)
    p_turns = pupil_turns(sm, hop)

    # timeline bins: about 240 bars whatever the lesson length
    bin_s = max(5.0, round(dur / 240 / 5) * 5) if dur else 5.0
    bf = int(bin_s / hop)
    timeline = []
    for i in range(0, n, bf):
        chunk = raw[i:i + bf]
        k = len(chunk) or 1
        timeline.append([round(chunk.count(1) / k, 3),
                         round(chunk.count(2) / k, 3)])

    # share by ten-minute phase
    phases = []
    pf = int(600 / hop)
    for i in range(0, n, pf):
        chunk = raw[i:i + pf]
        t, p = chunk.count(1), chunk.count(2)
        ps = rescale(t / (t + p), raw_share, tshare) if t + p else None
        if len(chunk) * hop < 120 and phases:  # fold a short tail in
            phases[-1]["end_s"] = round(dur, 1)
            continue
        phases.append({"start_s": round(i * hop, 1),
                       "end_s": round(min(n, i + pf) * hop, 1),
                       "teacher_share": None if ps is None else round(ps, 4)})

    gaps = [b[0] - a[1] for a, b in zip(frames["vad_segments"],
                                        frames["vad_segments"][1:])]

    def spans(rs, k):
        top = sorted(rs, key=lambda r: r[0] - r[1])[:k]
        return [{"start_s": round(a * hop, 1), "dur_s": round((b - a) * hop, 1)}
                for a, b in sorted(top)]

    sd = statistics.pstdev(d) if d else 0.0
    warnings = list(frames.get("warnings", []))
    if dead * hop >= 10 and dead / max(1, n) >= 0.01:
        warnings.append(
            f"a microphone was silent (cut out) for {dead * hop / 60:.1f} min "
            f"({dead / n:.0%} of the recording). Those moments are left out "
            "of every figure. Check the mic's battery and cable.")
    if sd < 1.0:
        warnings.append("the two channels are nearly identical, so teacher "
                        "and pupil speech cannot be told apart. Check the "
                        "lapel mic was plugged into its own input.")
    # The lapel jumps far above its noise floor when its wearer speaks, and
    # the room mic never gets far ahead of the lapel, because the lapel hears
    # the room too. So, referenced to each channel's floor, the top of the
    # delta during speech is large and the bottom small; swapped, mirrored.
    # On AMI p98 is +14..+22 dB and p2 -3..-4 dB at 47%, 75% and 85% teacher
    # share. (A median-centred version of this test misfired once the
    # teacher dominated: the median moves into the teacher's frames.)
    sd_speech = sorted(v for v, sp_ in zip(d, speech) if sp_)
    if len(sd_speech) > 100:
        k = len(sd_speech)
        top, bottom = sd_speech[49 * k // 50], -sd_speech[k // 50]
        if bottom > 1.5 * max(top, 0.5):
            warnings.append(
                "the teacher and room channels look the wrong way round "
                "(the 'teacher' channel does not jump when someone speaks "
                "close to it). Use Swap teacher / room and check the card "
                "again.")
    if speech_s and not 0.05 < tshare < 0.98:
        warnings.append("teacher share is extreme; if this lesson was not "
                        "like that, the channels may be the wrong way round.")

    return {
        "duration_s": round(dur, 1),
        "settings": {"teacher_recall": s.teacher_recall,
                     "calibrated": s.calibrated, "margin_db": MARGIN_DB,
                     "raw_teacher_share": round(raw_share, 4),
                     "min_silence_ms": MIN_SILENCE_MS},
        "tier1": {
            "teacher_share": round(tshare, 4),
            "teacher_s": round(teacher_s, 1),
            "pupil_s": round(pupil_s, 1),
            "silence_s": round(dur - speech_s - dead * hop, 1),
            "mic_dropout_s": round(dead * hop, 1),
            "longest_teacher_stretch_s": round(
                max((b - a for a, b in stretches), default=0) * hop, 1),
            "pupil_turns": len(p_turns),
        },
        # over-reported by construction: every VAD miss splits a run and
        # makes a phantom pause (SPEC-talk.md 4.1, tools/validate_ami.py)
        "tier2": {
            "speech_runs": len(frames["vad_segments"]),
            "pauses_over_3s": sum(1 for g in gaps if g >= 3.0),
        },
        "phases": phases,
        "timeline": {"bin_s": bin_s, "bins": timeline},
        "listen": {
            "teacher_stretches": spans(stretches, 3),
            "pupil_turns": spans(p_turns, 3),
        },
        "alignment": {"lag_s": round(frames.get("lag_frames", 0) * hop, 1),
                      "corr": frames.get("align_corr")},
        "warnings": warnings,
    }
