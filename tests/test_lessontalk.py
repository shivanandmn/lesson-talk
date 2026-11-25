"""Tests for the lesson-talk engine. Standard library unittest:

    python3 -m unittest discover -s tests -v

The last class runs the full pipeline on AMI audio and checks it against
hand-coded segments; it is skipped if the AMI files are not present.
"""

import array
import math
import os
import sys
import tempfile
import unittest
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lessontalk import audio, engine  # noqa: E402


def write_wav(path, chans, rate=16000, width=2):
    """chans: list of lists of floats in [-1, 1]."""
    n = len(chans[0])
    with wave.open(path, "wb") as w:
        w.setnchannels(len(chans))
        w.setsampwidth(width)
        w.setframerate(rate)
        full = 2 ** (8 * width - 1) - 1
        out = bytearray()
        for i in range(n):
            for c in chans:
                out += int(c[i] * full).to_bytes(width, "little", signed=True)
        w.writeframes(bytes(out))


def tone(seconds, amp, rate=16000, f=220.0):
    return [amp * math.sin(2 * math.pi * f * i / rate)
            for i in range(int(seconds * rate))]


class Audio(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_envelope_level_and_channels(self):
        p = os.path.join(self.d, "s.wav")
        write_wav(p, [tone(1, 0.5), tone(1, 0.05)])
        near, far = audio.envelopes(p)
        self.assertEqual(len(near), 10)
        # a sine of amplitude A has RMS A/sqrt(2): 0.5 -> -9.0 dBFS
        self.assertAlmostEqual(near[3], -9.03, delta=0.1)
        self.assertAlmostEqual(near[3] - far[3], 20.0, delta=0.1)

    def test_24bit_matches_16bit(self):
        a, b = os.path.join(self.d, "a.wav"), os.path.join(self.d, "b.wav")
        x = tone(0.5, 0.3)
        write_wav(a, [x], width=2)
        write_wav(b, [x], width=3)
        ea, eb = audio.envelopes(a)[0], audio.envelopes(b)[0]
        for u, v in zip(ea, eb):
            self.assertAlmostEqual(u, v, delta=0.01)

    def test_rejects_non_wav(self):
        p = os.path.join(self.d, "x.wav")
        with open(p, "w") as fh:
            fh.write("not audio")
        with self.assertRaises(audio.AudioError) as cm:
            audio.info(p)
        self.assertNotIn(self.d, str(cm.exception))  # no paths in messages

    def test_align_finds_offset(self):
        # an irregular on/off pattern, delayed by 2.3 s in the second copy
        import random
        rnd = random.Random(1)
        env = [rnd.choice([-60.0, -20.0]) for _ in range(1200)]
        shifted = [-60.0] * 23 + env
        lag, corr = audio.align(env, shifted)
        self.assertEqual(lag, 23)
        self.assertGreater(corr, 0.99)

    def test_write_stereo_shifts_and_interleaves(self):
        n, f = os.path.join(self.d, "n.wav"), os.path.join(self.d, "f.wav")
        o = os.path.join(self.d, "o.wav")
        write_wav(n, [tone(2, 0.5)])
        write_wav(f, [[0.0] * 1600 + tone(2, 0.1)])  # far started 0.1 s early
        audio.write_stereo(n, f, o, lag_frames=1)
        wi = audio.info(o)
        self.assertEqual((wi.channels, wi.width), (2, 2))
        self.assertAlmostEqual(wi.seconds, 2.0, delta=0.01)
        l, r = audio.envelopes(o)
        self.assertAlmostEqual(l[0] - r[0], 14.0, delta=0.2)  # 0.5 vs 0.1

    def test_mismatched_rates_refused(self):
        n, f = os.path.join(self.d, "n.wav"), os.path.join(self.d, "f.wav")
        write_wav(n, [tone(1, .5)], rate=16000)
        write_wav(f, [tone(1, .5, rate=8000)], rate=8000)
        with self.assertRaises(audio.AudioError):
            audio.write_stereo(n, f, os.path.join(self.d, "o.wav"), 0)


class Engine(unittest.TestCase):
    def test_runs_bridge_silence_but_not_other_speaker(self):
        lab = [1, 1, 0, 0, 1, 2, 1, 0, 0, 0, 0, 1]
        self.assertEqual(engine.runs(lab, 1, 2), [(0, 5), (6, 7), (11, 12)])

    def test_backchannel_does_not_break_stretch(self):
        hop = engine.HOP
        lab = [1] * 300 + [2] * 10 + [1] * 300   # 1 s "mm-hm" mid-stretch
        self.assertEqual(engine.teacher_stretches(lab, hop), [(0, 610)])
        lab = [1] * 300 + [2] * 30 + [1] * 300   # a 3 s pupil turn
        self.assertEqual(len(engine.teacher_stretches(lab, hop)), 2)
        self.assertEqual(len(engine.pupil_turns(lab, hop)), 1)

    def test_smoothing_removes_single_frame_flicker(self):
        speech = [True] * 21
        delta = [5.0] * 21
        delta[10] = -5.0
        raw, sm = engine.labels(speech, delta, 0.0)
        self.assertEqual(raw[10], 2)
        self.assertEqual(sm[10], 1)

    def _frames(self):
        # 70 s, floors at -60 dB. Teacher 40 s: lapel +30, room +15 above
        # floor. Pupils 20 s: lapel +10, room +13 (AMI: the room mic gets
        # only ~3 dB ahead of the lapel for others' speech). Then silence.
        near = [-30.0] * 400 + [-50.0] * 200 + [-60.0] * 100
        far = [-45.0] * 400 + [-47.0] * 200 + [-60.0] * 100
        return {"hop": 0.1, "near_db": near, "far_db": far,
                "vad_segments": [[0.0, 59.9]], "lag_frames": 0}

    def test_compute_shares(self):
        r = engine.compute(self._frames())
        raw = r["settings"]["raw_teacher_share"]
        self.assertAlmostEqual(raw, 400 / 600, delta=.01)
        p, q = engine.DEFAULT_TEACHER_RECALL, engine.PUPIL_LEAK
        self.assertAlmostEqual(r["tier1"]["teacher_share"],
                               (raw - q) / (p - q), delta=.001)
        self.assertEqual(r["tier1"]["pupil_turns"], 1)
        self.assertAlmostEqual(r["tier1"]["longest_teacher_stretch_s"], 40.0,
                               delta=0.2)

    def test_correction_inverts_confusion(self):
        # true 80% teacher, 76% of it caught, 5.5% of pupil talk leaked
        p, q, t = 0.76, 0.055, 0.80
        m = q + (p - q) * t
        self.assertAlmostEqual(engine.correct(m, p, q), t, places=6)
        self.assertEqual(engine.correct(0.01, p, q), 0.0)  # clamped

    def test_window_rescale_never_hits_100(self):
        # the ES2006c case: lesson raw 42% -> 53%, a window raw 85%
        w = engine.rescale(0.85, 0.42, 0.53)
        self.assertTrue(0.85 < w < 0.95)
        self.assertAlmostEqual(engine.rescale(0.42, 0.42, 0.53), 0.53)
        self.assertLess(engine.rescale(0.999, 0.42, 0.53), 1.0)
        self.assertEqual(engine.rescale(0.0, 0.42, 0.53), 0.0)

    def test_calibrate_reproduces_target(self):
        import random
        rnd = random.Random(0)
        f = self._frames()
        f["near_db"] = [v + rnd.gauss(0, 6) for v in f["near_db"]]
        recall = engine.calibrate(f, 0.75)
        r = engine.compute(f, engine.Settings(recall, True))
        self.assertAlmostEqual(r["tier1"]["teacher_share"], 0.75, delta=0.002)

    def test_calibrate_refuses_impossible_value(self):
        with self.assertRaises(ValueError):
            engine.calibrate(self._frames(), 0.05)

    def test_swap_warning(self):
        f = self._frames()
        self.assertFalse(any("wrong way" in w for w in
                             engine.compute(f)["warnings"]))
        self.assertTrue(any("wrong way" in w for w in
                            engine.compute(engine.swap_channels(f))["warnings"]))

    def test_identical_channels_warning(self):
        f = self._frames()
        f["far_db"] = list(f["near_db"])
        self.assertTrue(any("nearly identical" in w
                            for w in engine.compute(f)["warnings"]))


AMI = os.path.join(ROOT, "data/ami/audio/IS1001c.Lapel-0.wav")


@unittest.skipUnless(os.path.exists(AMI) and os.path.exists(engine.VAD_BIN),
                     "AMI audio or VAD binary not present")
class EndToEndAMI(unittest.TestCase):
    """The acceptance bars, run through the product code path."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from validate_ami import gold_labels
        cls.d = tempfile.mkdtemp()
        cls.frames = engine.prepare(
            cls.d, near=AMI,
            far=os.path.join(ROOT, "data/ami/audio/IS1001c.Array1-01.wav"))
        cls.gold = gold_labels("IS1001c", len(cls.frames["near_db"]))

    def test_alignment_is_zero_for_synced_mics(self):
        self.assertEqual(self.frames["lag_frames"], 0)

    def test_talk_share_within_5_points_uncalibrated(self):
        # Uncalibrated is the weaker claim, and holds here only because the
        # defaults were measured on AMI. Calibration is what a school does.
        g = self.gold
        gold = g.count(1) / sum(1 for v in g if v)
        got = engine.compute(self.frames)["tier1"]["teacher_share"]
        self.assertLess(abs(got - gold), 0.05)

    def test_stretch_and_turns_within_15_percent(self):
        g = self.gold
        r = engine.compute(self.frames)["tier1"]
        gs = max(b - a for a, b in engine.teacher_stretches(g)) * engine.HOP
        gt = len(engine.pupil_turns(g))
        self.assertLess(abs(r["longest_teacher_stretch_s"] - gs) / gs, 0.15)
        self.assertLess(abs(r["pupil_turns"] - gt) / gt, 0.15)


if __name__ == "__main__":
    unittest.main()
