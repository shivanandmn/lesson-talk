"""WAV reading, per-frame energy, and two-recorder alignment.

Standard library only. The machine this was built on has no numpy and no
ffmpeg, and a school laptop should not need either: a lesson reduces to one
energy value per 100 ms frame per channel, which is ~17k numbers.
"""

from __future__ import annotations

import array
import math
import sys
import wave
from dataclasses import dataclass

if sys.byteorder != "little":  # WAV is little-endian; arrays are native
    raise ImportError("lessontalk.audio assumes a little-endian CPU")

HOP = 0.1  # seconds per frame; every metric is computed on this grid


class AudioError(ValueError):
    """A recording we cannot use, with a message a teacher can act on."""


@dataclass
class WavInfo:
    path: str
    channels: int
    rate: int
    width: int
    seconds: float


def info(path: str) -> WavInfo:
    try:
        with wave.open(path, "rb") as w:
            return WavInfo(path, w.getnchannels(), w.getframerate(),
                           w.getsampwidth(), w.getnframes() / w.getframerate())
    except wave.Error as e:
        raise AudioError(
            "this file is not a WAV recording the tool can read "
            f"({e}). Export it as a 16- or 24-bit WAV and try again.") from e
    except EOFError as e:
        raise AudioError("this file is empty, cut short, or not a WAV "
                         "recording. Export it as a 16- or 24-bit WAV.") from e


def _to16(raw: bytes, width: int) -> array.array:
    """Any integer PCM width -> 16-bit samples, by keeping the top two bytes.

    Done with byte slicing rather than a Python loop so a 24-bit 48 kHz
    lesson converts at C speed. The low bits dropped are irrelevant to an
    energy measurement (the 16-bit floor is -96 dBFS).
    """
    if width == 2:
        raw = raw[:len(raw) - len(raw) % 2]
        a = array.array("h")
        a.frombytes(raw)
    elif width in (3, 4):
        n = len(raw) // width
        b = bytearray(n * 2)
        b[0::2] = raw[width - 2:n * width:width]
        b[1::2] = raw[width - 1:n * width:width]
        a = array.array("h")
        a.frombytes(bytes(b))
    else:
        raise AudioError(f"unsupported sample width {width * 8}-bit; "
                         "use 16- or 24-bit WAV")
    return a


def _db(samples: array.array) -> float:
    if not samples:
        return -120.0
    ms = math.sumprod(samples, samples) / len(samples)
    return 10.0 * math.log10(ms / (32768.0 ** 2) + 1e-12)


def envelopes(path: str, hop: float = HOP) -> list[list[float]]:
    """Per-channel energy in dBFS, one value per `hop` seconds."""
    wi = info(path)
    step = max(1, int(wi.rate * hop))
    out: list[list[float]] = [[] for _ in range(wi.channels)]
    with wave.open(path, "rb") as w:
        while True:
            raw = w.readframes(step)
            if not raw:
                break
            a = _to16(raw, wi.width)
            for c in range(wi.channels):
                out[c].append(_db(a[c::wi.channels]))
    return out


def align(near: list[float], far: list[float],
          max_lag_s: float = 30.0, hop: float = HOP) -> tuple[int, float]:
    """Offset, in frames, that best lines `far` up with `near`.

    Needed only when the lapel and the room mic are separate recorders that
    were started by hand at different moments. The teacher's speech is in
    both envelopes, so their correlation peaks at the true offset.

    Returns (lag, correlation). lag > 0 means `far` started `lag` frames
    earlier than `near`, i.e. far[i + lag] lines up with near[i].
    """
    L = int(max_lag_s / hop)

    def centred(x: list[float]) -> list[float]:
        m = sum(x) / len(x) if x else 0.0
        return [v - m for v in x]

    a, b = centred(near), centred(far)
    best = (0, -2.0)
    for lag in range(-L, L + 1):
        if lag >= 0:
            x, y = a, b[lag:]
        else:
            x, y = a[-lag:], b
        n = min(len(x), len(y))
        if n < 600:  # need at least a minute of overlap to trust it
            continue
        x, y = x[:n], y[:n]
        den = math.sqrt(math.sumprod(x, x) * math.sumprod(y, y)) or 1.0
        r = math.sumprod(x, y) / den
        if r > best[1]:
            best = (lag, r)
    return best


def write_stereo(near: str, far: str, out: str, lag_frames: int,
                 near_channel: int = 0, far_channel: int = 0,
                 hop: float = HOP) -> None:
    """Interleave two mono-or-multichannel recordings into one 16-bit stereo
    WAV (left = near, right = far), shifted by `lag_frames`, for the VAD.

    The VAD downmixes, so it hears everyone in the room at once.
    """
    ni, fi = info(near), info(far)
    if ni.rate != fi.rate:
        raise AudioError(
            f"the two recordings have different sample rates ({ni.rate} Hz "
            f"and {fi.rate} Hz). Record both at the same rate.")
    rate = ni.rate
    skip_near = max(0, -lag_frames) * int(rate * hop)
    skip_far = max(0, lag_frames) * int(rate * hop)
    chunk = rate * 10
    with wave.open(near, "rb") as wn, wave.open(far, "rb") as wf, \
            wave.open(out, "wb") as wo:
        wo.setnchannels(2)
        wo.setsampwidth(2)
        wo.setframerate(rate)
        wn.readframes(skip_near)
        wf.readframes(skip_far)
        while True:
            a = _to16(wn.readframes(chunk), ni.width)[near_channel::ni.channels]
            b = _to16(wf.readframes(chunk), fi.width)[far_channel::fi.channels]
            n = min(len(a), len(b))
            if n == 0:
                break
            ab, bb = a[:n].tobytes(), b[:n].tobytes()
            o = bytearray(n * 4)
            o[0::4], o[1::4] = ab[0::2], ab[1::2]
            o[2::4], o[3::4] = bb[0::2], bb[1::2]
            wo.writeframes(bytes(o))
