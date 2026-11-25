"""The lesson library: a folder on this computer, nothing else.

    $LESSONTALK_HOME (default ~/LessonTalk)
      setups.json                 mic setups and their calibration
      lessons/<id>/meta.json      label, group, date, status
      lessons/<id>/*.wav          the recording(s), deletable
      lessons/<id>/frames.json    energy + speech detection cache (no audio)
      lessons/<id>/result.json    the numbers on the card
      lessons/<id>/card.html      the card, a single self-contained file
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import secrets
import shutil
import threading

from . import engine, report

HOME = os.path.expanduser(os.environ.get("LESSONTALK_HOME", "~/LessonTalk"))
_lock = threading.RLock()
_work = threading.Lock()  # one analysis at a time; they are CPU-bound

ROLES = ("stereo", "near", "far")


def _p(*parts: str) -> str:
    return os.path.join(HOME, *parts)


def _read(path: str, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def _write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(tmp, path)


def _safe_id(lid: str) -> str:
    if not re.fullmatch(r"[0-9a-z-]{6,40}", lid or ""):
        raise KeyError(lid)
    return lid


# ------------------------------------------------------------------ setups
def setups() -> dict:
    with _lock:
        s = _read(_p("setups.json"), {})
        if not s:
            s = {"default": {"name": "Default setup",
                             "teacher_recall": engine.DEFAULT_TEACHER_RECALL,
                             "calibrated": False}}
            _write(_p("setups.json"), s)
        return s


def add_setup(name: str) -> str:
    with _lock:
        s = setups()
        sid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "setup"
        base, k = sid, 2
        while sid in s:
            sid, k = f"{base}-{k}", k + 1
        s[sid] = {"name": name.strip()[:80] or sid,
                  "teacher_recall": engine.DEFAULT_TEACHER_RECALL,
                  "calibrated": False}
        _write(_p("setups.json"), s)
        return sid


# ----------------------------------------------------------------- lessons
def lesson_dir(lid: str) -> str:
    return _p("lessons", _safe_id(lid))


def meta(lid: str) -> dict:
    m = _read(os.path.join(lesson_dir(lid), "meta.json"), None)
    if m is None:
        raise KeyError(lid)
    return m


def save_meta(m: dict) -> None:
    with _lock:
        _write(os.path.join(lesson_dir(m["id"]), "meta.json"), m)


def result(lid: str) -> dict | None:
    return _read(os.path.join(lesson_dir(lid), "result.json"), None)


def create(label: str, group: str, setup: str, recorded_on: str,
           mode: str, teacher_channel: int = 0) -> dict:
    if mode not in ("stereo", "pair"):
        raise ValueError("mode must be stereo or pair")
    if setup not in setups():
        raise ValueError(f"unknown setup {setup}")
    try:
        dt.date.fromisoformat(recorded_on)
    except (TypeError, ValueError):
        recorded_on = dt.date.today().isoformat()
    lid = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    m = {"id": lid, "label": (label or "Lesson").strip()[:120],
         "group": (group or "My class").strip()[:80], "setup": setup,
         "recorded_on": recorded_on, "mode": mode,
         "teacher_channel": 1 if teacher_channel else 0,
         "files": {}, "status": "awaiting audio", "error": None,
         "audio_deleted": False,
         "created": dt.datetime.now().isoformat(timespec="seconds")}
    os.makedirs(lesson_dir(lid), exist_ok=True)
    save_meta(m)
    return m


def audio_path(lid: str, role: str) -> str:
    if role not in ROLES + ("mix",):
        raise KeyError(role)
    return os.path.join(lesson_dir(lid), f"{role}.wav")


def save_upload(lid: str, role: str, stream, length: int) -> None:
    m = meta(lid)
    want = ("stereo",) if m["mode"] == "stereo" else ("near", "far")
    if role not in want:
        raise ValueError(f"this lesson expects {', '.join(want)}")
    dst = audio_path(lid, role)
    left = length
    with open(dst + ".part", "wb") as fh:
        while left > 0:
            buf = stream.read(min(1 << 20, left))
            if not buf:
                break
            fh.write(buf)
            left -= len(buf)
    if left:
        os.remove(dst + ".part")
        raise ValueError("upload was interrupted")
    os.replace(dst + ".part", dst)
    with _lock:
        m = meta(lid)
        m["files"][role] = os.path.getsize(dst)
        save_meta(m)


def all_lessons() -> list[dict]:
    out = []
    if not os.path.isdir(_p("lessons")):
        return out
    for lid in os.listdir(_p("lessons")):
        try:
            m = meta(lid)
        except (KeyError, json.JSONDecodeError):
            continue
        r = result(lid)
        if r:
            m["summary"] = {"teacher_share": r["tier1"]["teacher_share"],
                            "pupil_turns": r["tier1"]["pupil_turns"],
                            "duration_s": r["duration_s"],
                            "calibrated": r["settings"]["calibrated"]}
        out.append(m)
    out.sort(key=lambda m: (m["recorded_on"], m["created"]))
    return out


def history(group: str, setup: str | None = None) -> list[dict]:
    return [m for m in all_lessons()
            if m["group"] == group and "summary" in m]


# ---------------------------------------------------------------- analysis
def _settings(setup_id: str) -> engine.Settings:
    s = setups().get(setup_id) or setups()["default"]
    return engine.Settings(
        s.get("teacher_recall", engine.DEFAULT_TEACHER_RECALL), s["calibrated"])


def _render(lid: str) -> None:
    """Recompute numbers from the cache and rewrite the card. Cheap."""
    m = meta(lid)
    d = lesson_dir(lid)
    frames = engine.load_frames(d)
    if m.get("swapped"):
        frames = engine.swap_channels(frames)
    r = engine.compute(frames, _settings(m["setup"]))
    _write(os.path.join(d, "result.json"), r)
    hist = history(m["group"])
    setup = setups().get(m["setup"], {})
    # the card links its audio relatively, so it plays both when served by
    # the app and when opened straight from the lesson folder
    html = report.card(m, r, hist, setup, audio_file=frames["playback"])
    with open(os.path.join(d, "card.html"), "w") as fh:
        fh.write(html)


def analyse(lid: str) -> None:
    """Run in a background thread; status is written to meta.json."""
    def set_status(st, err=None):
        with _lock:
            m = meta(lid)
            m["status"], m["error"] = st, err
            save_meta(m)

    m = meta(lid)
    need = ("stereo",) if m["mode"] == "stereo" else ("near", "far")
    missing = [r for r in need if r not in m["files"]]
    if missing:
        raise ValueError(f"missing recording: {', '.join(missing)}")
    set_status("queued")
    with _work:
        set_status("analysing")
        try:
            d = lesson_dir(lid)
            if m["mode"] == "stereo":
                engine.prepare(d, stereo=audio_path(lid, "stereo"),
                               teacher_channel=m["teacher_channel"])
            else:
                engine.prepare(d, near=audio_path(lid, "near"),
                               far=audio_path(lid, "far"))
            _render(lid)
            set_status("ready")
            rerender_group(m["group"])
        except Exception as e:  # shown to the user, so keep it readable
            set_status("error", str(e))


def analyse_async(lid: str) -> None:
    meta(lid)
    threading.Thread(target=analyse, args=(lid,), daemon=True).start()


def swap(lid: str) -> None:
    with _lock:
        m = meta(lid)
        m["swapped"] = not m.get("swapped", False)
        save_meta(m)
    _render(lid)


def calibrate(lid: str, teacher_share_pct: float) -> dict:
    """One hand-coded lesson calibrates its whole mic setup, and
    every lesson recorded with that setup is recomputed."""
    m = meta(lid)
    frames = engine.load_frames(lesson_dir(lid))
    if m.get("swapped"):
        frames = engine.swap_channels(frames)
    recall = engine.calibrate(frames, teacher_share_pct / 100.0)
    with _lock:
        s = setups()
        s[m["setup"]].update({
            "teacher_recall": recall, "calibrated": True,
            "calibrated_from": lid, "hand_coded_share": teacher_share_pct / 100,
            "calibrated_at": dt.datetime.now().isoformat(timespec="seconds")})
        _write(_p("setups.json"), s)
    rerender_setup(m["setup"])
    return s[m["setup"]]


def rerender_setup(setup_id: str) -> None:
    for x in all_lessons():
        if x["setup"] == setup_id and x["status"] == "ready":
            _render(x["id"])


def rerender_group(group: str) -> None:
    """Cards show the class trend, so a new lesson updates the others."""
    for x in all_lessons():
        if x["group"] == group and x["status"] == "ready":
            _render(x["id"])


def delete_audio(lid: str) -> None:
    d = lesson_dir(lid)
    for role in ROLES + ("mix",):
        p = audio_path(lid, role)
        if os.path.exists(p):
            os.remove(p)
    with _lock:
        m = meta(lid)
        m["audio_deleted"] = True
        save_meta(m)
    if os.path.exists(os.path.join(d, "frames.json")):
        _render(lid)


def delete(lid: str) -> None:
    group = meta(lid)["group"]
    shutil.rmtree(lesson_dir(lid))
    rerender_group(group)
