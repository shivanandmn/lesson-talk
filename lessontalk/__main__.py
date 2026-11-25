"""lesson-talk: talk analytics for a lesson recording, on this computer.

    python3 -m lessontalk serve [--port 8765] [--no-browser]
    python3 -m lessontalk report STEREO.wav [--teacher-channel 0|1]
    python3 -m lessontalk report --near LAPEL.wav --far ROOM.wav
"""

import argparse
import json
import os
import sys

from . import engine, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lesson-talk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="start the app in your browser")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    r = sub.add_parser("report", help="analyse one recording, write a card")
    r.add_argument("stereo", nargs="?")
    r.add_argument("--teacher-channel", type=int, default=0, choices=(0, 1))
    r.add_argument("--near")
    r.add_argument("--far")
    r.add_argument("--teacher-recall", type=float,
                   help="calibrated teacher recall for this mic setup")
    r.add_argument("--out", help="output folder (default: next to the audio)")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        from .server import serve
        serve(a.port, not a.no_browser)
        return 0

    src = a.stereo or a.near
    if not src or (not a.stereo and not a.far):
        ap.error("give one stereo file, or --near and --far")
    base = os.path.splitext(os.path.basename(src))[0]
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(src)),
                                base + ".lesson")
    frames = engine.prepare(out, stereo=a.stereo,
                            teacher_channel=a.teacher_channel,
                            near=a.near, far=a.far)
    settings = engine.Settings(a.teacher_recall, True) \
        if a.teacher_recall is not None else engine.Settings()
    res = engine.compute(frames, settings)
    with open(os.path.join(out, "result.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    meta = {"id": "cli", "label": base, "group": "", "setup": "",
            "recorded_on": "", "created": ""}
    playback = frames["playback"]
    rel = os.path.relpath(a.stereo if a.stereo else os.path.join(out, playback),
                          out)
    with open(os.path.join(out, "card.html"), "w") as fh:
        fh.write(report.card(meta, res, [], {"name": "command line"},
                             audio_file=rel))
    t = res["tier1"]
    print(f"teacher share {t['teacher_share']:.0%}  "
          f"pupil turns {t['pupil_turns']}  "
          f"longest teacher stretch {t['longest_teacher_stretch_s']:.0f} s")
    for w in res["warnings"]:
        print("warning:", w, file=sys.stderr)
    print(f"wrote {os.path.join(out, 'card.html')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
