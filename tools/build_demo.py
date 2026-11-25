#!/usr/bin/env python3
"""Build a single read-only demo page from a Lesson Talk library.

The page shows the real lesson cards (rendered by lessontalk.report) for every
analysed lesson in one class, with a switcher between them. It holds numbers
only: no audio, nothing to upload, so it can be shared as a link.

    python3 tools/build_demo.py ~/LessonTalk "AMI demo" demo.html
"""

import html
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lessontalk import report  # noqa: E402

REPO = "https://github.com/shivanandmn/lesson-talk"

PAGE_CSS = """
main{max-width:900px;padding-block:28px 64px;
padding-left:max(16px,env(safe-area-inset-left,0px));
padding-right:max(16px,env(safe-area-inset-right,0px))}
.intro{display:grid;gap:10px;padding-bottom:22px;border-bottom:1px solid var(--line);
margin-bottom:22px}
.intro .name{font-size:28px;font-weight:700;letter-spacing:-.01em;margin:0;text-wrap:balance}
.intro p{margin:0;max-width:65ch;color:var(--fg)}
.intro .fine{font-size:13px;color:var(--mut)}
.intro a{color:var(--p)}
.intro code{font-size:13px;background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:2px 6px}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:18px}
.tabs button{font:inherit;font-size:14px;border:1px solid var(--line);background:var(--card);
color:var(--fg);border-radius:8px;padding:8px 12px;cursor:pointer;display:grid;gap:2px;
text-align:left}
.tabs button span{font-size:12px;color:var(--mut)}
.tabs button[aria-selected="true"]{border-color:var(--fg)}
.tabs button:focus-visible{outline:2px solid var(--p);outline-offset:2px}
.card-view{min-width:0}
"""


def load(home: str, group: str):
    lessons = []
    base = os.path.join(home, "lessons")
    for lid in sorted(os.listdir(base)):
        try:
            with open(os.path.join(base, lid, "meta.json")) as fh:
                m = json.load(fh)
            with open(os.path.join(base, lid, "result.json")) as fh:
                r = json.load(fh)
        except FileNotFoundError:
            continue
        if m["group"] == group and m["status"] == "ready":
            m["summary"] = {"teacher_share": r["tier1"]["teacher_share"]}
            lessons.append((m, r))
    lessons.sort(key=lambda x: (x[0]["recorded_on"], x[0]["created"]))
    with open(os.path.join(home, "setups.json")) as fh:
        setups = json.load(fh)
    return lessons, setups


def build(home: str, group: str) -> str:
    lessons, setups = load(home, group)
    if not lessons:
        raise SystemExit(f"no analysed lessons in class {group!r}")
    hist = [m for m, _ in lessons]
    E = html.escape
    tabs, views = [], []
    for i, (m, r) in enumerate(reversed(lessons)):
        vid = "lesson-" + m["id"][-4:]
        body, _ = report.card_parts(
            m, r, hist, setups.get(m["setup"], {}),
            no_audio_note="Audio is not included in this demo. In the app, "
                          "the timeline and the timestamps below play the "
                          "recording from that moment.")
        sel = "true" if i == 0 else "false"
        tabs.append(f'<button role="tab" id="t-{vid}" aria-controls="{vid}" '
                    f'aria-selected="{sel}" data-view="{vid}">{E(m["label"])}'
                    f'<span>{report.pct(r["tier1"]["teacher_share"])} teacher talk'
                    f' · {report.mmss(r["duration_s"])}</span></button>')
        views.append(f'<section class="card-view" role="tabpanel" id="{vid}" '
                     f'aria-labelledby="t-{vid}"{"" if i == 0 else " hidden"}>'
                     f'{body}</section>')
    return f"""<title>Lesson Talk Demo</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{report.CSS}{PAGE_CSS}</style>
<main>
<header class="intro">
  <p class="name">Lesson Talk</p>
  <p>How much of a lesson is teacher talk, measured from two microphones on
  the school's own computer. A lapel mic on the teacher and a room mic are
  enough to tell teacher speech from pupil speech, with no transcription, no
  cloud and no pupil identified.</p>
  <p class="fine">These are real cards from the app, but the recordings are
  two meetings from the AMI corpus (CC BY 4.0), with one participant's
  lapel mic standing in for the teacher's. They are uncalibrated and are not
  classroom data. Run it yourself with <code>./lesson-talk</code> from
  <a href="{REPO}" target="_blank" rel="noopener">the GitHub repo</a>.</p>
</header>
<div class="tabs" role="tablist" aria-label="Demo lessons">{"".join(tabs)}</div>
{"".join(views)}
</main>
<script>
const tabs=[...document.querySelectorAll('.tabs button')];
function show(id){{tabs.forEach(b=>{{const on=b.dataset.view===id;
b.setAttribute('aria-selected',on);document.getElementById(b.dataset.view).hidden=!on;}});}}
tabs.forEach(b=>b.onclick=()=>{{show(b.dataset.view);history.replaceState(null,'','#'+b.dataset.view);}});
function fromHash(){{const h=location.hash.slice(1);if(tabs.some(b=>b.dataset.view===h))show(h);}}
addEventListener('hashchange',fromHash);fromHash();
</script>
"""


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    home, group, out = sys.argv[1:]
    with open(out, "w") as fh:
        fh.write(build(os.path.expanduser(home), group))
    print(f"wrote {out}")
