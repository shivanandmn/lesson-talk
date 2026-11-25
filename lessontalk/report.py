"""The lesson card: one self-contained HTML file.

Design rule from the spec: every number on it is something the teacher can
check by listening, so every number comes with a place in the recording to
listen to, and a note on how far it can be trusted.
"""

from __future__ import annotations

import html
import json

E = html.escape

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--line:#e4e1da;--card:#fff;
--t:#c2562b;--p:#2f6f8f;--warn:#8a5a00;--wbg:#fff6df}
@media (prefers-color-scheme:dark){:root{--bg:#181816;--fg:#ecebe7;--mut:#a09e98;
--line:#34332f;--card:#22211e;--t:#e07a4f;--p:#5fa3c4;--warn:#f0c060;--wbg:#3a2f12}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);
margin:28px 0 10px;font-weight:600}
.sub{color:var(--mut);font-size:14px}
.badge{display:inline-block;font-size:12px;padding:2px 8px;border-radius:99px;
border:1px solid var(--line);color:var(--mut);margin-left:6px}
.badge.ok{color:var(--p);border-color:var(--p)}
.hero{display:grid;grid-template-columns:auto 1fr;gap:28px;align-items:center;
background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:22px;margin-top:18px}
.big{font-size:56px;font-weight:700;line-height:1;color:var(--t);
font-variant-numeric:tabular-nums}
.big small{font-size:22px}
.split{height:14px;border-radius:7px;overflow:hidden;display:flex;
background:var(--line);margin:10px 0 6px}
.split i{display:block;height:100%}
.legend{font-size:13px;color:var(--mut)}
.legend b{display:inline-block;width:10px;height:10px;border-radius:2px;
margin:0 4px 0 12px;vertical-align:-1px}
.legend b:first-child{margin-left:0}
.delta{font-size:14px;margin-top:8px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.stat .v{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
.stat .k{font-size:13px;color:var(--mut)}
.stat .n{font-size:12px;color:var(--mut);margin-top:6px}
.tl{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
svg{display:block;width:100%;height:auto}
.tl svg{cursor:pointer}
.axis{font-size:11px;fill:var(--mut)}
.jump{display:flex;flex-wrap:wrap;gap:8px}
button.j{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);
color:var(--fg);border-radius:8px;padding:6px 10px;cursor:pointer}
button.j:hover{border-color:var(--fg)}
button.j .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
audio{width:100%;margin-top:10px}
.warn{background:var(--wbg);color:var(--warn);border-radius:8px;padding:10px 14px;
margin-top:14px;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:14px}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:500}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.how{font-size:13px;color:var(--mut)}
.how li{margin:4px 0}
@media print{audio,.jump{display:none}.hero,.stat,.tl{break-inside:avoid}}
"""


def mmss(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else \
        f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def _timeline(r: dict) -> str:
    bins = r["timeline"]["bins"]
    bs = r["timeline"]["bin_s"]
    W, H, pad = 800, 90, 18
    n = max(1, len(bins))
    bw = W / n
    out = [f'<svg viewBox="0 0 {W} {H + pad}" id="tl" data-bin="{bs}" '
           f'data-n="{n}" role="img" aria-label="Who was talking across '
           f'the lesson">']
    for i, (t, p) in enumerate(bins):
        x = i * bw
        th, ph = t * H, p * H
        if th:
            out.append(f'<rect x="{x:.2f}" y="{H - th:.2f}" width="{bw + .3:.2f}"'
                       f' height="{th:.2f}" fill="var(--t)"/>')
        if ph:
            out.append(f'<rect x="{x:.2f}" y="{H - th - ph:.2f}" '
                       f'width="{bw + .3:.2f}" height="{ph:.2f}" '
                       f'fill="var(--p)"/>')
    dur = r["duration_s"]
    step = 300 if dur <= 3600 else 600
    for t in range(0, int(dur) + 1, step):
        x = t / dur * W if dur else 0
        anchor = "start" if t == 0 else "middle"
        out.append(f'<text class="axis" x="{x:.1f}" y="{H + 14}" '
                   f'text-anchor="{anchor}">{mmss(t)}</text>')
    out.append(f'<line x1="0" y1="{H}" x2="{W}" y2="{H}" stroke="var(--line)"/>'
               f'<line id="ph" x1="0" y1="0" x2="0" y2="{H}" '
               f'stroke="var(--fg)" stroke-width="1.5" opacity="0"/></svg>')
    return "".join(out)


def _trend(hist: list[dict], lid: str) -> str:
    pts = [(h["id"], h["recorded_on"], h["summary"]["teacher_share"])
           for h in hist]
    if len(pts) < 2:
        return ('<p class="sub">The trend appears once this class has two or '
                'more lessons. Trends are the most trustworthy thing this '
                'tool reports.</p>')
    W, H, l, b = 800, 160, 36, 22
    n = len(pts)
    def X(i): return l + (W - l - 12) * (i / (n - 1))
    def Y(v): return 8 + (H - b - 8) * (1 - v)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Teacher talk share across lessons">']
    for v in (0, .25, .5, .75, 1):
        out.append(f'<line x1="{l}" x2="{W - 12}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"'
                   f' stroke="var(--line)"/><text class="axis" x="{l - 6}" '
                   f'y="{Y(v) + 4:.1f}" text-anchor="end">{int(v * 100)}%</text>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(i):.1f},{Y(v):.1f}"
                 for i, (_, _, v) in enumerate(pts))
    out.append(f'<path d="{d}" fill="none" stroke="var(--t)" stroke-width="2"/>')
    for i, (hid, day, v) in enumerate(pts):
        cur = hid == lid
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="{6 if cur else 3.5}"'
                   f' fill="{"var(--t)" if cur else "var(--card)"}" '
                   f'stroke="var(--t)" stroke-width="2"><title>{E(day)}: '
                   f'{pct(v)}</title></circle>')
        if n <= 12 or cur:
            anchor = "end" if i == n - 1 else "start" if i == 0 else "middle"
            out.append(f'<text class="axis" x="{X(i):.1f}" y="{H - 4}" '
                       f'text-anchor="{anchor}">{E(day[5:])}</text>')
    out.append("</svg>")
    return "".join(out)


def card(m: dict, r: dict, hist: list[dict], setup: dict,
         audio_file: str | None = None) -> str:
    t1, t2 = r["tier1"], r["tier2"]
    share = t1["teacher_share"]
    sp = t1["teacher_s"] + t1["pupil_s"]
    dur = r["duration_s"] or 1
    cal = r["settings"]["calibrated"]

    prev = [h for h in hist if (h["recorded_on"], h["created"])
            < (m["recorded_on"], m["created"])]
    delta = ""
    if prev:
        p = prev[-1]["summary"]["teacher_share"]
        dd = (share - p) * 100
        word = "less" if dd < 0 else "more"
        delta = (f'<div class="delta">{abs(dd):.0f} points {word} teacher talk '
                 f'than the previous lesson ({E(prev[-1]["recorded_on"])}, '
                 f'{pct(p)}).</div>') if abs(dd) >= 1 else \
            '<div class="delta">About the same as the previous lesson.</div>'

    badge = ('<span class="badge ok">calibrated</span>' if cal else
             '<span class="badge" title="Absolute figures can be several '
             'points off until this mic setup is calibrated against one '
             'hand-coded lesson. Trends are unaffected.">not calibrated</span>')

    warn = "".join(f'<div class="warn">{E(w)}</div>' for w in r["warnings"])

    def jumps(items, colour, what):
        return "".join(
            f'<button class="j" data-t="{x["start_s"]}"><span class="dot" '
            f'style="background:var({colour})"></span>{mmss(x["start_s"])} · '
            f'{what} {mmss(x["dur_s"])}</button>' for x in items)

    player = ""
    if audio_file and not m.get("audio_deleted"):
        player = (f'<audio id="au" controls preload="metadata" '
                  f'src="{E(audio_file)}"></audio>')
    else:
        player = ('<p class="sub">The recording has been deleted; the numbers '
                  'above were kept.</p>')

    phases = "".join(
        f'<tr><td>{mmss(p["start_s"])}–{mmss(p["end_s"])}</td>'
        f'<td class="num">{pct(p["teacher_share"])}</td>'
        f'<td><div class="split" style="margin:0;height:8px">'
        f'<i style="width:{(p["teacher_share"] or 0) * 100:.1f}%;'
        f'background:var(--t)"></i><i style="flex:1;background:var(--p)"></i>'
        f'</div></td></tr>'
        for p in r["phases"] if p["teacher_share"] is not None)

    js = """
const au=document.getElementById('au'),tl=document.getElementById('tl'),
ph=document.getElementById('ph');
function go(t){if(!au)return;au.currentTime=t;au.play();}
document.querySelectorAll('button.j').forEach(b=>b.onclick=()=>go(+b.dataset.t));
if(tl){tl.onclick=e=>{const r=tl.getBoundingClientRect();
const f=(e.clientX-r.left)/r.width;go(f*DUR);};}
if(au&&ph){au.ontimeupdate=()=>{const x=au.currentTime/DUR*800;
ph.setAttribute('x1',x);ph.setAttribute('x2',x);ph.setAttribute('opacity',1);};}
""".replace("DUR", str(dur))

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{E(m["label"])} · lesson talk</title><style>{CSS}</style></head><body><main>
<h1>{E(m["label"])}{badge}</h1>
<div class="sub">{E(m["group"])} · {E(m["recorded_on"])} · {mmss(dur)} recorded ·
{E(setup.get("name", m["setup"]))}</div>
{warn}
<section class="hero">
  <div><div class="big">{share * 100:.0f}<small>%</small></div>
  <div class="sub">teacher talk</div></div>
  <div>
    <div class="split"><i style="width:{t1['teacher_s'] / dur * 100:.2f}%;
background:var(--t)"></i><i style="width:{t1['pupil_s'] / dur * 100:.2f}%;
background:var(--p)"></i></div>
    <div class="legend"><b style="background:var(--t)"></b>teacher {mmss(t1['teacher_s'])}
<b style="background:var(--p)"></b>pupils {mmss(t1['pupil_s'])}
<b style="background:var(--line)"></b>quiet {mmss(t1['silence_s'])}</div>
    {delta}
  </div>
</section>

<h2>Across the lesson</h2>
<div class="tl">{_timeline(r)}
<div class="legend" style="margin-top:6px"><b style="background:var(--t)"></b>teacher
<b style="background:var(--p)"></b>pupils · click anywhere to listen from there</div>
{player}</div>

<h2>Measures</h2>
<div class="grid">
  <div class="stat"><div class="v">{mmss(t1['longest_teacher_stretch_s'])}</div>
  <div class="k">longest teacher stretch</div>
  <div class="n">Teacher talking with no pupil turn of 2 s or more. Within 13% of
  hand-coded on test recordings.</div></div>
  <div class="stat"><div class="v">{t1['pupil_turns']}</div>
  <div class="k">pupil turns (2 s or longer)</div>
  <div class="n">{t1['pupil_turns'] / (dur / 60):.1f} per minute. Within 7% of
  hand-coded on test recordings.</div></div>
  <div class="stat"><div class="v">{pct(t1['pupil_s'] / sp if sp else 0)}</div>
  <div class="k">pupil share of talk</div>
  <div class="n">All pupils together. Individual pupils are never identified.</div></div>
</div>

<h2>Where to listen</h2>
<div class="jump">
{jumps(r['listen']['teacher_stretches'], '--t', 'teacher')}
{jumps(r['listen']['pupil_turns'], '--p', 'pupils')}
</div>

<h2>Teacher share by ten minutes</h2>
<table><tbody>{phases}</tbody></table>

<h2>This class over time</h2>
{_trend(hist, m["id"])}

<h2>How to read this</h2>
<ul class="how">
<li>Measured from two microphones only: the teacher's lapel mic and a room mic.
When the lapel is much louder than usual, it is the teacher. No words are
transcribed and nothing left this computer.</li>
<li>{"This mic setup was calibrated against a hand-coded lesson. On test recordings, calibrated readings were within 3.5 points for lessons with 70–90% teacher talk, and up to 5 points off for much more balanced lessons." if cal else "This mic setup is not calibrated yet, so treat the percentages as approximate: within about 5 points on test recordings, and a classroom may differ. Comparisons between lessons recorded the same way are more reliable than any single figure."}</li>
<li>Not measured: wait time, questions, pupil names, engagement, or teaching
quality. Pause counts ({t2['pauses_over_3s']} pauses of 3 s or more) are
over-reported by this method and are shown only here.</li>
</ul>
<script>{js}</script>
</main></body></html>"""


def as_json(r: dict) -> str:
    return json.dumps(r, indent=1)
