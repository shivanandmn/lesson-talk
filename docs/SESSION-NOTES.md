# Session notes: where Lesson Talk stands

Handoff for whoever (human or Claude) picks this up next. Last updated
2026-11-25, at commit `6d11773` plus this file. Read this, then `README.md`,
then `SPEC-talk.md` section 9.

## In one paragraph

Lesson Talk is a working v0: a local web app and CLI that turn a two-mic
lesson recording (teacher lapel + room mic) into a card showing teacher talk
share, longest teacher stretch, pupil turns, a clickable timeline and the
class trend. It is validated on AMI meeting audio, public on GitHub under
MIT, and running as a password-protected team test server on the machine
`sigmoid`. **It has never seen a real classroom recording.** That is still
the gate that decides whether the product idea survives (spec section 8).

## How we got here (2026-11-25, one session)

1. Started from research only: Kev CPU latency profile (`PROFILE.md`),
   AMI VAD and two-mic attribution experiments (`src/`), and a product spec
   (`SPEC-talk.md`). No product, no repo.
2. The user asked for "a product which can be used" and chose the
   **classroom** version (not a call/meeting analyser) as a **local web
   app**.
3. Built `lessontalk/`. Validating it end to end disproved three spec
   claims and changed the method (spec section 9):
   - speech-run counts were 85-195% high when counted consistently, so
     they were replaced by pupil turns >= 2 s and longest teacher stretch
   - the median channel baseline failed when the teacher dominates, so
     channels are now referenced to their 10th-percentile noise floor
   - one-number calibration could not fix the pull towards 50%, so the
     engine now inverts a confusion model, `true = (m - q) / (p - q)`, and
     calibration estimates teacher recall `p` per mic setup
4. Published: GitHub repo (public, MIT), README screenshot, and a read-only
   demo page as a private claude.ai artifact.
5. Fixed ten-minute shares clamping at 100% (now an odds-ratio rescale).
6. Deployed as a team server: accounts, lockout, host allowlist, systemd
   user service, Tailscale Funnel.
7. Added AMI ES2002a as an unseen test meeting. It **failed**: 56% shown
   against 11% true. Mic cut-outs were one cause (now detected and left
   out); a loud neighbour on the lapel is the other (**not solved**).
   Wrote `docs/TESTING.md` for teammates.

## Measured accuracy (AMI, `tools/validate_ami.py`)

| | IS1001c | ES2006c | ES2002a (unseen) |
|---|---|---|---|
| teacher share, calibrated on a ~75% lesson, other lessons 73-89% | within 1 pt | within 3.3 pts | calibration impossible |
| teacher share, uncalibrated | 44.9 vs 47.1 | 52.7 vs 47.3 | **56.0 vs 10.6** |
| longest teacher stretch | -0% | -13% | |
| pupil turns >= 2 s | -5% | -7% | |
| lapel separability (AUC) | 0.98 | 0.98 | 0.78 |

The 2 s turn rule, the 4 dB margin and the default recall (0.76) were all
chosen on IS1001c and ES2006c. ES2002a is the only honest out-of-sample
check, and it failed.

## Code map

| Path | What it is |
|---|---|
| `lessontalk/audio.py` | WAV reading (16/24/32-bit, stdlib only), 100 ms energy envelopes, two-recorder alignment, stereo mixdown |
| `lessontalk/engine.py` | `prepare()` (envelopes + VAD, cached in `frames.json`), `compute()` (all metrics, correction, warnings), `calibrate()` |
| `lessontalk/report.py` | The lesson card; `card_parts()` for embedding |
| `lessontalk/store.py` | Lesson library on disk, background analysis, calibration per setup |
| `lessontalk/server.py` | stdlib HTTP server: API, Range audio, host check, write header, auth |
| `lessontalk/auth.py` | Team accounts: generated passwords, PBKDF2, lockout |
| `lessontalk/app.html` | The single-page app |
| `tools/validate_ami.py` | End-to-end accuracy check against AMI hand-coding |
| `tools/build_demo.py` | Read-only demo page from a library |
| `tests/` | 27 tests; `python3 -m unittest discover -s tests` |

No Python packages and no numpy. The speech detector is the whisper.cpp
`whisper-vad-speech-segments` binary with the Silero model (README "Setup").

## Deployment (on `sigmoid`)

Deliberately not written down here, because the repo is public: the
server's address and the accounts. On `sigmoid` you can find them with:

- `systemctl --user cat lesson-talk` gives the service: port 8102, loopback
  only, library in `~/LessonTalk-team`, allowed host in the unit file
- `tailscale funnel status` gives the public address (Funnel on :8443)
- `~/Developer/PORTS.md` is the port registry entry
- `~/.config/lesson-talk/users.json` holds the accounts (hashes only).
  The owner's initial password is in `~/.config/lesson-talk/password-<name>.txt`

Operations:

```sh
systemctl --user restart lesson-talk     # after any code change: the service runs the repo directly
journalctl --user -u lesson-talk -f
./lesson-talk add-user NAME              # prints a generated password once
./lesson-talk remove-user NAME
./lesson-talk list-users
```

Changing the Funnel needs root, and `sudo` cannot prompt from Claude's `!`
shell. Run it in a real terminal, or run
`sudo tailscale set --operator=$USER` once so it no longer needs root.

The team library holds three samples (class "AMI demo"): IS1001c, ES2006c
and ES2002a. Only the owner's account exists so far; teammates' names were
never given.

## Not in git (on `sigmoid` only)

- `models/` (8 GB: Kev GGUFs, whisper and Silero models), `vendor/`
  (llama.cpp `abca692`, whisper.cpp `c1a2404`, built), `data/` (AMI audio
  and annotations, 1.2 GB), `audio/`
- `results/validate/*/` cached frames and mix WAVs (regenerated by
  `validate_ami.py`)
- the demo artifact: https://claude.ai/artifact/MTeEZ49hYd6qiyukLi1gnG
  (private to the owner; rebuild with `tools/build_demo.py ~/LessonTalk-team
  "AMI demo" out.html`)

## Open work, in priority order

1. **One real classroom lesson** (spec section 8): two mics, a teacher who
   moves around, 20 minutes hand-coded. Nothing else changes the picture as
   much. Pass bars: lapel separability AUC >= 0.95, share within 5 points
   after calibration, turns within 15%.
2. **Detect the loud-neighbour failure** (ES2002a). The card currently
   gives no warning. Leads: the speech-delta distribution in ES2002a has
   its median at +3.6 dB, close to the 4 dB margin, while the good meetings
   are bimodal. Needs more than three meetings to set a threshold honestly.
3. **More out-of-sample AMI meetings** with `Lapel-0` + `Array1-01`. Expect
   AMI's server to answer 503 under load and some meetings (e.g. TS3003a)
   to have no lapel. Run them through `tools/validate_ami.py`.
4. The timeline bars are uncorrected while the headline is corrected, so
   the timeline shows less teacher talk than the headline.
5. Teammate accounts, once names are known.
6. Publish (the career framework wants artifacts out): a post on the
   "building v0 disproved the spec" findings, and one on the Kev CPU profile.

## Gotchas learned the hard way

- `pkill -f "lessontalk serve"` kills the shell running it, because the
  pattern matches its own command line. Use `pkill -f "^python3 -m lessontalk"`.
- Firefox is a snap: it cannot read `/tmp`. For headless screenshots use a
  profile and output under `~/snap/firefox/common/`. It captures at the
  load event, before the app's fetch finishes, so serve a copy whose
  script loads state synchronously (see how `docs/screenshot.png` was made).
- The AMI mirror throttles: 503s come and go, and Range/HEAD requests fail
  more often than plain GETs.
- `auth.USERS` is read at import time. Tests must patch the attribute, not
  the environment variable (a test once hung by really starting a server).
- The whisper.cpp VAD prints centiseconds, not seconds.
- Git identity for this repo is set locally to the GitHub no-reply address,
  so the owner's personal email stays out of public history.
