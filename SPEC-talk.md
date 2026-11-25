# Lesson Talk Monitor — product spec

A local, CPU-only appliance that turns a lesson recording into a handful of
verifiable talk measurements, and never sends audio anywhere.

Status (2026-11-25): **v0 built** -- a local web app plus CLI in
`lessontalk/`, validated end to end on AMI (`tools/validate_ami.py`,
README). Building it corrected three claims below; see section 9, which
supersedes the speech-run, median-baseline and calibration statements in
sections 3-5. Next gate unchanged: one real lesson (section 8).

## 1. The user and the job

**User:** an instructional coach, head of department, or ITT mentor running
lesson observations, in a school with no ML budget and a hard rule that
pupil voice recordings cannot leave the premises.

**Job today:** they sit in the back of a lesson with a clipboard, or they
record and never review it. Observation is sampled (one lesson a term),
subjective, and arrives weeks later as prose.

**Job with this:** every lesson the teacher chooses to record produces, within
a minute of the bell, a one-page card: how much of the lesson was teacher
talk, how many pupil turns, how the balance is moving across lessons, and
where in the recording to listen. Teacher-owned, coach-shared.

**Why they buy:** talk ratio is the most actionable finding in
classroom-discourse research and it is invisible to the teacher in the
moment. The product makes it visible per lesson, as a trend.

Wait time is the other finding the literature points at, and this product
does **not** promise it. Measuring the pause *after a question* needs to know
a question was asked, which needs the words, which section 4 shows are the
unreliable layer. Silence measurement alone also over-reports (section 4.1).
Pause counts therefore sit in tier 2, flagged, and are not part of the
pitch.

## 2. Why local is the product, not a feature

Pupil voice is personal data about minors. In the UK and EU a school sending
classroom audio of children to a US inference API needs a DPIA, a lawful
basis and a transfer story, and that is a materially harder internal sell
than keeping the audio on site.

Two honest limits on this argument. **Local processing removes the transfer
problem, not the recording problem** -- recording pupils still needs a lawful
basis and a consent position, and this product does not change that. And the
competitive claim needs checking before it is made to anyone: TeachFX is
cloud, but IRIS Connect is an established UK schools incumbent with on-prem
and UK-hosted options, so "cloud competitors are excluded by procurement" is
a hypothesis about buyers, not a fact about the market. **To verify.**

This stack has no cloud component and needs no GPU. The measured budget on
one i7-12700K (see `PROFILE.md`):

| stage | cost for a 29-minute lesson | realtime factor |
|---|---|---|
| Silero VAD (whole lesson) | **10 s** | ~175x |
| whisper.cpp base.en transcript (optional) | **49 s** | ~30x |
| Kev-0.8B typed judgement per 90-word window | 0.31 s each | 1.5% duty cycle |

A lesson is processed in well under a minute on hardware a school already
owns. Marginal cost per lesson is electricity.

## 3. What it measures, and what it refuses to

Every metric is a quantity a human can verify by listening to the recording.
That is a design rule, not a preference: a teacher who disputes a number must
be able to check it. No composite "engagement score", no inferred affect, no
judgement of teaching quality.

**Tier 1 — from voice activity alone (no words), with an acceptance bar each:**

| metric | bar | status |
|---|---|---|
| teacher / pupil talk share | within 5 points of hand-coded, after calibration | met on 2 meetings (3.3 and 0.4 points when locally calibrated) |
| speech vs silence frames | F1 >= 0.90 | **met** (0.957 / 0.960 at the tuned setting) |
| speech-run count (turn proxy) | within 15% of hand-coded | **withdrawn** -- +85% / +195% when counted consistently (section 9) |
| longest uninterrupted teacher stretch | within 15% | **met** (-0%, -13%; section 9) |
| pupil turns >= 2 s | within 15% | **met** (-5%, -7%; section 9) |

**Tier 2 — flagged lower confidence, not in the pitch:**
- total silence and pause counts (over-reported; section 4.1)
- questions asked, open vs closed split (needs the words)
- jump-points: timestamps to review

**Tier 3 — explicitly out of scope:** pupil identification, pupil-level
anything, affect or attention inference, teacher ranking, performance
management. Tier 3 is where this category gets banned from schools, and the
product's privacy claim is worth more than the features.

## 4. Why tier 1 carries the product

Measured on AMI meeting audio, which is the closest available proxy
(multi-party, overlapping, single mixed channel):

**Words are unreliable.** `src/pipeline.py`, two meetings:

| | IS1001c | ES2006a |
|---|---|---|
| WER, whisper base.en | 33.8% | 28.3% |
| deletions : insertions | 655 : 37 | 494 : 15 |
| windows produced, manual -> ASR | 74 -> 52 | 51 -> 43 |
| Kev score correlation, manual vs ASR of the same audio | 0.499 | 0.287 |
| same side of 0.5 | 73.2% | 61.2% |

A third of word-level judgements flip between the manual transcript and the
ASR of the identical audio. Anything sold on words inherits that.

**Voice activity is reliable.** `src/talk.py` on IS1001c, Silero VAD over the
mixed channel vs the union of AMI's manual segments, 100 ms frames:

| | |
|---|---|
| frame F1 | **0.927** |
| precision | **0.981** |
| recall | 0.879 |
| IoU | 0.864 |

VAD almost never calls silence speech (98.1% precision) and recovers 86% of
speech time. Turn counts differ (436 VAD vs 361 manual) because VAD splits on
intra-turn pauses; speech ratio reads 71.2% against a manual 90.6%.

Replicated on a second meeting (ES2006c, 21817 frames): F1 **0.933**,
precision **0.996**, recall 0.879. The speech-recovery ratio came out at
**0.859 against IS1001c's 0.860** — two different rooms and sets of speakers
agreeing to three decimal places, which is the strongest evidence here that
the under-recovery is a fixed property of the VAD settings rather than
meeting-specific noise.

**The under-recovery is symmetric between near and far speakers.** This had
to be checked, because it is the assumption the headline metric rests on:
pupils are quieter and further from both mics, so if VAD missed their speech
more often than the teacher's, talk share would be **biased**, not merely
scaled, and no amount of trend-framing would rescue it. Measured:

| | lapel speaker | everyone else |
|---|---|---|
| IS1001c VAD recall | 0.883 | 0.885 |
| ES2006c VAD recall | 0.889 | 0.884 |

Symmetric to within half a point on both meetings. Talk share is scaled by a
constant, not skewed toward the loud participant.

**So absolute figures need calibration and relative figures do not.** The
bias is systematic (consistent under-recovery, near-zero false speech,
symmetric across speakers), which is the useful kind: "your talk share fell
from 78% to 61% across six lessons" survives it, "you talked for exactly
71.2%" does not. The product therefore ships **trends and comparisons as
primary** and absolute percentages as secondary, with the calibration factor
stated.

### 4.1 The parameter that had to be fixed, and the metric that could not be

The tool's default `-vsd 100` (min silence before a segment splits) treats
every breath as a turn boundary. A min-silence sweep over both meetings:

| `-vsd` | speech runs, IS1001c (gold 182) | ES2006c (gold 115) | silence, IS1001c (gold 222 s) | ES2006c (gold 211 s) | pauses >= 3 s (gold 12 / 18) | speech F1 |
|---|---|---|---|---|---|---|
| 100 ms | 436 | 461 | 414 s | 479 s | 15 / 26 | 0.928 / 0.933 |
| 500 ms | 253 | 216 | 354 s | 401 s | 15 / 25 | 0.942 / 0.947 |
| **800 ms** | **165** | **121** | 298 s | 340 s | 15 / 25 | **0.957 / 0.960** |
| 1200 ms | 84 | 77 | 221 s | 296 s | 14 / 24 | 0.962 / 0.966 |

**800 ms is now the default** (`DEFAULT_MIN_SILENCE_MS` in `src/talk.py`). It
puts speech-run counts within 10% of hand-coded on both meetings and lifts
speech F1 from 0.928 to 0.957. The tool default was simply wrong for this
purpose.

**Silence totals and long-pause counts survive none of this.** Total silence
stays over-reported by 34% and 61% at the chosen setting, and pauses >= 3 s
sit 25-39% high at *every* value swept -- the count barely moves, because it
comes from the 12% of speech VAD misses rather than from how turns are split.
Missed quiet speech becomes phantom silence, and no threshold fixes that.

That is why wait time left tier 1. It is measurable in principle and this
method does not measure it. Saying so is cheaper than shipping a pause count
that is 40% high.

A measurement bug found on the way, worth recording because it flattered the
result: the first gap figures were computed over **raw per-speaker segments,
which overlap**, so the interval between two consecutive segments was often
not silence at all. That understated manual silence (222 s reported as 135 s)
and made VAD look far closer than it is. Gaps are now computed on merged
speech on both sides.

## 5. Attribution: the two-mic shortcut

Teacher-versus-pupil is the whole product and one microphone cannot answer
it. Single-channel diarisation of a classroom — children, reverb, crosstalk —
is the hardest part of the problem and the least likely to work.

**So the design avoids it with hardware.** Two inputs:
- a **lapel mic on the teacher** (near)
- a **room mic** at the back (far)

The teacher is far louder on the lapel than in the room when speaking, and
only bleed-through otherwise. Attribution becomes a per-frame energy
comparison between two channels: no speaker model, no clustering, no
training, nothing to go stale. Pupils are never individually identified,
which is also the privacy posture.

One correction worth recording, because it is the obvious way to get this
wrong: comparing the two channels in **raw dB does not work at all**. The
channels have different gains, so a fixed 0 dB boundary just picks whichever
mic is hotter overall — measured, it attributed 1023 of 1024 seconds to one
side and scored F1 0.000. Each channel must be referenced to its own median
frame first. Then the gain offset cancels and what remains is how far this
moment departs from that channel's own baseline.

**Measured on IS1001c** — `Lapel-0` (near) vs `Array1-01` far-field array
(far), 436 VAD segments, scored against AMI's manual per-speaker segments
(speaker A, 41.9% of talk):

| | |
|---|---|
| separability of speaker A by channel delta | **AUC 0.989**, CI [0.979, 0.997] |
| best margin | +3.4 dB |
| frame F1 at that margin | **0.963** |
| precision / recall | 0.977 / 0.948 |
| talk share, measured vs gold | **38.6% vs 41.9%** |

Acceptance was frame F1 >= 0.85. **PASS**, and not marginally — an AUC of
0.989 means the separation is almost total and barely depends on where the
threshold goes. Measured talk share lands 3.3 points low, in the direction
and roughly the size VAD's 86% speech recovery predicts, which is the
calibration factor of section 4 rather than an attribution error.

**The mic-to-speaker convention checks out.** On ES2006c the channel delta
separates speaker A at AUC 0.977 while every other participant sits far below
chance (B 0.242, C 0.188, D 0.257). `Lapel-0` belongs to speaker A as a matter
of convention, not of picking whichever scored best.

**Transfer test — ES2006c, different room, different four participants, with
IS1001c's +3.4 dB margin held fixed:**

| | re-tuned here (+1.9 dB) | **carried margin (+3.4 dB)** |
|---|---|---|
| frame F1 | 0.929 | **0.887** |
| precision / recall | 0.926 / 0.931 | 0.961 / 0.823 |
| talk share, measured vs gold 38.6% | 38.2% | **32.5%** |

Both pass the F1 >= 0.85 bar, so **attribution transfers across rooms and
speakers without retuning.** But look at the last row: the carried margin
costs 6.1 points of talk share, against 0.4 when tuned locally. A margin set
too high silently reassigns the teacher's quieter speech to the room.

So the honest reading is split: **the method generalises, the calibration does
not.** Attribution needs no per-site training, but the absolute talk-share
figure needs one calibration per mic setup — which section 4 already requires
for the VAD factor, so it is the same step, not a new one. Trends within a
fixed setup are unaffected.

One caveat stays open. AMI lapel mics on seated adults in a quiet room are a
kinder case than a teacher walking around a classroom, and mic distance
varying with movement is exactly what an energy-difference method is
sensitive to. This cannot be settled with AMI; it needs one real lesson.

## 6. Shape of the thing

**A CLI is not shippable to the user in section 1.** An instructional coach
will not run `lesson-talk report`, so the CLI is the engineering artifact and
the demo, not the product. Stating this plainly because the gap is the kind
that gets discovered late:

- **v0, to learn (build this):** CLI plus a single-file HTML card. No account,
  no server, no upload. `lesson-talk report <recording>` writes
  `lesson-NN.json` and `lesson-NN.html` next to the audio. Audience: me, and
  one cooperative teacher.
- **v1, to sell:** a drop-folder or a double-clickable app. Recording stops,
  the card appears. Zero commands. The coach's view is the same HTML with
  several lessons on one axis.

Audio stays on disk and is deletable; the JSON holds no audio and no names.

**What this product does not use:** Kev, or any language model, in tier 1.
The latency profile that opened this project measured a System One model that
turned out to be at chance on the semantic task (AUC 0.527) and is not needed
for talk analytics. Tier 2 is the only place it would appear and tier 2 may
never ship. Worth being explicit: the viable product here is VAD plus two
microphones plus arithmetic.

## 7. Risks, honestly

1. **Classroom audio is harder than AMI, and that is now the main unknown.**
   AMI is close-miced seated adults in a quiet room. Children, reverb, 30
   bodies and a teacher who walks around will lower VAD recall and widen the
   channel-delta distribution that attribution depends on. Mic distance
   varying with movement is the specific failure mode. Untestable without a
   school.
2. **Calibration does not transfer, attribution does.** The transfer test
   passed on F1 (0.887 with a margin carried from another meeting) but cost
   6.1 points of talk share against 0.4 when calibrated locally. Each mic
   setup needs one hand-coded lesson. If a school will not provide one, the
   absolute share is not quotable -- only the trend.
3. **Longest-stretch metric is unmeasured.** It is in tier 1 on the
   assumption that it behaves like the other run-length statistics. That has
   not been checked against hand-coded data.
4. **Tier 2 may never ship.** Section 4 makes question detection and pause
   counts look unsalvageable with this stack. The product should be viable
   with tier 1 alone, and it is designed that way.
5. **The buyer is slow and the artifact is not yet theirs.** Schools procure
   annually and pay little; ITT providers and MATs are the faster path than
   individual schools. And v0 is a CLI (section 6), so there is a build step
   between the evidence and anything a coach can use.
6. **No pricing model yet.** Per-teacher or per-school, who buys the two
   mics, and what the hardware costs are all open. A two-mic requirement is a
   real adoption cost, not just a technical choice.

## 8. What would make this fail fast

The AMI gate is passed: VAD F1 0.957, attribution F1 0.963 locally and 0.887
transferred, recall symmetric across speakers, speech-run counts within 10%.
Nothing further can be learned from this corpus -- it is seated adults, and
every remaining risk is about a classroom.

**The next gate is one real lesson**, two mics, one teacher who agrees, and
twenty minutes of hand-coding to calibrate against. Pass conditions:

1. channel-delta separability AUC >= 0.95 with the teacher moving around
2. talk share within 5 points of the hand-coded value after calibration
3. speech-run count within 15%

If (1) fails, the two-mic shortcut does not survive movement and the product
needs real diarisation -- at which point this stops being a small local tool
and I would not continue. If (1) holds and (2) or (3) misses, that is a
calibration problem and the product survives as trends only.

Build order: v0 CLI (section 6) is the minimum needed to collect that lesson,
so it is the next thing to build -- as instrumentation, not as product.

## 9. What building v0 corrected

Running the product code path end to end on AMI (lapel and far array fed as
two separate recorders, `tools/validate_ami.py`) found three problems in the
claims above. They were found by measuring, not by review, which is the
argument for building v0 at all.

**1. Speech-run counts were compared against a looser gold.** The "within
10%" figure compared VAD runs (split only on silences >= 800 ms) with gold
runs split on *any* gap. Applying the same 800 ms rule to both, VAD
over-counts by 85% and 195%: every stretch of missed quiet speech splits a
run, which is the same mechanism that makes pause counts unusable (4.1).
Speech runs move to tier 2. Pupil turns of 2 s or more replace them (-5%,
-7%), and the longest teacher stretch is now measured (-0%, -13%) -- with
the rule that a pupil interjection under 2 s does not end a stretch.
Without that rule, backchannels cut the stretch by 70-80%. Both rules were chosen on these
same two meetings.

**2. The median baseline fails when the teacher dominates.** Each channel
was referenced to its median frame. AMI's lapel wearer talks under half the
time, so the median was a quiet frame. In a lesson where the teacher talks
most of the time the lapel's median frame *is* teacher speech. Simulated by
cutting other speakers' stretches out of AMI, a true 89% teacher share read
54%. Channels are now referenced to their 10th-percentile frame (the noise
floor), which holds frame accuracy at ~0.80.

**3. One-number calibration cannot fix a compression.** The method catches
~76% of teacher frames and gives 3-8% of pupil frames to the teacher, so
every measured share is pulled towards the middle (true 84% read as 62%).
Shifting the threshold to match one hand-coded lesson only fixes shares
near that lesson's. The engine now inverts the confusion,
`true = (measured - q) / (p - q)`, and calibration estimates `p` (teacher
recall) for the mic setup. Calibrated on a ~75% lesson, other lessons in
the same setup land within 3.3 points from 73% to 89% teacher share, and
5.2 points at a balanced 47%. Calibrated on a balanced lesson and
extrapolated to 89% it was 12 points off, so the app asks for a *typical*
lesson.

The channel-swap check was also rebuilt for the same reason as (2): tails
measured around the median misfired once the teacher dominated, whereas
extremes measured from the noise floor separate cleanly at every share
(p98 +14..+22 dB, p2 -3..-4 dB).

Still untested, and still the main risk: a real classroom, with a teacher
who moves around.
