# Testing Lesson Talk on the team server

This guide is for teammates trying the shared Lesson Talk server. You need
two things from Shivanand: **the server address** and **your own account**
(a username and a generated password). Neither is published here, on purpose.

The server is a test install. It is on the public internet behind a
sign-in, and anything you upload is stored on it and visible to everyone
with an account. **Use only the open sample recordings below, never real
recordings of pupils.**

## 1. Sign in and look around

1. Open the address you were sent. Your browser asks for a username and
   password: use your account. Ten wrong attempts lock you out for ten
   minutes.
2. The red line in the header says you are on a shared server and who you
   are signed in as.
3. Under **AMI DEMO** in the sidebar there are three sample lessons. Click
   each one:
   - **Click the timeline** to play the recording from that moment.
   - The **Where to listen** buttons jump to the longest teacher stretches
     and the longest pupil turns. Check by ear whether each is right.
   - **Open card** shows the card on its own page.

The samples are meetings from the [AMI corpus](https://groups.inf.ed.ac.uk/ami/download/)
(CC BY 4.0), not lessons. In each, one participant wears a lapel mic and
plays "teacher", and a microphone array in the room is the room mic.

## 2. What you should see

None of the samples is calibrated, so expect the "not calibrated" badge.
"Hand-coded" is the share computed from AMI's manual per-speaker
transcripts.

| Sample | Teacher talk shown | Hand-coded | Verdict |
|---|---|---|---|
| Demo: AMI meeting IS1001c | 45% | 47% | works |
| Demo: AMI meeting ES2006c | 53% | 47% | works, 6 points high uncalibrated |
| Sample: AMI meeting ES2002a (hard case) | 56% | 11% | **fails**, see below |

**ES2002a is deliberately included as a failure.** Two things go wrong:

- **The lapel mic cuts out** (digital silence) for 1.5 minutes. The app
  detects this, warns about it at the top of the card, and leaves those
  moments out. This part is handled.
- **The lapel wearer barely speaks (11% of talk), and the person next to
  them talks half the meeting.** That neighbour is loud on the wearer's
  lapel, so the app counts their speech as the "teacher's". This is **not**
  handled, and the card gives no warning. In a classroom the same thing
  would happen when a pupil sits right beside the teacher, or the teacher
  leans over a pupil's desk for a long time.

If you find a way to detect the second problem from the audio alone, that
is the most useful thing a tester could report.

## 3. Add a lesson yourself

This uses ES2002a, but any AMI meeting with both a `Lapel-0` and an
`Array1-01` file works the same way.

1. Download both files (about 40 MB each). The AMI server sometimes answers
   "503 Service Unavailable"; wait a minute and retry.
   - Teacher lapel: <https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Lapel-0.wav>
   - Room mic: <https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Array1-01.wav>
2. Click **New lesson**.
3. Give it a name that starts with your own name, e.g. "Asha: ES2002a",
   so others can tell it apart from the samples. Use a new **Class**, e.g.
   "Asha tests", or your lessons will appear in the samples' trend line.
4. Under **Recording**, choose **Two files**. Drop `Lapel-0` on **Teacher
   lapel mic** and `Array1-01` on **Room mic**.
5. Click **Analyse**. A 20-minute recording takes about 15 to 30 seconds,
   and longer if someone else is analysing at the same time (one analysis
   runs at a time).
6. You should get the same numbers as the ES2002a sample above.

Other things worth trying:

- **Swap teacher / room** on any lesson. The figures should flip, and on
  IS1001c and ES2006c a warning should appear saying the channels look the
  wrong way round.
- **Calibrate this mic setup…** with the hand-coded figure from the table
  (47 for IS1001c). Calibration applies to every lesson on the same mic
  setup, so **make a new setup first** (the **Mic setup** list, then **New
  setup…**) instead of changing the shared default.
- **Delete recording** keeps the numbers but removes the audio.
- Upload something that is not a two-channel WAV (an MP3, or a mono file
  as "one stereo file"). You should get a clear error, not a crash.

## 4. Reporting what you find

Open an issue at <https://github.com/shivanandmn/lesson-talk/issues> with:

- which lesson, and the timestamp if it is about a specific moment
- what the card said and what you heard when you listened
- for errors, the exact message on screen

Please don't put the server address or passwords in issues: the repo is
public.
