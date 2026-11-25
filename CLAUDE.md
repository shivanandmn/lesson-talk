# Lesson Talk

Local teacher-talk analytics from two microphones. Start every session by
reading `docs/SESSION-NOTES.md`: current state, measured accuracy,
deployment, open work and gotchas.

## Commands

```sh
python3 -m unittest discover -s tests          # 27 tests, ~15 s
python3 tools/validate_ami.py IS1001c ES2006c ES2002a   # accuracy vs AMI hand-coding
./lesson-talk                                  # local app on 127.0.0.1:8765
systemctl --user restart lesson-talk           # team server runs this repo; restart after changes
```

## Rules

- The repo is public. Never commit the team server's address, account
  names or passwords. They live only on the machine (see the session notes).
- Every accuracy claim in README, SPEC or the card must come from a
  measurement in `tools/validate_ami.py` or the tests. Say what a number was
  tuned on. When a result disproves an earlier claim, record it in
  `SPEC-talk.md` section 9 instead of quietly changing the number.
- Standard library only in `lessontalk/` (no numpy, no ffmpeg). The audio
  path must run on a school laptop.
- Recordings of real pupils must not go on the team server.
- Update `docs/SESSION-NOTES.md` at the end of a session that changes the
  state.
