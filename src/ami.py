"""
Reader for the AMI Meeting Corpus NXT stand-off annotation format.

AMI stores transcription and annotation in separate files that point into each
other by element id. A decision annotation does not contain any text; it holds
a list of spans like

    <nite:child href="ES2002a.B.words.xml#id(ES2002a.B.words624)..id(...700)"/>

so the text of a decision only exists once those spans are resolved against the
per-speaker word files.

Two details that a naive reader gets wrong:

1. Word files interleave <w> with <vocalsound>, <disfmarker> and <gap>, and all
   four share one id sequence. A span's endpoints are therefore resolved by
   DOCUMENT POSITION, never by parsing the trailing integer out of the id --
   `words624..words700` is not 77 words.

2. Spans are per-speaker, and one decision interleaves spans from several
   speakers. Reconstructing what was said means merging resolved spans from all
   speakers back into time order.

Licence: the AMI corpus is CC BY 4.0. See data/ami/annotations/extracted/LICENCE.txt.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

NITE = "{http://nite.sourceforge.net/}"

# Elements that carry spoken text; the others occupy ids but say nothing.
TEXT_TAGS = {"w"}
# Non-text elements that still matter as transcription events.
EVENT_TAGS = {"vocalsound", "disfmarker", "gap", "nonvocalsound", "pause"}

_HREF = re.compile(
    r"^(?P<file>[^#]+)#id\((?P<a>[^)]+)\)(?:\.\.id\((?P<b>[^)]+)\))?$"
)


@dataclass
class Elem:
    """One element of a word file, in document order."""
    id: str
    tag: str
    start: float | None
    end: float | None
    text: str
    speaker: str

    @property
    def is_word(self) -> bool:
        return self.tag in TEXT_TAGS


@dataclass
class Utterance:
    """A transcriber segment: the natural boundary for a streaming window."""
    id: str
    speaker: str
    start: float
    end: float
    words: list[Elem] = field(default_factory=list)

    @property
    def text(self) -> str:
        return _join(self.words)


def _join(elems: list[Elem]) -> str:
    """Join words, attaching punctuation to the preceding token."""
    out: list[str] = []
    for e in elems:
        if not e.is_word:
            continue
        t = e.text
        if not t:
            continue
        if out and (t in ".,?!;:" or t.startswith("'")):
            out[-1] += t
        else:
            out.append(t)
    return " ".join(out)


def parse_href(href: str) -> tuple[str, str, str]:
    """'F.xml#id(A)..id(B)' -> (F.xml, A, B). A single id gives B == A."""
    m = _HREF.match(href.strip())
    if not m:
        raise ValueError(f"unparseable href: {href!r}")
    a = m.group("a")
    return m.group("file"), a, m.group("b") or a


class Meeting:
    """All annotation for one AMI meeting, lazily loaded."""

    def __init__(self, root: str, meeting: str):
        self.root = root            # .../annotations/extracted
        self.meeting = meeting      # e.g. 'ES2002a'
        self._words: dict[str, list[Elem]] = {}     # filename -> elements
        self._index: dict[str, tuple[str, int]] = {}  # elem id -> (file, pos)

    # ---------------------------------------------------------------- words
    def _words_path(self, filename: str) -> str:
        return os.path.join(self.root, "words", filename)

    def load_words_file(self, filename: str) -> list[Elem]:
        if filename in self._words:
            return self._words[filename]
        path = self._words_path(filename)
        speaker = filename.split(".")[1]  # ES2002a.B.words.xml -> B
        elems: list[Elem] = []
        for node in ET.parse(path).getroot():
            tag = node.tag.split("}")[-1]
            eid = node.get(f"{NITE}id")
            if eid is None:
                continue
            st, en = node.get("starttime"), node.get("endtime")
            elems.append(Elem(
                id=eid,
                tag=tag,
                start=float(st) if st is not None else None,
                end=float(en) if en is not None else None,
                text=(node.text or "").strip(),
                speaker=speaker,
            ))
        for pos, e in enumerate(elems):
            self._index[e.id] = (filename, pos)
        self._words[filename] = elems
        return elems

    def speakers(self) -> list[str]:
        d = os.path.join(self.root, "words")
        out = []
        for fn in os.listdir(d):
            if fn.startswith(self.meeting + ".") and fn.endswith(".words.xml"):
                out.append(fn.split(".")[1])
        return sorted(out)

    def load_all_words(self) -> None:
        for sp in self.speakers():
            self.load_words_file(f"{self.meeting}.{sp}.words.xml")

    # ---------------------------------------------------------------- spans
    def resolve_span(self, href: str) -> list[Elem]:
        """Resolve one href to the elements it covers, by document position."""
        filename, a, b = parse_href(href)
        self.load_words_file(filename)
        elems = self._words[filename]
        try:
            _, ia = self._index[a]
            _, ib = self._index[b]
        except KeyError as exc:
            raise KeyError(f"{href}: id not in {filename}: {exc}") from exc
        if ib < ia:
            ia, ib = ib, ia
        return elems[ia:ib + 1]

    # ------------------------------------------------------------ decisions
    def decisions(self) -> list[dict]:
        """Decision annotations, each with its resolved words in time order."""
        path = os.path.join(
            self.root, "decision", "manual", f"{self.meeting}.decision.xml")
        if not os.path.exists(path):
            return []
        out = []
        for node in ET.parse(path).getroot():
            if node.tag.split("}")[-1] != "decision":
                continue
            elems: list[Elem] = []
            for child in node.findall(f"{NITE}child"):
                href = child.get("href")
                if href:
                    elems.extend(self.resolve_span(href))
            words = [e for e in elems if e.is_word and e.start is not None]
            words.sort(key=lambda e: (e.start, e.id))
            if not words:
                continue
            out.append({
                "id": node.get(f"{NITE}id"),
                "start": words[0].start,
                "end": max(w.end or w.start for w in words),
                "word_ids": {e.id for e in elems},
                "text": _join(words),
                "n_words": len(words),
            })
        out.sort(key=lambda d: d["start"])
        return out

    # ----------------------------------------------------------- utterances
    def utterances(self) -> list[Utterance]:
        """Transcriber segments across all speakers, merged into time order."""
        d = os.path.join(self.root, "segments")
        out: list[Utterance] = []
        for sp in self.speakers():
            path = os.path.join(d, f"{self.meeting}.{sp}.segments.xml")
            if not os.path.exists(path):
                continue
            for node in ET.parse(path).getroot():
                if node.tag.split("}")[-1] != "segment":
                    continue
                words: list[Elem] = []
                for child in node.findall(f"{NITE}child"):
                    href = child.get("href")
                    if href:
                        words.extend(self.resolve_span(href))
                timed = [w for w in words if w.start is not None]
                st = node.get("transcriber_start")
                en = node.get("transcriber_end")
                start = float(st) if st is not None else (
                    min(w.start for w in timed) if timed else 0.0)
                end = float(en) if en is not None else (
                    max(w.end or w.start for w in timed) if timed else start)
                out.append(Utterance(
                    id=node.get(f"{NITE}id") or "",
                    speaker=sp, start=start, end=end, words=words))
        out.sort(key=lambda u: (u.start, u.speaker))
        return [u for u in out if u.text]
