"""Text normalisation and quote matching against a page's words."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from ocr_pipeline.models.document import BBox, Word

# Markdown emphasis / headings / code / table pipes and inline HTML (HTML tables).
_MARKUP = re.compile(r"[*_`#|]+|<[^>]+>")
_EDGE_PUNCT = ".,;:!?()[]{}\"'«»“”‘’"
_WINDOW_SLACK = 2  # also try windows up to 2 tokens shorter/longer than the quote


def tokens(text: str) -> list[str]:
    """Lowercased words with markup and edge punctuation removed."""
    text = _MARKUP.sub(" ", unicodedata.normalize("NFKC", text)).lower()
    out = []
    for t in text.split():
        t = t.strip(_EDGE_PUNCT)
        if t and not set(t) <= {"-", ":"}:  # skip table rules like ---
            out.append(t)
    return out


@dataclass
class PageIndex:
    """A page's words flattened into tokens, remembering which word each came from."""

    words: list[Word]
    tokens: list[str]
    word_of: list[int]  # token index -> word index

    @classmethod
    def build(cls, words: list[Word]) -> PageIndex:
        toks, word_of = [], []
        for i, w in enumerate(words):
            for t in tokens(w.text):
                toks.append(t)
                word_of.append(i)
        return cls(words, toks, word_of)


@dataclass
class Match:
    start: int  # token range [start, end)
    end: int
    score: float  # 1.0 exact, else similarity in [0, 1]


def find_quote(quote: list[str], page: PageIndex, min_score: float) -> Match | None:
    n, p = len(quote), page.tokens
    if n == 0 or not p:
        return None
    for i in range(len(p) - n + 1):
        if p[i : i + n] == quote:
            return Match(i, i + n, 1.0)

    target = " ".join(quote)
    best: Match | None = None
    cutoff = min_score * 100
    for length in range(max(1, n - _WINDOW_SLACK), n + _WINDOW_SLACK + 1):
        for i in range(len(p) - length + 1):
            score = fuzz.ratio(target, " ".join(p[i : i + length]), score_cutoff=cutoff)
            if score and (best is None or score / 100 > best.score):
                best = Match(i, i + length, score / 100)
    return best


def matched_words(match: Match, page: PageIndex) -> list[Word]:
    idx = sorted(set(page.word_of[match.start : match.end]))
    return [page.words[i] for i in idx]


def line_boxes(words: list[Word]) -> list[BBox]:
    """One rectangle per text line covered by ``words``, top to bottom."""
    lines: dict[int, list[float]] = {}
    for w in words:
        x0, y0, x1, y1 = w.bbox
        box = lines.setdefault(w.line, [x0, y0, x1, y1])
        box[0], box[1] = min(box[0], x0), min(box[1], y0)
        box[2], box[3] = max(box[2], x1), max(box[3], y1)
    return sorted((tuple(b) for b in lines.values()), key=lambda b: (b[1], b[0]))
