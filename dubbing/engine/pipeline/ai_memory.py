"""
AI mode — memory before anything is spent (port of lekhak tm.ts lookupSegment)
==============================================================================
Consults the translation memory (tm.py pairs — every paragraph a human ever
approved) for each page, sentence by sentence, and returns Lekhak's tiers:

  tier 1  every sentence matched exactly     -> reuse, no model call
  tier 2  every sentence matched, char-weighted score >= 0.85
                                             -> model call, strongly anchored
  tier 3  some worked examples exist         -> model call with examples
  tier 4  nothing to go on                   -> model call

(Lekhak ships a tier-2 draft without a model call for a human to confirm. A
dub can run straight through with no human, and a fuzzy draft carries no
timing fit, so tier 2 goes to the model with every sentence anchored.)

Retrieval is Lekhak's shape without SQLite FTS5: an in-memory inverted index
over content words picks <= 60 candidates, then they are reranked with
similarity = 0.4 * token-set ratio + 0.6 * char ratio (text.ts:376-385).
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple

AI_FUZZY_THRESHOLD = 0.85    # lekhak tm.ts FUZZY_THRESHOLD (tier 2)
AI_ANCHOR_MIN = 0.75         # weakest fuzzy match still worth anchoring
AI_CANDIDATES = 60           # lekhak: top 60 FTS candidates
AI_EXAMPLES = 5              # lekhak examplesBlock: up to 5 neighbours

_AI_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+(?=\S)")
_AI_WORD = re.compile(r"\w+", re.UNICODE)


def _ai_mem_norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    return " ".join(_AI_WORD.findall(s))


def _ai_content_words(norm: str) -> set:
    return {w for w in norm.split() if len(w) >= 4}


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _AI_SENT_SPLIT.split((text or "").strip())
            if s.strip()]


def similarity(a_norm: str, b_norm: str) -> float:
    if not a_norm or not b_norm:
        return 0.0
    ta, tb = set(a_norm.split()), set(b_norm.split())
    tok = (2 * len(ta & tb) / (len(ta) + len(tb))) if (ta or tb) else 0.0
    ch = SequenceMatcher(None, a_norm, b_norm, autojunk=False).ratio()
    return 0.4 * tok + 0.6 * ch


class AiMemory:
    """One language's approved pairs, indexed once per run."""

    def __init__(self, pairs: Sequence[Tuple[str, str]]):
        self.rows: List[Tuple[str, str, str]] = []      # (norm, en, tr)
        self.exact: Dict[str, int] = {}
        self.index: Dict[str, List[int]] = {}
        for en, tr in pairs or []:
            n = _ai_mem_norm(en)
            if not n or not (tr or "").strip():
                continue
            i = len(self.rows)
            self.rows.append((n, en.strip(), tr.strip()))
            self.exact.setdefault(n, i)                 # newest wins
            for w in _ai_content_words(n):
                self.index.setdefault(w, []).append(i)

    def __len__(self):
        return len(self.rows)

    def _candidates(self, norm: str) -> List[int]:
        hits: Dict[int, int] = {}
        for w in _ai_content_words(norm):
            for i in self.index.get(w, ()):
                hits[i] = hits.get(i, 0) + 1
        return [i for i, _ in sorted(hits.items(), key=lambda kv: -kv[1])
                ][:AI_CANDIDATES]

    def best(self, sentence: str):
        """(score, en, tr) of the best match for one sentence, or None."""
        n = _ai_mem_norm(sentence)
        if not n:
            return None
        i = self.exact.get(n)
        if i is not None:
            return (1.0, self.rows[i][1], self.rows[i][2])
        best = None
        for i in self._candidates(n):
            sc = similarity(n, self.rows[i][0])
            if best is None or sc > best[0]:
                best = (sc, self.rows[i][1], self.rows[i][2])
        return best if best and best[0] >= AI_ANCHOR_MIN else None

    def lookup(self, page_text: str) -> dict:
        """{"tier", "score", "draft", "anchors": [{source, en, tr, score}],
        "examples": [(en, tr)]} for one page (lekhak lookupSegment)."""
        sents = split_sentences(page_text)
        anchors, used = [], set()
        weighted = total = 0.0
        all_matched = all_exact = bool(sents)
        for s in sents:
            m = self.best(s)
            total += len(s)
            if m is None:
                all_matched = all_exact = False
                continue
            if m[0] < 0.999:
                all_exact = False
            weighted += m[0] * len(s)
            anchors.append({"source": s, "en": m[1], "tr": m[2],
                            "score": round(m[0], 3)})
            used.add(_ai_mem_norm(m[1]))
        score = (weighted / total) if total else 0.0

        # Worked examples: neighbours sharing a content word, not already
        # anchored, best similarity first (lekhak styleExamples).
        page_norm = _ai_mem_norm(page_text)
        sent_norms = [_ai_mem_norm(s) for s in sents]
        scored = []
        for i in self._candidates(page_norm):
            n, en, tr = self.rows[i]
            if n in used:
                continue
            best_s = max((similarity(sn, n) for sn in sent_norms), default=0.0)
            scored.append((best_s, en, tr))
        scored.sort(key=lambda t: -t[0])
        examples = [(en, tr) for _s, en, tr in scored[:AI_EXAMPLES]]

        if all_exact:
            tier = 1
        elif all_matched and score >= AI_FUZZY_THRESHOLD:
            tier = 2
        elif examples or anchors:
            tier = 3
        else:
            tier = 4
        draft = " ".join(a["tr"] for a in anchors) if tier == 1 else ""
        return {"tier": tier, "score": round(score, 3), "draft": draft,
                "anchors": anchors, "examples": examples}
