"""
AI mode — quality gates that cost nothing (port of lekhak/src/server/checks.ts)
==============================================================================
Run on 100% of pages; only the failures escalate to a model (the repair
pass, then the critic panel). Same codes, severities and risk weights as
Lekhak, with two dubbing adaptations:

  * LENGTH (target/source char ratio learned from a book memory) is replaced
    by TIMING: each paragraph's estimated speech time (config.estimate_duration,
    the same per-language rate the rest of the engine uses) against the time
    window of the English cues it covers. For a dub, running long IS the
    failure that matters — the sync stage can only borrow so much.
  * ENCODING (legacy-font corruption in PDF extracts) is dropped: the source
    here is an ASR transcript, never a legacy-font document.
  * COVERAGE (warn) is new: the model's cue claims had to be repaired.
  * PAUSES (warn, v0.20): the ' | ' pause markers do not match the English
    cue boundaries. Markers never count toward any length (TIMING,
    UNDERSHOOT): they are stripped before the voice speaks.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence

from .ai_lang import script_share
from .config import estimate_duration

AI_SCRIPT_FLOOR = 0.9        # lekhak SCRIPT_FLOOR
AI_TIMING_TOLERANCE = 1.5    # a paragraph may run this x its cue window...
AI_TIMING_SLACK_S = 0.6      # ...plus this much, before it fails
AI_TIMING_MIN_WINDOW_S = 0.8

# Model preamble and translator commentary (lekhak checks.ts:44-52).
_AI_COMMENTARY = [
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\bi (?:cannot|can't|am unable to)\b", re.I),
    re.compile(r"\b(?:here(?:'s| is) (?:the|your)|below is the)\s+"
               r"(?:translation|translated)", re.I),
    re.compile(r"^\s*(?:translation|translated text|output|note|explanation)"
               r"\s*[:：]", re.I | re.M),
    re.compile(r"\btranslator'?s? note\b", re.I),
    re.compile(r"\[(?:note|translator|sic|untranslated)[^\]]*\]", re.I),
    re.compile(r"^\s*```", re.M),
]
_AI_SRC_ENDS = re.compile(r"[.!?\"'”’)\]]\s*$")
_AI_TGT_ENDS = re.compile(r"[.!?।॥\"'”’)\]]\s*$")


def _ai_unmarked(s: str) -> str:
    """Text without v0.20 pause markers — they are never spoken, so they
    must not count toward a length. (The reader strips them already; this
    keeps the checks honest for any caller that passes marked text.)"""
    s = s or ""
    return re.sub(r"\s*\|\s*", " ", s) if "|" in s else s


def _ai_normkey(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in "LNM")


def _ai_digit_runs(s: str) -> List[str]:
    """Digit runs with every script's digits folded to ASCII (lekhak
    normalises Devanagari; unicodedata covers all the Indic scripts)."""
    out, cur = [], []
    for ch in s or "":
        d = unicodedata.digit(ch, None)
        if d is not None:
            cur.append(str(d))
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def terms_in(source: str, glossary: Dict[str, str]) -> List[tuple]:
    """(english term, required target term) for glossary entries present in
    *source* — longest first, whole words (lekhak glossary.ts termsIn)."""
    found = []
    for en in sorted((glossary or {}).keys(), key=len, reverse=True):
        tr = str(glossary.get(en) or "").strip()
        if not en.strip() or not tr:
            continue
        if re.search(r"(?<!\w)" + re.escape(en.strip()) + r"(?!\w)",
                     source or "", re.I):
            found.append((en.strip(), tr))
    return found


AI_EXPECTED_RATIO = 0.8      # target/English chars when memory is too small
AI_RATIO_TOLERANCE = 1.6     # lekhak LENGTH_TOLERANCE
AI_UNDERSHOOT_MIN_EN = 60    # English shorter than this is not judged
AI_PAUSES_TOLERANCE = 0.5    # PAUSES warns past this share of the boundaries


def run_checks(source: str, target: str, language: str,
               rows: Optional[Sequence[dict]] = None,
               terms: Sequence[tuple] = (), coverage_repaired: bool = False,
               expected_ratio: Optional[float] = None) -> List[dict]:
    """[{"code", "severity": "fail"|"warn", "detail"}] for one page.
    *rows* are the page's paragraphs with their cue windows (start/end)."""
    target = target or ""
    if not target.strip():
        return [{"code": "EMPTY", "severity": "fail",
                 "detail": "The translation is empty."}]
    out = []
    if _ai_normkey(target) == _ai_normkey(source):
        out.append({"code": "PASSTHROUGH", "severity": "fail",
                    "detail": "The output is the English source, unchanged."})

    share = script_share(target, language)
    if share is not None and share < AI_SCRIPT_FLOOR:
        out.append({"code": "SCRIPT", "severity": "fail",
                    "detail": f"Only {share * 100:.0f}% of the letters are "
                              f"{language} script (need "
                              f"{AI_SCRIPT_FLOOR * 100:.0f}%). Either English "
                              "survived, or this is the wrong language."})

    wanted = _ai_digit_runs(source)
    if wanted:
        got = set(_ai_digit_runs(target))
        missing = sorted({d for d in wanted if d not in got})
        if missing:
            out.append({"code": "DIGITS", "severity": "fail",
                        "detail": "Numbers present in the source are missing "
                                  "from the translation: "
                                  + ", ".join(missing) + "."})

    # TIMING (dubbing replacement for LENGTH): per paragraph, against the
    # cue window it will be spoken in.
    over = []
    for i, r in enumerate(rows or [], 1):
        s0, s1 = r.get("start"), r.get("end")
        if s0 is None or s1 is None:
            continue
        win = max(AI_TIMING_MIN_WINDOW_S, float(s1) - float(s0))
        est = estimate_duration(_ai_unmarked(str(r.get("tr") or "")),
                                language)
        if est > win * AI_TIMING_TOLERANCE + AI_TIMING_SLACK_S:
            over.append(f"paragraph {i} (~{est:.1f}s of speech for a "
                        f"{win:.1f}s slot)")
    if over:
        out.append({"code": "TIMING", "severity": "fail",
                    "detail": "Too long to speak in the time the English "
                              "takes — shorten, keep the meaning: "
                              + "; ".join(over) + "."})

    # UNDERSHOOT (Lekhak's LENGTH, too-short side): a paragraph far shorter
    # than its English dropped content — the silent failure that left 50 s
    # of a talk undubbed. Ratio learned from approved pairs when available.
    ratio = expected_ratio or AI_EXPECTED_RATIO
    short = []
    for i, r in enumerate(rows or [], 1):
        en_len = len(str(r.get("en") or "").strip())
        tr_len = len(_ai_unmarked(str(r.get("tr") or "")).strip())
        if en_len >= AI_UNDERSHOOT_MIN_EN and \
           tr_len < en_len * ratio / AI_RATIO_TOLERANCE:
            short.append(f"paragraph {i} ({tr_len} chars for {en_len} "
                         f"English chars — about {100 * tr_len / (en_len * ratio):.0f}% "
                         "of the expected length)")
    if short:
        out.append({"code": "UNDERSHOOT", "severity": "fail",
                    "detail": "Content is missing — translate ALL of the "
                              "English, every sentence: " + "; ".join(short)
                              + "."})

    if _AI_SRC_ENDS.search(source.strip()) and \
       not _AI_TGT_ENDS.search(target.strip()):
        out.append({"code": "TRUNCATED", "severity": "fail",
                    "detail": "The translation does not end on a sentence "
                              "boundary; it was probably cut off."})

    for rx in _AI_COMMENTARY:
        m = rx.search(target)
        if m:
            out.append({"code": "COMMENTARY", "severity": "fail",
                        "detail": "The output contains commentary rather than "
                                  f"only the translation: "
                                  f"\"{m.group(0).strip()[:60]}\"."})
            break

    missing_terms = [(en, tr) for en, tr in terms if tr not in target]
    if missing_terms:
        out.append({"code": "GLOSSARY", "severity": "fail",
                    "detail": "Required terminology missing: " + "; ".join(
                        f"\"{en}\" should appear as \"{tr}\""
                        for en, tr in missing_terms) + "."})

    # PAUSES (v0.20, warn): the pause-aware prompt asks for one ' | ' per
    # English cue boundary inside a paragraph. Far fewer (or more) markers
    # than boundaries means the stretches will not line up with the
    # speaker's phrases, so anchor sync falls back on duration alone there.
    # Only rows that went through the marker reader carry "pauses".
    marked = [r for r in rows or [] if isinstance(r.get("pauses"), list)]
    if marked:
        want = sum(max(0, len(r.get("cues") or []) - 1) for r in marked)
        got = sum(len(r["pauses"]) for r in marked)
        if abs(got - want) > max(2, AI_PAUSES_TOLERANCE * want):
            out.append({"code": "PAUSES", "severity": "warn",
                        "detail": f"{got} pause marker(s) ' | ' for {want} "
                                  "English cue boundaries inside paragraphs — "
                                  "put one ' | ' where each cue ends and the "
                                  "next begins."})

    if coverage_repaired:
        out.append({"code": "COVERAGE", "severity": "warn",
                    "detail": "The model's cue numbering was inconsistent and "
                              "was repaired — check the paragraph boundaries."})
    return out


def has_failure(checks: Sequence[dict]) -> bool:
    return any(c.get("severity") == "fail" for c in checks)


def fail_count(checks: Sequence[dict]) -> int:
    return sum(1 for c in checks if c.get("severity") == "fail")


def risk_score(checks: Sequence[dict], tier: int, tm_score: float,
               source_chars: int, critic_flagged: bool = False) -> float:
    """How badly a human is needed here, 0..1 (lekhak checks.ts riskScore)."""
    risk = 0.0
    for c in checks:
        risk += 0.45 if c.get("severity") == "fail" else 0.12
    if critic_flagged:
        risk += 0.25
    risk += {1: 0.0, 2: 0.1, 3: 0.28, 4: 0.4}.get(int(tier), 0.4)
    risk += (1 - min(1.0, max(0.0, float(tm_score or 0)))) * 0.1
    if source_chars > 1500:
        risk += 0.05
    return round(min(1.0, risk), 3)
