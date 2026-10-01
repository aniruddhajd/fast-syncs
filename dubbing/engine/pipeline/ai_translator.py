"""
AI mode — self-learning script generation (v0.18)
==================================================
Port of the Lekhak agentic translation architecture
(D:\\isha automation\\lekhak, src/server/translate.ts + checks.ts +
critics.ts + proofer.ts + tm.ts) into the dubbing engine. It REPLACES the
Step1 -> Step2 -> Step3 prompt chain (S2a-S2c) for runs launched with
``--script-source ai``. Everything after the script — the sync stages
S2d..S3e, whichever sync mode is configured — is untouched.

Agents and stages (module = Lekhak file):

  1. Translate  ai_translate_script()  — the orchestrator (translate.ts)
       a. Memory       ai_memory.AiMemory (tm.ts lookupSegment): tier 1-4
                       per page; tier 1 reused with no model call; anchors
                       + worked examples injected otherwise.
       b. Translator   6 pages per call, batches in parallel, cached static
                       prefix (systemPrefix), temperature 0.3; a dropped
                       passage is retried alone once.
       c. Checks       ai_checks.run_checks (checks.ts) on 100% of pages,
                       free; TIMING replaces LENGTH for dubbing.
       d. Repair       one pass, check details as the rejection notes;
                       kept only if it fails no more checks.
       e. Critic panel ai_agents.run_critic_panel (critics.ts): 3 judges
                       at temperature 0 on still-failing pages + a 2%
                       audit; an errored judge approves nothing.
       f. Proofer      ai_agents.proof_pages (proofer.ts): advisory
                       sentence findings with drop-in suggestions.
       Output: paragraphs tied to their exact English cues (the review
       screen pairs them 1:1), plus a risk-sorted review report.

  2. Learn      learn_from_review()
       Called when a reviewed script is approved for dubbing (--steps dub).
       The approved rows go into the translation memory (tm.py — the
       capture half that was never wired), every row the human changed is
       kept as a draft -> final correction, and one LLM call MERGES the new
       evidence into the stored profile. Unlike Lekhak's learn-style, which
       re-derives the profile from scratch out of every locked page
       (overwrite, no history), this is incremental and bounded.
       v0.21: learn_from_final() does the same from the FINAL dub (the
       timeline after every regeneration) and is now the default moment;
       review-time learning runs only with ai_learn_at = "review". Both
       share _absorb_evidence().

  3. Guidelines + glossary  profile["guidelines"], profile["glossary"]
       Free-text house rules (Lekhak style.guidelines) and forced
       terminology {english: target} (Lekhak glossary table — verified by
       the GLOSSARY check). Hand-edited in the profile JSON; learning never
       overwrites either.

Storage: <repo>/dubbing/data/ai_learning/<language>.json (env
AI_LEARNING_DIR overrides). data/ is gitignored and the updater's overlay
never touches it, so learning survives updates. All learning writes are
fail-open — a storage or LLM error is logged and the dub carries on.
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .ai_agents import proof_pages, run_critic_panel
from .ai_checks import fail_count, has_failure, risk_score, run_checks, terms_in
from .ai_lang import glossary_directive, honorific_directive, translit_directive
from .ai_lang import house_rules as _house_rules_text
from .ai_memory import AiMemory
from .config import (DATA_DIR, DEFAULT_CHARS_PER_SEC, GEMINI_DEFAULT_MODEL,
                     LANG_CHARS_PER_SEC)
from .llm import (_llm_generate, _pair_review_rows, _tm_lang,
                  _split_translation_paragraphs, _strip_code_fence)

try:
    from . import tm as translation_memory
except Exception:                                       # pragma: no cover
    translation_memory = None

AI_LEARNING_DIR = os.getenv("AI_LEARNING_DIR",
                            os.path.join(DATA_DIR, "ai_learning"))

PAGE_CHARS = 1800            # translator's page size (App.tsx parseAndAddBook)
MAX_VOCAB = 80               # learned term mappings kept in the profile
MAX_CORRECTIONS = 40         # draft -> final corrections kept in the profile
PROMPT_CORRECTIONS = 8       # ... of which the newest N go into each prompt
LEARN_PAIRS_CAP = 24         # approved pairs sent to one learn call
LEARN_CHARS_CAP = 14000      # hard char budget for the learn call's evidence
PROFILE_VERSION = 1

StatusCb = Optional[Callable[[str], None]]
Row = Dict[str, object]      # {"en": str, "tr": str, "start": float|None,
                             #  "end": float|None, "cues": [int]}


# ─────────────────────────────────────────────────────────────────────────────
#  Profile storage
# ─────────────────────────────────────────────────────────────────────────────

def _lang_key(language: str) -> str:
    """Filename-safe language key — for the profile FILE only. Translation
    memory rows are keyed by llm._tm_lang, so both modes share one set."""
    return re.sub(r"[^a-z0-9_-]+", "_",
                  (language or "").strip().lower()) or "unknown"


def profile_path(language: str) -> str:
    return os.path.join(AI_LEARNING_DIR, _lang_key(language) + ".json")


def _empty_profile(language: str) -> dict:
    return {
        "version": PROFILE_VERSION,
        "language": language,
        "guidelines": "",
        "glossary": {},
        "toneDescription": "",
        "grammarRules": "",
        "vocabularyMappings": {},
        "corrections": [],
        "samples": 0,
        "runs": 0,
        "updated": "",
    }


def load_profile(language: str) -> dict:
    """The stored profile for *language*, or an empty one. Never raises.
    Unknown keys are kept and missing ones defaulted (additive format)."""
    prof = _empty_profile(language)
    try:
        with open(profile_path(language), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            prof.update(data)
    except Exception:
        pass
    if not isinstance(prof.get("vocabularyMappings"), dict):
        prof["vocabularyMappings"] = {}
    if not isinstance(prof.get("corrections"), list):
        prof["corrections"] = []
    if not isinstance(prof.get("glossary"), dict):
        prof["glossary"] = {}
    return prof


def save_profile(language: str, prof: dict) -> Optional[str]:
    """Atomic write (tmp + replace) so a crash never leaves half a JSON.
    Returns the path, or None on failure (fail-open)."""
    try:
        os.makedirs(AI_LEARNING_DIR, exist_ok=True)
        path = profile_path(language)
        fd, tmp = tempfile.mkstemp(prefix=".ai_", suffix=".json",
                                   dir=AI_LEARNING_DIR)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(prof, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return path
    except Exception:
        return None


def profile_summary(prof: dict) -> str:
    """One log line describing what the model is being taught."""
    return (f"{len(prof.get('vocabularyMappings') or {})} term(s), "
            f"{len(prof.get('corrections') or [])} correction(s), "
            f"{int(prof.get('samples') or 0)} approved row(s) over "
            f"{int(prof.get('runs') or 0)} review(s)"
            + (", house guidelines set"
               if (prof.get("guidelines") or "").strip() else ""))


# ─────────────────────────────────────────────────────────────────────────────
#  JSON from an LLM reply
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_reply(text: str):
    """Parse a model reply as JSON: fence-stripped first, then the widest
    {...} / [...] span (the translator's fallback). Raises ValueError."""
    raw = _strip_code_fence((text or "").strip())
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("model reply is not valid JSON: " + raw[:200])


# ─────────────────────────────────────────────────────────────────────────────
#  1. Translate — Lekhak's translateItems pipeline (lekhak translate.ts:288)
# ─────────────────────────────────────────────────────────────────────────────
#
#   memory ─► batched generation ─► checks ─► repair ─► critic panel ─► proofer
#   (free)    (6 pages / call,       (free,   (1 pass,  (failing pages  (advisory,
#              pages in parallel)     100%)    check     + 2% audit,     report
#                                              notes)    3 judges)       only)

AI_BATCH_SIZE = 6            # lekhak LEKHAK_BATCH_SIZE
AI_CONCURRENCY = 3           # batches in flight (lekhak LEKHAK_CONCURRENCY)
AI_AUDIT_RATE = 0.02         # lekhak run.auditRate
_AI_STRUCTURAL = {"EMPTY", "SCRIPT", "PASSTHROUGH"}   # lekhak unusableDraft

_PRINCIPLES = """You are a master literary translator specializing in translating English spiritual and philosophy talks into profound, elegant, and context-aware {lang}. Your translation is a DUBBING SCRIPT: a voice artist will speak it over the original video.

TRANSLATION PRINCIPLES (Broad Guidelines):
1. Prioritize Flow over Literalism: Always prefer natural, spoken-style phrasing over formal or academic structures. If a sentence sounds 'robotic' or 'translated,' rephrase it to match the rhythm of how a native speaker would express that same thought.
2. Capture the Intended Emotion: Focus on the speaker's core intent rather than individual words. Use evocative, culturally resonant terminology that conveys the same emotional weight as the original, even if the literal meaning differs.
3. Avoid Administrative/Clinical Tone: Replace overly formal, technical, or 'dictionary' words with simpler, more relatable vocabulary that fits a conversational yet reflective context.
4. Maintain Rhetorical Punch: Keep the speaker's wit and directness. If the original uses a rhetorical question or a punchy remark, replicate that structure rather than diluting it into a passive statement.
5. Contextual Accuracy: In philosophical or spiritual talks, use standard, culturally accepted terminology for cosmic concepts to maintain depth, but ensure they don't break the conversational flow.
6. Iterative Refinement: Be willing to discard 'correct' but stiff translations in favor of 'imperfect' but natural-sounding ones. Always prioritize the listener's ease of understanding the deeper meaning over grammatical rigidity.
7. Timing: every English cue carries its start time and duration. A paragraph must be speakable in roughly the time its cues take — shorten phrasing rather than overrun; never pad.

CONTEXTUAL PRECISION:
- {glossary}
- {translit}
- {honorific}

NEGATIVE CONSTRAINTS (DO NOT DO THIS):
- Never use robotic, mechanical, or textbook {lang}.
- Do not use archaic or overly obscure terms if they break the flow.
- Never output commentary, notes, explanations or pronunciation guides. Output ONLY the polished {lang} translation.
- Never leave any part of the passage in English.
"""

_OUTPUT_RULES = """OUTPUT FORMAT — JSON only:
{"passages": [{"n": <passage number>, "paragraphs": [{"cues": [<cue ids, ascending>], "text": "<LANG paragraph>"}]}]}
- Include every passage number exactly once. Do not merge, summarise or omit any passage.
- Within a passage, group consecutive cues into one paragraph per complete thought (usually one sentence; a cue split mid-sentence joins its neighbour).
- Every cue id of a passage appears in exactly one of its paragraphs, in order.
- Paragraph text is in LANG script only.
- NEVER put the double-quote character " inside a text value. For quoted speech use ‘ ’ or « » instead (e.g. मी म्हणालो, ‘ही फांदी कापून टाका.’). The reply must be valid JSON.
- PAUSES: each English cue is one spoken phrase, and the speaker pauses between cues. Inside a paragraph, write " | " (space, vertical bar, space) at the point where the English moves on to its next cue, so each stretch between markers is what is spoken during ONE cue. A paragraph of 3 cues has 2 markers. Never start or end a text with |, and never use | as sentence punctuation (use । or . as usual) — it is only this pause marker and is removed before the voice speaks.
- Each cue shows "~N chars": about how much LANG fits its duration at the voice's speed. Aim for that length per stretch (within about 10%); never pad.
"""


def build_static_prefix(language: str, prof: dict,
                        output_rules: bool = True,
                        house_rules: bool = False) -> str:
    """The invariant part of every request (lekhak systemPrefix): identical
    for every batch of a run, placed first so prompt caching hits."""
    parts = [_PRINCIPLES.format(lang=language,
                                glossary=glossary_directive(language),
                                translit=translit_directive(language),
                                honorific=honorific_directive(language))]
    g = (prof.get("guidelines") or "").strip()
    if g:
        parts.append("ADDITIONAL USER STYLE & TRANSLATION DIRECTIVES:\n" + g)
    tone = (prof.get("toneDescription") or "").strip()
    gram = (prof.get("grammarRules") or "").strip()
    vocab = prof.get("vocabularyMappings") or {}
    if tone or gram or vocab:
        # Compact JSON on purpose (lekhak): re-sent with every batch.
        parts.append("LEARNED STYLE PATTERNS (extracted from the user's own "
                     "proofread scripts):\n"
                     f"- Tone/Aesthetic: {tone or 'N/A'}\n"
                     "- Key Vocabulary Mappings:\n"
                     + json.dumps(vocab, ensure_ascii=False,
                                  separators=(",", ":")) + "\n"
                     f"- Grammatical/Pronoun Style: {gram or 'N/A'}")
    corr = [c for c in (prof.get("corrections") or [])
            if isinstance(c, dict) and c.get("draft") and c.get("final")]
    if corr:
        parts.append("RECENT HUMAN CORRECTIONS (the proofreader rewrote these "
                     "AI drafts — never repeat the rejected pattern):\n\n"
                     + "\n\n".join(f"English: {c.get('en', '')}\n"
                                   f"AI draft (rejected): {c['draft']}\n"
                                   f"Human final (approved): {c['final']}"
                                   for c in corr[-PROMPT_CORRECTIONS:]))
    # v0.24 "AI · test rules": the prompt-mode house rules come LAST, so they
    # win over the learned patterns and corrections above (user's choice).
    rules = _house_rules_text(language) if house_rules else ""
    if rules:
        parts.append("HOUSE DUBBING RULES — these override the learned "
                     "patterns and corrections above wherever they "
                     "conflict:\n" + rules)
    if output_rules:
        parts.append(_OUTPUT_RULES.replace("LANG", language))
    return "\n\n".join(parts) + "\n"


def _pages(cues: List[Tuple[int, float, float, str]]
           ) -> List[List[Tuple[int, float, float, str]]]:
    """Group cues into pages of <= PAGE_CHARS (lekhak packPages budget); a
    single long cue is its own page."""
    pages, cur, size = [], [], 0
    for c in cues:
        n = len(c[3]) + 1
        if cur and size + n > PAGE_CHARS:
            pages.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += n
    if cur:
        pages.append(cur)
    return pages


def _passage_block(n: int, page, next_start: Optional[float], lookup: dict,
                   terms, language: str, repair_note: str = "") -> str:
    """One passage (lekhak passageBlock): terminology, anchors, examples,
    then the timed English cues, then any repair note."""
    parts = [f"### PASSAGE {n}"]
    if terms:
        parts.append("REQUIRED TERMINOLOGY (use exactly these renderings):\n"
                     + "\n".join(f"  - \"{en}\" → \"{tr}\"" for en, tr in terms))
    anchors = lookup.get("anchors") or []
    if anchors:
        lines = []
        for a in anchors:
            exact = a["score"] >= 0.999
            lines.append(
                f"  • English: \"{a['source']}\"\n    {language}: \"{a['tr']}\"\n"
                + ("    This sentence was approved before. Reproduce this "
                   "rendering exactly." if exact else
                   f"    The English differs slightly ({a['score'] * 100:.0f}% "
                   "match). Keep this rendering and change only the words the "
                   "English changed."))
        parts.append("ALREADY APPROVED — these sentences of this very passage "
                     "are settled:\n" + "\n".join(lines))
    examples = lookup.get("examples") or []
    if examples:
        parts.append("HOW THIS SPEAKER HAS BEEN TRANSLATED BEFORE (approved "
                     "scripts — match this voice, vocabulary and spelling of "
                     "names):\n" + "\n".join(
                         f"  [{i}] English: \"{en}\"\n      {language}: \"{tr}\""
                         for i, (en, tr) in enumerate(examples, 1)))
    # v0.20: each cue carries a char budget (seconds x the language's
    # speaking rate) — per-phrase length control, as in isometric / length-
    # controlled dubbing MT (IWSLT 2022 isometric, VideoDubber).
    rate = LANG_CHARS_PER_SEC.get(language, DEFAULT_CHARS_PER_SEC)
    lines = []
    for k, (cid, s0, s1, text) in enumerate(page):
        nxt = page[k + 1][1] if k + 1 < len(page) else next_start
        gap = max(0.0, (nxt - s1)) if nxt is not None else 0.0
        dur = max(0.0, s1 - s0)
        lines.append(f'[{cid}] @{s0:.2f}s ({dur:.2f}s, pause after '
                     f'{gap:.2f}s, ~{max(1, int(round(dur * rate)))} chars): '
                     f'"{text}"')
    parts.append(f"ENGLISH CUES (ids {page[0][0]}-{page[-1][0]}):\n"
                 + "\n".join(lines))
    if repair_note:
        parts.append("THE PREVIOUS ATTEMPT AT THIS PASSAGE WAS REJECTED:\n"
                     f"{repair_note}\nFix exactly these problems. Everything "
                     "else about the passage was acceptable.")
    return "\n\n".join(parts)


def _as_int(x) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


_PAUSE_MARK = re.compile(r"\s*\|\s*")


def split_pause_markers(text: str) -> Tuple[str, List[int]]:
    """(clean text, pause offsets) for a translator paragraph (v0.20).

    The ' | ' markers the pause-aware prompt asks for must never reach the
    voice, the review screen or the learning memory, so they are removed
    here, where the reply is read. What is kept is WHERE they were: char
    offsets into the clean text (whitespace-normalised, stretches joined by
    one space) at which the next stretch starts — anchor_align.clause_units
    cuts there. A marker at the very start or end is dropped. Text with no
    '|' is returned untouched (byte-identical to before v0.20)."""
    if "|" not in (text or ""):
        return text, []
    segs = [" ".join(p.split()) for p in _PAUSE_MARK.split(text)]
    segs = [p for p in segs if p]
    offsets, pos = [], 0
    for k, seg in enumerate(segs):
        if k:
            offsets.append(pos)
        pos += len(seg) + 1
    return " ".join(segs), offsets


def _cue_list(v) -> list:
    """A paragraph's "cues" as a list, whatever shape the model wrote it in.
    Seen live: "cues": 12 (one bare number) killed a whole 25-minute run with
    "'int' object is not iterable". Also accepted: "12", "12, 13", "12-14"."""
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return list(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [v]
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(\d+)\s*[-–]\s*(\d+)\s*", v)
        if m and int(m.group(1)) <= int(m.group(2)) <= int(m.group(1)) + 200:
            return list(range(int(m.group(1)), int(m.group(2)) + 1))
        return re.findall(r"\d+", v)
    return []


def _rows_from_reply(paras, page) -> Tuple[List[Row], bool]:
    """(rows, repaired) for one passage. Rows cover EVERY cue of the page
    exactly once, in order, whatever the model got wrong: each paragraph
    owns the cues from its lowest valid id up to the next paragraph's;
    paragraphs whose claimed ranges overlap are merged, so English and
    translation always describe the same content. *repaired* is True when
    the model's claims needed any of this (-> COVERAGE warning).
    v0.20: pause markers are stripped from each text; every row carries
    "pauses" (char offsets of the markers in its clean "tr")."""
    if not isinstance(paras, list):
        raise ValueError("passage has no 'paragraphs' list")
    ids = [c[0] for c in page]
    idset = set(ids)
    claimed, seen, repaired = [], [], False
    for p in paras:
        if not isinstance(p, dict):
            continue
        text, pauses = split_pause_markers(str(p.get("text") or "").strip())
        if not text:
            continue
        raw = [_as_int(x) for x in _cue_list(p.get("cues"))]
        cues = [i for i in raw if i in idset]
        if len(cues) != len(raw):
            repaired = True
        seen.extend(cues)
        claimed.append([min(cues) if cues else None,
                        max(cues) if cues else None, text, pauses])
    if not claimed:
        raise ValueError("passage has no usable paragraphs")
    if sorted(seen) != ids:
        repaired = True
    last = ids[0]
    for a in claimed:
        if a[0] is None:
            a[0] = a[1] = last
            repaired = True
        last = a[0]
    merged: List[list] = []
    for lo, hi, text, pauses in sorted(claimed, key=lambda a: a[0]):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
            if pauses or merged[-1][3]:
                # offsets live in whitespace-normalised text
                head = " ".join(merged[-1][2].split())
                merged[-1][3] = merged[-1][3] + [o + len(head) + 1
                                                 for o in pauses]
                merged[-1][2] = head + " " + " ".join(text.split())
            else:
                merged[-1][2] += " " + text
            repaired = True
        else:
            merged.append([lo, hi, text, list(pauses)])
    merged[0][0] = ids[0]
    by_id = {c[0]: c for c in page}
    rows: List[Row] = []
    for k, (anchor, _hi, text, pauses) in enumerate(merged):
        stop = merged[k + 1][0] if k + 1 < len(merged) else ids[-1] + 1
        own = [by_id[i] for i in ids if anchor <= i < stop]
        rows.append({"en": " ".join(c[3] for c in own).strip(), "tr": text,
                     "start": own[0][1] if own else None,
                     "end": own[-1][2] if own else None,
                     "cues": [c[0] for c in own],
                     "pauses": pauses})
    return rows, repaired


_SALVAGE_N = re.compile(r'"n"\s*:\s*(\d+)')
_SALVAGE_PARA = re.compile(
    r'\{\s*"cues"\s*:\s*\[([^\]]*)\]\s*,\s*"text"\s*:\s*"(.*?)"\s*\}'
    r'(?=\s*[,\]])', re.S)


def _salvage_passages(raw: str):
    """Last-resort reader for a translator reply that is not valid JSON —
    typically a quote mark inside the Indic text that the model did not
    escape (मी म्हणालो, "ही फांदी…"). Each {"cues": [...], "text": "..."}
    object is pulled out directly (the text runs to the '"}' that closes
    it, so stray inner quotes survive) and assigned to the nearest preceding
    "n": marker. Returns {"passages": [...]} or None."""
    raw = _strip_code_fence(raw or "")
    marks = [(m.start(), int(m.group(1))) for m in _SALVAGE_N.finditer(raw)]
    by_n: Dict[int, list] = {}
    for m in _SALVAGE_PARA.finditer(raw):
        n = None
        for pos, num in marks:
            if pos < m.start():
                n = num
            else:
                break
        if n is None:
            n = 1
        cues = [int(x) for x in re.findall(r"\d+", m.group(1))]
        text = m.group(2).replace('\\"', '"').replace("\\n", " ").replace(
            "\\\\", "\\").strip()
        if text:
            by_n.setdefault(n, []).append({"cues": cues, "text": text})
    if not by_n:
        return None
    return {"passages": [{"n": n, "paragraphs": p}
                         for n, p in sorted(by_n.items())]}


def _complete_salvage(data, batch):
    """Keep only salvaged passages whose paragraphs claim EVERY cue of their
    page. A partial salvage silently dropped the paragraphs it could not
    read — whole sentences vanished from the dub — so an incomplete passage
    is discarded instead and its page goes to the solo retry."""
    if not data:
        return None
    keep = []
    for item in data.get("passages") or []:
        n = item.get("n")
        if not isinstance(n, int) or not 1 <= n <= len(batch):
            continue
        ids = {c[0] for c in batch[n - 1]["page"]}
        got = {_as_int(c) for p in item.get("paragraphs") or []
               if isinstance(p, dict) for c in _cue_list(p.get("cues"))}
        if ids <= got:
            keep.append(item)
    return {"passages": keep} if keep else None


def _call_batch(batch: List[dict], language: str, static: str, model: str,
                notes: Optional[Dict[int, str]] = None) -> Dict[int, tuple]:
    """One model call for up to AI_BATCH_SIZE passages (lekhak callBatch).
    Returns {page index: (rows, repaired)}; a passage the model dropped or
    garbled is simply absent (the caller retries it alone)."""
    blocks = [_passage_block(n, p["page"], p["next_start"], p["lookup"],
                             p["terms"], language,
                             (notes or {}).get(p["idx"], ""))
              for n, p in enumerate(batch, 1)]
    instruction = (f"\n\nTranslate each of the {len(batch)} passage(s) below "
                   f"into {language}. Return the JSON described above.\n\n")
    body = instruction + "\n\n".join(blocks)
    reply = _llm_generate(body, model, static_prefix=static,
                          role="translate", temperature=0.3)
    try:
        data = _parse_json_reply(reply)
    except ValueError:
        # Usually an unescaped quote inside the Indic text. Salvage the
        # paragraphs; failing that, ask once more, saying exactly why.
        data = _complete_salvage(_salvage_passages(reply), batch)
        if data is None:
            reply = _llm_generate(
                body + "\n\nYOUR PREVIOUS REPLY WAS NOT VALID JSON — most "
                "likely a double-quote character inside a text value. Reply "
                "again with valid JSON only; for quoted speech use ‘ ’ or « » "
                "instead of \".", model, static_prefix=static,
                role="translate", temperature=0.2)
            try:
                data = _parse_json_reply(reply)
            except ValueError:
                data = _complete_salvage(_salvage_passages(reply), batch)
                if data is None:
                    raise
    passages = data.get("passages") if isinstance(data, dict) else data
    out: Dict[int, tuple] = {}
    for item in passages if isinstance(passages, list) else []:
        if not isinstance(item, dict):
            continue
        n = _as_int(item.get("n"))
        if n is None or not 1 <= n <= len(batch):
            continue
        p = batch[n - 1]
        try:
            out[p["idx"]] = _rows_from_reply(item.get("paragraphs"), p["page"])
        except (ValueError, TypeError, KeyError):
            # One malformed passage must not sink the batch: it goes to the
            # solo retry like any other dropped passage.
            continue
    return out


def _run_batches(batches: list, fn, concurrency: int) -> list:
    if concurrency <= 1 or len(batches) <= 1:
        return [fn(b) for b in batches]
    with ThreadPoolExecutor(max_workers=min(concurrency, len(batches))) as ex:
        return list(ex.map(fn, batches))


def _page_report(o: dict) -> str:
    head = (f"Page {o['n']}  cues {o['cues'][0]}-{o['cues'][1]}  "
            f"tier {o['tier']}  risk {o['risk']:.2f}  "
            f"{'VERIFIED' if o['verified'] else 'needs review'}  "
            f"attempts {o['attempts']}")
    lines = [head]
    for c in o["checks"]:
        lines.append(f"  [{c['severity']}] {c['code']}: {c['detail']}")
    if o.get("critique"):
        for ln in o["critique"].splitlines():
            lines.append(f"  critic {ln}")
    for f in o.get("proof") or []:
        lines.append(f"  proof [{f['severity']}] \"{f['sentence']}\" — "
                     f"{f['reason']}")
        for s in f["suggestions"]:
            lines.append(f"      → {s}")
    if o.get("error"):
        lines.append(f"  error: {o['error']}")
    return "\n".join(lines)


def ai_translate_script(en_entries: List[Tuple[float, float, str]],
                        language: str, model: str = GEMINI_DEFAULT_MODEL,
                        status_cb: StatusCb = None,
                        options: Optional[dict] = None
                        ) -> Tuple[str, List[Row], dict]:
    """AI-mode script generation — Lekhak's agentic pipeline on dub pages.

    Returns (script_text, rows, info). *rows* pair every paragraph with its
    exact English cues and time window. *info* carries per-page outcomes
    (tier, checks, critique, verified, risk, proofer findings) and a
    human-readable ``report``. Raises only when a page ends with no
    translation at all — a half-translated script must not reach the voice.

    *options*: audit_rate (0.02), proofer (True), concurrency (3)."""
    say = status_cb or (lambda _m: None)
    opt = options or {}
    audit_rate = float(opt.get("audit_rate", AI_AUDIT_RATE))
    use_proofer = bool(opt.get("proofer", True))
    concurrency = max(1, int(opt.get("concurrency", AI_CONCURRENCY)))
    hr = bool(opt.get("house_rules", False))     # v0.24 "AI · test rules"

    cues = [(i + 1, float(s0), float(s1), (t or "").strip())
            for i, (s0, s1, t) in enumerate(en_entries) if (t or "").strip()]
    if not cues:
        raise RuntimeError("AI mode: the English transcript has no text to "
                           "translate.")
    prof = load_profile(language)
    guidelines = (prof.get("guidelines") or "").strip()
    glossary = prof.get("glossary") or {}
    pairs = []
    if translation_memory is not None:
        try:
            pairs = translation_memory.pairs_for(_tm_lang(language))
        except Exception:
            pairs = []
    memory = AiMemory(pairs)
    # Target/English length ratio from the approved pairs (Lekhak learns
    # its LENGTH band from memory the same way); None until 30 samples.
    rs = [len(tr) / len(en) for en, tr in pairs if len(en or "") > 40 and tr]
    exp_ratio = (sum(rs) / len(rs)) if len(rs) >= 30 else None
    static = build_static_prefix(language, prof, house_rules=hr)
    say(f"Learned profile: {profile_summary(prof)}; memory: {len(memory)} "
        f"approved pair(s); glossary: {len(glossary)} term(s).")

    pages = _pages(cues)
    outcomes: Dict[int, dict] = {}
    pending: List[dict] = []

    # ── 1. The memory, before anything is spent ──
    for idx, page in enumerate(pages):
        source = " ".join(c[3] for c in page)
        lookup = memory.lookup(source)
        terms = terms_in(source, glossary)
        base = {"n": idx + 1, "cues": (page[0][0], page[-1][0]),
                "source": source, "tier": lookup["tier"],
                "tm_score": lookup["score"], "critique": "",
                "verified": False, "proof": [], "error": ""}
        if lookup["tier"] == 1 and lookup["draft"]:
            rows = [{"en": source, "tr": lookup["draft"],
                     "start": page[0][1], "end": page[-1][2],
                     "cues": [c[0] for c in page]}]
            checks = run_checks(source, lookup["draft"], language, rows, terms,
                                expected_ratio=exp_ratio)
            # Structurally unfit memory text goes to the model instead
            # (lekhak unusableDraft); anything else ships, flagged if needed.
            if not any(c["code"] in _AI_STRUCTURAL and c["severity"] == "fail"
                       for c in checks):
                outcomes[idx] = dict(base, rows=rows, text=lookup["draft"],
                                     checks=checks, attempts=0,
                                     verified=not has_failure(checks),
                                     risk=risk_score(checks, 1,
                                                     lookup["score"],
                                                     len(source)))
                continue
            # The memory text itself is broken — never hand it back to the
            # model as "reproduce this exactly".
            lookup = dict(lookup, anchors=[])
        pending.append({"idx": idx, "page": page, "lookup": lookup,
                        "terms": terms, "base": base,
                        "next_start": pages[idx + 1][0][1]
                        if idx + 1 < len(pages) else None})
    reused = len(outcomes)
    if reused:
        say(f"Memory: {reused} page(s) reused verbatim (tier 1, no model "
            "call).")

    def _finish(p, rows, repaired, attempts, error=""):
        text = "\n\n".join(str(r["tr"]) for r in rows)
        checks = (run_checks(p["base"]["source"], text, language, rows,
                             p["terms"], coverage_repaired=repaired,
                             expected_ratio=exp_ratio)
                  if rows else [{"code": "EMPTY", "severity": "fail",
                                 "detail": error or "No output was produced."}])
        return dict(p["base"], rows=rows, text=text, checks=checks,
                    attempts=attempts, error=error,
                    risk=risk_score(checks, p["base"]["tier"],
                                    p["base"]["tm_score"],
                                    len(p["base"]["source"])))

    # ── 2. Batched generation (+ one solo retry for a dropped passage) ──
    if pending:
        batches = [pending[i:i + AI_BATCH_SIZE]
                   for i in range(0, len(pending), AI_BATCH_SIZE)]
        say(f"Translator: {len(pending)} page(s) in {len(batches)} batch(es), "
            f"{min(concurrency, len(batches))} in parallel…")

        def _gen(batch):
            try:
                return _call_batch(batch, language, static, model), ""
            except Exception as e:                   # noqa: BLE001
                return {}, str(e)[:200]

        results = _run_batches(batches, _gen, concurrency)
        got: Dict[int, tuple] = {}
        errors: Dict[int, str] = {}
        for batch, (res, err) in zip(batches, results):
            got.update(res)
            for p in batch:
                if p["idx"] not in res and err:
                    errors[p["idx"]] = err
        for p in pending:
            if p["idx"] in got:
                continue
            res, err = _gen([p])
            got.update(res)
            if err:
                errors[p["idx"]] = err
        for p in pending:
            rows, repaired = got.get(p["idx"], ([], False))
            outcomes[p["idx"]] = _finish(p, rows, repaired, 1,
                                         errors.get(p["idx"], ""))

    # ── 3-4. Repair, using the deterministic checks as the feedback ──
    broken = [p for p in pending
              if outcomes[p["idx"]]["rows"]
              and has_failure(outcomes[p["idx"]]["checks"])]
    if broken:
        say(f"Repair: {len(broken)} page(s) failed a check — one repair pass "
            "with the check notes…")
        notes = {p["idx"]: "\n".join(f"- {c['detail']}"
                                     for c in outcomes[p["idx"]]["checks"])
                 for p in broken}
        by_idx = {p["idx"]: p for p in broken}
        rbatches = [broken[i:i + AI_BATCH_SIZE]
                    for i in range(0, len(broken), AI_BATCH_SIZE)]

        def _rep(batch):
            try:
                return _call_batch(batch, language, static, model, notes)
            except Exception:                        # noqa: BLE001
                return {}

        for res in _run_batches(rbatches, _rep, concurrency):
            for idx, (rows, repaired) in res.items():
                prev = outcomes[idx]
                cand = _finish(by_idx[idx], rows, repaired,
                               prev["attempts"] + 1)
                # Only accept the repair if it is actually an improvement.
                if fail_count(cand["checks"]) <= fail_count(prev["checks"]):
                    outcomes[idx] = cand

    # ── 5. Critic panel: still failing + an audit sample ──
    judge = [p for p in pending if outcomes[p["idx"]]["rows"]
             and (has_failure(outcomes[p["idx"]]["checks"])
                  or random.random() < audit_rate)]
    if judge:
        say(f"Critic panel: judging {len(judge)} page(s) (3 critics each)…")

        def _panel(p):
            o = outcomes[p["idx"]]
            try:
                return p["idx"], run_critic_panel(
                    o["source"], o["text"], language, guidelines, prof, model,
                    house_rules=hr)
            except Exception as e:                   # noqa: BLE001
                return p["idx"], {"error": str(e)[:200]}

        for idx, panel in _run_batches(judge, _panel, concurrency):
            o = outcomes[idx]
            if panel.get("error"):
                o["critique"] = f"Critic panel failed: {panel['error']}"
                continue
            o["critique"] = (
                f"Not verified: {len(panel['degraded'])} of 3 critics could "
                f"not run ({', '.join(panel['degraded'])})."
                if panel["degraded"] else "\n".join(panel["feedbacks"]))
            o["verified"] = (panel["isValid"] and not panel["degraded"]
                             and not has_failure(o["checks"]))
            o["risk"] = risk_score(o["checks"], o["tier"], o["tm_score"],
                                   len(o["source"]),
                                   critic_flagged=not panel["isValid"])

    # Clean by the checks and untouched by the panel -> verified.
    for o in outcomes.values():
        if not o["critique"] and o["rows"] and not has_failure(o["checks"]):
            o["verified"] = True

    # ── 6. Proofer: sentence-level findings for the human pass ──
    if use_proofer:
        items = [{"id": i, "source": o["source"], "target": o["text"]}
                 for i, o in sorted(outcomes.items())
                 if o["rows"] and o["tier"] != 1]
        if items:
            say(f"Proofer: reading {len(items)} page(s)…")
            found = proof_pages(items, language, guidelines, prof, model,
                                status_cb=say, house_rules=hr)
            for i, findings in found.items():
                if not findings:
                    continue
                o = outcomes[i]
                o["proof"] = findings
                reviews = sum(1 for f in findings if f["severity"] == "review")
                o["risk"] = round(min(1.0, o["risk"]
                                      + min(0.18, reviews * 0.06)), 3)

    dead = sorted(i for i, o in outcomes.items() if not o["rows"])
    if dead:
        raise RuntimeError(
            f"AI mode: page(s) {[i + 1 for i in dead]} of {len(pages)} could "
            f"not be translated — {outcomes[dead[0]].get('error') or 'no output'}")

    ordered = [outcomes[i] for i in range(len(pages))]
    rows = [r for o in ordered for r in o["rows"]]
    script = "\n\n".join(str(r["tr"]) for r in rows)
    verified = sum(1 for o in ordered if o["verified"])
    findings = sum(len(o["proof"]) for o in ordered)
    summary = (f"{len(pages)} page(s): {reused} from memory, "
               f"{verified} verified, {len(pages) - verified} need review, "
               f"{sum(1 for o in ordered if o['critique'])} judged by critics, "
               f"{findings} proofer finding(s).")
    report = ("AI MODE REVIEW REPORT — " + language + "\n" + summary + "\n"
              "Pages sorted by risk (highest first). Proofer suggestions are "
              "drop-in replacements for the quoted sentence.\n\n"
              + "\n\n".join(_page_report(o) for o in
                            sorted(ordered, key=lambda o: -o["risk"])) + "\n")
    info = {
        "pages": len(pages), "paragraphs": len(rows), "reused": reused,
        "verified": verified, "findings": findings, "summary": summary,
        "report": report, "profile": profile_summary(prof),
        "page_outcomes": [{k: o[k] for k in
                           ("n", "cues", "tier", "tm_score", "risk",
                            "verified", "attempts", "checks", "critique",
                            "proof")} for o in ordered],
    }
    return script, rows, info


MIN_KEEP_RATIO = 0.5         # a "shorter" line below half the original lost meaning


def _digits(s: str) -> set:
    """Digit runs with every script's digits folded to ASCII."""
    import unicodedata
    out, cur = set(), ""
    for ch in s or "":
        d = unicodedata.digit(ch, None)
        if d is not None:
            cur += str(d)
        elif cur:
            out.add(cur)
            cur = ""
    if cur:
        out.add(cur)
    return out


def meaning_kept(original: str, shorter: str) -> bool:
    """Guard for every automatic shortening (fit suggestions, the anchor
    fit retry, learning): a shorter line must keep every number of the
    original and at least MIN_KEEP_RATIO of its length. A line that drops
    '108' or half its words is not a shorter rendering, it is a different
    (truncated) line — the failure that turned 'the distance between the
    moon and the earth IS 108 TIMES THE DIAMETER OF THE MOON' into 'the
    moon-earth distance'."""
    # v0.20: pause markers are never spoken — they must not count.
    o = split_pause_markers((original or "").strip())[0].strip()
    s = split_pause_markers((shorter or "").strip())[0].strip()
    if not s:
        return False
    if not _digits(o) <= _digits(s):
        return False
    return len(s) >= MIN_KEEP_RATIO * len(o)


_PARTIAL_EN_NOTE = (
    "The English shown may be only PART of the sentence (a subtitle "
    "fragment). Your rendering must still say EVERYTHING the current "
    "{lang} line says — every clause, name and number — just more "
    "compactly. Never drop a clause to save time.")


def fit_to_seconds(english: str, current: str, seconds: float,
                   language: str, model: str = GEMINI_DEFAULT_MODEL,
                   house_rules: bool = False) -> str:
    """Anchor sync's targeted retry (Lekhak's repair idea applied to
    timing): rewrite ONE line so it can be spoken in *seconds*, keeping its
    meaning, in the learned house voice. Returns the new line, or "" when
    the model gave nothing usable (the caller keeps the original)."""
    prof = load_profile(language)
    try:                              # v0.28: the measured speaking speed
        from .sync_learn import learned_cps
        # never above the old 11 chars/s guess: a shortened line keeps
        # room to spare (v0.28.3, the measured speed made them too long)
        cps = min(learned_cps(language) or 11.0, 11.0)
    except Exception:                                    # noqa: BLE001
        cps = 11.0
    target_chars = max(8, int(seconds * cps))
    prompt = (
        f"\n\nThis {language} dubbing line is too long to speak in "
        f"{seconds:.1f} seconds. Rewrite it so it fits (about "
        f"{target_chars} characters or fewer), keeping the meaning, the "
        "register and any names or numbers. Drop filler before meaning. "
        + _PARTIAL_EN_NOTE.format(lang=language) + "\n\n"
        f"ENGLISH: \"{english}\"\n{language} (too long): \"{current}\"\n\n"
        f"Reply with JSON only: {{\"text\": \"<shorter {language} line>\"}}")
    try:
        data = _parse_json_reply(_llm_generate(
            prompt, model,
            static_prefix=build_static_prefix(language, prof,
                                              output_rules=False,
                                              house_rules=house_rules),
            role="translate", temperature=0.3))
        text = str((data or {}).get("text") or "").strip() \
            if isinstance(data, dict) else ""
    except Exception:                                # noqa: BLE001
        return ""
    return text if (text and len(text) < len(current)
                    and meaning_kept(current, text)) else ""


def suggest_fits(english: str, current: str, speech_s: float, hard_s: float,
                 language: str, model: str = GEMINI_DEFAULT_MODEL,
                 n: int = 2, house_rules: bool = False) -> List[str]:
    """Review-screen fit suggestions: up to *n* shorter renderings of one
    line, aimed at its speech slot (*speech_s*: how long the English
    speaker talks) and never past its hard slot (*hard_s*: that plus the
    pause after). Written in the learned house voice when a profile exists.
    Returns [] when nothing usable came back — the reviewer keeps typing."""
    from .config import DEFAULT_CHARS_PER_SEC, LANG_CHARS_PER_SEC, \
        estimate_duration
    rate = LANG_CHARS_PER_SEC.get(language, DEFAULT_CHARS_PER_SEC)
    target = max(6, int(max(0.3, speech_s) * rate))
    prompt = (
        f"\n\nThis {language} dubbing line is too long for its slot. The "
        f"English speaker talks for {speech_s:.1f} s (with the pause after "
        f"it: {hard_s:.1f} s). Write {n} different shorter {language} "
        f"renderings, each about {target} characters or fewer, that keep the "
        "meaning, the register, and every name and number. Prefer dropping "
        "filler and repetition over dropping meaning; keep it natural to "
        "speak. " + _PARTIAL_EN_NOTE.format(lang=language) + "\n\n"
        f"ENGLISH: \"{english}\"\n{language} (too long, {len(current)} "
        f"chars): \"{current}\"\n\n"
        f"Reply with JSON only: {{\"options\": [\"<{language} line>\", ...]}}")
    prof = load_profile(language)
    try:
        data = _parse_json_reply(_llm_generate(
            prompt, model,
            static_prefix=build_static_prefix(language, prof,
                                              output_rules=False,
                                              house_rules=house_rules),
            role="translate", temperature=0.5))
    except Exception:                                # noqa: BLE001
        return []
    raw = data.get("options") if isinstance(data, dict) else data
    out, seen = [], set()
    for o in raw if isinstance(raw, list) else []:
        t = " ".join(str(o or "").split())
        if not t or t in seen or len(t) >= len(current):
            continue
        if not meaning_kept(current, t):            # dropped a clause/number
            continue
        seen.add(t)
        out.append(t)
    # Lines that fit the hard slot first, then the shortest overrun.
    out.sort(key=lambda t: (estimate_duration(t, language) > hard_s,
                            abs(estimate_duration(t, language) - speech_s)))
    return out[:n]


def fit_chunks(rows, language: str, chars_per_sec: float = 0.0,
               prev_line: str = "", next_line: str = "",
               model: str = GEMINI_DEFAULT_MODEL,
               house_rules: bool = False) -> dict:
    """v0.32 Regenerate tab, "Write a line for each chunk": the reviewer has
    placed several consecutive dub chunks on the timeline where the English
    is spoken. Write ONE fresh line per chunk that says what the English
    says inside that chunk's window, in the English order, sized to the
    chunk's placed length. One call for the whole selection so the lines
    read as one continuous speech.

    rows: [{"k": int, "len": seconds, "en": str, "cur": str}] in timeline
    order. *chars_per_sec* is the speaking rate measured on the timeline
    (0 = the language default). Returns {k: text}; a chunk the model
    skipped is simply absent (the panel keeps its current line)."""
    from .config import DEFAULT_CHARS_PER_SEC, LANG_CHARS_PER_SEC
    rows = [r for r in rows or [] if (r.get("en") or r.get("cur"))]
    if not rows:
        return {}
    rate = chars_per_sec if chars_per_sec and chars_per_sec > 1 else \
        LANG_CHARS_PER_SEC.get(language, DEFAULT_CHARS_PER_SEC)
    no_en = "(no English under it: continue the thought of its neighbours)"
    listing = "\n".join(
        f"[{r['k']}] {r['len']:.1f} s, ~{max(4, int(r['len'] * rate))} chars"
        f"\n  ENGLISH: \"{r.get('en') or no_en}\""
        + (f"\n  CURRENT {language}: \"{r['cur']}\"" if r.get("cur") else "")
        for r in rows)
    prompt = (
        f"\n\nThe reviewer selected {len(rows)} consecutive {language} dub "
        "chunks. Each chunk's time is how long its ENGLISH takes in the "
        "source audio (start to end of the English speech), so the "
        f"{language} line must be spoken in that same time. "
        "Write ONE new line per chunk:\n"
        "- It says what the ENGLISH of that chunk says — nothing from a "
        "neighbouring chunk, nothing left out (every name and number).\n"
        "- It follows the English order of ideas and key words as closely "
        f"as {language} grammar allows.\n"
        "- It FILLS the chunk's time: about the given character count "
        "(within about 10%). If a literal rendering is too short, use "
        "fuller natural phrasing (a connecting word, the full form instead "
        "of a pronoun) — never new meaning. If too long, compact the "
        "wording — never drop meaning.\n"
        "- [pause N s] in the English marks where the speaker pauses "
        "inside the chunk: put a comma, or … for a pause of a second or "
        "more, at the same point of the line. Never write the [pause] "
        "tag itself.\n"
        "- Read together, the lines are one natural continuous speech: a "
        "sentence may run across chunks, so each line is the part of it "
        "spoken in that window.\n"
        "- CURRENT is only context; rewrite freely.\n"
        "- No pause markers (|), no commentary.\n\n"
        + (f"LINE BEFORE THE SELECTION: \"{prev_line}\"\n" if prev_line else "")
        + "CHUNKS:\n" + listing + "\n"
        + (f"LINE AFTER THE SELECTION: \"{next_line}\"\n" if next_line else "")
        + f"\nReply with JSON only: {{\"lines\": [{{\"k\": <chunk number>, "
        f"\"text\": \"<{language} line>\"}}]}}")
    prof = load_profile(language)
    try:
        data = _parse_json_reply(_llm_generate(
            prompt, model,
            static_prefix=build_static_prefix(language, prof,
                                              output_rules=False,
                                              house_rules=house_rules),
            role="translate", temperature=0.3))
    except Exception:                                # noqa: BLE001
        return {}
    wanted = {int(r["k"]) for r in rows}
    out = {}
    lines = data.get("lines") if isinstance(data, dict) else data
    for ln in lines if isinstance(lines, list) else []:
        if not isinstance(ln, dict):
            continue
        try:
            k = int(ln.get("k"))
        except (TypeError, ValueError):
            continue
        text = str(ln.get("text") or "")
        # Never voice a copied [pause N s] tag.
        text = re.sub(r"\[\s*pause[^\]]*\]", " ", text, flags=re.I)
        text = " ".join(split_pause_markers(text)[0].split())
        if k in wanted and text and k not in out:
            out[k] = text
    return out


def recommend_voices(language: str, voices, line: str = "", english: str = "",
                     current: str = "", speaker: str = "", n: int = 3,
                     model: str = GEMINI_DEFAULT_MODEL) -> List[dict]:
    """Pick the *n* voices from the account catalogue that best suit this
    line: language support, the speaker (e.g. a calm male spiritual
    teacher), the line's tone, and continuity with the current voice. Uses
    only what the catalogue carries (name + label). No prompt files.
    voices: [(id, name)]. Returns [{"id", "why"}] with ids from *voices*;
    [] on any failure."""
    cat = [(str(i), str(nm)) for i, nm in voices if str(i).strip()][:300]
    if not cat:
        return []
    listing = "\n".join(f"{i} | {nm}" for i, nm in cat)
    prompt = (
        f"\n\nChoose the {n} best ElevenLabs voices from the catalogue below "
        f"for dubbing this {language} line.\n"
        f"SPEAKER: {speaker or 'the original English speaker'}\n"
        f"CURRENT VOICE: {current or 'none'}\n"
        f"ENGLISH: \"{english}\"\n{language} LINE: \"{line}\"\n\n"
        "Judge from each voice's name and label: native or strong "
        f"{language} support first (names/labels mentioning {language} or "
        "its region, or a ✦ mark), a gender/age/character that suits the "
        "speaker, a tone that suits the line (warm, calm, narrator-like for "
        "spiritual talks), and continuity with the current voice. Never pick "
        "an obviously wrong gender or a comic/character voice for a "
        "narrator.\n\nCATALOGUE (id | name):\n" + listing + "\n\n"
        "JSON only: {\"picks\": [{\"id\": \"<voice id>\", \"why\": \"<one "
        "short English reason>\"}]}")
    try:
        data = _parse_json_reply(_llm_generate(prompt, model, role="translate",
                                               temperature=0.2))
    except Exception:                                # noqa: BLE001
        return []
    ids = {i for i, _nm in cat}
    out, seen = [], set()
    picks = data.get("picks") if isinstance(data, dict) else None
    for p in picks if isinstance(picks, list) else []:
        if not isinstance(p, dict):
            continue
        vid = str(p.get("id") or "").strip()
        if vid in ids and vid not in seen:
            seen.add(vid)
            out.append({"id": vid,
                        "why": " ".join(str(p.get("why") or "").split())})
    return out[:n]


ASSIST_SPEED_MIN, ASSIST_SPEED_MAX = 0.80, 1.25


def review_assist(english: str, current: str, speech_s: float, hard_s: float,
                  instruction: str, language: str, history=(),
                  prev_line: str = "", next_line: str = "",
                  model: str = GEMINI_DEFAULT_MODEL,
                  house_rules: bool = False) -> dict:
    """The review screen's assistant for ONE line: follows the reviewer's
    instruction (shorter / longer / speed up / slow down / more formal /
    alternatives ...) in the learned house voice. No prompt files.

    Returns {"reply": str, "alternatives": [str], "speed": float|None}.
    Speed is a playback factor for this paragraph (1.10 = 10 % faster,
    0.90 = slower), clamped to [0.80, 1.25]. Never raises."""
    from .config import DEFAULT_CHARS_PER_SEC, LANG_CHARS_PER_SEC, \
        estimate_duration
    rate = LANG_CHARS_PER_SEC.get(language, DEFAULT_CHARS_PER_SEC)
    est = estimate_duration(current, language)
    hist = "\n".join(f"{'Reviewer' if r == 'user' else 'Assistant'}: {t}"
                     for r, t in list(history)[-8:])
    prompt = (
        f"\n\nYou are the dubbing-script assistant on the review screen. The "
        f"reviewer is working on ONE {language} line of a voice-over.\n"
        f"TIMING: the English speaker talks {speech_s:.1f} s here "
        f"({hard_s:.1f} s including the pause after). The current line is "
        f"about {est:.1f} s at {rate:.1f} chars/s ({len(current)} chars; "
        f"~{int(speech_s * rate)} chars would fill the speech slot exactly).\n"
        + (f"PREVIOUS LINE: \"{prev_line}\"\n" if prev_line else "")
        + f"ENGLISH: \"{english}\"\nCURRENT {language}: \"{current}\"\n"
        + (f"NEXT LINE: \"{next_line}\"\n" if next_line else "")
        + (f"\nCONVERSATION SO FAR:\n{hist}\n" if hist else "")
        + f"\nREVIEWER'S INSTRUCTION: \"{instruction}\"\n\n"
        "Do exactly what the reviewer asks:\n"
        "- Shorter / longer / a target length or seconds: give 2-3 "
        "alternatives at that length. Keep names and numbers unless told "
        "otherwise; a longer line may add natural connecting words, never "
        "new meaning.\n"
        "- TRANSLATE instructions: write new renderings FROM THE ENGLISH (the "
        "current line may be wrong or incomplete); every one must carry the "
        "whole English meaning.\n"
        "- SPEED ONLY instructions: never rewrite the line; return an empty "
        "\"alternatives\" list and set only \"speed\".\n"
        "- Speed up / slow down / faster / slower: set \"speed\" to a factor "
        f"between {ASSIST_SPEED_MIN} and {ASSIST_SPEED_MAX} (1.10 = 10% "
        "faster, 0.90 = 10% slower); if a number is given use it, otherwise "
        "choose the factor that makes the line fit its slot. Alternatives "
        "are optional then.\n"
        "- Tone, word choice, alternatives, questions: answer, and give 2-3 "
        "alternatives when useful.\n"
        f"- Alternatives are complete drop-in {language} lines. Never put "
        "the \" character inside them; use ‘ ’ for quotes.\n"
        "- \"reply\" is one or two short English sentences explaining what "
        "you did or recommend.\n\n"
        "JSON only: {\"reply\": \"...\", \"alternatives\": [\"...\"], "
        "\"speed\": null}")
    out = {"reply": "", "alternatives": [], "speed": None}
    try:
        data = _parse_json_reply(_llm_generate(
            prompt, model,
            static_prefix=build_static_prefix(language, load_profile(language),
                                              output_rules=False,
                                              house_rules=house_rules),
            role="translate", temperature=0.5))
    except Exception as e:                           # noqa: BLE001
        out["reply"] = f"The assistant could not answer ({str(e)[:120]})."
        return out
    if not isinstance(data, dict):
        return out
    out["reply"] = " ".join(str(data.get("reply") or "").split())
    # v0.23: the panel's Speed up / Slow down send "SPEED ONLY" — the script
    # must not change, so any rewritten line the model offers is dropped here
    # rather than trusted to the prompt.
    speed_only = "SPEED ONLY" in (instruction or "").upper()
    seen = set()
    for a in ([] if speed_only else (data.get("alternatives") or [])):
        t = " ".join(str(a or "").split())
        if t and t != current and t not in seen:
            seen.add(t)
            out["alternatives"].append(t)
    try:
        sp = data.get("speed")
        if sp is not None:
            out["speed"] = round(min(ASSIST_SPEED_MAX,
                                     max(ASSIST_SPEED_MIN, float(sp))), 2)
    except (TypeError, ValueError):
        out["speed"] = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  2. Learn
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return " ".join((s or "").split())


def pair_final_with_draft(draft_rows: List[Row], final_paras: List[str],
                          en_entries: List[Tuple[float, float, str]]
                          ) -> Tuple[List[Tuple[str, str]], List[dict]]:
    """(approved_pairs, corrections) for a reviewed script.

    With an AI draft of the same paragraph count (the review screen keeps
    rows 1:1) pairing is exact and every changed row is a correction.
    Otherwise — rows added/removed, or no draft (TM hit, prompt-mode
    translate) — the pipeline's char-share pairing is used and no
    corrections are claimed, because a row cannot be matched to its draft."""
    final = [p.strip() for p in final_paras if p and p.strip()]
    if draft_rows and len(draft_rows) == len(final):
        pairs, corr = [], []
        for r, fin in zip(draft_rows, final):
            en = str(r.get("en") or "").strip()
            if not en:
                continue
            # A row whose English was only estimated (memory rows paired by
            # length: "cues" present but empty) must not be memorised as an
            # English -> translation pair — that is how a split sentence got
            # stored under half its English.
            if "cues" in r and not r.get("cues"):
                continue
            draft = str(r.get("tr") or "")
            # A final that lost numbers or half the draft is a truncation
            # (e.g. an accepted over-eager fit suggestion), not the house
            # style: never memorise it or teach it as a correction.
            if not meaning_kept(draft, fin):
                continue
            pairs.append((en, fin))
            if _norm(draft) != _norm(fin):
                corr.append({"en": en, "draft": draft, "final": fin})
        return pairs, corr
    rows = _pair_review_rows(en_entries, final)
    return [(en, tr) for (en, tr, _a, _b) in rows if en and tr], []


_LEARN_PROMPT = """You are a linguistic analyst maintaining the style profile that teaches an AI dubbing translator to write LANG exactly like one human proofreader.

You get (1) the CURRENT profile, (2) CORRECTIONS — rows where the proofreader rewrote the AI draft (strongest evidence), and (3) APPROVED PAIRS — finalized English -> LANG rows.

MERGE the new evidence into the profile:
- Keep existing rules and mappings unless new evidence contradicts them; a correction overrides older guidance.
- toneDescription: register, warmth, sentence rhythm, how spiritual ideas are voiced.
- grammarRules: honorific plural/singular choices, pronouns, verb endings, active vs passive, sentence length, punctuation habits — concrete and imperative.
- vocabularyMappings: English word/phrase -> the exact LANG term the proofreader uses. Only terms with evidence. At most MAXVOCAB entries; drop the least useful if over.
- Be concise: tone and grammar under 120 words each.

Reply with JSON only:
{"toneDescription": "...", "grammarRules": "...", "vocabularyMappings": {"english": "LANG term"}}
"""


def _learn_evidence(prof: dict, corrections: List[dict],
                    pairs: List[Tuple[str, str]], language: str) -> str:
    cur = {k: prof.get(k) for k in
           ("toneDescription", "grammarRules", "vocabularyMappings")}
    budget = LEARN_CHARS_CAP
    parts = ["=== CURRENT PROFILE ===\n"
             + json.dumps(cur, ensure_ascii=False, indent=1)]
    budget -= len(parts[0])
    blk = []
    for i, c in enumerate(corrections, 1):
        s = (f"[Correction {i}]\nEnglish: {c['en']}\n"
             f"AI draft: {c['draft']}\nHuman final: {c['final']}")
        if budget - len(s) < 0:
            break
        budget -= len(s)
        blk.append(s)
    if blk:
        parts.append("=== CORRECTIONS ===\n" + "\n\n".join(blk))
    blk = []
    for i, (en, tr) in enumerate(pairs[:LEARN_PAIRS_CAP], 1):
        s = f"[Pair {i}]\nEnglish: \"{en}\"\n{language}: \"{tr}\""
        if budget - len(s) < 0:
            break
        budget -= len(s)
        blk.append(s)
    if blk:
        parts.append("=== APPROVED PAIRS ===\n" + "\n\n".join(blk))
    return "\n\n" + "\n\n".join(parts) + "\n\nTask: return the merged profile."


def learn_from_review(language: str, draft_rows: List[Row],
                      final_script: str,
                      en_entries: List[Tuple[float, float, str]],
                      model: str = GEMINI_DEFAULT_MODEL, source: str = "",
                      status_cb: StatusCb = None) -> dict:
    """Self-learning step, run when a reviewed script is approved.

    Stores the approved rows in the translation memory, records the
    corrections, and merges a new style profile. Never raises: returns
    {"ok", "pairs", "tm_stored", "corrections", "profile_path", "error"}."""
    say = status_cb or (lambda _m: None)
    out = {"ok": False, "pairs": 0, "tm_stored": 0, "corrections": 0,
           "profile_path": "", "error": ""}
    try:
        final_paras = _split_translation_paragraphs(final_script)
        pairs, corrections = pair_final_with_draft(draft_rows, final_paras,
                                                   en_entries)
        out["pairs"], out["corrections"] = len(pairs), len(corrections)
        held = [i + 1 for i, (r, fin) in enumerate(zip(draft_rows or [],
                                                       final_paras))
                if len(draft_rows or []) == len(final_paras)
                and not meaning_kept(str(r.get("tr") or ""), fin)]
        out["held_back"] = held
        if held:
            say(f"WARNING: paragraph(s) {held} lost numbers or more than half "
                "their text versus the draft — held back from memory and "
                "corrections (and the whole script is not memorised), so a "
                "truncation is never reused or learned.")
        if not pairs:
            out["error"] = "no approved English/translation pairs"
            return out
        src = " ".join(t.strip() for (_a, _b, t) in en_entries
                       if t and t.strip())
        full = (src, final_script.strip()) if src and not held else None
        return _absorb_evidence(language, pairs, corrections, full, model,
                                source, say, out, "review")
    except Exception as e:
        out["error"] = str(e)[:200]
        say(f"WARNING: AI learning failed ({out['error']}) — the dub "
            "continues unaffected.")
        return out


def _absorb_evidence(language: str, pairs: List[Tuple[str, str]],
                     corrections: List[dict],
                     full: Optional[Tuple[str, str]], model: str,
                     source: str, say: Callable[[str], None], out: dict,
                     what: str) -> dict:
    """The shared half of both learners (review, final dub): translation
    memory, corrections (newest MAX_CORRECTIONS kept), one profile-merge LLM
    call over bounded evidence, save. *full* = (english_doc, final_doc) to
    memorise the whole document, or None. Fills and returns *out*; never
    raises (the callers wrap it too)."""
    try:
        lang = _tm_lang(language)

        # (a) Translation memory — exact reuse next time.
        if translation_memory is not None:
            try:
                if full and full[0] and full[1]:
                    translation_memory.store_full(lang, full[0], full[1],
                                                  source)
                out["tm_stored"] = translation_memory.store_pairs(
                    lang, pairs, source) or 0
            except Exception as e:
                say(f"WARNING: translation memory not updated ({e}).")

        # (b) Profile merge — one LLM call, bounded evidence.
        prof = load_profile(language)
        stamp = time.strftime("%Y-%m-%d")
        prof["corrections"] = ((prof.get("corrections") or [])
                               + [dict(c, at=stamp) for c in corrections]
                               )[-MAX_CORRECTIONS:]
        # Most-edited rows first — the learner sees the strongest signal
        # even when the char budget cuts the tail.
        ranked = sorted(corrections,
                        key=lambda c: -abs(len(c["final"]) - len(c["draft"])))
        prefix = (_LEARN_PROMPT.replace("LANG", language)
                  .replace("MAXVOCAB", str(MAX_VOCAB)))
        try:
            reply = _llm_generate(
                _learn_evidence(prof, ranked, pairs, language), model,
                static_prefix=prefix, role="translate")
            data = _parse_json_reply(reply)
            if not isinstance(data, dict):
                raise ValueError("profile reply is not an object")
            for k in ("toneDescription", "grammarRules"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    prof[k] = v.strip()
            vocab = data.get("vocabularyMappings")
            if isinstance(vocab, dict) and vocab:
                clean = {str(k).strip(): str(v).strip()
                         for k, v in vocab.items()
                         if str(k).strip() and str(v).strip()}
                prof["vocabularyMappings"] = dict(
                    list(clean.items())[:MAX_VOCAB])
        except Exception as e:
            # Corrections + TM are still saved; only the merge is skipped.
            out["error"] = f"profile merge skipped ({str(e)[:160]})"
            say(f"WARNING: {out['error']}")
        prof["samples"] = int(prof.get("samples") or 0) + len(pairs)
        prof["runs"] = int(prof.get("runs") or 0) + 1
        prof["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        prof["language"] = language
        prof["version"] = PROFILE_VERSION
        path = save_profile(language, prof)
        out["profile_path"] = path or ""
        out["ok"] = bool(path)
        if not path:
            out["error"] = out["error"] or "profile file could not be written"
        say(f"Learned from {what}: {len(pairs)} approved row(s), "
            f"{len(corrections)} correction(s), {out['tm_stored']} stored "
            f"in memory — profile now {profile_summary(prof)}.")
        return out
    except Exception as e:
        out["error"] = str(e)[:200]
        say(f"WARNING: AI learning failed ({out['error']}) — the dub "
            "continues unaffected.")
        return out


def learn_from_final(language: str, pairs: List[Tuple[str, str]],
                     corrections: List[dict], full_text: str = "",
                     en_entries: Optional[List[Tuple[float, float, str]]]
                     = None, model: str = GEMINI_DEFAULT_MODEL,
                     source: str = "", status_cb: StatusCb = None) -> dict:
    """v0.21 self-learning from the FINAL dub (the timeline after every
    regeneration), launched by the panel's "Learn from final dub".

    *pairs* = [(english, final_text)], *corrections* = [{"en", "draft",
    "final"}] built by the engine from the timeline, the AI draft and the
    panel's regen log. Same guard as the review learner: a correction whose
    final lost numbers or more than half its draft is a truncation — it is
    HELD BACK, and so is any pair carrying that final text. *full_text* is
    memorised against the English of *en_entries* only when nothing was
    held back (pass en_entries=None when the timeline does not cover the
    whole English). Never raises: returns {"ok", "pairs", "tm_stored",
    "corrections", "held_back", "profile_path", "error"}."""
    say = status_cb or (lambda _m: None)
    out = {"ok": False, "pairs": 0, "tm_stored": 0, "corrections": 0,
           "held_back": [], "profile_path": "", "error": ""}
    try:
        keep_c, held = [], []
        for c in corrections or []:
            draft = str(c.get("draft") or "").strip()
            fin = str(c.get("final") or "").strip()
            if not fin or _norm(draft) == _norm(fin):
                continue
            if not meaning_kept(draft, fin):
                held.append(fin)
                continue
            keep_c.append({"en": str(c.get("en") or "").strip(),
                           "draft": draft, "final": fin})
        held_n = [_norm(h) for h in held]
        keep_p = []
        for en, fin in pairs or []:
            en, fin = (en or "").strip(), (fin or "").strip()
            if not en or not fin:
                continue
            nf = _norm(fin)
            if any(h and (h == nf or h in nf) for h in held_n):
                continue
            keep_p.append((en, fin))
        out["held_back"] = held
        out["pairs"], out["corrections"] = len(keep_p), len(keep_c)
        if held:
            say(f"WARNING: {len(held)} edit(s) lost numbers or more than half "
                "their text versus the earlier wording — held back from "
                "memory and corrections, so a truncation is never learned.")
        if not keep_p and not keep_c:
            out["error"] = "nothing to learn (no usable pairs or corrections)"
            return out
        src = " ".join(t.strip() for (_a, _b, t) in (en_entries or [])
                       if t and t.strip())
        full = ((src, full_text.strip())
                if src and (full_text or "").strip() and not held else None)
        return _absorb_evidence(language, keep_p, keep_c, full, model,
                                source, say, out, "the final dub")
    except Exception as e:
        out["error"] = str(e)[:200]
        say(f"WARNING: AI learning failed ({out['error']}).")
        return out
