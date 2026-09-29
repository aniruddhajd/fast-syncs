"""
Prompt agents (v0.25) — the Prompt chain as an agent team, AI-mode sync
========================================================================
The "Prompt agents · test" script source (--script-source prompt_agents).
It BYPASSES the Lekhak pipeline (no memory, no learned style, no critic
panel, no proofer) and instead runs the hand-written Prompt-chain files as
three agents, page by page:

  1. Translator   Step1_Translation_Prompt_<lang> — full file, unchanged
  2. Reviewer     Step2_Review_Prompt_<lang>      — English + the draft
  3. Punctuator   Step3_Punctuation_Prompt_<lang> — the reviewed script

Each agent gets the Prompt chain's own input block (same prompt file as the
cached static prefix), only for one page of the English, so pages run in
parallel. Between the agents the free checks (ai_checks.run_checks) act as
the supervisor:

  * a Translator draft that fails a check gets ONE retry, with the check
    notes appended; the retry is kept only if it is no worse;
  * a Reviewer or Punctuator output is kept only if it fails no more checks
    than its input and keeps every number and at least half the text
    (meaning_kept) — otherwise the page keeps the previous agent's text.

v0.25.1 — LINE TAGS. The first version paired the prompts' plain-text
paragraphs to the English by length, and a live run put words under the
wrong English: Step1 groups by THOUGHT UNIT and reorders words, so its
paragraph breaks do not fall where the English lines break, and one clause
drifting across a boundary shifted every following line. Now every English
line is numbered (#N) and ONE output rule is appended after the untouched
prompt: each paragraph starts with the numbers it covers, "[12-14] ...".
The Reviewer and Punctuator are told to keep the tags. The tags are read
with Lekhak's own parser (_rows_from_reply: every cue exactly once, in
order, overlaps merged), so rows carry EXACT English windows — the strong
anchor-sync hint Lekhak's rows have. Missing tags fail a check (TAGS) and
the page is retried. Tags never reach the review screen or the voice.

Nothing is learned and nothing is read from AI memory: a pure test of the
prompts. Step4 (emotion tags) is not used — anchor sync voices clean text.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

from .ai_checks import fail_count, has_failure, run_checks
from .ai_translator import _pages, _rows_from_reply, meaning_kept
from .config import GEMINI_DEFAULT_MODEL
from .llm import (_llm_generate, _load_lang_prompt,
                  _split_translation_paragraphs, _strip_code_fence)

PA_CONCURRENCY = 3
StatusCb = Optional[Callable[[str], None]]

_TAG = re.compile(r"^\s*\[\s*#?(\d+)(?:\s*[-–]\s*#?(\d+))?\s*\]\s*(.*)$", re.S)

TAG_RULE = (
    "\n\nOUTPUT RULE FOR SYNC (required, overrides any output format above): "
    "every English line above is numbered #N. Begin EVERY paragraph of your "
    "script with the numbers of the English lines it translates, as [N] or "
    "[N-M], e.g. \"[12-14] <your text>\". Cover every English line exactly "
    "once, in order; separate paragraphs with one blank line; write nothing "
    "outside the paragraphs.")
KEEP_TAGS = (
    "\n\nKEEP THE [N-M] TAGS: every paragraph starts with a tag such as "
    "[12-14] naming the English lines it translates. Keep each tag exactly "
    "as it is at the start of its paragraph; do not merge, split, add or "
    "renumber paragraphs.")


def _numbered(page, next_start) -> str:
    """The chain's '[dur] text [gap]' lines, each prefixed with its #N."""
    lines = []
    for k, (i, s0, s1, text) in enumerate(page):
        nxt = page[k + 1][1] if k + 1 < len(page) else next_start
        gap = (nxt - s1) if nxt is not None else 0.0
        lines.append(f"#{i} [{s1 - s0:.3f}s] {text} [{gap:.3f}s]")
    return "\n\n".join(lines)


def _parse(text: str):
    """[{"cues": [...], "text": ...}] from tagged paragraphs; [] if untagged."""
    out = []
    for para in _split_translation_paragraphs(text):
        m = _TAG.match(para)
        if not m:
            if out:                       # untagged tail joins the previous
                out[-1]["text"] += " " + " ".join(para.split())
            continue
        a = int(m.group(1))
        b = int(m.group(2) or a)
        body = " ".join(m.group(3).split())
        if body:
            out.append({"cues": list(range(min(a, b), max(a, b) + 1)),
                        "text": body})
    return out


def _tagged(rows) -> str:
    """Rows back to tagged text for the next agent."""
    return "\n\n".join(
        f"[{r['cues'][0]}-{r['cues'][-1]}] {r['tr']}" if r.get("cues")
        else r["tr"] for r in rows)


def _plain(rows) -> str:
    return "\n\n".join(r["tr"] for r in rows)


def _judge(page, text: str, language: str):
    """(rows, checks) for one agent's output. Rows carry exact windows."""
    source = " ".join(c[3] for c in page)
    paras = _parse(text)
    if not paras:
        return [], [{"code": "TAGS", "severity": "fail",
                     "detail": "The paragraphs have no [N-M] English line "
                               "tags — start every paragraph with the "
                               "numbers of the English lines it covers."}]
    try:
        rows, repaired = _rows_from_reply(paras, page)
    except (ValueError, TypeError, KeyError):
        return [], [{"code": "TAGS", "severity": "fail",
                     "detail": "The [N-M] tags could not be matched to the "
                               "English lines of this page."}]
    for r in rows:
        r.pop("pauses", None)             # the prompts use no " | " marks
    return rows, run_checks(source, _plain(rows), language, rows,
                            coverage_repaired=repaired)


def _notes(checks) -> str:
    return "\n".join(f"- {c['detail']}" for c in checks
                     if c.get("severity") == "fail")


def _call(dynamic: str, prompt: str, model: str) -> str:
    return _strip_code_fence(_llm_generate(dynamic, model,
                                           static_prefix=prompt,
                                           role="translate") or "").strip()


def _run_page(idx: int, page, next_start, language: str, model: str,
              prompts: dict) -> dict:
    """Translator -> (retry) -> Reviewer -> Punctuator for one page."""
    english = _numbered(page, next_start)
    log, attempts = [], 0

    # ── 1. Translator (Step1 + the chain's input block + the tag rule) ──
    dyn1 = f"\n\n=== Formatted SRT Content ===\n{english}" + TAG_RULE
    rows, checks = _judge(page, _call(dyn1, prompts["p1"], model), language)
    attempts += 1
    if has_failure(checks):
        retry = _call(dyn1 + "\n\nYOUR PREVIOUS DRAFT FAILED THESE CHECKS — "
                      "translate again and fix them:\n" + _notes(checks),
                      prompts["p1"], model)
        attempts += 1
        r_rows, r_checks = _judge(page, retry, language)
        if r_rows and (not rows or fail_count(r_checks) <= fail_count(checks)):
            rows, checks = r_rows, r_checks
            log.append("translator: retried with the check notes (kept)")
        else:
            log.append("translator: retry was no better (first draft kept)")
    if not rows:
        raise RuntimeError("the translator returned no tagged paragraphs")

    # ── 2 + 3. Reviewer and Punctuator, each gated by the checks ──
    for step, key, dyn in (
            ("reviewer", "p2", f"\n\nEnglish text\n{english}\n\n"
                               f"{language} Script for Tuning\n{{text}}"),
            ("punctuator", "p3", "\n\n{text}")):
        out = _call(dyn.replace("{text}", _tagged(rows)) + KEEP_TAGS,
                    prompts[key], model)
        attempts += 1
        o_rows, o_checks = _judge(page, out, language)
        if (o_rows and meaning_kept(_plain(rows), _plain(o_rows))
                and fail_count(o_checks) <= fail_count(checks)):
            rows, checks = o_rows, o_checks
            log.append(f"{step}: kept")
        else:
            log.append(f"{step}: rejected (lost its line tags, dropped "
                       "content or failed more checks) — previous text kept")
    return {"n": idx + 1, "cues": (page[0][0], page[-1][0]), "rows": rows,
            "checks": checks, "attempts": attempts, "log": log}


def prompt_agents_translate(en_entries: List[Tuple[float, float, str]],
                            language: str, model: str = GEMINI_DEFAULT_MODEL,
                            status_cb: StatusCb = None,
                            options: Optional[dict] = None
                            ) -> Tuple[str, List[dict], dict]:
    """(script, rows, info) — the same shape as ai_translate_script, so the
    engine's review files and anchor sync take it unchanged."""
    say = status_cb or (lambda _m: None)
    opt = options or {}
    conc = max(1, int(opt.get("concurrency", PA_CONCURRENCY)))
    cues = [(i + 1, float(s0), float(s1), (t or "").strip())
            for i, (s0, s1, t) in enumerate(en_entries) if (t or "").strip()]
    if not cues:
        raise RuntimeError("Prompt agents: the English transcript has no "
                           "text to translate.")
    prompts = {k: _load_lang_prompt(name, language) for k, name in (
        ("p1", "Step1_Translation_Prompt"), ("p2", "Step2_Review_Prompt"),
        ("p3", "Step3_Punctuation_Prompt"))}
    pages = _pages(cues)
    say(f"Prompt agents: {len(pages)} page(s), {min(conc, len(pages))} in "
        "parallel — Translator (Step1) -> Reviewer (Step2) -> Punctuator "
        "(Step3), checks between each; paragraphs tagged with their English "
        "lines.")

    def work(idx):
        page = pages[idx]
        nxt = pages[idx + 1][0][1] if idx + 1 < len(pages) else None
        try:
            return _run_page(idx, page, nxt, language, model, prompts)
        except Exception as e:                       # noqa: BLE001
            return {"n": idx + 1, "cues": (page[0][0], page[-1][0]),
                    "rows": [], "checks": [], "attempts": 0,
                    "log": [f"error: {str(e)[:200]}"]}

    with ThreadPoolExecutor(max_workers=conc) as ex:
        outs = list(ex.map(work, range(len(pages))))
    dead = [o["n"] for o in outs if not o["rows"]]
    if dead:
        raise RuntimeError(f"Prompt agents: page(s) {dead} of {len(pages)} "
                           f"could not be translated — "
                           f"{outs[dead[0] - 1]['log'][-1:]}")

    rows = [r for o in outs for r in o["rows"]]
    script = _plain(rows)
    flagged = [o for o in outs if has_failure(o["checks"])]
    calls = sum(o["attempts"] for o in outs)
    summary = (f"{len(pages)} page(s), {calls} agent call(s); "
               f"{len(pages) - len(flagged)} clean, {len(flagged)} still "
               "failing a check")
    report = ["PROMPT AGENTS REVIEW REPORT — " + language, summary, ""]
    for o in sorted(outs, key=lambda o: -fail_count(o["checks"])):
        report.append(f"== Page {o['n']} (cues {o['cues'][0]}-{o['cues'][1]})")
        report += ["  " + ln for ln in o["log"]]
        report += [f"  [{c['severity']}] {c['code']}: {c['detail']}"
                   for c in o["checks"]]
        report.append("")
    say("Prompt agents: " + summary + ".")
    return script, rows, {"summary": summary, "report": "\n".join(report),
                          "page_outcomes": [
                              {"n": o["n"], "cues": list(o["cues"]),
                               "checks": o["checks"], "log": o["log"]}
                              for o in outs]}
