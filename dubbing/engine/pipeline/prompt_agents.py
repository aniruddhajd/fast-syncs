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

Each agent gets exactly the input the Prompt chain gives it (same prompt
file as the cached static prefix, same dynamic block), only for one page of
the English instead of the whole talk, so pages run in parallel. Between the
agents the free checks (ai_checks.run_checks: timing, numbers, script,
dropped content, truncation, commentary) act as the supervisor:

  * the Translator's draft that fails a check gets ONE retry, with the check
    notes appended to its input; the retry is kept only if it is no worse;
  * a Reviewer or Punctuator output is kept only if it fails no more checks
    than its input and keeps every number and at least half the text
    (meaning_kept) — otherwise the page keeps the previous agent's text.

Output rows are paired to the English phrase cues (_pair_review_rows), so
the dub stage runs AI mode's anchor sync on them unchanged. Nothing is
learned and nothing is read from AI memory: a pure test of the prompts.
Step4 (emotion tags) is not used — anchor sync voices clean text.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

from .ai_checks import fail_count, has_failure, run_checks
from .ai_translator import _pages, meaning_kept
from .config import GEMINI_DEFAULT_MODEL
from .llm import (_llm_generate, _load_lang_prompt, _pair_review_rows,
                  _split_translation_paragraphs, _strip_code_fence)
from .srt_tools import _parse_srt_to_analysis_format, _srt_ts

PA_CONCURRENCY = 3
StatusCb = Optional[Callable[[str], None]]


def _page_srt(page) -> str:
    lines = []
    for n, (_i, s0, s1, text) in enumerate(page, 1):
        lines += [str(n), f"{_srt_ts(s0)} --> {_srt_ts(s1)}", text, ""]
    return "\n".join(lines)


def _rows(page, text: str) -> List[dict]:
    """Paragraphs of *text* paired to this page's English cues."""
    en = [(float(s0), float(s1), t) for (_i, s0, s1, t) in page]
    paras = _split_translation_paragraphs(text)
    return [{"en": e, "tr": t, "start": a, "end": b}
            for (e, t, a, b) in _pair_review_rows(en, paras)]


def _judge(page, text: str, language: str):
    """(rows, checks) — the free checks on one page's text."""
    source = " ".join(c[3] for c in page)
    rows = _rows(page, text) if text.strip() else []
    return rows, run_checks(source, text, language, rows)


def _notes(checks) -> str:
    return "\n".join(f"- {c['detail']}" for c in checks
                     if c.get("severity") == "fail")


def _call(dynamic: str, prompt: str, model: str) -> str:
    return _strip_code_fence(_llm_generate(dynamic, model,
                                           static_prefix=prompt,
                                           role="translate") or "").strip()


def _run_page(idx: int, page, language: str, model: str,
              prompts: dict) -> dict:
    """Translator -> (retry) -> Reviewer -> Punctuator for one page."""
    formatted = _parse_srt_to_analysis_format(_page_srt(page))
    log, attempts = [], 0

    # ── 1. Translator (the Step1 prompt, the chain's own input block) ──
    dyn1 = f"\n\n=== Formatted SRT Content ===\n{formatted}"
    draft = _call(dyn1, prompts["p1"], model)
    attempts += 1
    rows, checks = _judge(page, draft, language)
    if has_failure(checks):
        retry = _call(dyn1 + "\n\nYOUR PREVIOUS DRAFT FAILED THESE CHECKS — "
                      "translate again and fix them:\n" + _notes(checks),
                      prompts["p1"], model)
        attempts += 1
        r_rows, r_checks = _judge(page, retry, language)
        if retry and fail_count(r_checks) <= fail_count(checks):
            draft, rows, checks = retry, r_rows, r_checks
            log.append("translator: retried with the check notes (kept)")
        else:
            log.append("translator: retry was no better (first draft kept)")

    # ── 2 + 3. Reviewer and Punctuator, each gated by the checks ──
    text = draft
    for step, key, dyn in (
            ("reviewer", "p2", f"\n\nEnglish text\n{formatted}\n\n"
                               f"{language} Script for Tuning\n{{text}}"),
            ("punctuator", "p3", "\n\n{text}")):
        out = _call(dyn.replace("{text}", text), prompts[key], model)
        attempts += 1
        o_rows, o_checks = _judge(page, out, language)
        if (out and meaning_kept(text, out)
                and fail_count(o_checks) <= fail_count(checks)):
            text, rows, checks = out, o_rows, o_checks
            log.append(f"{step}: kept")
        else:
            log.append(f"{step}: rejected (it dropped content or failed "
                       "more checks) — previous text kept")
    return {"n": idx + 1, "cues": (page[0][0], page[-1][0]), "rows": rows,
            "text": text, "checks": checks, "attempts": attempts,
            "log": log}


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
        "(Step3), checks between each.")

    def work(item):
        idx, page = item
        try:
            return _run_page(idx, page, language, model, prompts)
        except Exception as e:                       # noqa: BLE001
            return {"n": idx + 1, "cues": (page[0][0], page[-1][0]),
                    "rows": [], "text": "", "checks": [], "attempts": 0,
                    "log": [f"error: {str(e)[:200]}"]}

    with ThreadPoolExecutor(max_workers=conc) as ex:
        outs = list(ex.map(work, list(enumerate(pages))))
    dead = [o["n"] for o in outs if not o["rows"]]
    if dead:
        raise RuntimeError(f"Prompt agents: page(s) {dead} of {len(pages)} "
                           f"could not be translated — "
                           f"{outs[dead[0] - 1]['log'][-1:]}")

    rows = [r for o in outs for r in o["rows"]]
    script = "\n\n".join(r["tr"] for r in rows)
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
