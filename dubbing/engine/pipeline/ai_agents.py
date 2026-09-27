"""
AI mode — judgement agents (port of lekhak critics.ts + proofer.ts)
===================================================================
Critic panel — three judges, run in parallel at temperature 0, on the pages
a deterministic check still fails after the repair pass plus a random audit
sample. Lekhak's rule kept verbatim: a critic that threw has approved
NOTHING — errors and passes are kept apart, so unverified output is never
reported as verified. The panel does not loop back into generation; its
feedback is the critique a human reads (here: the review report).

Proofer — advisory, sentence-level. Marks the few sentences a reviewer
should look at ("review" = likely wrong, "polish" = optional) with drop-in
replacements. Batched 6 pages per call; a failed batch yields no findings;
it can never be the reason a run fails.

Model roles "critic" / "proofer" (llm_settings model_critic /
model_proofer; blank = the main model).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

from .ai_lang import glossary_directive, honorific_directive, translit_directive
from .config import GEMINI_DEFAULT_MODEL
from .llm import _llm_generate, _strip_code_fence

AI_PROOF_BATCH = 6           # lekhak BATCH_SIZE
AI_MAX_FINDINGS = 6          # lekhak proofer MAX_FINDINGS
AI_MAX_SUGGESTIONS = 3


def _ai_json(text: str):
    raw = _strip_code_fence((text or "").strip())
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", raw)
        if m:
            return json.loads(m.group(0))
        raise


def _ai_call(prompt: str, model: str, role: str, temperature: float,
             attempts: int = 2, static_prefix: Optional[str] = None):
    """_llm_generate + JSON parse, *attempts* tries (lekhak call attempts)."""
    err: Exception = RuntimeError("no attempt made")
    for _ in range(max(1, attempts)):
        try:
            return _ai_json(_llm_generate(prompt, model,
                                          static_prefix=static_prefix,
                                          role=role, temperature=temperature))
        except Exception as e:                      # noqa: BLE001
            err = e
    raise err


# ── The three critics (lekhak critics.ts:87-148) ────────────────────────────

def _grammar_prompt(src, tgt, lang):
    return f"""You are a professional {lang} proofreader and grammarian. Check whether the translation contains grammatical issues, typos, unnatural word flow, or incorrect honorifics.

ENGLISH ORIGINAL:
"{src}"

{lang} TRANSLATION:
"{tgt}"

CRITERIA:
1. Standard {lang} script spelling rules must be followed.
2. Grammar must feel natural and flow smoothly when SPOKEN — this is a voice-over script.
3. {honorific_directive(lang)}

Return JSON: {{ "isValid": boolean, "feedback": "If isValid is false, state exactly which rule is broken and how to correct it. Otherwise empty." }}"""


def _glossary_prompt(src, tgt, lang, guidelines, prof):
    extra = ""
    if guidelines:
        extra += f"- Custom user directives:\n{guidelines}\n"
    vocab = (prof or {}).get("vocabularyMappings") or {}
    gloss = (prof or {}).get("glossary") or {}
    if vocab:
        extra += ("- Learned style mappings:\n"
                  + json.dumps(vocab, ensure_ascii=False) + "\n")
    if gloss:
        extra += ("- Required glossary (forced):\n"
                  + json.dumps(gloss, ensure_ascii=False) + "\n")
    return f"""You are a terminology and glossary auditor. Verify that vocabulary constraints are strictly followed.

ENGLISH ORIGINAL:
"{src}"

{lang} TRANSLATION:
"{tgt}"

GLOSSARY & MAPPING DIRECTIVES:
- {glossary_directive(lang)}
{extra}
If the translation violates a mapping, uses a forbidden or negative term, or ignores a custom mapping, set isValid false and name the offending word and its correct replacement.

Return JSON: {{ "isValid": boolean, "feedback": "Details of any vocabulary violation." }}"""


def _tone_prompt(src, tgt, lang, guidelines, prof):
    extra = ""
    if guidelines:
        extra += f"- Custom user guidelines:\n{guidelines}\n"
    if prof and (prof.get("toneDescription") or prof.get("grammarRules")):
        extra += (f"- Learned style profile: Tone: "
                  f"{prof.get('toneDescription') or 'N/A'}, Grammar: "
                  f"{prof.get('grammarRules') or 'N/A'}\n")
    return f"""You are a literary tone editor for dubbing scripts. Verify the translation holds the intended register and is free of formatting problems or translator commentary.

ENGLISH ORIGINAL:
"{src}"

{lang} TRANSLATION:
"{tgt}"

TONE & FORMAT CONSTRAINTS:
1. High-register, natural, evocative spoken prose. Not robotic, literal or textbook.
2. The output must be the translation and nothing else — no commentary, footnotes or introductions.
{extra}
Return JSON: {{ "isValid": boolean, "feedback": "Critique details." }}"""


AI_CRITIC_NAMES = ("Grammar & Honorifics Critic", "Glossary & Mappings Guard",
                   "Tone & Format Critic")


def _judge(prompt: str, model: str) -> dict:
    try:
        v = _ai_call(prompt, model, "critic", 0.0, attempts=2)
        if not isinstance(v, dict):
            raise ValueError("verdict is not an object")
        return {"isValid": bool(v.get("isValid")),
                "feedback": str(v.get("feedback") or "")}
    except Exception as e:                           # noqa: BLE001
        return {"isValid": True, "feedback": "", "errored": True,
                "error": str(e)[:200]}


def run_critic_panel(source: str, target: str, language: str,
                     guidelines: str = "", profile: Optional[dict] = None,
                     model: str = GEMINI_DEFAULT_MODEL) -> dict:
    """{"isValid", "feedbacks": [...], "degraded": [critic names]}."""
    prompts = (_grammar_prompt(source, target, language),
               _glossary_prompt(source, target, language, guidelines, profile),
               _tone_prompt(source, target, language, guidelines, profile))
    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(lambda p: _judge(p, model), prompts))
    feedbacks, degraded = [], []
    for name, r in zip(AI_CRITIC_NAMES, results):
        if r.get("errored"):
            degraded.append(name)
        elif not r["isValid"] and r["feedback"]:
            feedbacks.append(f"[{name}]: {r['feedback']}")
    return {"isValid": not feedbacks, "feedbacks": feedbacks,
            "degraded": degraded}


# ── The proofer (lekhak proofer.ts:75-246) ──────────────────────────────────

def proofer_prefix(language: str, guidelines: str = "",
                   profile: Optional[dict] = None) -> str:
    prof = profile or {}
    team = (f"\nTHIS TEAM'S OWN STYLE DIRECTIVES (a translation that follows "
            f"these is correct, even if you would have phrased it "
            f"differently):\n{guidelines}\n" if guidelines else "")
    voice = ""
    if prof.get("toneDescription") or prof.get("vocabularyMappings") \
       or prof.get("grammarRules"):
        voice = ("\nTHE HOUSE VOICE, LEARNED FROM THIS TEAM'S OWN PROOFREAD "
                 "SCRIPTS — treat it as the standard this page is measured "
                 "against, not as a suggestion:\n"
                 f"- Tone/Aesthetic: {prof.get('toneDescription') or 'N/A'}\n"
                 "- Key Vocabulary Mappings: "
                 + json.dumps(prof.get("vocabularyMappings") or {},
                              ensure_ascii=False) + "\n"
                 f"- Grammatical/Pronoun Style: "
                 f"{prof.get('grammarRules') or 'N/A'}\n"
                 "Do not flag a sentence for matching this voice, and never "
                 "suggest a replacement that departs from these mappings.\n")
    return f"""You are the final proofreader of {language} dubbing scripts translated from English spiritual and philosophical talks. For each page you receive the English source and the current {language} translation. Your job is NOT to retranslate the page — it is to mark the few sentences a human reviewer should spend their limited attention on.

WHAT TO FLAG:
- severity "review": the sentence is likely wrong — meaning drifted from the English, a grammar or spelling fault, a broken honorific, terminology that violates the directives below, or robotic/clinical register.
- severity "polish": the sentence is acceptable as it stands, but you can offer a rendering with better flow, rhythm or emotional weight when spoken. These are optional improvements, not corrections.

TERMINOLOGY & REGISTER DIRECTIVES:
- {glossary_directive(language)}
- {translit_directive(language)}
- {honorific_directive(language)}
{team}{voice}
HARD RULES:
- "sentence" must be copied character-for-character from the {language} TRANSLATION — an exact substring, complete sentences only. Never quote the English.
- Each suggestion must be a complete drop-in replacement for exactly that sentence, in {language}, obeying the directives above.
- "reason" is one short English sentence a reviewer can act on.
- Flag only what matters: typically 0 to 3 findings per page, never more than {AI_MAX_FINDINGS}. A clean page returns an empty findings list — most pages are clean.
- Do not flag a sentence merely because a synonym exists. Flag it when a reviewer would genuinely thank you.
"""


def sanitise_findings(raw, target: str) -> List[dict]:
    """Keep only findings a reviewer can anchor and act on
    (lekhak sanitiseFindings)."""
    if not isinstance(raw, list):
        return []
    out, seen = [], set()
    for f in raw:
        if not isinstance(f, dict):
            continue
        sentence = str(f.get("sentence") or "").strip()
        if not sentence or sentence not in target:
            continue
        if len(sentence) > max(400, len(target) * 0.7):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        raw_sugg = f.get("suggestions")
        sugg = [str(s or "").strip() for s in raw_sugg] \
            if isinstance(raw_sugg, list) else []
        sugg = [s for s in sugg if s and s != sentence][:AI_MAX_SUGGESTIONS]
        if not sugg:
            continue
        out.append({"sentence": sentence,
                    "severity": "review" if f.get("severity") == "review"
                    else "polish",
                    "reason": str(f.get("reason") or "").strip()[:300],
                    "suggestions": sugg})
        if len(out) >= AI_MAX_FINDINGS:
            break
    return out


def proof_pages(items: Sequence[dict], language: str, guidelines: str = "",
                profile: Optional[dict] = None,
                model: str = GEMINI_DEFAULT_MODEL,
                status_cb=None) -> Dict[int, List[dict]]:
    """items: [{"id", "source", "target"}] -> {id: findings}. Never raises."""
    say = status_cb or (lambda _m: None)
    prefix = proofer_prefix(language, guidelines, profile)
    found: Dict[int, List[dict]] = {}
    for i in range(0, len(items), AI_PROOF_BATCH):
        sl = list(items[i:i + AI_PROOF_BATCH])
        blocks = "\n\n".join(
            f"### PAGE {n}\nENGLISH SOURCE:\n\"\"\"\n{p['source']}\n\"\"\"\n\n"
            f"{language} TRANSLATION:\n\"\"\"\n{p['target']}\n\"\"\""
            for n, p in enumerate(sl, 1))
        instruction = (
            f"\n\nProofread each of the {len(sl)} page(s) below.\n"
            "Return JSON: {\"pages\":[{\"n\":<page number>,\"findings\":"
            f"[{{\"sentence\":\"<exact {language} sentence>\",\"severity\":"
            "\"review\"|\"polish\",\"reason\":\"<short English reason>\","
            f"\"suggestions\":[\"<replacement {language} sentence>\"]}}]}}]}}\n"
            "Include every page number exactly once, with an empty findings "
            "list when the page is clean.\n\n")
        try:
            data = _ai_call(instruction + blocks, model, "proofer", 0.2,
                            attempts=2, static_prefix=prefix)
            pages = data.get("pages") if isinstance(data, dict) else None
            for page in pages or []:
                try:
                    n = int(page.get("n"))
                except Exception:                    # noqa: BLE001
                    continue
                if not 1 <= n <= len(sl):
                    continue
                it = sl[n - 1]
                found[it["id"]] = sanitise_findings(page.get("findings"),
                                                    it["target"])
        except Exception as e:                       # noqa: BLE001
            say(f"WARNING: proofer batch of {len(sl)} page(s) did not run "
                f"({str(e)[:120]}) — those pages have no findings.")
    return found
