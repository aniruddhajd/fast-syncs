"""
AI mode — language registry (port of lekhak/src/languages.ts)
=============================================================
One entry per target language. The translator prefix, the critic panel and
the proofer all derive their language-specific rules from it, so adding a
language is one entry here. Entries are Lekhak's, verbatim, for the twelve
languages the dubbing engine speaks.

A language with no entry (a user-added one — dub_engine accepts any safe
name) still works: it gets generic directives and the SCRIPT check is
skipped, because its script is unknown.
"""

from __future__ import annotations

import os
import unicodedata
from typing import Dict, Optional

# Unicode blocks per script family (lekhak ScriptFamily).
AI_SCRIPT_RANGES = {
    "devanagari": ((0x0900, 0x097F), (0xA8E0, 0xA8FF)),
    "bengali":    ((0x0980, 0x09FF),),
    "gurmukhi":   ((0x0A00, 0x0A7F),),
    "gujarati":   ((0x0A80, 0x0AFF),),
    "odia":       ((0x0B00, 0x0B7F),),
    "tamil":      ((0x0B80, 0x0BFF),),
    "telugu":     ((0x0C00, 0x0C7F),),
    "kannada":    ((0x0C80, 0x0CFF),),
    "malayalam":  ((0x0D00, 0x0D7F),),
}

AI_LANGS: Dict[str, dict] = {
    "Marathi": {
        "script": "devanagari",
        "honorificRule": "use respectful plural conjugations (आदरार्थी बहुवचन, e.g. 'ते म्हणाले' not 'तो म्हणाला')",
        "positiveVocab": "विस्मय, गहिरी ओढ, अलौकिक, अद्वैत",
        "avoidTerm": "मोह",
        "translitExample": "अरुंधती",
    },
    "Hindi": {
        "script": "devanagari",
        "honorificRule": "use respectful plural forms (आदरसूचक बहुवचन, e.g. 'वे बोले' not 'वह बोला')",
        "positiveVocab": "विस्मय, गहिरा लगाव, अलौकिक, अद्वैत",
        "avoidTerm": "मोह",
        "translitExample": "अरुंधति",
    },
    "Bengali": {
        "script": "bengali",
        "honorificRule": "use the honorific pronoun and verb forms (e.g. 'তিনি বললেন' not 'সে বলল')",
        "positiveVocab": "বিস্ময়, গভীর টান, অলৌকিক",
        "translitExample": "অরুন্ধতী",
    },
    "Assamese": {
        "script": "bengali",
        "honorificRule": "use the honorific pronoun তেওঁ with honorific verb endings (e.g. 'তেওঁ ক’লে' not 'সি ক’লে')",
        "positiveVocab": "বিস্ময়, গভীৰ টান, অলৌকিক",
        "translitExample": "অৰুন্ধতী",
    },
    "Odia": {
        "script": "odia",
        "honorificRule": "use honorific plural verb endings (e.g. 'ସେ କହିଲେ' not 'ସେ କହିଲା')",
        "positiveVocab": "ବିସ୍ମୟ, ଗଭୀର ଆକର୍ଷଣ, ଅଲୌକିକ",
        "translitExample": "ଅରୁନ୍ଧତୀ",
    },
    "Punjabi": {
        "script": "gurmukhi",
        "honorificRule": "use respectful plural forms (e.g. 'ਉਹਨਾਂ ਨੇ ਕਿਹਾ' not 'ਉਸ ਨੇ ਕਿਹਾ')",
        "positiveVocab": "ਵਿਸਮਾਦ, ਡੂੰਘੀ ਖਿੱਚ, ਅਲੌਕਿਕ",
        "translitExample": "ਅਰੁੰਧਤੀ",
    },
    "Gujarati": {
        "script": "gujarati",
        "honorificRule": "use respectful pronouns and verbs (e.g. 'તેમણે કહ્યું' rather than the singular form, or adding '-જી')",
        "positiveVocab": "વિસ્મય, ઊંડો લગાવ, અલૌકિક, અદ્ભુત",
        "avoidTerm": "મોહ",
        "translitExample": "અરુંધતી",
    },
    "Tamil": {
        "script": "tamil",
        "honorificRule": "use respectful pronouns and verb forms (e.g. 'அவர்கள்' and a verb ending in '-ஆர்' like 'சொன்னார்' instead of the singular '-ஆன்')",
        "positiveVocab": "வியப்பு, ஆன்மீக ஈர்ப்பு, பேரதிசயம்",
        "translitExample": "அருந்ததி",
    },
    "Telugu": {
        "script": "telugu",
        "honorificRule": "use the respectful pronoun 'వారు' and plural verb endings such as '-చారు' (e.g. 'చెప్పారు' not 'చెప్పాడు')",
        "positiveVocab": "విస్మయం, ఆధ్యాత్మిక అనుబంధం, అద్భుతం",
        "translitExample": "అరుంధతి",
    },
    "Kannada": {
        "script": "kannada",
        "honorificRule": "use the respectful plural form (e.g. 'ಅವರು ಹೇಳಿದರು' not 'ಅವನು ಹೇಳಿದನು')",
        "positiveVocab": "ವಿಸ್ಮಯ, ಆಧ್ಯಾತ್ಮಿಕ ಸೆಳೆತ, ಅಲೌಕಿಕ",
        "translitExample": "ಅರುಂಧತಿ",
    },
    "Malayalam": {
        "script": "malayalam",
        "honorificRule": "use the respectful pronoun 'അദ്ദേഹം' and honorific verb forms rather than familiar ones",
        "positiveVocab": "വിസ്മയം, ആത്മീയ ആകർഷണം, അലൗകികം",
        "translitExample": "അരുന്ധതി",
    },
    "Nepali": {
        "script": "devanagari",
        "honorificRule": "use the high honorific forms (e.g. 'उहाँले भन्नुभयो' not 'उसले भन्यो')",
        "positiveVocab": "विस्मय, गहिरो आकर्षण, अलौकिक",
        "translitExample": "अरुन्धती",
    },
}

_AI_GENERIC_LANG = {
    "script": None,
    "honorificRule": "use the respectful, honorific forms of the language",
    "positiveVocab": "the language's elevated words for wonder and the sacred",
    "translitExample": "the standard spelling of the name",
}


def ai_lang_spec(language: str) -> dict:
    return AI_LANGS.get((language or "").strip()) or _AI_GENERIC_LANG


# ── Shared directives (lekhak critics.ts:33-50) ─────────────────────────────

def honorific_directive(language: str) -> str:
    return (f"If referring to elders or spiritual figures, "
            f"{ai_lang_spec(language)['honorificRule']}.")


def glossary_directive(language: str) -> str:
    spec = ai_lang_spec(language)
    avoid = (f" (AVOID '{spec['avoidTerm']}' for positive fascination)"
             if spec.get("avoidTerm") else "")
    return ("Words like 'fascination', 'incredulity', 'awe', and 'miracle' must "
            f"use positive spiritual vocabulary (e.g. '{spec['positiveVocab']}') "
            "rather than clinical, negative or plain dictionary equivalents"
            f"{avoid}.")


def translit_directive(language: str) -> str:
    return (f"Transliterate proper names using standard {language} literary "
            f"usage (e.g., '{ai_lang_spec(language)['translitExample']}').")


# ── v0.24 house rules (the "AI · test rules" source only) ────────────────────
# Prompt-mode know-how (Step1-3 prompt files) distilled into ai_rules/:
# _common.md (structural rules shared by every language) + <Language>.md.
# The prompt files themselves are never read here — AI mode still opens no
# prompt file; these are separate, compact, tracked rule files.
AI_RULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "ai_rules")
_RULES_CACHE: Dict[str, str] = {}


def _read_rule(name: str) -> str:
    try:
        with open(os.path.join(AI_RULES_DIR, name), encoding="utf-8") as f:
            # The provenance comment at the top is for people, not the model.
            return "\n".join(ln for ln in f.read().splitlines()
                             if not ln.lstrip().startswith("<!--")).strip()
    except OSError:
        return ""


def house_rules(language: str) -> str:
    """_common.md + <language>.md, cached. "" when the language has no rules
    file (a user-added language) — never raises."""
    lang = (language or "").strip()
    if lang not in _RULES_CACHE:
        own = _read_rule(lang + ".md") if lang else ""
        _RULES_CACHE[lang] = ("\n\n".join(p for p in (_read_rule("_common.md"),
                                                        own) if p)
                              if own else "")
    return _RULES_CACHE[lang]


def script_share(text: str, language: str) -> Optional[float]:
    """Share of letters (incl. combining vowel signs) in the language's own
    script, or None when the script is unknown / there are no letters."""
    ranges = AI_SCRIPT_RANGES.get(ai_lang_spec(language).get("script") or "")
    if not ranges:
        return None
    total = inside = 0
    for ch in text or "":
        if unicodedata.category(ch)[0] not in ("L", "M"):
            continue
        total += 1
        cp = ord(ch)
        if any(a <= cp <= b for a, b in ranges):
            inside += 1
    return (inside / total) if total else None
