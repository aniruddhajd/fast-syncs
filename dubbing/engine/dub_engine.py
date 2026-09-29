#!/usr/bin/env python3
"""
Headless dubbing worker (contract v0.3) — single-voice pipeline with an
optional review pause, standalone chunk regeneration, and settings/voice
utility modes.

STANDALONE as of v0.3: all pipeline logic lives in the local pipeline/
package (extracted from the bulk app's Translation_and_Syncing_App.py —
see each module's provenance header and ENGINE_NOTES.md). The bulk app is
no longer imported and --app-dir is accepted only for backward
compatibility (ignored, with a warning). The stage sequence is unchanged:

    [S1a] ElevenLabs Scribe transcription of the English audio
    [S1b] Region detection + English SRT + analysis format
    [S2a] Translation  (Step1 prompt)   \
    [S2b] Review       (Step2 prompt)    > one _run_gemini_pipeline() call
    [S2c] Punctuation  (Step3 prompt)   /
    [S2d] Step-4 emotion enrichment (default ON) + ElevenLabs TTS (single voice)
    [S3a] English sync SRT (reuses the S1a transcription)
    [S3b] TTS-audio regions + transcription -> target-language SRT
    [S3c] LLM subtitle mapping
    [S3d] Sync algorithm (+ synced SRT + timestamps file)
    [S3e] Synced audio render

Modes (v0.1 + v0.2 modes keep IDENTICAL flags and manifests):
  --steps full        S1a..S3e in one shot (v0.1 behaviour, unchanged).
  --steps translate   S1a..S2c only. Writes the app-convention outputs plus
                      a pair of aligned plain-text paragraph files (English
                      left / translation right, one paragraph per review
                      row, blank-line separated, SAME count and order in
                      both files) and a "status":"review" manifest, then
                      exits 0 so the panel can pause for human review.
  --steps dub         Resume after review. Re-derives out_dir from --audio,
                      validates the translate-stage artifacts, reads the
                      (possibly user-edited) translation from --script,
                      runs emotion (unless --no-emotion) + TTS + S3a..S3e
                      and writes the normal "status":"ok" manifest,
                      overwriting the stale review manifest.
  --regen-chunk       Synthesize ONE chunk text (--text-file, UTF-8; Indic
                      text never travels on argv) with the ElevenLabs
                      helpers into --out-wav. No emotion pass — the chunk
                      text is already final. Manifest:
                      {"status":"ok","regen_wav":"<abs>"}.
  --test-llm          NEW in v0.3. One tiny LLM call on the configured
                      provider. Manifest: {"status":"ok","provider":"…",
                      "model":"…","reply":"…"} (or status error).
  --list-voices       NEW in v0.3 (requires --language). Fetch the account
                      voice catalogue from ElevenLabs, language-token
                      sorted like the app (voices matching the language
                      first). Manifest: {"status":"ok","voices":
                      [{"id","name"},…]} (or status error).

Scope notes: multi-speaker dubbing IS supported since v0.16 — a saved
cast (<base>_speakers.json, written by the panel's review screen) names a
voice per script paragraph, and both sync modes honour it; without one the
run is single-voice exactly as before. NO translation-memory WRITES, NO
run-history recording. Translation-memory READS (from
this repo's data/translation_memory.db) and Step-4 emotion enrichment
mirror the bulk app's defaults (both ON). Emotion is toggleable:
--no-emotion on the CLI, or {"emotion": false} in engine_settings.json.
All outputs go to the app-convention per-file output folder next to the
input audio (regen output goes wherever --out-wav points, typically
<out_dir>/regen/).

Secrets: read from this repo's gitignored config/ directory only —
config/llm_settings.json + config/tts_settings.json (plus the key files
they point at). Nothing sensitive is passed on argv.

On success AND on failure a result manifest (engine_done.json) is written
to <engine>/status/engine_done.json, and additionally to
<out_dir>/engine_done.json whenever an out_dir is known (regen/test-llm/
list-voices runs have no out_dir of their own, so they write the status
copy only).

Run with --selfcheck to verify the pipeline package imports, every
required symbol exists and every per-language prompt file is present
(no network, no audio processing). A missing config/ settings file is
reported as a WARNING, not a failure.
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
import types

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
# DUB_STATUS_DIR is set by run_dub.py when the panel runs with a per-project
# status dir (concurrent runs from two REAPER instances must not share one).
STATUS_DIR = os.environ.get("DUB_STATUS_DIR") or os.path.join(ENGINE_DIR, "status")
ENGINE_SETTINGS_FILE = os.path.join(ENGINE_DIR, "engine_settings.json")

LANGUAGES = ["Bengali", "Hindi", "Kannada", "Malayalam", "Tamil", "Telugu",
             "Gujarati", "Marathi", "Punjabi", "Assamese", "Odia", "Nepali"]


# KEEP IN SYNC with run_dub.py, pipeline/config.py and
# dubbing/reaper/Dub_Pipeline_Panel.lua (V5._is_safe_lang_name).
# See run_dub.py for why all four readers carry the same rule.
# Charset only -- length and edge-whitespace are checked in _lang_name_ok so
# the rule stays readable and matches the Lua predicate exactly.
_LANG_NAME_OK = re.compile("^[0-9A-Za-z \-_.()\u0080-\U0010FFFF]+$")


# Unicode whitespace, rejected anywhere in a name. Mirrors
# V5._has_unicode_space in Dub_Pipeline_Panel.lua, which matches the same
# code points as UTF-8 byte sequences because Lua's %s is ASCII-only.
_LANG_NAME_UNICODE_WS = re.compile(
    "[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]")


def _lang_name_ok(name: str) -> bool:
    """True if *name* is usable EXACTLY as written. Never raises.

    Mirrors V5._is_safe_lang_name in dubbing/reaper/Dub_Pipeline_Panel.lua.
    The bound is 64 UTF-8 BYTES because that is what Lua's #s measures.

    Edge whitespace is compared against ASCII whitespace ONLY -- the exact set
    Lua's %s matches. Plain str.strip() would also strip U+00A0, U+2003 and
    other Unicode spaces that Lua does not recognise, and the two sides would
    then disagree about names starting with one.

    Leading/trailing whitespace is rejected, not stripped: stripping is a
    rewrite, and a rewritten name is a second spelling of the same entry.
    """
    if not isinstance(name, str) or not name:
        return False
    if name != name.strip(" \t\n\r\v\f"):
        return False
    if _LANG_NAME_UNICODE_WS.search(name):
        return False
    try:
        if len(name.encode("utf-8")) > 64:
            return False
    except (UnicodeEncodeError, UnicodeError):
        # Lone surrogate from a hand-edited "\udXXX" escape. json.load()
        # hands these back happily; encoding them raises. Return False rather
        # than propagating -- the caller in pipeline/config.py has no guard.
        return False
    return bool(_LANG_NAME_OK.match(name))


def _custom_language_names():
    """Names from config/custom_languages.json (v0.7).

    Read here WITHOUT importing the pipeline: argparse builds its --language
    choices before any heavy import, and a user-added language must be a
    valid choice or the run dies at the command line. pipeline/config.py
    reads the same file and merges the full entries into TTS_LANGUAGES.

    Names that fail _LANG_NAME_OK are skipped, matching run_dub.py,
    pipeline/config.py and the REAPER panel. See run_dub.py for why all four
    readers carry the same rule.
    """
    path = os.path.join(ENGINE_DIR, os.pardir, "config",
                        "custom_languages.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for e in (data.get("languages") or []):
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "")
            if _lang_name_ok(name):
                out.append(name)
        return out
    except Exception:
        return []



for _name in _custom_language_names():
    if _name not in LANGUAGES:
        LANGUAGES.append(_name)
LANGUAGES.sort()          # --language choices read alphabetically in --help

# The five per-language prompt stages the pipeline loads from prompts/.
PROMPT_STAGES = ["Step1_Translation_Prompt", "Step2_Review_Prompt",
                 "Step3_Punctuation_Prompt", "Step4_Emotion_Prompt",
                 "SyncingPrompt"]

# Every pipeline symbol this worker relies on, verified by _check_symbols
# (and by --selfcheck). Defined across pipeline/config|stt|srt_tools|llm|
# tts|sync — see ENGINE_NOTES.md for the module map.
REQUIRED_FUNCTIONS = [
    "_prepare_output_dir",           # per-file output folder + audio copy
    "_get_api_key",                  # ElevenLabs key from config/tts_settings.json
    "_validate_llm_config",          # fail fast if the LLM provider is unusable
    "_transcribe_audio",             # ElevenLabs Scribe STT
    "_load_audio_any",               # audio/video -> mono float32 + sr
    "_detect_regions_from_audio",    # waveform speech-region detection
    "_build_subtitle_srt",           # Stage-1 English SRT (for translation)
    "_parse_srt_to_analysis_format", # LLM input format
    "_run_gemini_pipeline",          # Step1 -> Step2 -> Step3 chain
    "_run_emotion_enrichment",       # Step4 emotion tags (strict=True from here)
    "_load_lang_prompt",             # per-language prompt file (pre-spend check)
    "_extract_srt_entries",          # SRT -> (start, end, text) rows (review pairing)
    "_split_translation_paragraphs", # dub script -> blank-line paragraphs
    "_pair_review_rows",             # EN segments <-> translation paragraphs, aligned
    "_tts_output_name",              # output naming convention
    "_sanitize_voice_id",            # strict ElevenLabs voice_id validation
    "_fetch_voices_for_language",    # account voice catalogue, matches first
    "synthesize_tts_elevenlabs",     # single-voice TTS -> WAV
    "ensure_writable_output",        # locked-output divert (open REAPER project)
    "voice_change_elevenlabs",       # speech-to-speech re-voice (--voice-change)
    "_build_english_subtitle_srt",   # sync-quality English SRT
    "_build_target_subtitle_srt",    # target-language SRT from TTS audio
    "_call_gemini_mapping",          # EN<->target subtitle mapping
    "run_sync_from_strings",         # sync algorithm
    "_write_srt_from_dict",          # synced subs -> SRT text
    "_build_timestamps",             # synced subs -> timestamp entries
    "_format_timestamps_as_text",    # timestamps file format (contract)
    "sync_audio_with_timestamps",    # cut/overlay TTS audio -> synced WAV
    "_llm_generate",                 # provider-agnostic generation (--test-llm)
    "_active_provider_and_model",    # provider/model names (--test-llm manifest)
    "_llm_provider_label",           # provider + credential state (startup banner)
    # v0.7 match sync mode (pipeline/match.py + tts sectioned synthesis)
    "call_match_sections",           # Gemini section match (script <-> EN cues)
    "build_chunks",                  # match result -> TTS chunk list
    "place_chunks",                  # slot placement + order sweep + statuses
    "synthesize_sections_elevenlabs",# per-section TTS -> one wav + spans
    "_split_script_into_sentences",  # script -> sentence match units
    "_srt_ts",                       # seconds -> SRT timestamp (synced SRT build)
    # v0.8 sentence-timed pieces
    "build_pieces",                  # sections -> one piece per sentence
    "place_pieces",                  # windowed placement + bounded borrowing
    "synthesize_sentences_elevenlabs",# /with-timestamps TTS -> spans per sentence
    "_split_script_into_units",      # v0.12 clause-level units
    # v0.13 pause-aware sync (pipeline/pausechunk.py + preview_html.py)
    "pause_chunks_from_regions",     # speech regions -> pause-delimited chunks
    "source_text_for_chunks",        # Scribe words bucketed per chunk
    "assign_script_to_chunks",       # target script spread by duration share
    "build_plan",                    # chunks + texts + fit analysis -> rows
    "format_plan_text",              # plan rows -> the editable plan file
    "parse_plan_text",               # plan file -> the approved TR: lines
    "summarize_plan",                # one-line plan summary for the log
    "plan_counts",                   # verdict histogram (manifest + preview)
    "render_plan_html",              # the readable timeline preview
    "stretch_wav_atempo",            # ffmpeg time-stretch to fit a slot
    # v0.18 AI mode — Lekhak agentic pipeline (pipeline/ai_*.py)
    "ai_translate_script",           # orchestrator: memory->translate->checks->repair->critics->proofer
    "AiMemory",                      # TM tiers, anchors, worked examples
    "run_checks",                    # free deterministic quality gates
    "risk_score",                    # 0..1 review priority per page
    "run_critic_panel",              # 3 judges, temperature 0
    "proof_pages",                   # advisory sentence-level proofer
    "fit_to_seconds",                # anchor sync: targeted line-fit retry
    "suggest_fits",                  # review screen: shorter options for overflows
    "review_assist",                 # review screen: instruction-following assistant
    "recommend_voices",              # regen tab: AI voice recommendations
    "sync_check",                    # v0.18 sync check agent: verify + tighten offsets
    "format_sync_check",             # sync check report text
    "clause_units",                  # anchor sync: paragraph -> clause units
    "align_units",                   # anchor sync: phrase-level duration alignment
    # v0.20 sync accuracy (anchor_align phrase cues + pipeline/spectral_vad.py)
    "phrase_cues",                   # Scribe words -> fine English phrase cues
    "cues_from_items",               # the user's timeline chunks -> cues
    "spectral_pauses",               # STFT speech-band pause detector
    "onset_after",                   # first speech frame after t (onset snap)
    "split_pause_markers",           # strip ' | ' pause markers, keep offsets
    "learn_from_review",             # approved review -> TM + style profile
    "learn_from_final",              # v0.21 final dub (after regens) -> TM + profile
    "load_profile",                  # per-language learned style profile
    # v0.22 ElevenLabs Dub (--script-source eleven), pipeline/el_dubbing.py
    "el_lang_code", "create_dub", "wait_dub", "get_resource",
    "parse_resource", "update_segment_text", "update_speaker_voice",
    "redub", "wait_redub", "render", "wait_render", "download",
    "download_dubbed_audio", "speaker_voices",
]
REQUIRED_ATTRIBUTES = [
    "GEMINI_DEFAULT_MODEL",
    "TTS_LANGUAGES",
    "PROMPTS_DIR", "LLM_SETTINGS_FILE", "TTS_SETTINGS_FILE",
    "DEFAULT_THR_DB", "DEFAULT_HYS_DB", "DEFAULT_MIN_MS",          # EN regions
    "DEFAULT_BN_THR_DB", "DEFAULT_BN_HYS_DB", "DEFAULT_BN_MIN_MS", # TTS regions
    "CLAUSE_MAX_CHARS",              # v0.12 clause subdivision threshold
    "CLAUSE_CHARS_PER_SEC",          # v0.13 shared rate behind CLAUSE_MAX_CHARS
    "LANG_CHARS_PER_SEC",            # v0.13 per-language speaking rate table
    "PAUSE_MIN_S", "MAX_ATEMPO",     # v0.13 pause gate + stretch ceiling
]

# Per-mode manifest key sets (contract v0.1 + v0.2 + v0.3). _write_manifest
# filters by the active set, so stray working keys never leak into the JSON.
# v0.7 adds sync_texts / synced_count / unsynced_count — written by the
# match sync mode, "" otherwise (consumers skip empties, per contract).
MANIFEST_KEYS = ["status", "error", "variant", "audio", "language", "out_dir",
                 "en_audio", "en_srt", "tts_wav", "timestamps_txt",
                 "synced_wav", "synced_srt", "sync_texts",
                 "synced_count", "unsynced_count"]
REVIEW_MANIFEST_KEYS = ["status", "error", "variant", "audio", "language", "out_dir",
                        "en_srt", "en_text", "translation_text",
                        "final_script"]
REGEN_MANIFEST_KEYS = ["status", "error", "regen_wav"]
# v0.13 pause-aware dry run. plan_txt is the editable artifact, plan_html the
# readable one; chunk_count / the verdict tallies let the panel draw its
# summary strip without parsing the plan file itself.
PLAN_MANIFEST_KEYS = ["status", "error", "audio", "language", "out_dir",
                      "en_srt", "plan_txt", "plan_html", "chunk_count",
                      "fits_count", "tight_count", "over_count",
                      "short_count", "empty_count"]
TEST_LLM_MANIFEST_KEYS = ["status", "error", "provider", "model", "reply"]
# v0.18 --suggest-fit: fit suggestions for the review screen
SUGGEST_MANIFEST_KEYS = ["status", "error", "suggest_txt", "suggest_count"]
# v0.18.6 --review-assist: the review screen assistant
ASSIST_MANIFEST_KEYS = ["status", "error", "assist_txt"]
# v0.19 --recommend-voice: voice recommendations for the regen tab
RECO_MANIFEST_KEYS = ["status", "error", "reco_txt"]
# v0.21 --learn-final: learn from the final dub (Regenerate tab button)
LEARNF_MANIFEST_KEYS = ["status", "error", "learn_txt"]
VOICES_MANIFEST_KEYS = ["status", "error", "voices"]
VOICE_CHANGE_MANIFEST_KEYS = ["status", "error", "vc_wav"]

# Stand-in for a review row that has no text on one side (e.g. more
# translation paragraphs than English segments). A visible placeholder keeps
# the blank-line paragraph count identical in both files — an empty string
# would merge two separators and desync the panes.
EMPTY_PARAGRAPH_PLACEHOLDER = "—"


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Headless dubbing worker — normally spawned by "
                    "run_dub.py, which owns log/pid/done markers.")
    ap.add_argument("--app-dir", default=None,
                    help="DEPRECATED (v0.3): accepted for backward "
                         "compatibility and IGNORED — the engine is "
                         "standalone and no longer imports the bulk app")
    ap.add_argument("--audio", default=None,
                    help="Path to the English source audio file")
    ap.add_argument("--language", default=None, choices=LANGUAGES,
                    help="Target language display name")
    ap.add_argument("--voice-id", default=None,
                    help="Optional ElevenLabs voice_id; auto-resolved from "
                         "the account voice catalogue when omitted")
    ap.add_argument("--el-model", default="eleven_v3",
                    help="ElevenLabs TTS model id (default: eleven_v3)")
    ap.add_argument("--steps", default="full",
                    choices=["full", "translate", "dub", "plan", "dubplan"],
                    help="Pipeline scope: 'full' = one shot (v0.1), "
                         "'translate' = stop after S2c for script review, "
                         "'dub' = resume from a reviewed script "
                         "(requires --script), "
                         "'plan' = pause-aware dry run: detect the source "
                         "pauses, split --provided-script across them and "
                         "estimate the fit, writing an editable plan + an "
                         "HTML preview. NO TTS, NO LLM, no credits. "
                         "'dubplan' = generate from an approved plan "
                         "(requires --plan)")
    ap.add_argument("--plan", dest="plan", default=None,
                    help="Approved sync plan file for --steps dubplan "
                         "(as written by --steps plan, possibly user-edited). "
                         "Only its TR: lines are read — every timing is "
                         "re-derived from the audio, so a hand-edited "
                         "timestamp cannot desync the run.")
    ap.add_argument("--script", default=None,
                    help="Reviewed translation text file for --steps dub "
                         "(blank-line paragraph format, as written by the "
                         "translate stage, possibly user-edited)")
    ap.add_argument("--provided-script", dest="provided_script", default=None,
                    help="User-provided translation text file (UTF-8). With "
                         "--steps translate/full the S2a-S2c LLM translation "
                         "chain is SKIPPED and this text is used as the "
                         "translation. Transcription (S1a/S1b) still runs — "
                         "the sync stages need it.")
    ap.add_argument("--script-source", dest="script_source",
                    default="prompt", choices=["prompt", "ai", "ai_test", "eleven"],
                    help="Who writes the script (v0.18). 'prompt' "
                         "(default): the Step1 -> Step2 -> Step3 prompt "
                         "chain. 'ai': the self-learning translator "
                         "(pipeline/ai_translator.py) — pages of timed "
                         "cues translated with the learned style profile, "
                         "approved exemplars and past corrections. It "
                         "LEARNS (translation memory + profile merge) from "
                         "the final dub via --learn-final (v0.21), or at "
                         "--steps dub when engine_settings ai_learn_at is "
                         "'review'. The sync stages are identical in both.")
    ap.add_argument("--voice-change", dest="voice_change",
                    action="store_true",
                    help="Re-voice an existing audio file with the "
                         "ElevenLabs voice changer (speech-to-speech): "
                         "convert --in-wav to the --voice-id voice and "
                         "write --out-wav. No other pipeline stages.")
    ap.add_argument("--in-wav", dest="in_wav", default=None,
                    help="Input audio file for --voice-change")
    ap.add_argument("--sts-model", dest="sts_model",
                    default="eleven_multilingual_sts_v2",
                    help="ElevenLabs speech-to-speech model for "
                         "--voice-change (default: eleven_multilingual_sts_v2)")
    ap.add_argument("--regen-chunk", dest="regen_chunk", action="store_true",
                    help="Regenerate ONE chunk: synthesize the text in "
                         "--text-file with ElevenLabs and write --out-wav. "
                         "No emotion pass, no other pipeline stages.")
    ap.add_argument("--text-file", dest="text_file", default=None,
                    help="UTF-8 chunk text file for --regen-chunk (Indic "
                         "text never travels on argv)")
    ap.add_argument("--out-wav", dest="out_wav", default=None,
                    help="Output WAV path for --regen-chunk")
    ap.add_argument("--sync-mode", dest="sync_mode", default=None,
                    choices=["match", "legacy"],
                    help="How dub chunks get their timeline positions "
                         "(v0.7). 'match' (default): Gemini section-matches "
                         "the script sentences to the English cues BEFORE "
                         "TTS, synthesizes per section, places each chunk "
                         "in its English slot and marks the leftovers "
                         "unsync (Auto-Sync-style). 'legacy': the v0.1-v0.6 "
                         "whole-script TTS + re-transcription + mapping "
                         "path. Also settable via a 'sync_mode' key in "
                         "engine_settings.json; the CLI wins.")
    ap.add_argument("--chunk-mode", dest="chunk_mode", default=None,
                    choices=["clause", "sentence", "section"],
                    help="Piece size for match mode. 'clause' (default, "
                         "v0.12): sentences, with any sentence longer than "
                         "~90 chars subdivided at ; : , or a dash — the "
                         "granularity the pre-v0.7 pipeline got by cutting "
                         "the TTS audio at every silence. 'sentence' "
                         "(v0.8): one piece per sentence, no subdivision. "
                         "'section' (v0.7): one piece per matched thought. "
                         "Also settable via a 'chunk_mode' key in "
                         "engine_settings.json; the CLI wins.")
    ap.add_argument("--emotion", dest="emotion", action="store_true",
                    default=None,
                    help="Run the Step-4 emotion enrichment before TTS "
                         "(default: ON, matching the bulk app; also settable "
                         "via an 'emotion' key in engine_settings.json)")
    ap.add_argument("--no-emotion", dest="emotion", action="store_false",
                    help="Skip Step-4 emotion enrichment (send the bare "
                         "punctuated text to TTS)")
    ap.add_argument("--suggest-fit", dest="suggest_fit", action="store_true",
                    help="v0.18 review-screen helper: read the rows in "
                         "--text-file ('@@ <index> <speech_s> <hard_s>' then "
                         "'EN: ...' and 'TR: ...' lines, UTF-8), ask the LLM "
                         "for shorter renderings of each, and write them to "
                         "<text-file>.out ('@@ <index>' then one option per "
                         "line). No audio, no TTS.")
    ap.add_argument("--recommend-voice", dest="recommend_voice",
                    action="store_true",
                    help="v0.19: recommend voices from the account catalogue "
                         "for one line. --text-file (UTF-8): LANG:/CURRENT:/"
                         "SPEAKER:/TR:/EN: lines and 'V: <id>|<name>' lines; "
                         "writes <text-file>.out ('PICK: <id>|<why>').")
    ap.add_argument("--learn-final", dest="learn_final",
                    action="store_true",
                    help="v0.21 AI mode: learn from the FINAL dub (the "
                         "timeline after every regeneration). --text-file "
                         "(UTF-8): 'BASE: <run base path>' and one "
                         "'C: <start_s>|<len_s>|<text>' line per dub chunk; "
                         "writes <text-file>.out (summary lines). Text-only "
                         "LLM call, no audio, no TTS, no prompt files.")
    ap.add_argument("--review-assist", dest="review_assist",
                    action="store_true",
                    help="v0.18.6 review-screen assistant: one line + the "
                         "reviewer's instruction in --text-file (UTF-8: "
                         "'@@ <index> <speech_s> <hard_s>', EN:/TR:/PREV:/"
                         "NEXT:/ASK: lines, H: history lines); writes "
                         "<text-file>.out (REPLY:/SPEED:/ALT: lines). No "
                         "audio, no TTS, no prompt files.")
    ap.add_argument("--test-llm", dest="test_llm", action="store_true",
                    help="Make one tiny LLM call on the configured provider "
                         "and write a {status, provider, model, reply} "
                         "manifest. No audio, no TTS.")
    ap.add_argument("--list-voices", dest="list_voices", action="store_true",
                    help="Fetch the ElevenLabs account voice catalogue "
                         "(language-token sorted, requires --language) and "
                         "write a {status, voices:[{id,name},…]} manifest.")
    ap.add_argument("--selfcheck", action="store_true",
                    help="Import the pipeline package, verify every required "
                         "symbol and prompt file exists, print SELFCHECK OK "
                         "and exit. Missing config/ settings files are "
                         "warnings, not failures. No network, no audio "
                         "processing.")
    args = ap.parse_args()

    if args.selfcheck:
        return args
    if args.review_assist or args.recommend_voice or args.learn_final:
        if not args.language or not args.text_file:
            ap.error("--review-assist/--recommend-voice/--learn-final "
                     "require --language and --text-file")
        return args
    if args.suggest_fit:
        if args.test_llm or args.regen_chunk or args.list_voices \
           or args.voice_change:
            ap.error("--suggest-fit cannot be combined with other modes")
        if not args.language:
            ap.error("--suggest-fit requires --language")
        if not args.text_file:
            ap.error("--suggest-fit requires --text-file <utf-8 rows file>")
        return args
    if args.test_llm:
        if args.regen_chunk or args.list_voices or args.voice_change:
            ap.error("--test-llm cannot be combined with other modes")
        return args
    if args.list_voices:
        if args.regen_chunk or args.voice_change:
            ap.error("--list-voices cannot be combined with other modes")
        if not args.language:
            ap.error("--list-voices requires --language")
        return args
    if args.voice_change:
        if args.regen_chunk:
            ap.error("--voice-change cannot be combined with --regen-chunk")
        if not args.language:
            ap.error("--voice-change requires --language")
        if not args.in_wav:
            ap.error("--voice-change requires --in-wav <input audio path>")
        if not args.out_wav:
            ap.error("--voice-change requires --out-wav <output wav path>")
        if args.script or args.provided_script or args.text_file:
            ap.error("--script/--provided-script/--text-file are not valid "
                     "with --voice-change")
        return args
    if args.regen_chunk:
        if not args.language:
            ap.error("--regen-chunk requires --language")
        if not args.text_file:
            ap.error("--regen-chunk requires --text-file <utf-8 chunk text>")
        if not args.out_wav:
            ap.error("--regen-chunk requires --out-wav <output wav path>")
        if args.script or args.provided_script:
            ap.error("--script/--provided-script are not valid with "
                     "--regen-chunk")
        return args
    if not args.audio or not args.language:
        ap.error("--audio and --language are required unless --selfcheck, "
                 "--test-llm, --list-voices, --voice-change or --regen-chunk")
    if args.steps == "dub" and not args.script:
        ap.error("--steps dub requires --script <translation text file> — "
                 "run '--steps translate' first, review/edit its "
                 "translation_text file, then pass that file here")
    if args.script and args.steps != "dub":
        ap.error("--script is only valid with --steps dub")
    if args.provided_script and args.steps == "dub":
        ap.error("--provided-script is only valid with --steps translate/"
                 "full (--steps dub already takes the script via --script)")
    # v0.13 pause-aware sync. The plan stage takes the target script the same
    # way Paste Translation does (--provided-script); the generate stage takes
    # the approved plan instead, because the plan IS the script by then.
    if args.steps == "plan" and not (args.provided_script or args.plan):
        ap.error("--steps plan requires --provided-script <utf-8 target "
                 "script> — the pasted target-language text to lay out "
                 "across the detected pauses — or --plan <sync plan file> "
                 "to re-measure text that is already assigned per chunk")
    if args.steps == "dubplan" and not args.plan:
        ap.error("--steps dubplan requires --plan <sync plan file> — run "
                 "'--steps plan' first, review/edit its plan file, then pass "
                 "that file here")
    if args.plan and args.steps not in ("dubplan", "plan"):
        ap.error("--plan is only valid with --steps dubplan (generate) or "
                 "--steps plan (re-measure the corrected TR: lines)")
    if args.provided_script and args.steps == "dubplan":
        ap.error("--provided-script is not valid with --steps dubplan (the "
                 "approved plan already carries the target text)")
    if args.text_file or args.out_wav or args.in_wav:
        ap.error("--text-file/--out-wav/--in-wav are only valid with "
                 "--regen-chunk / --voice-change")
    if args.script_source in ("ai", "ai_test", "eleven"):
        if args.provided_script:
            ap.error(f"--script-source {args.script_source} writes the script "
                     "itself — it cannot be combined with --provided-script")
        if args.steps in ("plan", "dubplan"):
            ap.error(f"--script-source {args.script_source} is only valid "
                     "with --steps full/translate/dub")
    return args


# v0.24 "AI · test rules": AI mode + the distilled prompt-mode house rules,
# run read-only (never learns) into its own <audio>_TEST folder. Every AI-mode
# behaviour (phrase cues, anchor sync, no prompt files) applies to it.
TEST_SUFFIX = "_TEST"


def _ai_source(args):
    """"ai" / "ai_test" for the AI translator sources, else None."""
    s = getattr(args, "script_source", "prompt")
    return s if s in ("ai", "ai_test") else None


def _is_test(args) -> bool:
    return getattr(args, "script_source", "prompt") == "ai_test"


def _is_test_base(base: str) -> bool:
    """A run base inside an <x>_TEST folder, or whose AI draft says test."""
    if os.path.basename(os.path.dirname(base or "")).endswith(TEST_SUFFIX):
        return True
    try:
        with open((base or "") + "_ai_draft.json", "r", encoding="utf-8") as f:
            return json.load(f).get("mode") == "test"
    except Exception:
        return False


def _emotion_enabled(args) -> bool:
    """Step-4 emotion enrichment toggle.

    Priority: --emotion/--no-emotion CLI flag > 'emotion' key in
    engine_settings.json > ON (the bulk app's own default).
    """
    if args.emotion is not None:
        return bool(args.emotion)
    try:
        with open(ENGINE_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "emotion" in data:
            return bool(data["emotion"])
    except Exception:
        pass
    return True


def _sync_mode(args) -> str:
    """v0.7 sync-mode toggle: --sync-mode CLI flag > 'sync_mode' key in
    engine_settings.json > 'match' (the new default)."""
    if getattr(args, "sync_mode", None) in ("match", "legacy"):
        return args.sync_mode
    try:
        with open(ENGINE_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = (data.get("sync_mode") or "").strip().lower() \
            if isinstance(data, dict) else ""
        if v in ("match", "legacy"):
            return v
    except Exception:
        pass
    return "match"


CHUNK_MODES = ("clause", "sentence", "section")


def _chunk_mode(args) -> str:
    """Piece-size toggle for match mode: --chunk-mode CLI flag >
    'chunk_mode' in engine_settings.json > 'clause' (the v0.12 default).

    'clause'   = sentences, and any sentence longer than
                 tts.CLAUSE_MAX_CHARS subdivided at ; : , or a dash. This
                 is the granularity the pre-v0.7 pipeline got by cutting
                 the TTS audio at every silence — Indic scripts chain
                 clauses with those marks, so sentence-only splitting left
                 15-second blocks on the timeline.
    'sentence' = one piece per sentence, no subdivision (v0.8).
    'section'  = one piece per matched thought (v0.7)."""
    if getattr(args, "chunk_mode", None) in CHUNK_MODES:
        return args.chunk_mode
    try:
        with open(ENGINE_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = (data.get("chunk_mode") or "").strip().lower() \
            if isinstance(data, dict) else ""
        if v in CHUNK_MODES:
            return v
    except Exception:
        pass
    return "clause"


def _chunk_max_chars(args, pl) -> int:
    """Clause-subdivision threshold: 'chunk_max_chars' in
    engine_settings.json, else tts.CLAUSE_MAX_CHARS. Exposed as a setting so
    a talk that still feels blocky (or too choppy) can be retuned without a
    code change. Values below 20 are ignored — that only produces fragments
    the merge step would glue back together anyway."""
    try:
        with open(ENGINE_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = int(data.get("chunk_max_chars") or 0) if isinstance(data, dict) else 0
        if v >= 20:
            return v
    except Exception:
        pass
    return pl.CLAUSE_MAX_CHARS


def _engine_setting(key, default, cast=float, minimum=None, maximum=None):
    """One scalar from engine_settings.json, validated, else *default*.

    Same read-with-fallback shape as _chunk_mode / _chunk_max_chars: the file
    is optional, a broken value is ignored rather than fatal, and the caller
    always gets a usable number.
    """
    try:
        with open(ENGINE_SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or key not in data:
            return default
        raw = data.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return default
        v = cast(raw)
        if minimum is not None and v < minimum:
            return default
        if maximum is not None and v > maximum:
            return default
        return v
    except Exception:
        return default


def _plan_settings(pl):
    """The v0.13 pause-aware knobs, resolved once per run.

    Returns (pause_min_s, thr_db, max_atempo, rate_override). All are
    optional keys in engine_settings.json — an install that never touches
    that file gets the tuned defaults.
    """
    pause_min_s = _engine_setting("pause_min_ms", pl.PAUSE_MIN_S * 1000.0,
                                  float, minimum=0.0, maximum=5000.0) / 1000.0
    thr_db = _engine_setting("pause_thr_db", pl.DEFAULT_THR_DB, float,
                             minimum=-90.0, maximum=-6.0)
    max_atempo = _engine_setting("max_atempo", pl.MAX_ATEMPO, float,
                                 minimum=1.0, maximum=2.0)
    rate_override = _engine_setting("plan_rate_override", 0.0, float,
                                    minimum=1.0, maximum=60.0)
    return pause_min_s, thr_db, max_atempo, rate_override


def _app_version() -> str:
    """Fast-syncs VERSION file (repo root, two levels above engine/)."""
    try:
        p = os.path.join(os.path.dirname(os.path.dirname(ENGINE_DIR)),
                         "VERSION")
        with open(p, "r", encoding="utf-8") as f:
            return (f.readline() or "").strip()
    except Exception:
        return ""


def _say(tag, msg):
    """Progress line in the contract format: '[Sxx] message'."""
    print(f"[{tag}] {msg}", flush=True)


def _note(msg):
    """Untagged informational log line (does not change the panel stage)."""
    print(f"[engine] {msg}", flush=True)


def _import_pipeline():
    """Import the local pipeline package and return a flat facade namespace.

    All pipeline symbols (functions, constants, module attributes like
    the decode helpers) are aggregated onto one SimpleNamespace so the stage code can
    keep addressing them the way it addressed the app module in v0.1/v0.2
    (pl.<symbol>). Name collisions across modules are shared imports of the
    same objects (e.g. TTS_LANGUAGES), so the aggregation order is safe.
    """
    if ENGINE_DIR not in sys.path:
        sys.path.insert(0, ENGINE_DIR)
    from pipeline import (config, stt, srt_tools, llm, tts, sync, tm,  # noqa: F401
                          match, agent_splitter, agent_aligner,
                          pausechunk, preview_html, ai_translator,
                          ai_lang, ai_checks, ai_memory, ai_agents,
                          sync_check, anchor_align, spectral_vad,
                          el_dubbing)
    ns = types.SimpleNamespace()
    for mod in (config, stt, srt_tools, llm, tts, sync, match, agent_splitter,
                agent_aligner, pausechunk, preview_html, ai_lang, ai_checks,
                ai_memory, ai_agents, ai_translator, sync_check,
                anchor_align, spectral_vad, el_dubbing):
        for name, value in vars(mod).items():
            if name.startswith("__"):
                continue
            setattr(ns, name, value)
    return ns


def _check_symbols(pl):
    """Assert every pipeline symbol this worker calls actually exists."""
    missing = []
    for name in REQUIRED_FUNCTIONS:
        fn = getattr(pl, name, None)
        if fn is None or not callable(fn):
            missing.append(name + " (function)")
    for name in REQUIRED_ATTRIBUTES:
        if getattr(pl, name, None) is None:
            missing.append(name + " (attribute)")
    if missing:
        raise RuntimeError(
            "Pipeline package is missing required symbols: "
            + ", ".join(missing))


def _selfcheck(args) -> int:
    pl = _import_pipeline()
    _check_symbols(pl)

    # Every per-language prompt file must exist (hard failure when missing —
    # prompts ship with the repo, so absence means a broken checkout). A
    # USER-ADDED language (v0.7) is different: its prompts are created by the
    # panel, so a gap there is a warning naming the files to write, never a
    # failed selfcheck that blocks setup for the other eleven languages.
    custom = set(_custom_language_names())
    missing_prompts, missing_custom = [], []
    for lang in LANGUAGES:
        for stage in PROMPT_STAGES:
            p = os.path.join(pl.PROMPTS_DIR, f"{stage}_{lang}.txt")
            if not os.path.isfile(p):
                (missing_custom if lang in custom
                 else missing_prompts).append(f"prompts/{stage}_{lang}.txt")
    if missing_prompts:
        raise RuntimeError(
            f"{len(missing_prompts)} prompt file(s) missing from "
            f"{pl.PROMPTS_DIR}: " + ", ".join(missing_prompts[:10])
            + ("…" if len(missing_prompts) > 10 else ""))
    if missing_custom:
        print(f"WARNING: {len(missing_custom)} prompt file(s) missing for "
              "user-added language(s) — create them in the panel's Settings "
              "tab (Prompts → copy from an existing language) before dubbing "
              "into them: " + ", ".join(missing_custom[:10])
              + ("…" if len(missing_custom) > 10 else ""), flush=True)

    # v0.24: "AI · test rules" needs a rules file per language (a warning:
    # without one the test source simply runs as plain AI mode).
    no_rules = [lang for lang in LANGUAGES if lang not in custom
                and not pl.house_rules(lang)]
    if no_rules:
        print("WARNING: no AI house rules (pipeline/ai_rules/) for: "
              + ", ".join(no_rules), flush=True)

    # Config presence is a WARNING, not a failure: a fresh clone passes
    # selfcheck and the user configures keys afterwards (setup / panel).
    for label, path in (("LLM settings", pl.LLM_SETTINGS_FILE),
                        ("TTS settings", pl.TTS_SETTINGS_FILE)):
        if not os.path.isfile(path):
            print(f"WARNING: {label} file missing: {path} — run "
                  "setup_mac.command or open the REAPER panel's Settings "
                  "section before the first real run.", flush=True)

    print("SELFCHECK OK", flush=True)
    return 0


def _write_manifest(manifest, out_dir, keys=MANIFEST_KEYS):
    """Write engine_done.json to the status dir and (if known) out_dir.

    *keys* selects the mode's manifest shape (full/dub, review, regen,
    test-llm, voices); anything else in *manifest* is a working value and
    is filtered out.
    """
    payload = {k: manifest.get(k, "") for k in keys}
    targets = [os.path.join(STATUS_DIR, "engine_done.json")]
    if out_dir:
        targets.append(os.path.join(out_dir, "engine_done.json"))
    os.makedirs(STATUS_DIR, exist_ok=True)
    for path in targets:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            _note(f"WARNING: could not write manifest {path}: {e}")


def _write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _resolve_voice(pl, api_key, language, cli_voice_id):
    """Return (voice_id, description) for the dub voice.

    Explicit --voice-id wins (after strict sanitizing). Otherwise the
    account's voice catalogue is fetched with the pipeline helper, which
    sorts voices that advertise support for *language* to the top and marks
    them with a leading '✦'. We pick the first language-matched voice;
    if none advertises the language, fall back to the first voice on the
    account (eleven_v3 auto-detects the target language from the text).
    Hard-errors with a clear message when no voice is available.
    """
    if cli_voice_id and cli_voice_id.strip():
        vid = pl._sanitize_voice_id(cli_voice_id)
        if not vid:
            raise RuntimeError(
                f"--voice-id {cli_voice_id!r} is not a valid ElevenLabs "
                "voice_id (expected a 12-40 char alphanumeric token, not a "
                "display label).")
        return vid, "from --voice-id"

    _note(f"No --voice-id given — resolving a {language} voice from the "
          "ElevenLabs account catalogue…")
    try:
        voices = pl._fetch_voices_for_language(api_key, language)
    except Exception as e:
        raise RuntimeError(f"Could not fetch the ElevenLabs voice "
                           f"catalogue: {e}")
    if not voices:
        raise RuntimeError(
            "The ElevenLabs account has no voices — add a voice on "
            "elevenlabs.io or pass --voice-id explicitly.")
    matched = [v for v in voices
               if str(v.get("label", "")).startswith("✦")]
    if matched:
        pick = matched[0]
        return pick["voice_id"], (f"auto: first {language}-matched voice "
                                  f"'{pick['name']}'")
    pick = voices[0]
    _note(f"WARNING: no voice on the account advertises {language} support "
          f"— falling back to the first account voice '{pick['name']}' "
          "(eleven_v3 auto-detects the language from the text).")
    return pick["voice_id"], f"auto-fallback: first account voice '{pick['name']}'"


def _load_pipeline_and_keys(args, need_llm=True):
    """Import + symbol-check the pipeline and fail fast on credentials.

    Returns (pl, api_key). The LLM check is skipped for --regen-chunk,
    which never calls a language model.
    """
    _note("Importing pipeline modules…")
    pl = _import_pipeline()
    _check_symbols(pl)
    if args.language not in pl.TTS_LANGUAGES:
        raise RuntimeError(f"Language {args.language!r} is not registered "
                           "in the pipeline's TTS_LANGUAGES table.")
    try:
        api_key = pl._get_api_key()
    except Exception as e:
        raise RuntimeError(f"ElevenLabs API key unavailable: {e}")
    if need_llm:
        try:
            pl._validate_llm_config()
        except Exception as e:
            raise RuntimeError(f"LLM provider not usable: {e}")
    return pl, api_key


def _prepare_out_dir(pl, audio_path, manifest, test=False):
    """Create/reuse the app-convention output folder next to the audio.

    Done BEFORE any paid API work (cheap mkdir+copy): an unwritable input
    location (mounted DMG, read-only share) must fail fast, not after the
    S1a transcription spend. Setting manifest["out_dir"] up front also
    means even early failures write the manifest copy next to the audio.
    Returns (out_dir, base).
    """
    if test:
        return _prepare_test_dir(audio_path, manifest)
    out_dir = pl._prepare_output_dir(audio_path)
    if not os.access(out_dir, os.W_OK):
        raise RuntimeError(f"Output folder is not writable: {out_dir} — "
                           "move the audio to a writable location.")
    base = os.path.join(
        out_dir, os.path.splitext(os.path.basename(audio_path))[0])
    manifest["out_dir"] = out_dir
    copied_audio = os.path.join(out_dir, os.path.basename(audio_path))
    manifest["en_audio"] = copied_audio if os.path.exists(copied_audio) else ""
    return out_dir, base


def _prepare_test_dir(audio_path, manifest):
    """v0.24 "AI · test rules": <src>/<stem>_TEST/ with every file named
    <stem>_TEST_*, exactly as if the audio were called <stem>_TEST — so every
    rule that says "the folder name is the base name" (the panel's cast file,
    Regenerate's English lookup, the importers) works unchanged, and a test
    run never overwrites the AI-mode or prompt-mode files of the same audio."""
    src_dir = os.path.dirname(os.path.abspath(audio_path))
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    tname = stem + TEST_SUFFIX
    if os.path.basename(src_dir) == tname:          # the copy inside it
        out_dir = src_dir
    elif os.path.basename(src_dir) == stem:          # the normal-run copy
        out_dir = os.path.join(os.path.dirname(src_dir), tname)
    else:
        out_dir = os.path.join(src_dir, tname)
    os.makedirs(out_dir, exist_ok=True)
    if not os.access(out_dir, os.W_OK):
        raise RuntimeError(f"Output folder is not writable: {out_dir} — "
                           "move the audio to a writable location.")
    copied = os.path.join(out_dir, os.path.basename(audio_path))
    try:
        if (os.path.abspath(audio_path) != os.path.abspath(copied)
                and not os.path.exists(copied)):
            import shutil
            shutil.copy2(audio_path, copied)
    except Exception:
        pass
    manifest["out_dir"] = out_dir
    manifest["en_audio"] = copied if os.path.exists(copied) else ""
    manifest["variant"] = "test"
    return out_dir, os.path.join(out_dir, tname)


def _stage_translate(pl, args, api_key, manifest, ctx):
    """S1a..S2c: transcription, regions/SRT, translation chain.

    Fills *ctx* with everything the dub half (or the review-file writer)
    needs: out_dir, base, raw_eng, words, regions, en_audio_dur, final_srt,
    punc_result. Shared verbatim between --steps full and --steps translate.
    """
    audio_path = ctx["audio_path"]
    language = args.language
    gemini_model = pl.GEMINI_DEFAULT_MODEL

    # A user-provided translation (--provided-script) replaces the whole
    # S2a-S2c LLM chain. Read + validate it BEFORE any paid API work so a
    # bad file fails fast, not after the S1a transcription spend.
    provided_text = None
    if getattr(args, "provided_script", None):
        p = os.path.abspath(os.path.expanduser(args.provided_script))
        if not os.path.isfile(p):
            raise RuntimeError(f"--provided-script file not found: {p}")
        provided_text = _read_text(p).strip()
        if not provided_text:
            raise RuntimeError(f"--provided-script file is empty: {p}")
        _note(f"Provided translation loaded: {p} "
              f"({len(provided_text)} chars) — the LLM translation chain "
              "will be skipped.")

    out_dir, base = _prepare_out_dir(pl, audio_path, manifest,
                                     test=_is_test(args))
    ctx["out_dir"], ctx["base"] = out_dir, base
    if _is_test(args):
        _note(f"Test mode (AI · test rules): house rules ON, nothing is "
              f"learned; results go to {os.path.basename(out_dir)}.")

    # ── [S1a] Transcribe the English audio ─────────────────────────────────
    _say("S1a", "Transcribing English audio (ElevenLabs Scribe)…")
    result = pl._transcribe_audio(audio_path, api_key)
    words = result.get("words", [])
    raw_eng = (result.get("text", "") or "").strip()
    if not raw_eng and words:
        raw_eng = " ".join(w.get("text", "").strip() for w in words
                           if w.get("type", "word") == "word")
    if not words:
        raise RuntimeError("No word data from ElevenLabs for the English "
                           "audio.")
    _say("S1a", f"Transcription done ({len(words)} word tokens).")
    ctx["words"], ctx["raw_eng"] = words, raw_eng

    # ── [S1b] Regions + English SRT + analysis format ───────────────────────
    _say("S1b", "Detecting speech regions in the English audio…")
    y_data, sr = pl._load_audio_any(audio_path)
    regions = pl._detect_regions_from_audio(
        y_data, sr, pl.DEFAULT_THR_DB, pl.DEFAULT_HYS_DB,
        pl.DEFAULT_MIN_MS)
    if not regions:
        raise RuntimeError("No speech regions detected in the English audio "
                           "(is the file silent?).")
    try:
        ctx["en_audio_dur"] = float(len(y_data)) / float(sr) if sr else 0.0
    except Exception:
        ctx["en_audio_dur"] = 0.0
    ctx["regions"] = regions

    final_srt = pl._build_subtitle_srt(regions, words)
    srt_path = base + ".srt"
    _write_text(srt_path, final_srt)
    manifest["en_srt"] = srt_path
    formatted_srt = pl._parse_srt_to_analysis_format(final_srt)
    _write_text(base + "_analyzed.txt", formatted_srt)
    _say("S1b", f"{len(regions)} regions — English SRT saved: "
                f"{os.path.basename(srt_path)}")
    ctx["final_srt"] = final_srt

    # ── [S2a-c] Translation chain (Step1 -> Step2 -> Step3) ────────────────
    # The pipeline runs all three prompt steps inside one call, so S2b/S2c
    # are announced when the chain returns rather than in real time.
    # Translation memory: the READ-ONLY half of the TM flow (ON by default,
    # like the app) — an exact proofed match reuses the human-proofed script
    # with zero LLM cost, partial matches become a prompt glossary. No
    # _tm_capture — the engine records nothing. The lookup helpers stay
    # getattr-guarded so a broken/absent tm module degrades gracefully.
    ai_mode = (_ai_source(args) is not None and provided_text is None)
    # The AI draft is what --steps dub learns corrections against, so a
    # stale one from an earlier AI run must never outlive this translation.
    ai_draft_path = base + "_ai_draft.json"
    ai_report_path = base + "_ai_report.txt"
    ai_phrases_path = base + AI_PHRASES_SUFFIX
    for stale in (ai_draft_path, ai_report_path, ai_phrases_path,
                  base + EL_DUB_SUFFIX):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except Exception:
                pass
    # v0.20: AI mode times its translation against fine English phrases
    # (word timestamps + spectral pauses) instead of the loudness regions.
    # Fail-open: any problem keeps the region SRT, exactly as before.
    ai_srt = final_srt
    if ai_mode:
        try:
            cues = _ai_phrase_cues(pl, words, regions, audio_path, y_data, sr)
            if cues:
                ai_srt = _cues_to_srt(pl, cues)
                _write_text(ai_phrases_path, ai_srt)
                ctx["ai_phrases_srt"] = ai_srt
        except Exception as e:                           # noqa: BLE001
            ai_srt = final_srt
            _note(f"WARNING: AI phrase cues failed ({e}) — using the speech-"
                  "region SRT.")
    tm_cached = None
    tm_glossary = ""
    _tm_full = getattr(pl, "_tm_lookup_full", None)
    _tm_block = getattr(pl, "_tm_glossary_block", None)
    if callable(_tm_full) and provided_text is None:
        try:
            en_entries_tm = pl._extract_srt_entries(final_srt)
            tm_cached = _tm_full(language, en_entries_tm)
            if not tm_cached and callable(_tm_block):
                tm_glossary = _tm_block(language, en_entries_tm) or ""
        except Exception as e:
            tm_cached, tm_glossary = None, ""
            _note(f"WARNING: translation-memory lookup failed ({e}) — "
                  "running the full LLM chain.")
    if provided_text is not None:
        tr_result = rev_result = punc_result = provided_text
        _say("S2a", "Using the provided translation — LLM translation "
                    "skipped.")
        _say("S2b", "Review step skipped (translation provided by user).")
        _say("S2c", "Punctuation step skipped (translation provided by "
                    "user).")
    elif tm_cached:
        tr_result = rev_result = punc_result = tm_cached
        _say("S2a", "Translation memory hit — reusing the human-proofed "
                    "script (no LLM call).")
        if ai_mode:
            # v0.18: a memory hit still gets timed rows, so anchor sync (and
            # 1:1 review pairing) work on a re-run of an already-learned
            # script instead of falling back to the prompt sync.
            m_rows, m_exact = _memory_rows(pl, language, ai_srt, tm_cached)
            if m_rows:
                _note("AI mode: memory paragraphs paired "
                      + ("by their original English (exact)." if m_exact else
                         "by length, snapped to English sentence ends."))
                ctx["ai_rows"] = m_rows
                prs = m_rows
                punc_result = tr_result = rev_result = "\n\n".join(
                    r["tr"] for r in ctx["ai_rows"])
                _write_text(ai_draft_path, json.dumps({
                    "language": language,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "rows": ctx["ai_rows"], "pages": [],
                    "source": "memory",
                    "mode": "test" if _is_test(args) else "ai"},
                    ensure_ascii=False, indent=2))
                _note(f"AI mode: {len(prs)} memory paragraph(s) timed "
                      "against the English — anchor sync can place them.")
        _say("S2b", "Review step skipped (proofed script from memory).")
        _say("S2c", "Punctuation step skipped (proofed script from memory).")
    elif ai_mode:
        # v0.18 AI mode: the self-learning translator replaces the whole
        # Step1-3 prompt chain. Its rows pair each paragraph with its exact
        # English, so review and learning need no char-share guessing.
        # Lekhak pipeline: memory -> batched translator -> checks -> repair
        # -> critic panel -> proofer. The [S2a-c] tags stay (the panel's
        # step list greps them); the agent-by-agent detail is [engine] lines.
        _say("S2a", "AI mode: agentic translator running (memory -> "
                    "translator -> checks -> repair -> critics -> proofer)…")
        punc_result, ai_rows, ai_info = pl.ai_translate_script(
            pl._extract_srt_entries(ai_srt), language, gemini_model,
            status_cb=_note, options={
                "audit_rate": _engine_setting("ai_audit_rate", 0.02, float,
                                              0.0, 1.0),
                "proofer": bool(_engine_setting("ai_proofer", 1, int, 0, 1)),
                "concurrency": _engine_setting("ai_concurrency", 3, int,
                                               1, 12),
                "house_rules": _is_test(args),
            })
        tr_result = rev_result = punc_result
        ctx["ai_rows"] = ai_rows
        _write_text(ai_draft_path, json.dumps({
            "language": language,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rows": ai_rows,
            "pages": ai_info.get("page_outcomes", []),
            "mode": "test" if _is_test(args) else "ai",
        }, ensure_ascii=False, indent=2))
        _write_text(ai_report_path, ai_info.get("report", ""))
        _say("S2b", f"AI agents done: {ai_info['summary']}")
        for po in sorted(ai_info.get("page_outcomes", []),
                         key=lambda o: -o["risk"])[:3]:
            if po["verified"] and not po["proof"]:
                continue
            _note(f"  review first: page {po['n']} (cues {po['cues'][0]}-"
                  f"{po['cues'][1]}, risk {po['risk']:.2f})"
                  + (" — " + "; ".join(c["code"] for c in po["checks"])
                     if po["checks"] else ""))
        _say("S2c", f"Review report saved: {os.path.basename(ai_report_path)}"
                    " (prompt-chain review/punctuation skipped in AI mode).")
    else:
        if tm_glossary:
            _note("Translation memory: partial matches found — injecting "
                  "the approved-translations glossary into the prompt.")
        _say("S2a", f"Translation chain running on {gemini_model} "
                    f"(Step1 translate -> Step2 review -> Step3 punctuation)…")
        (tr_result, rev_result, punc_result,
         _tr_in, _rev_in, _punc_in) = pl._run_gemini_pipeline(
            formatted_srt, gemini_model, language=language, steps=3,
            tm_glossary=tm_glossary)
        _say("S2b", "Review step done (ran inside the translation chain).")
        _say("S2c", "Punctuation step done (ran inside the translation chain).")

    _write_text(base + "_TranslationStep.txt", tr_result)
    _write_text(base + "_ReviewStep.txt", rev_result)
    combined = (f"=== ENGLISH TRANSCRIPTION ===\n{raw_eng}\n\n"
                f"=== {language.upper()} TRANSLATION ===\n{punc_result}")
    _write_text(base + "_FinalScript.txt", combined)
    _say("S2c", "FinalScript saved.")
    ctx["punc_result"] = punc_result


def _paragraphs(text):
    """The script's paragraphs, by the same rule everything else uses:
    blank-line separated, empties dropped. This is the unit the cast is
    keyed by — one paragraph is one row of the panel's review screen."""
    import re as _re
    return [p.strip() for p in _re.split(r"\n\s*\n", text or "")
            if p.strip()]


# v0.20: AI mode's own English phrase cues, persisted by the translate stage
# so --steps dub aligns against the same phrases the translator was timed on.
AI_PHRASES_SUFFIX = "_ai_phrases.srt"
# v0.22: the ElevenLabs Dubbing Studio project behind an --script-source
# eleven run (dubbing_id + speakers + timed segments). Deleted at the start of
# every translate, like the AI draft.
EL_DUB_SUFFIX = "_el_dub.json"


def _region_items(audio_path):
    """The user's English chunks, [(start_s, length_s)] from the audio's own
    0:00, read from the additive "items" list of <wav>.dubregion.json (the
    panel writes it when it renders a region). Fewer than two items says
    nothing about phrasing, so [] — as for a plain file with no sidecar."""
    try:
        data = json.loads(_read_text(audio_path + ".dubregion.json"))
        items = [(float(it["start"]), float(it["len"]))
                 for it in (data.get("items") or [])
                 if isinstance(it, dict) and float(it.get("len") or 0) > 0.05]
    except Exception:
        return []
    return sorted(items) if len(items) >= 2 else []


def _ai_phrase_cues(pl, words, regions, audio_path, y=None, sr=None):
    """v0.20 English phrase cues for AI mode: [(start_s, end_s, text)].

    The user's own timeline chunks when the region sidecar lists them
    (cues_from_items), else word-timestamp phrases (phrase_cues) cut at word
    gaps, sentence ends and the STFT pauses of spectral_pauses. Knobs, all
    optional in engine_settings.json: ai_phrase_gap_ms (300),
    ai_phrase_max_s (5.0), ai_phrase_min_s (0.8), ai_spectral_vad (1/0)."""
    gap_s = _engine_setting("ai_phrase_gap_ms", 300.0, float, 80.0,
                            2000.0) / 1000.0
    max_s = _engine_setting("ai_phrase_max_s", 5.0, float, 1.5, 30.0)
    min_s = _engine_setting("ai_phrase_min_s", 0.8, float, 0.0, max_s)
    pauses = []
    if y is not None and sr and _engine_setting("ai_spectral_vad", 1, int,
                                                0, 1):
        try:
            pauses = pl.spectral_pauses(y, sr)
        except Exception as e:                           # noqa: BLE001
            _note(f"WARNING: spectral pause detection failed ({e}) — "
                  "phrases from word gaps only.")
    items = _region_items(audio_path) if audio_path else []
    if items:
        cues = pl.cues_from_items(words, items, pauses, gap_s, max_s, min_s)
        how = f"your {len(items)} timeline chunk(s)"
    else:
        cues = pl.phrase_cues(words, regions, pauses, gap_s, max_s, min_s)
        how = (f"word timestamps + {len(pauses)} spectral pause(s)"
               if pauses else "word timestamps")
    if cues:
        _note(f"AI mode: {len(cues)} English phrase(s) from {how} (longest "
              f"{max(e - s for s, e, _t in cues):.1f}s).")
    return cues


def _cues_to_srt(pl, cues):
    """(start, end, text) cues -> SRT text, the sync SRT's layout."""
    lines = []
    for n, (s, e, t) in enumerate(cues, 1):
        lines += [str(n), f"{pl._srt_ts(s)} --> {pl._srt_ts(e)}", t, ""]
    return "\n".join(lines)


def _load_ai_phrases(ctx):
    """--steps dub: pick up the phrases the translate stage saved, so anchor
    sync uses the translator's own cues (not the coarser sync SRT)."""
    p = ctx["base"] + AI_PHRASES_SUFFIX
    if os.path.isfile(p):
        try:
            ctx["ai_phrases_srt"] = _read_text(p)
        except Exception:
            pass


def _cast_map(pl, base):
    """{paragraph_index: voice_id} from <base>_speakers.json, or {}.

    The panel writes this file when the review screen casts a paragraph to
    a second voice, and writes NO file (and deletes a stale one) for a
    single-voice run — so an empty map here means "one voice", not "the
    cast is missing"."""
    fn = getattr(pl, "_speakers_voice_map", None)
    if not callable(fn):
        return {}
    try:
        return fn(base) or {}
    except Exception:
        return {}


def _cast_note(cast, voices, what):
    """One log line that says who is speaking how much — the last chance to
    notice a mis-cast script before the credits are spent on it."""
    tally = {}
    for v in voices:
        tally[v] = tally.get(v, 0) + 1
    parts = ", ".join(f"{v}: {n}" for v, n in
                      sorted(tally.items(), key=lambda kv: -kv[1]))
    return (f"Multi-speaker cast: {len(tally)} voice(s) over {len(voices)} "
            f"{what} ({parts}); map covers "
            f"{len(cast)} paragraph(s).")


def _stage_dub(pl, args, api_key, manifest, ctx, voice_id):
    """S2d..S3e dispatcher (v0.7): 'match' = Gemini section matching before
    per-section TTS (Auto-Sync-style placement + Un sync statuses);
    'legacy' = the v0.1-v0.6 whole-script TTS + re-transcription path."""
    # v0.22 ElevenLabs Dub: ElevenLabs voices and times every segment
    # itself; the render is already aligned to the English timeline.
    if getattr(args, "script_source", "prompt") == "eleven":
        _note("Sync mode: ElevenLabs Dub (segments are timed by ElevenLabs; "
              "no matching, no anchor alignment)")
        _stage_dub_eleven(pl, args, api_key, manifest, ctx, voice_id)
        return
    # v0.18 AI mode: every paragraph already knows its English lines, so
    # no matching prompt is needed — anchor sync, when the rows line up.
    rows = _anchor_rows(pl, args, ctx)
    if rows:
        _note("Sync mode: anchor (AI mode — each paragraph already knows its "
              "English lines; no matching or mapping prompt)")
        _stage_dub_anchor(pl, args, api_key, manifest, ctx, voice_id, rows)
        return
    mode = _sync_mode(args)
    if _ai_source(args) and mode != "match":
        # AI mode never uses the prompt-file sync (SyncingPrompt/Step4):
        # match mode builds its matcher prompt inline in code.
        _note(f"Sync mode: match (AI mode never uses the '{mode}' prompt-"
              "file sync)")
        mode = "match"
    else:
        _note(f"Sync mode: {mode}")
    if mode == "match":
        _stage_dub_match(pl, args, api_key, manifest, ctx, voice_id)
    else:
        _stage_dub_legacy(pl, args, api_key, manifest, ctx, voice_id)


def _stage_dub_match(pl, args, api_key, manifest, ctx, voice_id):
    """v0.7/v0.8 match mode, S2d..S3e:

      S2d  script -> sentences -> ONE Gemini section-match call against the
           English sync-SRT cues -> TTS. Piece size is _chunk_mode():
             sentence (v0.8 default) — long natural stretches through the
               /with-timestamps endpoint, cut into ONE PIECE PER SENTENCE
               at the character times ElevenLabs itself reports; each
               sentence targets its own slice of its section's window
             section (v0.7)  — one piece per matched thought
      S3a  English sync SRT persisted (it fed the matcher)
      S3b  piece offsets come from synthesis — NO second Scribe pass
      S3c  match summary echo
      S3d  placement + order sweep (sentence mode may borrow bounded slack
           from a neighbour before demoting) -> synced/unsync statuses,
           timestamps txt (6th [status] field), texts sidecar, synced SRT
      S3e  synced-only render

    Step-4 emotion enrichment is NOT applied in this mode: pieces are
    matched and synthesized on the clean reviewed text (enriching each
    piece separately would multiply LLM calls; enriching the whole script
    first would break the sentence-id mapping).
    """
    language = args.language
    out_dir, base = ctx["out_dir"], ctx["base"]
    audio_path = ctx["audio_path"]
    script_text = ctx["script_text"]
    grain = _chunk_mode(args)

    # English cue list: reuse the persisted sync SRT (dub resume) or build
    # it from the in-memory S1a transcription (full run).
    en_sync_path = base + "_sync_en.srt"
    if ctx.get("en_srt_text"):
        en_srt = ctx["en_srt_text"]
    else:
        en_srt = pl._build_english_subtitle_srt(ctx["regions"], ctx["words"])
        _write_text(en_sync_path, en_srt)
    en_entries = pl._extract_srt_entries(en_srt)
    if not en_entries:
        raise RuntimeError("The English sync SRT contains no cues — cannot "
                           "match the script against it.")

    # ── [S2d] unit split + Gemini section match + TTS ──────────────────────
    # 'clause' subdivides an over-long sentence at ; : , or a dash so a
    # single unit never becomes a 15-second block on the timeline.
    cast = _cast_map(pl, base)
    unit_paras = None
    if cast and hasattr(pl, "_split_script_into_units_with_paras"):
        # The cast is keyed by paragraph, so the units have to remember which
        # paragraph they came from. Same split as below, walked once.
        if grain == "clause":
            sentences, unit_paras = pl._split_script_into_units_with_paras(
                script_text, _chunk_max_chars(args, pl))
        else:
            sentences, unit_paras = \
                pl._split_script_into_sentences_with_paras(script_text)
    elif grain == "clause":
        sentences = pl._split_script_into_units(script_text,
                                                _chunk_max_chars(args, pl))
    else:
        sentences = pl._split_script_into_sentences(script_text)
    if not sentences:
        raise RuntimeError("The dub script produced no units to match.")
    _longest = max(len(s) for s in sentences)
    _say("S2d", f"Script split into {len(sentences)} unit(s) "
                f"(longest {_longest} chars ≈ "
                f"{_longest / pl.CLAUSE_CHARS_PER_SEC:.1f}s of speech).")
    if _emotion_enabled(args):
        _say("S2d", "note: Step-4 emotion enrichment is skipped in match "
                    "sync mode (pieces are synthesized from the clean "
                    "reviewed text).")
    _say("S2d", f"Piece size: {grain}.")
    _say("S2d", f"Matching {len(sentences)} script sentence(s) to "
                f"{len(en_entries)} English cue(s) with Gemini…")
    sections, unmatched_tr, unmatched_en, sentences = pl.agentic_split_match(
        en_entries, sentences, language, pl.GEMINI_DEFAULT_MODEL,
        status_cb=lambda m: _say("S2d", m))

    tts_path = os.path.join(
        out_dir, pl._tts_output_name(language, audio_path, "_tts"))

    # A piece speaks in the voice cast to the paragraph its first sentence
    # came from. agentic_split_match rewrites sentences in place and never
    # renumbers them, so the sentence -> paragraph list built above is still
    # valid here; a piece whose paragraph nobody cast falls to the main voice.
    def _piece_voice(tr_ids):
        if not unit_paras:
            return voice_id
        for j in tr_ids or ():
            if 0 < j <= len(unit_paras):
                v = cast.get(unit_paras[j - 1])
                if v:
                    return v
        return voice_id

    if grain != "section":
        pieces = pl.build_pieces(sections, unmatched_tr, sentences,
                                 en_entries)
        if not pieces:
            raise RuntimeError("Section matching produced no pieces.")
        texts = [p["text"] for p in pieces]
        n_matched = sum(1 for p in pieces if p.get("win"))
        _say("S2d", f"Match result: {n_matched} matched sentence(s), "
                    f"{len(pieces) - n_matched} unmatched, "
                    f"{len(unmatched_en)} English cue(s) without a "
                    "translation.")
        p_voices = [_piece_voice(p.get("tr_ids")) for p in pieces]
        if len(set(p_voices)) > 1:
            _say("S2d", _cast_note(cast, p_voices, "sentence piece(s)"))
        else:
            p_voices = None
        _say("S2d", f"Synthesizing {len(texts)} sentence(s) in long "
                    f"stretches ({language}, voice {voice_id}, model "
                    f"{args.el_model})…")
        tts_path, spans = pl.synthesize_sentences_elevenlabs(
            texts, tts_path, api_key=api_key, voice_id=voice_id,
            model_id=args.el_model, voices=p_voices,
            status_cb=lambda m: _say("S2d", m))
    else:
        chunks = pl.build_chunks(sections, unmatched_tr, sentences)
        if not chunks:
            raise RuntimeError("Section matching produced no TTS chunks.")
        texts = [c["text"] for c in chunks]
        n_matched = sum(1 for c in chunks if c["en_ids"])
        _say("S2d", f"Match result: {n_matched} matched chunk(s), "
                    f"{len(chunks) - n_matched} unmatched chunk(s), "
                    f"{len(unmatched_en)} English cue(s) without a "
                    "translation.")
        c_voices = [_piece_voice(c.get("tr_ids")) for c in chunks]
        if len(set(c_voices)) > 1:
            _say("S2d", _cast_note(cast, c_voices, "section(s)"))
            # A section is a THOUGHT, and Gemini can group one across a
            # paragraph break — which, with a cast, can mean across a
            # speaker. The section then speaks in its first paragraph's
            # voice. Said out loud here because 'sentence' or 'clause' piece
            # size does not have the problem, and switching is the fix.
            _mixed = sum(1 for c in chunks
                         if len({_piece_voice([j]) for j in
                                 (c.get("tr_ids") or ())}) > 1)
            if _mixed:
                _say("S2d", f"WARNING: {_mixed} matched section(s) span more "
                            "than one cast voice; each is spoken by the "
                            "voice of its first paragraph. Piece size "
                            "'clause' or 'sentence' keeps every speaker "
                            "separate.")
        else:
            c_voices = None
        _say("S2d", f"Synthesizing {len(chunks)} section(s) ({language}, "
                    f"voice {voice_id}, model {args.el_model})…")
        tts_path, spans = pl.synthesize_sections_elevenlabs(
            texts, tts_path, api_key=api_key, voice_id=voice_id,
            model_id=args.el_model, voices=c_voices,
            status_cb=lambda m: _say("S2d", m))
    manifest["tts_wav"] = tts_path
    _say("S2d", f"TTS audio saved: {os.path.basename(tts_path)} "
                f"({len(spans)} piece span(s)).")

    # ── [S3a..S3c] bookkeeping stages (the heavy work already happened) ────
    _say("S3a", "English sync SRT ready (it drove the matching).")
    _say("S3b", "Piece offsets taken from synthesis — no TTS "
                "re-transcription needed in match mode.")
    _say("S3c", "EN <-> script mapping done by the Gemini section match.")

    # ── [S3d] placement + order sweep + files ───────────────────────────────
    _say("S3d", "Placing pieces into their English slots…")
    durations = [(e - s) / 1000.0 for (s, e) in spans]
    if grain != "section":
        placed = pl.agentic_place_pieces(pieces, durations, en_entries, language,
                                         pl.GEMINI_DEFAULT_MODEL, api_key=api_key,
                                         voice_id=voice_id, el_model=args.el_model,
                                         log=lambda m: _say("S3d", m))
        placed = _run_sync_check(pl, ctx, pieces, placed, durations)
    else:
        placed = pl.place_chunks(chunks, en_entries, durations,
                                 log=lambda m: _say("S3d", m))

    entries = []
    for i, (span, p) in enumerate(zip(spans, placed), 1):
        entries.append({
            "index":           i,
            "orig_start_ms":   int(span[0]),
            "orig_end_ms":     int(span[1]),
            "synced_start_ms": int(round(p["position"] * 1000)),
            "sync_status":     p["status"],
        })
    sync_ts_path = base + "_sync_timestamps.txt"
    _write_text(sync_ts_path, pl._format_timestamps_as_text(entries))
    manifest["timestamps_txt"] = sync_ts_path

    # Texts sidecar: block N (blank-line separated) = timestamps index N.
    # The importers use it for item notes on BOTH tracks (the synced SRT
    # below only covers the synced pieces).
    texts_path = base + "_sync_texts.txt"
    _write_text(texts_path, "\n\n".join(
        " ".join((t or "").split()) or EMPTY_PARAGRAPH_PLACEHOLDER
        for t in texts) + "\n")
    manifest["sync_texts"] = texts_path

    synced_entries = [e for e in entries if e["sync_status"] == "synced"]
    unsynced_n = len(entries) - len(synced_entries)
    manifest["synced_count"] = str(len(synced_entries))
    manifest["unsynced_count"] = str(unsynced_n)

    # Synced SRT: cues for the synced chunks at their timeline positions
    # (importer fallback for the per-item chunk text; the Un sync chunks
    # are not in here, they come from the texts sidecar).
    srt_lines = []
    for n, e in enumerate(
            sorted(synced_entries, key=lambda x: x["synced_start_ms"]), 1):
        start_s = e["synced_start_ms"] / 1000.0
        end_s = start_s + (e["orig_end_ms"] - e["orig_start_ms"]) / 1000.0
        text = " ".join((texts[e["index"] - 1] or "").split())
        srt_lines += [str(n), f"{pl._srt_ts(start_s)} --> {pl._srt_ts(end_s)}",
                      text, ""]
    synced_srt_path = base + "_sync_synced.srt"
    _write_text(synced_srt_path, "\n".join(srt_lines))
    manifest["synced_srt"] = synced_srt_path
    _say("S3d", f"{len(synced_entries)} synced / {unsynced_n} unsync — "
                "timestamps, texts and synced SRT saved.")

    # ── [S3e] synced-only render ────────────────────────────────────────────
    if not synced_entries:
        _say("S3e", "WARNING: no synced sections — the synced render is "
                    "skipped (all chunks go to the Un sync track).")
        return
    _say("S3e", "Rendering the synced audio…")
    synced_path = os.path.join(
        out_dir, pl._tts_output_name(language, audio_path, "_synced"))
    synced_path = pl.ensure_writable_output(
        synced_path, status_cb=lambda m: _say("S3e", m))
    pl.sync_audio_with_timestamps(
        tts_path, synced_entries, synced_path,
        status_cb=lambda m: _say("S3e", m), extend_last=False)
    manifest["synced_wav"] = synced_path
    _say("S3e", f"Synced audio saved: {os.path.basename(synced_path)}")


def _memory_rows(pl, language, final_srt, doc):
    """v0.18.2: timed rows for a script reused whole from memory.

    1. Exact: each paragraph's ORIGINAL English is looked up in memory
       (the pair stored when it was approved, keyed by its translation), and
       consumed cue by cue from the English SRT until the letters match.
       Rows then cover exactly the English each paragraph translates.
    2. Fallback: length-share pairing, but every row boundary is moved to
       the nearest English SENTENCE end (within 3 cues), so a row never
       holds half a sentence — the split that left 'is 108 times the
       diameter of the moon' in the next row and got the paragraph
       shortened into a fragment.
    Returns (rows, exact) or (None, False)."""
    en = pl._extract_srt_entries(final_srt)
    paras = pl._split_translation_paragraphs(doc)
    if not en or not paras:
        return None, False

    def nk(s):
        return "".join(ch for ch in (s or "").lower() if ch.isalnum())

    tm = getattr(pl, "translation_memory", None)
    lookup = {}
    if tm is not None:
        try:
            for en_text, tr in tm.pairs_for(pl._tm_lang(language)):
                lookup.setdefault(nk(tr), en_text)      # newest first
        except Exception:
            lookup = {}
    rows, j, exact = [], 0, bool(lookup)
    for p in paras if exact else []:
        target = nk(lookup.get(nk(p), ""))
        if not target:
            exact = False
            break
        acc, start = "", j
        while j < len(en) and len(acc) < len(target):
            acc += nk(en[j][2])
            j += 1
        if acc != target:
            exact = False
            break
        rows.append({"en": " ".join(e[2] for e in en[start:j]), "tr": p,
                     "start": en[start][0], "end": en[j - 1][1],
                     "cues": list(range(start + 1, j + 1))})
    if exact and j == len(en) and len(rows) == len(paras):
        return rows, True

    # Fallback: length share, boundaries snapped to sentence ends.
    if len(paras) > len(en):
        return None, False
    ends_sentence = [bool(re.search(r"[.?!][\"')\]]*\s*$", e[2])) for e in en]
    en_len = [max(1, len(e[2])) for e in en]
    tot_en, tot_tr = float(sum(en_len)), float(sum(len(p) for p in paras))
    bounds, cum_tr, cum_en, k = [], 0.0, 0.0, 0
    for p in paras[:-1]:
        cum_tr += len(p) / tot_tr
        while k < len(en) and cum_en + en_len[k] / tot_en <= cum_tr:
            cum_en += en_len[k] / tot_en
            k += 1
        bounds.append(k)
    snapped, prev = [], 0
    for n, b in enumerate(bounds):
        lo = prev + 1
        hi = len(en) - (len(bounds) - n)
        cands = [c for c in range(max(lo, b - 3), min(hi, b + 3) + 1)
                 if ends_sentence[c - 1]]
        b2 = min(cands, key=lambda c: abs(c - b)) if cands else \
            min(max(b, lo), hi)
        snapped.append(b2)
        prev = b2
    edges = [0] + snapped + [len(en)]
    rows = []
    for i, p in enumerate(paras):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            return None, False
        rows.append({"en": " ".join(e[2] for e in en[a:b]), "tr": p,
                     "start": en[a][0], "end": en[b - 1][1], "cues": []})
    return rows, False


def _anchor_rows(pl, args, ctx):
    """AI-mode rows carrying the APPROVED text (and, when known, their
    English windows as soft hints), or None outside AI mode.

    v0.18.3: AI mode never falls back to a prompt-file sync. The draft rows
    (translator or memory) only HINT the phrase aligner; when they cannot be
    used — the reviewer added/removed a paragraph, no draft was saved, a row
    has no window — the approved paragraphs are returned WITHOUT windows and
    anchor sync aligns them to the English phrases purely by duration.
    ai_anchor_sync=0 is honoured by returning None; the dispatcher then uses
    match mode (inline prompt, no prompt files), never legacy."""
    if not _ai_source(args):
        return None
    if not _engine_setting("ai_anchor_sync", 1, int, 0, 1):
        _note("Anchor sync is off (ai_anchor_sync=0) — AI mode uses match "
              "sync (no prompt files).")
        return None
    paras = pl._split_translation_paragraphs(ctx.get("script_text") or "")
    if not paras:
        return None
    rows = ctx.get("ai_rows")
    if not rows:
        draft = ctx["base"] + "_ai_draft.json"
        try:
            rows = json.loads(_read_text(draft)).get("rows") or [] \
                if os.path.isfile(draft) else []
        except Exception:
            rows = []
    why = ""
    if not rows:
        why = "no AI draft rows for this script"
    elif len(paras) != len(rows):
        why = (f"the approved script has {len(paras)} paragraph(s) but the "
               f"AI draft has {len(rows)}")
    elif any(r.get("start") is None or r.get("end") is None for r in rows):
        why = "a draft row has no time window"
    if why:
        _note(f"Anchor sync: {why} — aligning the approved paragraphs to the "
              "English phrases by duration alone (no row hints, no prompt "
              "files).")
        return [{"en": "", "tr": t, "start": None, "end": None,
                 "exact": False} for t in paras]
    out = []
    for r, text in zip(rows, paras):
        s0, s1 = float(r["start"]), float(r["end"])
        # v0.20: the translator's pause markers (char offsets into its own
        # text) still fit the approved paragraph only if the reviewer left
        # it unchanged; an edited row splits at clauses as before.
        pauses = r.get("pauses")
        if not (isinstance(pauses, list) and pauses and " ".join(
                str(r.get("tr") or "").split()) == " ".join(text.split())):
            pauses = None
        out.append({"en": str(r.get("en") or ""), "tr": text,
                    "start": s0, "end": max(s0, s1),
                    # translator rows name their cues (exact windows);
                    # memory rows were paired by length (a weak hint only)
                    "exact": bool(r.get("cues")),
                    "pauses": pauses})
    return out


def _stage_dub_anchor(pl, args, api_key, manifest, ctx, voice_id, rows):
    """v0.18 anchor sync — AI mode's replacement for the sync prompts.

      S2d  approved paragraphs -> clause units (anchor_align.clause_units;
           'section' piece size keeps one unit per paragraph) -> TTS of every
           unit with measured spans -> PHRASE-LEVEL alignment
           (anchor_align.align_units): consecutive units are grouped onto
           consecutive English sync-SRT phrases so the group's REAL duration
           matches the phrase span; each paragraph's row window is only a
           soft hint (strong for translator rows, weak for memory rows).
           Each group becomes one piece whose window is its phrases' span,
           so the dub pauses where the English pauses. No matching LLM call.
      S3a-c bookkeeping (no re-transcription, no mapping prompt)
      S3d  fit, cheapest first: slot = own window + the pause before the
           next piece; still over at max_atempo -> ONE targeted agent retry
           (fit_to_seconds, re-voice only those pieces); time-stretch up to
           max_atempo; place_pieces (spring + order sweep); sync check agent.
      S3e  synced-only render.
    Output files and manifest keys are match mode's, so the importers are
    unchanged."""
    language = args.language
    out_dir, base = ctx["out_dir"], ctx["base"]
    audio_path = ctx["audio_path"]
    grain = _chunk_mode(args)
    max_atempo = _engine_setting("max_atempo", pl.MAX_ATEMPO, float,
                                 minimum=1.0, maximum=2.0)
    retry_on = bool(_engine_setting("ai_fit_retry", 1, int, 0, 1))
    retry_cap = _engine_setting("ai_fit_retry_max", 12, int, 0, 200)
    cast = _cast_map(pl, base)

    # English phrase cues: v0.20 AI phrases (word timestamps + spectral
    # pauses) when the translate stage made them; else the persisted sync SRT
    # (dub resume) or one built from the in-memory S1a transcription (full
    # run) — as match mode does.
    fine = bool(ctx.get("ai_phrases_srt"))
    if fine:
        en_srt = ctx["ai_phrases_srt"]
    elif ctx.get("en_srt_text"):
        en_srt = ctx["en_srt_text"]
    elif ctx.get("regions") and ctx.get("words"):
        en_srt = pl._build_english_subtitle_srt(ctx["regions"], ctx["words"])
        _write_text(base + "_sync_en.srt", en_srt)
    else:
        en_srt = ""
    cues = pl._extract_srt_entries(en_srt) if en_srt else []

    # ── [S2d] clause units per approved paragraph ─────────────────────────
    live = [(k, r) for k, r in enumerate(rows, 1) if re.search(r"\w", r["tr"])]
    if not live:
        raise RuntimeError("The approved script has no text to dub.")
    windowless = any(r.get("start") is None for (_k, r) in live)
    if windowless and not cues:
        # Nothing to anchor to: no row windows and no English phrases. Match
        # mode (inline prompt, no prompt files) places the script instead.
        _note("Anchor sync: no row windows and no English phrases — using "
              "match sync (no prompt files).")
        _stage_dub_match(pl, args, api_key, manifest, ctx, voice_id)
        return
    # Without row windows, paragraph pieces have nowhere to go: phrase-level.
    phrase_level = bool(cues) and (grain != "section" or windowless)
    units, unit_row, unit_para = [], [], []
    n_marked = 0
    # v0.20.1: splitting at the translator's pause markers is OFF by default.
    # A real 13-minute run showed why: one unit per English phrase chopped
    # Marathi sentences (different word order) into 27 fragments under 1 s,
    # placed with gaps — choppy speech the reviewer rated worse than the
    # punctuation split. engine_settings "ai_marker_split": 1 turns it on.
    use_markers = bool(_engine_setting("ai_marker_split", 0, int, 0, 1))
    for li, (k, r) in enumerate(live):
        if phrase_level and use_markers and r.get("pauses"):
            n_marked += 1
        parts = (pl.clause_units(r["tr"],
                                 pauses=r.get("pauses") if use_markers else None)
                 if phrase_level else [" ".join(r["tr"].split())])
        for p in parts or [" ".join(r["tr"].split())]:
            units.append(p)
            unit_row.append(li)
            unit_para.append(k)
    if n_marked:
        _note(f"Anchor sync: {n_marked} paragraph(s) split at the "
              "translator's pause markers (one unit per English phrase).")
    if _emotion_enabled(args):
        _say("S2d", "note: Step-4 emotion enrichment is skipped in anchor "
                    "sync (pieces are synthesized from the approved text).")

    u_voices = [(cast.get(unit_para[i]) or voice_id) for i in range(len(units))]
    if len(set(u_voices)) > 1:
        _say("S2d", _cast_note(cast, u_voices, "clause unit(s)"))
    else:
        u_voices = None
    tts_path = os.path.join(
        out_dir, pl._tts_output_name(language, audio_path, "_tts"))
    _say("S2d", f"Synthesizing {len(units)} clause unit(s) from "
                f"{len(live)} paragraph(s) ({language}, voice {voice_id}, "
                f"model {args.el_model})…")
    tts_path, u_spans = pl.synthesize_sentences_elevenlabs(
        units, tts_path, api_key=api_key, voice_id=voice_id,
        model_id=args.el_model, voices=u_voices,
        status_cb=lambda m: _say("S2d", m))
    if len(u_spans) != len(units):
        raise RuntimeError(f"Synthesis returned {len(u_spans)} span(s) for "
                           f"{len(units)} unit(s) — refusing to place audio "
                           "that cannot be matched to its text.")
    manifest["tts_wav"] = tts_path
    _say("S2d", f"TTS audio saved: {os.path.basename(tts_path)} "
                f"({len(u_spans)} unit span(s)).")

    # ── phrase-level alignment of units onto English phrases ──────────────
    u_durs = [max(0.05, (e - s) / 1000.0) for (s, e) in u_spans]
    en_speech = sum(max(0.0, e - s) for (s, e, _t) in cues)
    if en_speech > 5 and sum(u_durs) < 0.6 * en_speech:
        _say("S2d", f"WARNING: the dub script speaks {sum(u_durs):.0f}s for "
                    f"{en_speech:.0f}s of English speech — it is probably "
                    "missing content (check the review report); lines will "
                    "be spread out with gaps.")
    row_windows = [((r["start"], r["end"]) if r.get("start") is not None
                    else None) for (_k, r) in live]
    if phrase_level:
        hints = [(3.0 if r.get("exact") else 1.0) for (_k, r) in live]
        groups = pl.align_units(u_durs, unit_row,
                                [(s, e) for (s, e, _t) in cues],
                                row_windows, hints)
    else:
        groups = []
        for li in range(len(live)):
            idx = [i for i in range(len(units)) if unit_row[i] == li]
            groups.append((idx[0], idx[-1], -1, -1))
    # v0.20.1: never leave tiny fragments — a piece with under
    # ai_min_piece_s of speech is merged into its neighbour in the same
    # paragraph (the smaller joined piece wins), so sentences keep flowing.
    min_piece = _engine_setting("ai_min_piece_s", 1.2, float, 0.0, 5.0)
    if phrase_level and min_piece > 0 and len(groups) > 1:
        before = len(groups)
        groups = _merge_short_groups(groups, u_durs, unit_row, min_piece)
        if len(groups) < before:
            _note(f"Anchor sync: {before - len(groups)} short piece(s) "
                  f"(< {min_piece:.1f}s of speech) merged into their "
                  "neighbours — no choppy fragments.")
    pieces, texts, spans, p_voices = [], [], [], []
    for (ul, uh, cl, ch) in groups:
        li = unit_row[ul]
        if cl >= 0:
            win = (cues[cl][0], cues[ch][1])
            en = " ".join(c[2] for c in cues[cl:ch + 1])
        else:
            win = row_windows[li]
            en = live[li][1]["en"]
        text = " ".join(units[ul:uh + 1])
        pieces.append({"tr_ids": list(range(ul + 1, uh + 2)), "text": text,
                       "win": win, "en": en, "para": unit_para[ul]})
        texts.append(text)
        spans.append((u_spans[ul][0], u_spans[uh][1]))
        p_voices.append(u_voices[ul] if u_voices else voice_id)
    if not u_voices:
        p_voices = None
    # Cut between pieces at the quietest point, never through a word.
    spans, n_snap = _snap_cuts(pl, tts_path, spans)
    if n_snap:
        _say("S2d", f"{n_snap} piece cut(s) moved to the nearest word gap "
                    "(no clipped words).")
    skipped = (len(cues) - sum(ch - cl + 1 for (_a, _b, cl, ch) in groups
                               if cl >= 0)) if phrase_level else 0
    if phrase_level:
        _say("S2d", f"Anchor sync: {len(live)} paragraph(s) -> {len(units)} "
                    f"clause unit(s) -> {len(pieces)} piece(s) aligned to "
                    f"{len(cues)} English phrase(s)"
                    + (f" ({skipped} left undubbed: fillers/pauses)"
                       if skipped else "") + " — no matching call.")
    else:
        _say("S2d", f"Anchor sync: {len(pieces)} paragraph piece(s) on their "
                    "row windows (section piece size, or no English phrases).")

    _say("S3a", ("English phrases taken from the AI phrase SRT (word "
                 "timestamps)" if fine else "English phrases taken from the "
                 "sync SRT") + " — no re-transcription.")
    _say("S3b", "Piece offsets taken from synthesis.")
    _say("S3c", "No EN <-> script mapping prompt: phrase alignment by "
                "duration, guided by the translator's paragraph windows.")

    # ── [S3d] fit: stretch -> targeted retry -> stretch, then place ─────────
    _say("S3d", "Fitting pieces to their slots…")
    wins = [p["win"] for p in pieces]
    audio_end = float(ctx.get("en_audio_dur") or 0.0)

    def _slot(i):
        a, b = wins[i]
        if i + 1 < len(wins):
            nxt = wins[i + 1][0]
        else:
            nxt = min(audio_end, b + 2.0) if audio_end > b else b + 1.0
        return max(0.3, max(b, nxt) - a)

    if not pl.PYDUB_AVAILABLE:
        _say("S3d", "WARNING: pydub unavailable — pieces are placed at their "
                    "synthesized length (no stretch, no fit retry).")
        durations = [(e - s) / 1000.0 for (s, e) in spans]
        fitted_spans = spans
    else:
        source = pl._AudioSegment.from_file(tts_path)
        segs = [source[int(s):int(e)] for (s, e) in spans]
        # v0.18.6: the reviewer's per-paragraph speed (review assistant /
        # speed buttons) is applied first, so fitting sees the real length.
        speeds = _row_speeds(base)
        if speeds:
            n_sp = 0
            for i, p in enumerate(pieces):
                f = speeds.get(p.get("para"))
                if f:
                    segs[i] = _stretch_segment(pl, segs[i], f, out_dir,
                                               1000 + i)
                    n_sp += 1
            _say("S3d", f"Reviewer speed applied to {n_sp} piece(s) in "
                        f"{len(speeds)} paragraph(s).")
        changes = []
        over = [i for i, sg in enumerate(segs)
                if len(sg) / 1000.0 > _slot(i) * max_atempo + 0.25]
        if over and retry_on and retry_cap > 0:
            todo = over[:retry_cap]
            _say("S3d", f"{len(over)} piece(s) too long even at "
                        f"{max_atempo:.2f}x — asking the translator agent to "
                        f"fit {len(todo)} of them…")
            new_texts, idxs = [], []
            for i in todo:
                budget = _slot(i) * max_atempo * 0.95
                en = pieces[i]["en"]
                shorter = pl.fit_to_seconds(en, texts[i], budget, language,
                                            pl.GEMINI_DEFAULT_MODEL,
                                            house_rules=_is_test(args))
                if shorter:
                    idxs.append(i)
                    new_texts.append(shorter)
            if idxs:
                retry_path = os.path.join(out_dir, "_fit", "retry_tts.wav")
                os.makedirs(os.path.dirname(retry_path), exist_ok=True)
                r_voices = ([p_voices[i] for i in idxs] if p_voices else None)
                try:
                    retry_path, r_spans = pl.synthesize_sentences_elevenlabs(
                        new_texts, retry_path, api_key=api_key,
                        voice_id=voice_id, model_id=args.el_model,
                        voices=r_voices, status_cb=lambda m: _say("S3d", m))
                    if len(r_spans) == len(idxs):
                        r_audio = pl._AudioSegment.from_file(retry_path)
                        for i, t, (s, e) in zip(idxs, new_texts, r_spans):
                            changes.append((i, texts[i], t))
                            texts[i] = t
                            segs[i] = r_audio[int(s):int(e)]
                    else:
                        _say("S3d", "WARNING: re-voicing returned the wrong "
                                    "number of lines — keeping the originals.")
                except Exception as e:                       # noqa: BLE001
                    _say("S3d", f"WARNING: fit retry synthesis failed ({e}) — "
                                "keeping the original lines.")
            if changes:
                _write_text(base + "_ai_fit_changes.txt", "\n\n".join(
                    f"Piece {i + 1}\nAPPROVED: {old}\nSPOKEN:   {new}"
                    for i, old, new in changes) + "\n")
                _say("S3d", f"{len(changes)} line(s) shortened to fit — the "
                            "approved and spoken wording is listed in "
                            f"{os.path.basename(base)}_ai_fit_changes.txt.")

        fitted, fitted_spans, stretched = None, [], 0
        gap = pl._AudioSegment.silent(duration=pl.SECTION_GAP_MS,
                                      frame_rate=source.frame_rate)
        for i, sg in enumerate(segs):
            dur_ms = max(1.0, float(len(sg)))
            slot_ms = _slot(i) * 1000.0
            if dur_ms > slot_ms:
                sg = _stretch_segment(pl, sg, min(dur_ms / slot_ms, max_atempo),
                                      out_dir, i + 1)
                stretched += 1
            if fitted is None:
                fitted, start = sg, 0
            else:
                fitted = fitted + gap
                start = len(fitted)
                fitted = fitted + sg
            fitted_spans.append((start, len(fitted)))
        fitted.export(tts_path, format="wav")
        durations = [(e - s) / 1000.0 for (s, e) in fitted_spans]
        _say("S3d", f"{stretched} piece(s) time-stretched (ceiling "
                    f"{max_atempo:.2f}x).")

    placed = pl.place_pieces(pieces, durations, log=lambda m: _say("S3d", m))
    placed = _run_sync_check(pl, ctx, pieces, placed, durations)
    _write_sync_outputs(pl, args, manifest, ctx, tts_path, fitted_spans,
                        placed, texts)


def _el_settings():
    """engine_settings.json knobs for --script-source eleven (all optional):
    el_num_speakers (0 = auto), el_drop_background (1), el_disable_cloning
    (0; 1 asks ElevenLabs for a similar library voice instead of a clone —
    needs the workspace permission)."""
    return {
        "num_speakers": _engine_setting("el_num_speakers", 0, int, 0, 32),
        "drop_background": bool(_engine_setting("el_drop_background", 1, int,
                                                0, 1)),
        "disable_cloning": bool(_engine_setting("el_disable_cloning", 0, int,
                                                0, 1)),
    }


def _stage_translate_eleven(pl, args, api_key, manifest, ctx):
    """v0.22 S1a..S2c for --script-source eleven: ElevenLabs Dubbing Studio
    transcribes, splits speakers, translates and voices the English. We read
    its segments back as the review rows (one paragraph per segment) and
    pre-fill the cast with the speakers it detected, so the reviewer can edit
    the text and pick a voice per speaker before --steps dub renders.

    Fills ctx like _stage_translate: out_dir, base, raw_eng, final_srt,
    en_srt_text, en_audio_dur, punc_result, plus el_rows (review rows)."""
    audio_path = ctx["audio_path"]
    language = args.language
    out_dir, base = _prepare_out_dir(pl, audio_path, manifest)
    ctx["out_dir"], ctx["base"] = out_dir, base
    el_path = base + EL_DUB_SUFFIX
    for stale in (el_path, base + "_ai_draft.json", base + "_ai_report.txt",
                  base + AI_PHRASES_SUFFIX):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except Exception:
                pass
    iso = pl.el_lang_code(pl.TTS_LANGUAGES.get(language) or {})
    if not iso:
        raise RuntimeError(f"No language code for {language!r} — cannot ask "
                           "ElevenLabs Dubbing for it.")
    opts = _el_settings()

    _say("S1a", f"ElevenLabs Dub: sending the English audio to Dubbing "
                f"Studio (target '{iso}', speakers "
                f"{opts['num_speakers'] or 'auto'})…")
    dub_id, expected = pl.create_dub(
        audio_path, iso, api_key, num_speakers=opts["num_speakers"],
        drop_background=opts["drop_background"],
        disable_cloning=opts["disable_cloning"],
        status_cb=lambda m: _say("S1a", m))
    created = time.strftime("%Y-%m-%d %H:%M:%S")
    # Saved at once: the dub is paid for now, so a crash below must never
    # lose the id (the project can still be opened on elevenlabs.io).
    _write_text(el_path, json.dumps({
        "dubbing_id": dub_id, "language": language, "iso": iso,
        "created": created, "speakers": [], "segments": []},
        ensure_ascii=False, indent=2))
    _note(f"ElevenLabs dubbing_id: {dub_id}")

    _say("S1b", "ElevenLabs is transcribing, translating and voicing…")
    pl.wait_dub(dub_id, api_key, expected, status_cb=lambda m: _say("S1b", m))
    speakers, segs = pl.parse_resource(pl.get_resource(dub_id, api_key), iso)
    for n, sp in enumerate(speakers, 1):
        sp["key"] = f"s{n}"
    _write_text(el_path, json.dumps({
        "dubbing_id": dub_id, "language": language, "iso": iso,
        "created": created, "speakers": speakers, "segments": segs},
        ensure_ascii=False, indent=2))
    _say("S1b", f"ElevenLabs: {len(segs)} segment(s), {len(speakers)} "
                "speaker(s) detected.")

    # The English side, timed by ElevenLabs' segments: the review timing and
    # --steps dub's artifact check both read these.
    en_srt = _cues_to_srt(pl, [(s["start"], s["end"],
                                s["en"] or EMPTY_PARAGRAPH_PLACEHOLDER)
                               for s in segs])
    _write_text(base + ".srt", en_srt)
    _write_text(base + "_sync_en.srt", en_srt)
    manifest["en_srt"] = base + ".srt"
    ctx["final_srt"] = ctx["en_srt_text"] = en_srt
    ctx["raw_eng"] = " ".join(s["en"] for s in segs if s["en"])
    try:
        y_data, sr = pl._load_audio_any(audio_path)
        ctx["en_audio_dur"] = float(len(y_data)) / float(sr) if sr else 0.0
    except Exception:
        ctx["en_audio_dur"] = 0.0

    ctx["el_rows"] = [{"en": s["en"], "tr": s["tr"], "start": s["start"],
                       "end": s["end"]} for s in segs]
    punc = "\n\n".join(s["tr"] or EMPTY_PARAGRAPH_PLACEHOLDER for s in segs)
    _say("S2a", "Translation by ElevenLabs Dubbing — LLM chain skipped.")
    _say("S2b", "Review step skipped (ElevenLabs Dub).")
    _write_text(base + "_TranslationStep.txt", punc)
    _write_text(base + "_ReviewStep.txt", punc)
    _write_text(base + "_FinalScript.txt",
                f"=== ENGLISH TRANSCRIPTION ===\n{ctx['raw_eng']}\n\n"
                f"=== {language.upper()} TRANSLATION ===\n{punc}")
    ctx["punc_result"] = punc

    # Pre-fill the cast: one speaker per ElevenLabs speaker, each paragraph
    # (1-based = segment order) assigned to its segment's speaker. The panel's
    # review screen loads this file, so the reviewer only picks voices. A
    # single-speaker dub writes no file (and removes a stale one) — the same
    # convention the panel uses.
    cast_path = base + "_speakers.json"
    if len(speakers) > 1:
        dv = pl._sanitize_voice_id(args.voice_id) if args.voice_id else ""
        key_of = {sp["el_id"]: sp["key"] for sp in speakers}
        _write_text(cast_path, json.dumps({
            "version": 1, "language": language, "default_voice_id": dv,
            "speakers": [{"key": sp["key"], "name": sp["name"],
                          "voice_id": dv} for sp in speakers],
            "assignments": {str(n): {"speaker": key_of.get(s["speaker"],
                                                           "s1"),
                                     "voice_id": dv}
                            for n, s in enumerate(segs, 1)},
        }, ensure_ascii=False, indent=2))
        _say("S2c", f"Cast pre-filled with {len(speakers)} speaker(s) — pick "
                    "a voice for each on the review screen.")
    elif os.path.exists(cast_path):
        try:
            os.remove(cast_path)
        except Exception:
            pass
    _say("S2c", "FinalScript saved.")


def _stage_dub_eleven(pl, args, api_key, manifest, ctx, voice_id):
    """v0.22 S2d..S3e for --script-source eleven: push the reviewed text and
    the cast's voices into the Dubbing Studio project, re-voice every
    segment, render one full-length WAV and lay it out as pieces.

    The render is already aligned to the English timeline, so each segment
    becomes a piece cut from it at its own time and placed at that same time
    — the timestamps/texts/SRT files and manifest keys are match mode's, and
    the importers are unchanged."""
    language = args.language
    out_dir, base = ctx["out_dir"], ctx["base"]
    el_path = base + EL_DUB_SUFFIX
    try:
        el = json.loads(_read_text(el_path))
    except Exception:
        raise RuntimeError(f"No ElevenLabs Dub project for this audio "
                           f"({os.path.basename(el_path)}) — run the "
                           "translate step with ElevenLabs Dub first.")
    dub_id, iso = el.get("dubbing_id"), el.get("iso")
    segs = el.get("segments") or []
    if not (dub_id and iso and segs):
        raise RuntimeError(f"{os.path.basename(el_path)} is incomplete (no "
                           "dubbing_id or segments) — run translate again.")
    paras = pl._split_translation_paragraphs(ctx.get("script_text") or "")
    if len(paras) != len(segs):
        raise RuntimeError(
            f"The approved script has {len(paras)} paragraph(s) but the "
            f"ElevenLabs dub has {len(segs)} segment(s). In ElevenLabs Dub "
            "mode paragraphs can be edited but not added, removed or merged.")

    # ── [S2d] reviewed text + voices -> the studio project ─────────────────
    texts, edits = [], 0
    for seg, para in zip(segs, paras):
        t = " ".join(para.split())
        if not t or t == EMPTY_PARAGRAPH_PLACEHOLDER:
            texts.append(seg.get("tr") or "")
            continue
        if t != " ".join(str(seg.get("tr") or "").split()):
            pl.update_segment_text(dub_id, seg["id"], iso, t, api_key)
            seg["tr"] = t
            edits += 1
        texts.append(t)
    _say("S2d", f"{edits} edited segment(s) sent to ElevenLabs.")

    vmap, warns = pl.speaker_voices(segs, _cast_map(pl, base), voice_id)
    for w in warns:
        _say("S2d", "WARNING: " + w)
    names = {sp.get("el_id"): sp.get("name") for sp in el.get("speakers", [])}
    for spk, vid in vmap.items():
        pl.update_speaker_voice(dub_id, spk, vid, iso, api_key)
        _say("S2d", f"Voice for {names.get(spk) or spk}: {vid}")

    ids = [s["id"] for s in segs]
    _say("S2d", f"ElevenLabs re-voicing {len(ids)} segment(s) with the chosen "
                "voice(s)…")
    pl.redub(dub_id, ids, iso, api_key)
    pl.wait_redub(dub_id, ids, iso, api_key,
                  status_cb=lambda m: _say("S2d", m))

    _say("S2d", "ElevenLabs rendering the dub (WAV)…")
    render_id = pl.render(dub_id, iso, api_key, "wav")
    url = pl.wait_render(dub_id, render_id, iso, api_key,
                         status_cb=lambda m: _say("S2d", m))
    raw_path = base + "_el_render.bin"
    if url:
        pl.download(url, raw_path, api_key)
    else:
        _say("S2d", "Render has no download link — fetching the dubbed "
                    "audio instead.")
        pl.download_dubbed_audio(dub_id, iso, raw_path, api_key)
    tts_path = os.path.join(
        out_dir, pl._tts_output_name(language, ctx["audio_path"], "_tts"))
    tts_path = pl.ensure_writable_output(tts_path,
                                         status_cb=lambda m: _say("S2d", m))
    if pl.PYDUB_AVAILABLE:
        pl._AudioSegment.from_file(raw_path).export(tts_path, format="wav")
        try:
            os.remove(raw_path)
        except Exception:
            pass
    else:
        os.replace(raw_path, tts_path)
    manifest["tts_wav"] = tts_path
    _say("S2d", f"ElevenLabs dub saved: {os.path.basename(tts_path)}")
    try:
        _write_text(el_path, json.dumps(el, ensure_ascii=False, indent=2))
    except Exception:
        pass

    _say("S3a", "English segments taken from ElevenLabs — no "
                "re-transcription.")
    _say("S3b", "Piece offsets taken from the ElevenLabs render (aligned to "
                "the English timeline).")
    _say("S3c", "No matching prompt: ElevenLabs timed every segment.")
    _say("S3d", "Laying out one piece per segment…")
    spans = [(int(round(s["start"] * 1000)), int(round(s["end"] * 1000)))
             for s in segs]
    placed = [{"position": float(s["start"]), "status": "synced"}
              for s in segs]
    _write_sync_outputs(pl, args, manifest, ctx, tts_path, spans, placed,
                        texts)


def _merge_short_groups(groups, u_durs, unit_row, min_s):
    """Merge alignment groups whose speech is shorter than *min_s* into a
    neighbour of the SAME paragraph (never across paragraphs — the cast is
    per paragraph). The neighbour whose merged piece is shorter is chosen.
    Groups are (u_lo, u_hi, c_lo, c_hi); a merged group spans both unit and
    cue ranges (c = -1 sides are handled)."""
    g = [list(x) for x in groups]

    def dur(x):
        return sum(u_durs[x[0]:x[1] + 1])

    def join(a, b):
        cl = [c for c in (a[2], b[2]) if c >= 0]
        ch = [c for c in (a[3], b[3]) if c >= 0]
        return [a[0], b[1], min(cl) if cl else -1, max(ch) if ch else -1]

    changed = True
    while changed and len(g) > 1:
        changed = False
        for i, x in enumerate(g):
            if dur(x) >= min_s:
                continue
            row = unit_row[x[0]]
            cands = []
            if i > 0 and unit_row[g[i - 1][1]] == row:
                cands.append((dur(g[i - 1]) + dur(x), i - 1))
            if i + 1 < len(g) and unit_row[g[i + 1][0]] == row:
                cands.append((dur(g[i + 1]) + dur(x), i))
            if not cands:
                continue
            _d, left = min(cands)
            g[left:left + 2] = [join(g[left], g[left + 1])]
            changed = True
            break
    return [tuple(x) for x in g]


def _pause_cut(rms, lo, frame_ms, target, min_run=6, rel=0.12):
    """v0.23.5: where to cut in a window of frame loudness values.

    *rms* are consecutive frame_ms RMS values starting at *lo* (ms). A pause
    is a run of >= min_run frames (60 ms at 10 ms frames) quieter than rel x
    the window's speech level (90th percentile). The cut goes to the centre
    of the pause nearest *target*; a longer pause wins over a slightly nearer
    blip. Returns the cut in ms, or None when the window holds no real pause
    (the caller then falls back to the quietest single frame)."""
    if len(rms) < min_run:
        return None
    level = sorted(rms)[int(0.9 * (len(rms) - 1))]
    if level <= 0:
        return None
    thr = level * rel
    runs, start = [], None
    for k, v in enumerate(list(rms) + [thr + 1]):     # sentinel closes a run
        if v <= thr:
            if start is None:
                start = k
        elif start is not None:
            if k - start >= min_run:
                runs.append((start, k))
            start = None
    if not runs:
        return None

    def centre(r):
        return lo + ((r[0] + r[1]) * frame_ms) // 2

    # Distance to the timestamp boundary, discounted by pause length.
    best = min(runs, key=lambda r: abs(centre(r) - target)
               - 2 * (r[1] - r[0]) * frame_ms)
    return centre(best)


def _snap_cuts(pl, tts_path, spans, search_ms=250, frame_ms=10,
               apart_ms=1200, back_ms=300, ahead_ms=700):
    """v0.18.1: move each cut between neighbouring pieces to the quietest
    point near it, and make the neighbours share that cut.

    Piece spans come from ElevenLabs' character timings (first char start
    -> last char end). Where a cut falls INSIDE continuous speech — clause
    and mid-clause splits — those timings are often 100-300 ms off, and the
    audio between two spans belonged to neither: word onsets and endings
    were clipped ("words are missing"). The quietest 10 ms frame within
    +-search_ms of the boundary is the gap between two words; cutting there
    and letting both pieces meet at it loses nothing. Pieces already far
    apart (a request gap, real silence) are left alone.

    v0.23.5: the quietest single frame within +-250 ms was often a dip INSIDE
    a word — eleven_v3's character times run early by more than that — so a
    sentence's last word was split, its tail showing up as a stray blob at
    the start of the next chunk. Now the cut goes into a real pause (see
    _pause_cut) searched from back_ms before to ahead_ms after the boundary
    (biased later, where the true end usually is); the old quietest-frame
    rule is only the fallback when that window has no pause at all."""
    if not pl.PYDUB_AVAILABLE or len(spans) < 2:
        return spans, 0
    try:
        audio = pl._AudioSegment.from_file(tts_path)
    except Exception:
        return spans, 0
    out = [[int(s), int(e)] for (s, e) in spans]
    snapped = 0
    for i in range(len(out) - 1):
        e_i, s_n = out[i][1], out[i + 1][0]
        if s_n - e_i > apart_ms or s_n < out[i][0]:
            continue
        target = (e_i + s_n) // 2
        # A real pause first, in a wider window biased later.
        plo = max(out[i][0] + 40, min(e_i, s_n) - back_ms)
        phi = min(out[i + 1][1] - 40, max(e_i, s_n) + ahead_ms)
        cut = None
        if phi - plo >= 6 * frame_ms:
            frames = [audio[pos:pos + frame_ms].rms
                      for pos in range(plo, phi - frame_ms + 1, frame_ms)]
            cut = _pause_cut(frames, plo, frame_ms, target)
        if cut is None:
            lo = max(out[i][0] + 40, min(e_i, s_n) - search_ms)
            hi = min(out[i + 1][1] - 40, max(e_i, s_n) + search_ms)
            if hi - lo < frame_ms:
                cut = target
            else:
                best, cut = None, target
                for pos in range(lo, hi - frame_ms + 1, frame_ms):
                    rms = audio[pos:pos + frame_ms].rms
                    if best is None or rms < best:
                        best, cut = rms, pos + frame_ms // 2
        if cut != e_i or cut != s_n:
            snapped += 1
        out[i][1] = cut
        out[i + 1][0] = cut
    return [(s, e) for (s, e) in out], snapped


def _run_sync_check(pl, ctx, pieces, placed, durations):
    """v0.18 sync check agent: tighten late/early pieces toward their
    English start, rescue Un sync pieces that have room, and report the
    offsets (<base>_sync_check.txt + one log line). Never makes a
    placement worse; any error keeps the placement as it was.
    engine_settings.json: sync_check (1/0), sync_tolerance_ms (300)."""
    if not _engine_setting("sync_check", 1, int, 0, 1):
        return placed
    tol = _engine_setting("sync_tolerance_ms", 300.0, float, 50.0,
                          2000.0) / 1000.0
    try:
        fixed, rep = pl.sync_check(pieces, placed, durations,
                                   float(ctx.get("en_audio_dur") or 0.0), tol,
                                   log=lambda m: _say("S3d", m))
        _write_text(ctx["base"] + "_sync_check.txt",
                    pl.format_sync_check(rep))
        _say("S3d", f"Sync check: {rep['within_tol']}/{rep['synced']} on time "
                    f"(|offset| <= {rep['tol_ms']} ms), mean "
                    f"{rep['mean_offset_ms']} ms, max {rep['max_offset_ms']} ms"
                    f", end offset {rep.get('mean_end_offset_ms', 0)} ms"
                    f"; tightened {rep['moved']}, rescued {rep['rescued']}"
                    + (f"; {len(rep['drift'])} still off — see "
                       f"{os.path.basename(ctx['base'])}_sync_check.txt"
                       if rep['drift'] else "."))
        return fixed
    except Exception as e:                               # noqa: BLE001
        _say("S3d", f"WARNING: sync check skipped ({e}) — placement kept.")
        return placed


def _write_sync_outputs(pl, args, manifest, ctx, tts_path, spans, placed,
                        texts):
    """S3d files + S3e render for a placed piece list — the same artifacts
    and manifest keys match mode writes (timestamps with [synced]/[unsync],
    texts sidecar, synced SRT, synced-only wav)."""
    base, out_dir = ctx["base"], ctx["out_dir"]
    entries = []
    for i, (span, p) in enumerate(zip(spans, placed), 1):
        entries.append({
            "index":           i,
            "orig_start_ms":   int(span[0]),
            "orig_end_ms":     int(span[1]),
            "synced_start_ms": int(round(p["position"] * 1000)),
            "sync_status":     p["status"],
        })
    sync_ts_path = base + "_sync_timestamps.txt"
    _write_text(sync_ts_path, pl._format_timestamps_as_text(entries))
    manifest["timestamps_txt"] = sync_ts_path
    texts_path = base + "_sync_texts.txt"
    _write_text(texts_path, "\n\n".join(
        " ".join((t or "").split()) or EMPTY_PARAGRAPH_PLACEHOLDER
        for t in texts) + "\n")
    manifest["sync_texts"] = texts_path
    synced_entries = [e for e in entries if e["sync_status"] == "synced"]
    unsynced_n = len(entries) - len(synced_entries)
    manifest["synced_count"] = str(len(synced_entries))
    manifest["unsynced_count"] = str(unsynced_n)
    srt_lines = []
    for n, e in enumerate(
            sorted(synced_entries, key=lambda x: x["synced_start_ms"]), 1):
        start_s = e["synced_start_ms"] / 1000.0
        end_s = start_s + (e["orig_end_ms"] - e["orig_start_ms"]) / 1000.0
        text = " ".join((texts[e["index"] - 1] or "").split())
        srt_lines += [str(n), f"{pl._srt_ts(start_s)} --> {pl._srt_ts(end_s)}",
                      text, ""]
    synced_srt_path = base + "_sync_synced.srt"
    _write_text(synced_srt_path, "\n".join(srt_lines))
    manifest["synced_srt"] = synced_srt_path
    _say("S3d", f"{len(synced_entries)} synced / {unsynced_n} unsync — "
                "timestamps, texts and synced SRT saved.")

    if not synced_entries:
        _say("S3e", "WARNING: no synced pieces — the synced render is skipped "
                    "(all pieces go to the Un sync track).")
        return
    _say("S3e", "Rendering the synced audio…")
    synced_path = os.path.join(
        out_dir, pl._tts_output_name(args.language, ctx["audio_path"],
                                     "_synced"))
    synced_path = pl.ensure_writable_output(
        synced_path, status_cb=lambda m: _say("S3e", m))
    pl.sync_audio_with_timestamps(
        tts_path, synced_entries, synced_path,
        status_cb=lambda m: _say("S3e", m), extend_last=False)
    manifest["synced_wav"] = synced_path
    _say("S3e", f"Synced audio saved: {os.path.basename(synced_path)}")


def _stage_dub_legacy(pl, args, api_key, manifest, ctx, voice_id):
    """S2d..S3e: emotion + TTS + sync + render (v0.1-v0.6 behaviour).

    Consumes from *ctx*: out_dir, base, audio_path, en_audio_dur and the
    dub script text (script_text). The English sync SRT comes from the
    in-memory transcription (regions+words, full run) or from the
    _sync_en.srt file the translate stage persisted (dub resume) —
    ctx["en_srt_text"] set means "already on disk, reuse it".
    """
    language = args.language
    gemini_model = pl.GEMINI_DEFAULT_MODEL
    out_dir, base = ctx["out_dir"], ctx["base"]
    audio_path = ctx["audio_path"]
    script_text = ctx["script_text"]

    # ── [S2d] Step-4 emotion enrichment + TTS (ElevenLabs) ─────────────
    # v0.16: a saved cast is HONOURED here, not just reported. The script is
    # spoken in runs of consecutive paragraphs that share a voice, one
    # ElevenLabs request per run, concatenated into the one TTS wav the rest
    # of this stage expects — S3b re-transcribes that wav, so nothing
    # downstream needs to know how many voices made it.
    cast = _cast_map(pl, base)

    # Step-4 emotion enrichment before ElevenLabs TTS, exactly like the bulk
    # app's single-voice worker (ON by default there; the app only strips
    # the tags for its non-ElevenLabs Google TTS path, which this engine
    # does not have). Best-effort: on any LLM failure the original text
    # comes back unchanged.
    if _emotion_enabled(args):
        _say("S2d", f"Step-4 emotion enrichment on {gemini_model}…")
        # strict=True: ANY failure here stops the run. This is the last free
        # moment before the ElevenLabs spend, and every sync mode still needs
        # a mandatory LLM call afterwards.
        tts_text = pl._run_emotion_enrichment(
            script_text, language=language, model=gemini_model,
            status_cb=lambda m: _say("S2d", m), strict=True)
        if not (tts_text or "").strip():
            tts_text = script_text
    else:
        tts_text = script_text
        _say("S2d", "Emotion enrichment disabled (--no-emotion / settings) "
                    "— sending the bare punctuated text to TTS.")

    # Which paragraph is whose. Emotion enrichment rewrites the script, and
    # the cast is keyed by paragraph NUMBER — so if enrichment came back with
    # a different number of paragraphs the map can no longer be trusted, and
    # guessing which line moved is exactly how a talk gets the wrong voice.
    run_texts, run_voices = None, None
    if cast:
        p_out = _paragraphs(tts_text)
        p_in = _paragraphs(script_text)
        if len(p_out) != len(p_in):
            _say("S2d", "WARNING: emotion enrichment changed the paragraph "
                        f"count ({len(p_in)} → {len(p_out)}), so the saved "
                        "cast cannot be matched to the script — the whole "
                        f"script is dubbed with {voice_id}. Re-run with "
                        "--no-emotion to keep the cast.")
        else:
            voices = [cast.get(i + 1) or voice_id for i in range(len(p_out))]
            if len(set(voices)) > 1:
                runs = []
                for para_text, v in zip(p_out, voices):
                    if runs and runs[-1][1] == v:
                        runs[-1][0].append(para_text)
                    else:
                        runs.append(([para_text], v))
                run_texts = ["\n\n".join(t) for t, _ in runs]
                run_voices = [v for _, v in runs]
                _say("S2d", _cast_note(cast, voices, "paragraph(s)")
                            + f" Speaking them as {len(run_texts)} run(s).")

    tts_path = os.path.join(
        out_dir, pl._tts_output_name(language, audio_path, "_tts"))
    if run_texts:
        _say("S2d", f"Synthesizing {language} speech in "
                    f"{len(set(run_voices))} voices (model "
                    f"{args.el_model})…")
        tts_path, _spans = pl.synthesize_sections_elevenlabs(
            run_texts, tts_path, api_key=api_key, voice_id=voice_id,
            model_id=args.el_model, voices=run_voices,
            status_cb=lambda m: _say("S2d", m))
    else:
        _say("S2d", f"Synthesizing {language} speech (voice {voice_id}, "
                    f"model {args.el_model})…")
        # synthesize_tts_elevenlabs may divert to a "-2" name when the
        # previous wav is still locked (open REAPER project) — use the
        # returned path.
        tts_path = pl.synthesize_tts_elevenlabs(
            tts_text, tts_path, api_key=api_key, voice_id=voice_id,
            model_id=args.el_model,
            status_cb=lambda m: _say("S2d", m))
    manifest["tts_wav"] = tts_path
    _say("S2d", f"TTS audio saved: {os.path.basename(tts_path)}")

    # ── [S3a] English sync SRT ──────────────────────────────────────────────
    en_sync_path = base + "_sync_en.srt"
    if ctx.get("en_srt_text"):
        # Dub resume: the translate stage already built and persisted this
        # SRT from its S1a transcription — no second Scribe call needed.
        en_srt = ctx["en_srt_text"]
        _say("S3a", "Using the English sync SRT from the translate stage.")
    else:
        _say("S3a", "Building English sync SRT…")
        en_srt = pl._build_english_subtitle_srt(ctx["regions"], ctx["words"])
        _write_text(en_sync_path, en_srt)
        _say("S3a", "English sync SRT saved.")

    # ── [S3b] Target-language SRT from the TTS audio ───────────────────────
    _say("S3b", "Loading TTS audio and detecting regions…")
    # tts_path is a WAV this pipeline just wrote (tts.py exports format="wav"),
    # so the shared loader handles it — same mono float32 at the native rate
    # that librosa.load(sr=None, mono=True) returned here before.
    te_y, te_sr = pl._load_audio_any(tts_path)
    te_regions = pl._detect_regions_from_audio(
        te_y, te_sr, pl.DEFAULT_BN_THR_DB, pl.DEFAULT_BN_HYS_DB,
        pl.DEFAULT_BN_MIN_MS)
    if not te_regions:
        raise RuntimeError("No regions detected in the TTS audio.")
    _say("S3b", f"Transcribing TTS audio ({len(te_regions)} regions)…")
    te_result = pl._transcribe_audio(tts_path, api_key)
    te_words = te_result.get("words", [])
    if not te_words:
        raise RuntimeError("No word data from ElevenLabs for the TTS audio.")
    te_srt = pl._build_target_subtitle_srt(te_regions, te_words)
    _write_text(base + "_sync_te.srt", te_srt)
    _say("S3b", f"{language} sync SRT saved.")

    # ── [S3c] LLM subtitle mapping ──────────────────────────────────────────
    _say("S3c", "Calling the LLM for EN <-> target subtitle mapping…")
    mapping_text = pl._call_gemini_mapping(
        en_srt, te_srt, script_text, gemini_model, language=language)
    _write_text(base + "_sync_mapping.txt", mapping_text)
    _say("S3c", "Mapping received and saved.")

    # ── [S3d] Sync algorithm ────────────────────────────────────────────────
    _say("S3d", "Running the sync algorithm…")
    synced_subs, orig_te_subs, sync_log = pl.run_sync_from_strings(
        en_srt, te_srt, mapping_text,
        en_audio_duration=ctx.get("en_audio_dur", 0.0))
    _write_text(base + "_sync_log.txt", sync_log)
    n_bleed = sync_log.count("[bleed-over]")
    if n_bleed:
        _say("S3d", f"{len(synced_subs)} subtitles synced — {n_bleed} long "
                    "section(s) anchored to their English start (see sync "
                    "log).")
    else:
        _say("S3d", f"{len(synced_subs)} subtitles synced.")

    synced_srt_text = pl._write_srt_from_dict(synced_subs)
    synced_srt_path = base + "_sync_synced.srt"
    _write_text(synced_srt_path, synced_srt_text)
    manifest["synced_srt"] = synced_srt_path

    ts_list = pl._build_timestamps(orig_te_subs, synced_subs)
    sync_ts_path = base + "_sync_timestamps.txt"
    _write_text(sync_ts_path, pl._format_timestamps_as_text(ts_list))
    manifest["timestamps_txt"] = sync_ts_path
    _say("S3d", "Synced SRT + timestamps saved.")

    # ── [S3e] Render the synced audio ───────────────────────────────────────
    _say("S3e", "Rendering the synced audio…")
    synced_path = os.path.join(
        out_dir, pl._tts_output_name(language, audio_path, "_synced"))
    # Same lock hazard as the TTS wav: a previous _synced.wav imported into
    # an open REAPER project holds a share lock — divert instead of Errno 13.
    synced_path = pl.ensure_writable_output(
        synced_path, status_cb=lambda m: _say("S3e", m))
    pl.sync_audio_with_timestamps(
        tts_path, ts_list, synced_path,
        status_cb=lambda m: _say("S3e", m))
    manifest["synced_wav"] = synced_path
    _say("S3e", f"Synced audio saved: {os.path.basename(synced_path)}")


_FFMPEG_MSG = ("ffmpeg not found — the engine needs it to decode TTS audio "
               "(pydub shells out to it). Re-run the setup script "
               "(setup_windows.bat / setup_mac.command); it installs ffmpeg "
               "automatically. Then start the run again.")


def _require_ffmpeg(pl, hard):
    """Fail (or warn) BEFORE any LLM/TTS spend when ffmpeg is missing.

    Without this, a run burns transcription + translation + TTS credits and
    then dies at the audio-save step with a bare WinError 2 (real report
    from a Windows install where the setup's ffmpeg step was skipped).
    """
    if getattr(pl, "FFMPEG_PATH", None):
        return
    if hard:
        raise RuntimeError(_FFMPEG_MSG)
    _note("WARNING: " + _FFMPEG_MSG + " (The dub stage will need it; "
          "translate-only runs on WAV input can proceed.)")


def _required_prompts(args):
    """The prompt files THIS run will load — not all five.

    Which ones depend on the run: a --provided-script run never enters the
    translation chain, and match mode neither enriches emotion nor maps
    subtitles. Listing more than the run needs would block a language whose
    missing prompt does not matter here.
    """
    need = []
    # AI mode (v0.18.3) uses NO prompt files at all: the translator, critics,
    # proofer and fit agents build their prompts inline, and its sync is
    # anchor (or match, whose matcher prompt is inline) — never legacy.
    # ElevenLabs Dub (v0.22) hands the whole job to ElevenLabs: no prompts.
    if getattr(args, "script_source", "prompt") in ("ai", "ai_test",
                                                    "eleven"):
        return need
    if args.steps in ("full", "translate") and not args.provided_script:
        need += ["Step1_Translation_Prompt", "Step2_Review_Prompt",
                 "Step3_Punctuation_Prompt"]
    if args.steps in ("full", "dub") and _sync_mode(args) != "match":
        # match mode builds its matcher prompt inline and skips Step 4.
        if _emotion_enabled(args):
            need.append("Step4_Emotion_Prompt")
        need.append("SyncingPrompt")
    return need


def _preflight_prompts(pl, args):
    """Read every prompt file this run needs BEFORE anything bills.

    A missing or empty prompt used to surface only when the stage that loads
    it ran — and in a legacy run both Step 4 and SyncingPrompt sit AFTER the
    ElevenLabs spend, so adding a language without its SyncingPrompt meant
    paying for a TTS synthesis and a Scribe pass before finding out. Reading
    them up front costs a few filesystem stats.
    """
    needed = _required_prompts(args)
    missing = []
    for stage in needed:
        fname = f"{stage}_{args.language}.txt"
        try:
            text = pl._load_lang_prompt(stage, args.language)
        except Exception as e:
            missing.append(f"{fname} — {e.__class__.__name__}")
            continue
        if not (text or "").strip():
            missing.append(f"{fname} — the file is empty")
    if missing:
        raise RuntimeError(
            f"Prompt file(s) missing or empty for {args.language}, stopping "
            "before any paid transcription or speech synthesis:\n  "
            + "\n  ".join(missing)
            + "\nAdd them under dubbing/prompts/ (adapt the _Bengali.txt "
              "copies), then run again.")
    if needed:
        _note(f"Prompts present: {len(needed)} file(s) for {args.language}.")


def _preflight_llm(pl):
    """One tiny LLM call BEFORE any paid API work. Raises if it fails.

    Every --steps full / dub run needs the LLM for something it cannot skip:
    the section-match call in 'match' mode, the EN<->target mapping in
    'legacy'. Both sit AFTER Scribe transcription and AFTER ElevenLabs
    speech synthesis, so an unreachable endpoint used to be discovered only
    once the expensive half had already been paid for a run that could never
    finish. On 2026-08-24 a Kannada run logged the gateway as unreachable at
    S2d, carried on to spend a full TTS synthesis plus an 11.3 MB Scribe
    pass, and only then died at S3c on the same endpoint.

    Same probe as --test-llm, same failure surface. Cost is one sub-token
    reply, which is why it can run unconditionally.

    Deliberately placed alongside the voice resolution: that already fails
    fast "before any expensive transcription/translation work happens", and
    the LLM simply never got the same treatment.
    """
    provider, model = pl._active_provider_and_model()
    _note(f"Checking the LLM is reachable ({provider}, {model})…")
    reply = pl._llm_generate(
        "Reply with the single word OK and nothing else.", model)
    if not (reply or "").strip():
        raise RuntimeError(
            f"The LLM at {provider} ({model}) accepted the connection but "
            "returned an empty reply. Stopping before any paid "
            "transcription or speech synthesis — fix the LLM provider in "
            "the panel's Settings tab, then run again.")
    _note("LLM reachable.")


def _begin_run(args, manifest, need_llm=True):
    """Common head of full/translate/dub: audio checks + pipeline import +
    keys.

    *need_llm* is False for the v0.13 pause-aware modes, which never call a
    language model — requiring a working LLM provider there would block a
    free preview on a machine that only has an ElevenLabs key, which is
    exactly the setup this feature is meant to serve.

    Returns (pl, api_key, ctx) with ctx pre-seeded with the absolute
    audio path.
    """
    audio_path = os.path.abspath(os.path.expanduser(args.audio))
    manifest["audio"] = audio_path
    manifest["language"] = args.language
    if not os.path.isfile(audio_path):
        raise RuntimeError(f"Audio file not found: {audio_path}")
    # v0.22 ElevenLabs Dub: ElevenLabs translates — no LLM is ever called.
    if getattr(args, "script_source", "prompt") == "eleven":
        need_llm = False
    pl, api_key = _load_pipeline_and_keys(args, need_llm=need_llm)
    # _llm_provider_label(), not GEMINI_DEFAULT_MODEL: the constant is the
    # Gemini default and says nothing about the provider this install actually
    # calls, so a gateway run used to advertise "gemini-2.5-pro" in its log.
    _note(f"Pipeline loaded. LLM: {pl._llm_provider_label()}; "
          f"TTS model: {args.el_model}.")
    _roles = getattr(pl, "_llm_role_overrides_label", None)
    if callable(_roles) and _roles():
        _note(f"Per-stage model overrides: {_roles()}")
    _require_ffmpeg(pl, hard=(args.steps in ("full", "dub", "dubplan")))
    # v0.15.1: prove the LLM answers before anything bills. full/dub always
    # need it (see _preflight_llm); a translate run handed --provided-script
    # skips the whole LLM chain, so it must not be blocked by this. The v0.13
    # pause-aware modes pass need_llm=False and never call a model at all —
    # requiring one there would block a free preview.
    if need_llm and (args.steps in ("full", "dub") or not args.provided_script):
        _preflight_prompts(pl, args)
        _preflight_llm(pl)
    return pl, api_key, {"audio_path": audio_path}


def _run_full(args, manifest):
    """--steps full: v0.1 behaviour, unchanged — S1a..S3e in one process."""
    pl, api_key, ctx = _begin_run(args, manifest)
    # Resolve the dub voice up-front so a bad voice fails before any
    # expensive transcription/translation work happens.
    voice_id, voice_how = _resolve_voice(pl, api_key, args.language,
                                         args.voice_id)
    _note(f"Dub voice: {voice_id} ({voice_how})")
    if getattr(args, "script_source", "prompt") == "eleven":
        _stage_translate_eleven(pl, args, api_key, manifest, ctx)
    else:
        _stage_translate(pl, args, api_key, manifest, ctx)
    ctx["script_text"] = ctx["punc_result"]
    _stage_dub(pl, args, api_key, manifest, ctx, voice_id)


def _stage_pause_plan(pl, args, api_key, manifest, ctx):
    """S1a/S1b for the pause-aware modes: transcribe, detect, chunk, assign.

    Shared by --steps plan and --steps dubplan. dubplan re-runs this rather
    than trusting the plan file's numbers: the timings must come from the
    audio every time, so a hand-edited (or stale) timestamp in the plan can
    never desync a paid run. Only the TR: text is taken from the file.

    Fills *ctx* with out_dir, base, chunks, en_texts, total_dur_s and the
    resolved settings tuple.
    """
    audio_path = ctx["audio_path"]
    pause_min_s, thr_db, max_atempo, rate_override = _plan_settings(pl)

    out_dir, base = _prepare_out_dir(pl, audio_path, manifest)
    ctx["out_dir"], ctx["base"] = out_dir, base

    # ── [S1a] Transcription — disk-cached, so Reload is free ────────────────
    _say("S1a", "Transcribing source audio (ElevenLabs Scribe)…")
    result = pl._transcribe_audio(audio_path, api_key)
    words = result.get("words", [])
    if not words:
        raise RuntimeError("No word data from ElevenLabs for the source "
                           "audio.")
    _say("S1a", f"Transcription done ({len(words)} word tokens).")

    # ── [S1b] Pause detection -> the chunk grid ─────────────────────────────
    _say("S1b", f"Detecting pauses (floor {thr_db:.0f} dB, minimum gap "
                f"{pause_min_s * 1000:.0f} ms)…")
    y_data, sr = pl._load_audio_any(audio_path)
    total_dur_s = float(len(y_data)) / float(sr) if sr else 0.0
    regions = pl._detect_regions_from_audio(
        y_data, sr, thr_db, pl.DEFAULT_HYS_DB, pl.DEFAULT_MIN_MS)
    if not regions:
        raise RuntimeError("No speech detected in the source audio (is the "
                           "file silent, or is the level below the "
                           f"{thr_db:.0f} dB floor?).")
    chunks = pl.pause_chunks_from_regions(regions, total_dur_s, pause_min_s)
    if not chunks:
        raise RuntimeError("Pause detection produced no chunks.")

    # The English SRT is written for the same reason the other modes write
    # it: it is the human-readable record of what the source said where.
    final_srt = pl._build_subtitle_srt(regions, words)
    srt_path = base + ".srt"
    _write_text(srt_path, final_srt)
    manifest["en_srt"] = srt_path

    en_texts = pl.source_text_for_chunks(chunks, words)
    _longest = max(c["dur_s"] for c in chunks)
    _pauses = sorted(c["pause_after_s"] for c in chunks[:-1])
    _line = (f"{len(chunks)} pause-delimited chunk(s) from {len(regions)} "
             f"region(s) — source {total_dur_s:.1f}s, longest chunk "
             f"{_longest:.1f}s")
    if _pauses:
        _line += f", median pause {_pauses[len(_pauses) // 2]:.2f}s"
    _say("S1b", _line)

    ctx["chunks"] = chunks
    ctx["en_texts"] = en_texts
    ctx["total_dur_s"] = total_dur_s
    ctx["plan_settings"] = (pause_min_s, thr_db, max_atempo, rate_override)
    return chunks


def _run_plan(args, manifest):
    """--steps plan: the free dry run.

    Detect the source's pauses, lay the target script across them, estimate
    each chunk's spoken duration from character count and the per-language
    rate, and write two artifacts: an editable plan file and a self-contained
    HTML review page. No TTS request, no LLM call, no credits.

    Two text sources, and which one is used decides whether a review survives:

      --provided-script  the flowing pasted script, spread across the chunks
                         by duration share. The FIRST look at a run.
      --plan             a plan file whose TR: lines are already assigned per
                         chunk. Re-measures exactly those lines against
                         freshly detected pauses, so corrections made in the
                         review page (or by hand) are preserved. Reload used
                         to re-spread the original script here, which threw
                         every correction away.
    """
    pl, api_key, ctx = _begin_run(args, manifest, need_llm=False)
    _stage_pause_plan(pl, args, api_key, manifest, ctx)
    base = ctx["base"]
    chunks, en_texts = ctx["chunks"], ctx["en_texts"]
    _pause_min, _thr, max_atempo, rate_override = ctx["plan_settings"]

    if args.plan:
        # Re-measure pass. Timings still come from the audio — only the TR:
        # text is taken from the file, exactly as --steps dubplan does it.
        plan_path = os.path.abspath(os.path.expanduser(args.plan))
        if not os.path.isfile(plan_path):
            raise RuntimeError(f"--plan file not found: {plan_path}")
        corrected = pl.parse_plan_text(_read_text(plan_path))
        if not corrected or not any(t.strip() for t in corrected):
            raise RuntimeError(
                f"No TR: lines found in the plan file: {plan_path} — it must "
                "be a plan written by '--steps plan' (target text on the TR: "
                "lines).")
        if len(corrected) != len(chunks):
            _note(f"WARNING: the plan has {len(corrected)} chunk(s) but the "
                  f"audio now yields {len(chunks)} — pairing by index and "
                  "padding/truncating. Re-run the preview from the pasted "
                  "script if the audio changed.")
        tr_texts = [(corrected[i] if i < len(corrected) else "")
                    for i in range(len(chunks))]
        _say("S1b", f"Re-measuring {len(chunks)} chunk(s) against the "
                    "corrected plan…")
        _note(f"Target text loaded per chunk from: {plan_path}")
    else:
        script_path = os.path.abspath(os.path.expanduser(args.provided_script))
        if not os.path.isfile(script_path):
            raise RuntimeError(
                f"--provided-script file not found: {script_path}")
        script_text = _read_text(script_path).strip()
        if not script_text:
            raise RuntimeError(f"--provided-script file is empty: "
                               f"{script_path}")
        _note(f"Target script loaded: {script_path} ({len(script_text)} "
              "chars).")
        _say("S1b", f"Spreading the script across {len(chunks)} chunk(s) by "
                    "duration share…")
        tr_texts = pl.assign_script_to_chunks(script_text, chunks,
                                              args.language)

    plan = pl.build_plan(chunks, en_texts, tr_texts, args.language,
                         max_atempo, rate_override)

    plan_txt = base + "_sync_plan.txt"
    _write_text(plan_txt, pl.format_plan_text(
        plan, ctx["audio_path"], args.language, ctx["total_dur_s"],
        max_atempo, rate_override))
    manifest["plan_txt"] = plan_txt

    plan_html = base + "_sync_plan.html"
    try:
        pl.render_plan_html(plan, plan_html, ctx["audio_path"], args.language,
                            ctx["total_dur_s"], max_atempo, rate_override)
        manifest["plan_html"] = plan_html
    except Exception as e:
        # The HTML is the readable, editable half; the plan file is the
        # contract. Losing the review page must not lose the analysis.
        _note(f"WARNING: could not write the HTML review page ({e}) — the "
              "plan file is still valid.")

    counts = pl.plan_counts(plan)
    manifest["chunk_count"] = str(len(plan))
    for key in ("fits", "tight", "over", "short", "empty"):
        manifest[f"{key}_count"] = str(counts.get(key, 0))

    _say("S1b", pl.summarize_plan(plan))
    _note("Plan written — NO audio was generated and no credits were spent. "
          f"Correct the target text in {os.path.basename(plan_html)} (the "
          "only surface that renders Indic script properly), paste the "
          "corrections back in the panel and reload to re-check — or approve "
          "to generate.")


def _run_dubplan(args, manifest):
    """--steps dubplan --plan <file>: generate from an approved plan.

    The whole matching/placement stack is bypassed. Every chunk already has
    a home — the timestamp where the source speaker started — so this only
    has to synthesize, fit each chunk to its slot, and lay the pieces down
    at their original starts. Nothing can be demoted to Un sync here,
    because nothing has to compete for a position.
    """
    pl, api_key, ctx = _begin_run(args, manifest, need_llm=False)

    # Read and validate the plan BEFORE resolving the voice or transcribing:
    # a plan the user pointed at by mistake is a local file check, and it
    # must not cost an API round trip to discover.
    plan_path = os.path.abspath(os.path.expanduser(args.plan))
    if not os.path.isfile(plan_path):
        raise RuntimeError(f"--plan file not found: {plan_path}")
    approved = pl.parse_plan_text(_read_text(plan_path))
    if not approved or not any(t.strip() for t in approved):
        raise RuntimeError(
            f"No TR: lines found in the plan file: {plan_path} — it must be "
            "a plan written by '--steps plan' (target text on the TR: "
            "lines).")

    voice_id, voice_how = _resolve_voice(pl, api_key, args.language,
                                         args.voice_id)
    _note(f"Dub voice: {voice_id} ({voice_how})")

    _stage_pause_plan(pl, args, api_key, manifest, ctx)
    out_dir, base = ctx["out_dir"], ctx["base"]
    chunks, en_texts = ctx["chunks"], ctx["en_texts"]
    _pause_min, _thr, max_atempo, rate_override = ctx["plan_settings"]

    if len(approved) != len(chunks):
        _note(f"WARNING: the plan has {len(approved)} chunk(s) but the audio "
              f"now yields {len(chunks)} — pairing by index and padding/"
              "truncating. Re-run '--steps plan' if the audio changed.")
    tr_texts = [(approved[i] if i < len(approved) else "")
                for i in range(len(chunks))]
    plan = pl.build_plan(chunks, en_texts, tr_texts, args.language,
                         max_atempo, rate_override)
    _say("S2d", "Approved plan: " + pl.summarize_plan(plan))

    live = [(i, row) for i, row in enumerate(plan) if row["tr"].strip()]
    if not live:
        raise RuntimeError("Every chunk in the plan has empty TR: text — "
                           "nothing to synthesize.")

    # ── [S2d] One stitched synthesis pass over the chunks that have text ────
    _say("S2d", f"Synthesizing {len(live)} chunk(s) ({args.language}, voice "
                f"{voice_id}, model {args.el_model})…")
    tts_path = os.path.join(
        out_dir, pl._tts_output_name(args.language, ctx["audio_path"], "_tts"))
    tts_path = pl.ensure_writable_output(tts_path,
                                         status_cb=lambda m: _say("S2d", m))
    tts_path, spans = pl.synthesize_sections_elevenlabs(
        [row["tr"] for (_i, row) in live], tts_path, api_key=api_key,
        voice_id=voice_id, model_id=args.el_model,
        status_cb=lambda m: _say("S2d", m))
    if len(spans) != len(live):
        raise RuntimeError(
            f"Synthesis returned {len(spans)} span(s) for {len(live)} chunk"
            "(s) — refusing to place audio that cannot be matched to the "
            "plan.")
    _say("S2d", f"TTS audio saved: {os.path.basename(tts_path)}.")

    _say("S3a", "Chunk grid came from the source pauses — no matching pass.")
    _say("S3b", "Chunk offsets taken from synthesis — no re-transcription.")
    _say("S3c", "No EN <-> script mapping needed: each chunk owns its own "
                "source timestamp.")

    # ── [S3d] Fit each chunk to its slot, then lay them down ────────────────
    _say("S3d", "Fitting chunks to their slots…")
    entries, texts, stretched, still_over = [], [], 0, []
    use_pydub = bool(pl.PYDUB_AVAILABLE)
    fitted, source_audio, gap = None, None, None

    for n, ((ci, row), (s_ms, e_ms)) in enumerate(zip(live, spans), 1):
        chunk = chunks[ci]
        hard_slot_ms = (chunk["dur_s"] + chunk["pause_after_s"]) * 1000.0
        measured_ms = max(1.0, e_ms - s_ms)
        ratio = 1.0
        if hard_slot_ms > 0 and measured_ms > hard_slot_ms:
            ratio = min(measured_ms / hard_slot_ms, max_atempo)
            if measured_ms / hard_slot_ms > max_atempo + 0.005:
                still_over.append((chunk["index"],
                                   (measured_ms - hard_slot_ms) / 1000.0))

        if use_pydub:
            if source_audio is None:
                source_audio = pl._AudioSegment.from_file(tts_path)
                gap = pl._AudioSegment.silent(
                    duration=pl.SECTION_GAP_MS,
                    frame_rate=source_audio.frame_rate)
            seg = source_audio[int(s_ms):int(e_ms)]
            if ratio > 1.005:
                seg = _stretch_segment(pl, seg, ratio, out_dir, n)
                stretched += 1
            if fitted is None:
                fitted, start_ms = seg, 0
            else:
                # Same SECTION_GAP_MS cushion synthesize_sections_elevenlabs
                # leaves between sections, and excluded from the spans for the
                # same reason: an item nudged a few ms in REAPER must not pull
                # in its neighbour's audio.
                fitted = fitted + gap
                start_ms = len(fitted)
                fitted = fitted + seg
            end_ms = len(fitted)
        else:
            # No pydub: fall back to the unstretched spans in the raw wav.
            start_ms, end_ms = int(s_ms), int(e_ms)

        entries.append({
            "index":           n,
            "orig_start_ms":   int(start_ms),
            "orig_end_ms":     int(end_ms),
            "synced_start_ms": int(round(chunk["start_s"] * 1000)),
            "sync_status":     "synced",
        })
        texts.append(row["tr"])

    if fitted is not None:
        fitted.export(tts_path, format="wav")
    manifest["tts_wav"] = tts_path
    _say("S3d", f"{len(entries)} chunk(s) placed at their source "
                f"timestamps; {stretched} time-stretched to fit.")
    if still_over:
        _say("S3d", "WARNING: "
             + f"{len(still_over)} chunk(s) still overrun their slot after "
               f"the {max_atempo:.2f}x ceiling and were left long rather "
               "than squashed — "
             + ", ".join(f"#{i} by {o:.1f}s" for i, o in still_over[:8])
             + ("…" if len(still_over) > 8 else "")
             + ". Shorten their TR: lines and re-run the plan to fix.")

    # ── Standard artifacts — unchanged contract, so the importers just work ─
    sync_ts_path = base + "_sync_timestamps.txt"
    _write_text(sync_ts_path, pl._format_timestamps_as_text(entries))
    manifest["timestamps_txt"] = sync_ts_path

    texts_path = base + "_sync_texts.txt"
    _write_text(texts_path, "\n\n".join(
        " ".join((t or "").split()) or EMPTY_PARAGRAPH_PLACEHOLDER
        for t in texts) + "\n")
    manifest["sync_texts"] = texts_path

    manifest["synced_count"] = str(len(entries))
    manifest["unsynced_count"] = "0"

    srt_lines = []
    for n, e in enumerate(sorted(entries,
                                 key=lambda x: x["synced_start_ms"]), 1):
        start_s = e["synced_start_ms"] / 1000.0
        end_s = start_s + (e["orig_end_ms"] - e["orig_start_ms"]) / 1000.0
        srt_lines += [str(n),
                      f"{pl._srt_ts(start_s)} --> {pl._srt_ts(end_s)}",
                      " ".join((texts[e["index"] - 1] or "").split()), ""]
    synced_srt_path = base + "_sync_synced.srt"
    _write_text(synced_srt_path, "\n".join(srt_lines))
    manifest["synced_srt"] = synced_srt_path

    # ── [S3e] Render ────────────────────────────────────────────────────────
    _say("S3e", "Rendering the synced audio…")
    synced_path = os.path.join(
        out_dir, pl._tts_output_name(args.language, ctx["audio_path"],
                                     "_synced"))
    synced_path = pl.ensure_writable_output(
        synced_path, status_cb=lambda m: _say("S3e", m))
    pl.sync_audio_with_timestamps(
        tts_path, entries, synced_path,
        status_cb=lambda m: _say("S3e", m), extend_last=False)
    manifest["synced_wav"] = synced_path
    _say("S3e", f"Synced audio saved: {os.path.basename(synced_path)}")


def _stretch_segment(pl, seg, ratio, out_dir, n):
    """Time-stretch one pydub segment via ffmpeg; return the new segment.

    Round-trips through a scratch wav because atempo is an ffmpeg filter,
    not a pydub operation. Any failure returns the segment unchanged —
    stretch_wav_atempo logs why, and an overlong chunk is a far better
    outcome than a run that dies after the credits are spent.
    """
    scratch = os.path.join(out_dir, "_fit")
    try:
        os.makedirs(scratch, exist_ok=True)
        raw = os.path.join(scratch, f"chunk_{n:04d}.wav")
        fit = os.path.join(scratch, f"chunk_{n:04d}_fit.wav")
        seg.export(raw, format="wav")
        got = pl.stretch_wav_atempo(raw, fit, ratio,
                                    status_cb=lambda m: _say("S3d", m))
        if got and os.path.isfile(got) and got != raw:
            return pl._AudioSegment.from_file(got)
    except Exception as e:
        _say("S3d", f"WARNING: could not time-stretch chunk {n} ({e}) — "
                    "left at its synthesized length.")
    return seg


def _paired_paragraph_texts(pl, final_srt, punc_result, ai_rows=None):
    """Build the aligned EN/translation paragraph texts for review.

    Uses the pipeline's review pairing (_pair_review_rows): English SRT
    segments are grouped onto translation paragraphs proportionally by
    character share, so both sides come back with the SAME row count in the
    same order. Each row becomes one paragraph; paragraphs are joined with
    exactly one blank line. A row with no text on one side gets a visible
    placeholder so the paragraph counts stay equal after a blank-line
    re-split. Internal blank lines inside a paragraph are collapsed for the
    same reason. The translation side preserves every translation paragraph
    verbatim and in order, so joining it back reproduces the dub script.

    *ai_rows* (v0.18 AI mode) already pair every paragraph with its exact
    English cues, so they are used as-is instead of the char-share guess.
    """
    if ai_rows:
        rows = [(str(r.get("en") or ""), str(r.get("tr") or ""),
                 r.get("start"), r.get("end")) for r in ai_rows]
    else:
        en_entries = pl._extract_srt_entries(final_srt)
        tr_paras = pl._split_translation_paragraphs(punc_result)
        rows = pl._pair_review_rows(en_entries, tr_paras)
    if not rows:
        raise RuntimeError("Could not build review rows — the translation "
                           "result is empty.")

    def _block(text):
        text = (text or "").strip()
        if not text:
            return EMPTY_PARAGRAPH_PLACEHOLDER
        # One paragraph = one blank-line-free block.
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines) or EMPTY_PARAGRAPH_PLACEHOLDER

    en_text = "\n\n".join(_block(en) for (en, _tr, _s0, _s1) in rows) + "\n"
    tr_text = "\n\n".join(_block(tr) for (_en, tr, _s0, _s1) in rows) + "\n"
    return en_text, tr_text, len(rows)


def _run_translate(args, manifest):
    """--steps translate: S1a..S2c, then stop for review.

    Extra outputs beyond the app-convention files:
      <base>_sync_en.srt          persisted here (instead of at S3a) so the
                                  dub resume never re-transcribes the audio
      <base>_review_en.txt        EN paragraphs (one per review row)
      <base>_review_translation.txt  translation paragraphs, SAME count/order
    """
    pl, api_key, ctx = _begin_run(args, manifest)
    eleven = getattr(args, "script_source", "prompt") == "eleven"
    if eleven:
        # v0.22: ElevenLabs transcribed + translated; its segments are the
        # review rows and it already wrote <base>_sync_en.srt.
        _stage_translate_eleven(pl, args, api_key, manifest, ctx)
    else:
        _stage_translate(pl, args, api_key, manifest, ctx)
    base = ctx["base"]

    # Persist the sync-quality English SRT now (the pipeline builds it at
    # S3a from the same S1a transcription) — this is what lets --steps dub
    # resume without a second paid Scribe call.
    if not eleven:
        en_sync_srt = pl._build_english_subtitle_srt(ctx["regions"],
                                                     ctx["words"])
        _write_text(base + "_sync_en.srt", en_sync_srt)

    # Delete any stale edited translation script from a previous run
    edited_path = base + "_translation_edited.txt"
    if os.path.exists(edited_path):
        try:
            os.remove(edited_path)
        except Exception:
            pass

    en_text, tr_text, n_rows = _paired_paragraph_texts(
        pl, ctx["final_srt"], ctx["punc_result"],
        ctx.get("el_rows") or ctx.get("ai_rows"))
    en_text_path = base + "_review_en.txt"
    tr_text_path = base + "_review_translation.txt"
    _write_text(en_text_path, en_text)
    _write_text(tr_text_path, tr_text)
    manifest["en_text"] = en_text_path
    manifest["translation_text"] = tr_text_path
    manifest["final_script"] = base + "_FinalScript.txt"
    _say("S2c", f"Review files saved ({n_rows} paragraph pair(s)) — "
                "waiting for script review.")


def _ai_learn(pl, args, base, script_text):
    """v0.18 AI mode: learn from the script the user just approved.

    Pairs the approved paragraphs with the AI draft (<base>_ai_draft.json,
    written by the translate stage) and hands them to learn_from_review:
    translation memory + corrections + style-profile merge. Fail-open — a
    learning problem is logged and the dub goes on. Idempotent: the hash of
    the learned script is kept in <base>_ai_learned.json, so re-dubbing the
    same approved script (a voice or sync retry) never counts the same
    corrections twice.
    """
    import hashlib
    digest = hashlib.sha256(script_text.strip().encode("utf-8")).hexdigest()
    marker_path = base + "_ai_learned.json"
    try:
        if os.path.isfile(marker_path) and \
           json.loads(_read_text(marker_path)).get("sha256") == digest:
            _note("AI learning: this approved script was already learned — "
                  "skipped.")
            return
    except Exception:
        pass
    draft_rows = []
    draft_path = base + "_ai_draft.json"
    if os.path.isfile(draft_path):
        try:
            draft_rows = json.loads(_read_text(draft_path)).get("rows") or []
        except Exception as e:
            _note(f"WARNING: AI draft unreadable ({e}) — learning without "
                  "corrections.")
    else:
        _note("AI learning: no AI draft for this script (memory hit or "
              "prompt-mode translation) — learning approved rows only.")
    try:
        en_entries = pl._extract_srt_entries(_read_text(base + ".srt"))
    except Exception:
        en_entries = []
    _note("AI learning: absorbing the reviewed script…")
    res = pl.learn_from_review(args.language, draft_rows, script_text,
                               en_entries, pl.GEMINI_DEFAULT_MODEL,
                               source=os.path.basename(base),
                               status_cb=_note)
    if res.get("ok"):
        try:
            _write_text(marker_path, json.dumps({
                "sha256": digest,
                "learned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pairs": res.get("pairs", 0),
                "corrections": res.get("corrections", 0),
                "profile": res.get("profile_path", ""),
            }, ensure_ascii=False, indent=2))
        except Exception:
            pass


def _ai_learn_moment():
    """v0.21: WHEN AI mode learns. "final" (default): only from the final
    dub, when the user presses "Learn from final dub" on the Regenerate tab
    (--learn-final) — lines fixed there after the review are learned too.
    "review": the pre-v0.21 behaviour, learning at --steps dub.
    engine_settings.json key ai_learn_at; anything else means "final"."""
    v = _engine_setting("ai_learn_at", "final", str)
    return "review" if str(v).strip().lower() == "review" else "final"


def _maybe_ai_learn(pl, args, base, script_text):
    """--steps dub of an AI run: learn now only when ai_learn_at=review."""
    if _ai_learn_moment() == "review":
        _ai_learn(pl, args, base, script_text)
    else:
        _note("AI learning waits for 'Learn from final dub' (Regenerate tab) "
              "— set ai_learn_at: \"review\" in engine_settings.json to learn "
              "at review time instead.")


def _run_dub(args, manifest):
    """--steps dub --script <file>: resume after review, S2d..S3e.

    Re-derives out_dir from --audio exactly like a full run, validates the
    translate-stage artifacts, reads the (possibly edited) translation from
    --script, rewrites FinalScript to match what actually gets dubbed
    (mirroring the bulk app's post-review behaviour), then runs the dub
    half. The stale "status":"review" manifest is overwritten by the normal
    ok manifest at the end.
    """
    pl, api_key, ctx = _begin_run(args, manifest)
    voice_id, voice_how = _resolve_voice(pl, api_key, args.language,
                                         args.voice_id)
    _note(f"Dub voice: {voice_id} ({voice_how})")

    out_dir, base = _prepare_out_dir(pl, ctx["audio_path"], manifest,
                                     test=_is_test(args))
    ctx["out_dir"], ctx["base"] = out_dir, base

    # The translate stage must have run first in this out_dir.
    required = {
        "English SRT": base + ".srt",
        "English sync SRT": base + "_sync_en.srt",
        "FinalScript": base + "_FinalScript.txt",
    }
    missing = [f"{name}: {path}" for name, path in required.items()
               if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            "Translate-stage artifacts are missing from "
            f"{out_dir} — run '--steps translate' on this audio first. "
            "Missing: " + "; ".join(missing))
    manifest["en_srt"] = required["English SRT"]
    ctx["en_srt_text"] = _read_text(required["English sync SRT"])
    if _ai_source(args):
        _load_ai_phrases(ctx)            # v0.20: the translator's phrases

    script_path = os.path.abspath(os.path.expanduser(args.script))
    if not os.path.isfile(script_path):
        raise RuntimeError(f"--script file not found: {script_path}")
    script_text = _read_text(script_path).strip()
    if not script_text:
        raise RuntimeError(f"--script file is empty: {script_path}")
    n_paras = len(pl._split_translation_paragraphs(script_text))
    _note(f"Dub script loaded: {script_path} ({n_paras} paragraph(s)).")
    ctx["script_text"] = script_text

    # Rewrite FinalScript so the saved file matches what actually gets
    # dubbed — the same thing the bulk app does after its review window.
    marker = f"=== {args.language.upper()} TRANSLATION ==="
    fs_path = required["FinalScript"]
    fs_text = _read_text(fs_path)
    if marker in fs_text:
        _write_text(fs_path,
                    fs_text.split(marker, 1)[0] + marker + "\n" + script_text)
        _note("FinalScript updated with the reviewed translation.")
    else:
        _note(f"WARNING: marker {marker!r} not found in "
              f"{os.path.basename(fs_path)} — FinalScript left unchanged.")

    if _is_test(args):
        _note("Test mode (AI · test rules): nothing is learned from this run.")
    elif _ai_source(args):
        _maybe_ai_learn(pl, args, base, script_text)

    # English audio duration for the sync algorithm (local decode, no API).
    try:
        y_data, sr = pl._load_audio_any(ctx["audio_path"])
        ctx["en_audio_dur"] = float(len(y_data)) / float(sr) if sr else 0.0
    except Exception as e:
        ctx["en_audio_dur"] = 0.0
        _note(f"WARNING: could not measure the English audio duration "
              f"({e}) — sync runs without it.")

    _stage_dub(pl, args, api_key, manifest, ctx, voice_id)


def _run_regen(args, manifest):
    """--regen-chunk: synthesize one chunk text into --out-wav.

    Single-voice ElevenLabs synthesis with the same voice-resolution rule
    as the pipeline. No emotion pass — the chunk text is already final
    (it came out of a dubbed script; re-enriching would double the tags).
    synthesize_tts_elevenlabs converts the returned MP3 bytes to WAV via
    pydub, and also drops its usual per-chunk side files (<out>_chunks.txt,
    <out>_chunk_NN.mp3) next to the WAV.
    """
    manifest["regen_wav"] = ""
    text_path = os.path.abspath(os.path.expanduser(args.text_file))
    if not os.path.isfile(text_path):
        raise RuntimeError(f"--text-file not found: {text_path}")
    text = _read_text(text_path).strip()
    if not text:
        raise RuntimeError(f"--text-file is empty: {text_path}")

    pl, api_key = _load_pipeline_and_keys(args, need_llm=False)
    _require_ffmpeg(pl, hard=True)
    voice_id, voice_how = _resolve_voice(pl, api_key, args.language,
                                         args.voice_id)
    _note(f"Regen voice: {voice_id} ({voice_how})")

    out_wav = os.path.abspath(os.path.expanduser(args.out_wav))
    # Hard rule kept from v0.2: never write into the bulk-app folder. The
    # engine no longer knows APP_DIR on its own, but if a legacy caller
    # still passes --app-dir, honour the protection.
    if args.app_dir:
        app_dir = os.path.abspath(os.path.expanduser(args.app_dir))
        if out_wav == app_dir or out_wav.startswith(app_dir + os.sep):
            raise RuntimeError(f"--out-wav must not be inside the bulk-app "
                               f"folder: {out_wav}")
    out_parent = os.path.dirname(out_wav)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    _say("S2d", f"Regenerating chunk ({len(text)} chars, voice {voice_id}, "
                f"model {args.el_model})…")
    # A locked --out-wav (previous regen still loaded in REAPER) diverts to
    # a "-2" name; the panel applies whatever path the manifest reports.
    out_wav = pl.synthesize_tts_elevenlabs(
        text, out_wav, api_key=api_key, voice_id=voice_id,
        model_id=args.el_model,
        status_cb=lambda m: _say("S2d", m))
    manifest["regen_wav"] = out_wav
    _say("S2d", f"Regen chunk saved: {os.path.basename(out_wav)}")


def _run_voice_change(args, manifest):
    """--voice-change: re-voice --in-wav with the ElevenLabs voice changer.

    Speech-to-speech conversion to the --voice-id voice (auto-resolved from
    the account catalogue when omitted, same rule as every other mode).
    Keeps the source timing/pacing, so a synced dub stays synced. Manifest:
    {"status":"ok","vc_wav":"<abs>"}.
    """
    manifest["vc_wav"] = ""
    in_wav = os.path.abspath(os.path.expanduser(args.in_wav))
    if not os.path.isfile(in_wav):
        raise RuntimeError(f"--in-wav not found: {in_wav}")
    out_wav = os.path.abspath(os.path.expanduser(args.out_wav))

    pl, api_key = _load_pipeline_and_keys(args, need_llm=False)
    _require_ffmpeg(pl, hard=True)
    voice_id, voice_how = _resolve_voice(pl, api_key, args.language,
                                         args.voice_id)
    _note(f"Voice-change target voice: {voice_id} ({voice_how})")

    # Same lock hazard as the dub outputs: a previous vc wav still loaded in
    # an open REAPER project would fail the save AFTER the STS credits are
    # spent — divert to a "-2" name up front instead.
    out_wav = pl.ensure_writable_output(out_wav, status_cb=lambda m: _note(m))
    pl.voice_change_elevenlabs(
        in_wav, out_wav, api_key=api_key, voice_id=voice_id,
        model_id=args.sts_model,
        status_cb=lambda m: _note(m))
    manifest["vc_wav"] = out_wav
    _note(f"Voice-changed audio saved: {os.path.basename(out_wav)}")


def _run_test_llm(args, manifest):
    """--test-llm: one tiny call on the configured LLM provider.

    Fills provider/model/reply in the manifest. Any failure (missing
    settings, bad key, unreachable endpoint) surfaces as the usual error
    manifest via main()'s handler.
    """
    manifest["provider"] = ""
    manifest["model"] = ""
    manifest["reply"] = ""
    _note("Importing pipeline modules…")
    pl = _import_pipeline()
    _check_symbols(pl)
    pl._validate_llm_config()
    provider, model = pl._active_provider_and_model()
    manifest["provider"], manifest["model"] = provider, model
    _note(f"Testing LLM connection: provider={provider}, model={model}…")
    reply = pl._llm_generate(
        "Reply with the single word OK and nothing else.", model)
    reply = (reply or "").strip()
    if not reply:
        raise RuntimeError("The LLM returned an empty reply.")
    manifest["reply"] = reply[:200]
    _note(f"LLM reply: {manifest['reply']}")


def _parse_suggest_rows(text):
    """Rows from the panel's suggest request file:
    '@@ <index> <speech_s> <hard_s>' then 'EN: ...' / 'TR: ...' lines."""
    rows, cur = [], None
    for line in (text or "").splitlines():
        if line.startswith("@@"):
            parts = line[2:].split()
            try:
                cur = {"index": int(parts[0]), "speech": float(parts[1]),
                       "hard": float(parts[2]), "en": "", "tr": ""}
                rows.append(cur)
            except (IndexError, ValueError):
                cur = None
        elif cur is not None and line.startswith("EN:"):
            cur["en"] = line[3:].strip()
        elif cur is not None and line.startswith("TR:"):
            cur["tr"] = line[3:].strip()
    return [r for r in rows if r["tr"]]


def _run_suggest_fit(args, manifest):
    """--suggest-fit: shorter renderings for the review screen's too-long
    rows. Text-only LLM calls; Indic text travels by UTF-8 file both ways
    (never argv). One row that fails simply gets no options."""
    manifest["suggest_txt"] = ""
    manifest["suggest_count"] = "0"
    in_path = os.path.abspath(os.path.expanduser(args.text_file))
    if not os.path.isfile(in_path):
        raise RuntimeError(f"--text-file not found: {in_path}")
    rows = _parse_suggest_rows(_read_text(in_path))
    if not rows:
        raise RuntimeError("The suggest request has no rows with text.")
    _note("Importing pipeline modules…")
    pl = _import_pipeline()
    _check_symbols(pl)
    pl._validate_llm_config()
    _note(f"Asking for shorter {args.language} renderings of "
          f"{len(rows)} line(s)…")
    blocks, total = [], 0
    for r in rows:
        opts = pl.suggest_fits(r["en"], r["tr"], r["speech"], r["hard"],
                               args.language, pl.GEMINI_DEFAULT_MODEL,
                               house_rules=_is_test(args))
        _note(f"  line {r['index']}: {len(opts)} option(s)")
        if opts:
            blocks.append("@@ " + str(r["index"]) + "\n" + "\n".join(opts))
            total += len(opts)
    out_path = in_path + ".out"
    _write_text(out_path, "\n\n".join(blocks) + "\n")
    manifest["suggest_txt"] = out_path
    manifest["suggest_count"] = str(total)
    _note(f"{total} suggestion(s) written.")


def _run_review_assist(args, manifest):
    """--review-assist: answer the reviewer's instruction for one line.
    Request and answer travel as UTF-8 files (Indic text never on argv)."""
    manifest["assist_txt"] = ""
    in_path = os.path.abspath(os.path.expanduser(args.text_file))
    if not os.path.isfile(in_path):
        raise RuntimeError(f"--text-file not found: {in_path}")
    req = {"index": 0, "speech": 0.0, "hard": 0.0, "en": "", "tr": "",
           "prev": "", "next": "", "ask": "", "history": []}
    for line in _read_text(in_path).splitlines():
        if line.startswith("@@"):
            parts = line[2:].split()
            try:
                req["index"] = int(parts[0])
                req["speech"], req["hard"] = float(parts[1]), float(parts[2])
            except (IndexError, ValueError):
                pass
        elif line.startswith("H:"):
            role, _, text = line[2:].strip().partition("|")
            req["history"].append((role.strip(), text.strip()))
        else:
            for key, tag in (("en", "EN:"), ("tr", "TR:"), ("prev", "PREV:"),
                             ("next", "NEXT:"), ("ask", "ASK:")):
                if line.startswith(tag):
                    req[key] = line[len(tag):].strip()
    if not req["ask"]:
        raise RuntimeError("The assistant request has no instruction.")
    _note("Importing pipeline modules…")
    pl = _import_pipeline()
    _check_symbols(pl)
    pl._validate_llm_config()
    _note(f"Assistant: line {req['index']} — \"{req['ask'][:80]}\"")
    res = pl.review_assist(req["en"], req["tr"], req["speech"], req["hard"],
                           req["ask"], args.language, req["history"],
                           req["prev"], req["next"], pl.GEMINI_DEFAULT_MODEL,
                           house_rules=_is_test(args))
    lines = ["REPLY: " + (res.get("reply") or "")]
    if res.get("speed") is not None:
        lines.append(f"SPEED: {res['speed']:.2f}")
    lines += ["ALT: " + a for a in res.get("alternatives") or []]
    out_path = in_path + ".out"
    _write_text(out_path, "\n".join(lines) + "\n")
    manifest["assist_txt"] = out_path
    _note(f"Assistant answered: {len(res.get('alternatives') or [])} "
          f"alternative(s)"
          + (f", speed x{res['speed']:.2f}" if res.get("speed") else "") + ".")


def _run_recommend_voice(args, manifest):
    """--recommend-voice: pick the best catalogue voices for one line."""
    manifest["reco_txt"] = ""
    in_path = os.path.abspath(os.path.expanduser(args.text_file))
    if not os.path.isfile(in_path):
        raise RuntimeError(f"--text-file not found: {in_path}")
    req = {"current": "", "speaker": "", "tr": "", "en": ""}
    voices = []
    for line in _read_text(in_path).splitlines():
        if line.startswith("V:"):
            vid, _, nm = line[2:].strip().partition("|")
            if vid.strip():
                voices.append((vid.strip(), nm.strip()))
        else:
            for key, tag in (("current", "CURRENT:"), ("speaker", "SPEAKER:"),
                             ("tr", "TR:"), ("en", "EN:")):
                if line.startswith(tag):
                    req[key] = line[len(tag):].strip()
    if not voices:
        raise RuntimeError("No voices in the request — fetch the voice list "
                           "first.")
    _note("Importing pipeline modules…")
    pl = _import_pipeline()
    _check_symbols(pl)
    pl._validate_llm_config()
    _note(f"Recommending voices from {len(voices)} for {args.language}…")
    picks = pl.recommend_voices(args.language, voices, req["tr"], req["en"],
                                req["current"], req["speaker"], 3,
                                pl.GEMINI_DEFAULT_MODEL)
    out_path = in_path + ".out"
    _write_text(out_path, "\n".join(f"PICK: {p['id']}|{p['why']}"
                                     for p in picks) + "\n")
    manifest["reco_txt"] = out_path
    _note(f"{len(picks)} voice(s) recommended.")


# ─── v0.21 --learn-final: learn from the final dub ─────────────────────────
# The review screen is not the last word on a script: lines are regenerated
# on the Regenerate tab afterwards, and those fixes never reached the
# learner. The panel's "Learn from final dub" sends the timeline as it is
# NOW (every dub chunk: start, length, text) and the engine learns from that,
# the AI draft and the panel's regen log (<base>_regen_edits.jsonl).
LEARN_FINAL_SUFFIX = "_ai_learned_final.json"
REGEN_EDITS_SUFFIX = "_regen_edits.jsonl"
LEARN_KEYS_MAX = 5000            # learned pair/correction keys remembered


def _parse_learn_request(text):
    """Request file -> (base, [(start_s, len_s, text)] in timeline order).
    'BASE: <abs run base path, no suffix>' and 'C: <start>|<len>|<text>'
    lines; the text may itself contain '|'. Malformed lines are skipped."""
    base, chunks = "", []
    for line in (text or "").lstrip("﻿").splitlines():
        line = line.rstrip("\r")
        if line.startswith("BASE:"):
            base = line[5:].strip()
        elif line.startswith("C:"):
            parts = line[2:].strip().split("|", 2)
            if len(parts) < 3:
                continue
            try:
                s, n = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            t = " ".join(parts[2].split())
            if t and n >= 0:
                chunks.append((s, n, t))
    chunks.sort(key=lambda c: c[0])
    return base, chunks


def _learn_norm(s):
    return " ".join((s or "").split())


def _load_final_english(pl, base):
    """(entries, file name) of the run's English cues: the translator's
    phrases first (finest), then the sync SRT, then the Stage-1 SRT."""
    for suffix in (AI_PHRASES_SUFFIX, "_sync_en.srt", ".srt"):
        p = base + suffix
        if os.path.isfile(p):
            try:
                entries = pl._extract_srt_entries(_read_text(p))
            except Exception:
                entries = []
            if entries:
                return entries, os.path.basename(p)
    return [], ""


def _english_under(en_entries, start, end):
    """English spoken under [start, end]: every cue whose midpoint falls in
    the window (so a cue is never split between two neighbouring chunks).
    The English source starts at project start — the same assumption as the
    panel's V5.regen_en_under."""
    parts = []
    for (s, e, t) in en_entries or []:
        mid = (float(s) + float(e)) / 2.0
        if start - 0.05 <= mid <= end + 0.05 and (t or "").strip():
            parts.append(t.strip())
    return " ".join(parts)


def _load_regen_edits(base, chunks):
    """The panel's regen log, chains collapsed (A->B then B->C at the same
    spot = A->C), kept only while the final wording is still on the
    timeline (an edit undone with Ctrl+Z is not learned)."""
    p = base + REGEN_EDITS_SUFFIX
    if not os.path.isfile(p):
        return []
    chains = []
    for line in _read_text(p).splitlines():
        try:
            e = json.loads(line)
            t, n = float(e.get("t", 0)), float(e.get("len", 0))
        except Exception:
            continue
        old, new = _learn_norm(e.get("old")), _learn_norm(e.get("new"))
        if not new:
            continue
        for c in chains:
            if abs(c["t"] - t) < 0.25 and c["new"] == old:
                c["new"], c["len"] = new, n
                break
        else:
            chains.append({"t": t, "len": n, "old": old, "new": new})
    final = [_learn_norm(c[2]) for c in chunks]
    return [c for c in chains if c["old"] != c["new"]
            and any(c["new"] in f for f in final)]


def _fit_spoken(base):
    """SPOKEN lines of <base>_ai_fit_changes.txt — the engine's own fit
    shortenings, which are not a human's correction."""
    p = base + "_ai_fit_changes.txt"
    out = []
    try:
        for line in _read_text(p).splitlines() if os.path.isfile(p) else []:
            if line.startswith("SPOKEN:"):
                t = _learn_norm(line[7:])
                if t:
                    out.append(t)
    except Exception:
        pass
    return out


def _final_pairs_and_corrections(pl, base, chunks, en_entries, edits):
    """(pairs, corrections) from the final timeline.

    (a) every regen edit old -> new, with the English under it;
    (b) AI draft paragraph -> the final text of the chunks inside its
        window, only when that is unambiguous: the paragraph was timed by
        the translator (real cues), no chunk of it was regenerated (those
        are learned by (a)), no chunk is the engine's own fit shortening,
        no chunk reads like another paragraph's text, and the two versions
        are clearly the same passage (similarity >= 0.6).
    Pairs: (draft English, final paragraph) for every accepted paragraph,
    else (English under the chunk, chunk text) per remaining chunk."""
    import difflib
    corrections = [{"en": _english_under(en_entries, e["t"], e["t"] + e["len"]),
                    "draft": e["old"], "final": e["new"]} for e in edits]
    touched = set()
    for i, (s, n, t) in enumerate(chunks):
        nt = _learn_norm(t)
        for e in edits:
            if e["new"] in nt or (abs(e["t"] - s) < 0.25):
                touched.add(i)
    rows = []
    p = base + "_ai_draft.json"
    if os.path.isfile(p):
        try:
            rows = json.loads(_read_text(p)).get("rows") or []
        except Exception:
            rows = []
    split = getattr(pl, "split_pause_markers", None)

    def clean(tr):
        tr = str(tr or "")
        if callable(split):
            try:
                tr = split(tr)[0]
            except Exception:
                pass
        return _learn_norm(tr)

    timed = []
    for k, r in enumerate(rows):
        try:
            s0, s1 = float(r.get("start")), float(r.get("end"))
        except (TypeError, ValueError):
            continue
        if r.get("cues") and clean(r.get("tr")) and \
           str(r.get("en") or "").strip():
            timed.append((k, s0, max(s0, s1), clean(r.get("tr")),
                          str(r.get("en")).strip()))
    groups = {}
    for i, (s, n, _t) in enumerate(chunks):
        best, best_ov = None, 0.0
        for (k, s0, s1, _tr, _en) in timed:
            ov = min(s + n, s1) - max(s, s0)
            if ov > best_ov:
                best, best_ov = k, ov
        if best is not None:
            groups.setdefault(best, []).append(i)
    spoken = _fit_spoken(base)
    drafts = {k: tr for (k, _a, _b, tr, _en) in timed}
    row_of = {i: k for k, idx in groups.items() for i in idx}
    norm = [_learn_norm(c[2]) for c in chunks]

    def mine(t, k):
        # A chunk text long enough to be unambiguous, found in row k's draft.
        return len(t) >= 12 and t in drafts.get(k, "")

    pairs, covered = [], set()
    for (k, s0, s1, draft, en) in timed:
        idx = groups.get(k)
        if not idx:
            continue
        # A regenerated chunk anywhere in this window (a merge can pull a
        # piece of this paragraph into a neighbour): (a) owns it.
        if any(min(chunks[i][0] + chunks[i][1], s1) > max(chunks[i][0], s0)
               for i in touched):
            continue
        texts = [norm[i] for i in idx]
        if any(sp and any(sp in t for t in texts) for sp in spoken):
            continue
        # A chunk that reads like ANOTHER paragraph landed here, or one of
        # this paragraph's pieces landed in a neighbour: timing is ambiguous.
        if any(not mine(norm[i], k) and any(mine(norm[i], kk)
                                            for kk in drafts if kk != k)
               for i in idx):
            continue
        if any(i not in idx and mine(norm[i], k)
               and not mine(norm[i], row_of.get(i))
               for i in range(len(chunks))):
            continue
        final = " ".join(texts)
        if final != draft and difflib.SequenceMatcher(
                None, draft, final).ratio() < 0.6:
            continue
        pairs.append((en, final))
        covered.update(idx)
        if final != draft:
            corrections.append({"en": en, "draft": draft, "final": final})
    for i, (s, n, t) in enumerate(chunks):
        if i in covered:
            continue
        en = _english_under(en_entries, s, s + n)
        if en:
            pairs.append((en, _learn_norm(t)))
    return pairs, corrections


def _learn_key(*parts):
    import hashlib
    return hashlib.sha1("\x00".join(_learn_norm(p) for p in parts)
                        .encode("utf-8")).hexdigest()[:20]


def _run_learn_final(args, manifest):
    """--learn-final: learn from the final dub (see the block comment)."""
    import hashlib
    manifest["learn_txt"] = ""
    in_path = os.path.abspath(os.path.expanduser(args.text_file))
    if not os.path.isfile(in_path):
        raise RuntimeError(f"--text-file not found: {in_path}")
    base, chunks = _parse_learn_request(_read_text(in_path))
    if not base or not os.path.isdir(os.path.dirname(base) or "."):
        raise RuntimeError("The learn request names no usable run folder "
                           f"(BASE: {base or '(missing)'}).")
    if _is_test_base(base):
        raise RuntimeError(
            "This dub was made with 'AI · test rules' (a _TEST folder). Test "
            "runs never teach AI mode — dub it with 'AI · learns' to learn "
            "from it.")
    if not chunks:
        raise RuntimeError("The learn request has no dub chunks with text.")
    out_path = in_path + ".out"

    def finish(lines):
        _write_text(out_path, "\n".join(lines) + "\n")
        manifest["learn_txt"] = out_path

    final_text = "\n\n".join(c[2] for c in chunks)
    digest = hashlib.sha256(final_text.strip().encode("utf-8")).hexdigest()
    marker_path = base + LEARN_FINAL_SUFFIX
    marker = {}
    try:
        if os.path.isfile(marker_path):
            marker = json.loads(_read_text(marker_path)) or {}
    except Exception:
        marker = {}
    if marker.get("sha256") == digest:
        _note("AI learning: this final dub was already learned — skipped.")
        finish(["SKIPPED: 1", "PAIRS: 0", "CORRECTIONS: 0", "HELD: 0",
                "NOTE: this final dub was already learned"])
        return

    _note("Importing pipeline modules…")
    pl = _import_pipeline()
    _check_symbols(pl)
    en_entries, en_name = _load_final_english(pl, base)
    _note(f"AI learning from the final dub: {len(chunks)} chunk(s), English "
          f"from {en_name or '(none found)'}.")
    edits = _load_regen_edits(base, chunks)
    pairs, corrections = _final_pairs_and_corrections(pl, base, chunks,
                                                      en_entries, edits)
    # A second press after more fixes learns only what is NEW.
    seen = set(marker.get("learned_keys") or [])
    pairs = [(e, f) for (e, f) in pairs if _learn_key("p", e, f) not in seen]
    corrections = [c for c in corrections
                   if _learn_key("c", c["draft"], c["final"]) not in seen]
    # The whole document is memorised only when the timeline covers all of
    # the English (every cue has a chunk within 2 s) — a region-only or
    # half-imported timeline must not stand in for the full script.
    covers = bool(en_entries) and all(
        any(min(e + 2.0, cs + cn) > max(s - 2.0, cs) for (cs, cn, _t) in chunks)
        for (s, e, _t) in en_entries)
    _note(f"AI learning: {len(edits)} regeneration(s) in the log, "
          f"{len(pairs)} new pair(s), {len(corrections)} new correction(s).")
    res = pl.learn_from_final(args.language, pairs, corrections,
                              final_text if covers else "",
                              en_entries if covers else None,
                              pl.GEMINI_DEFAULT_MODEL,
                              source=os.path.basename(base) + " (final)",
                              status_cb=_note)
    held = res.get("held_back") or []
    nothing = not pairs and not corrections
    if res.get("ok") or nothing:
        held_n = [_learn_norm(h) for h in held]
        keys = list(marker.get("learned_keys") or [])
        keys += [_learn_key("p", e, f) for (e, f) in pairs
                 if not any(h and h in _learn_norm(f) for h in held_n)]
        keys += [_learn_key("c", c["draft"], c["final"]) for c in corrections
                 if _learn_norm(c["final"]) not in held_n]
        try:
            _write_text(marker_path, json.dumps({
                "sha256": digest,
                "learned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pairs": res.get("pairs", 0),
                "corrections": res.get("corrections", 0),
                "held_back": held,
                "profile": res.get("profile_path", ""),
                "learned_keys": keys[-LEARN_KEYS_MAX:],
            }, ensure_ascii=False, indent=2))
        except Exception as e:
            _note(f"WARNING: could not write {marker_path}: {e}")
    elif res.get("error"):
        raise RuntimeError("AI learning failed: " + res["error"])
    note = ("nothing new to learn" if nothing
            else res.get("error") or "learned")
    finish(["SKIPPED: 0",
            f"PAIRS: {res.get('pairs', 0)}",
            f"CORRECTIONS: {res.get('corrections', 0)}",
            f"HELD: {len(held)}",
            f"REGEN_EDITS: {len(edits)}",
            f"TM_STORED: {res.get('tm_stored', 0)}",
            "NOTE: " + " ".join(str(note).split())])
    _note(f"AI learning done: {res.get('pairs', 0)} pair(s), "
          f"{res.get('corrections', 0)} correction(s), {len(held)} held back.")


def _row_speeds(base):
    """{paragraph_number: factor} from <base>_row_speed.json (written by
    the review screen), or {}. 1.10 = 10 % faster, 0.90 = slower."""
    p = base + "_row_speed.json"
    try:
        data = json.loads(_read_text(p)) if os.path.isfile(p) else {}
        out = {}
        for k, v in (data.get("speeds") or {}).items():
            f = float(v)
            if 0.5 <= f <= 2.0 and abs(f - 1.0) > 0.005:
                out[int(k)] = f
        return out
    except Exception:
        return {}


def _run_list_voices(args, manifest):
    """--list-voices --language <Lang>: fetch the ElevenLabs voice catalogue.

    Voices come back language-token sorted exactly like the app's dropdown
    (voices advertising the language first, each bucket alphabetical); the
    manifest keeps that order as [{"id","name"},…].
    """
    manifest["voices"] = []
    _note("Importing pipeline modules…")
    pl = _import_pipeline()
    _check_symbols(pl)
    if args.language not in pl.TTS_LANGUAGES:
        raise RuntimeError(f"Language {args.language!r} is not registered "
                           "in the pipeline's TTS_LANGUAGES table.")
    api_key = pl._get_api_key()
    _note(f"Fetching the ElevenLabs voice catalogue ({args.language})…")
    voices = pl._fetch_voices_for_language(api_key, args.language,
                                           force_refresh=True)
    manifest["voices"] = [{"id": v["voice_id"], "name": v["name"]}
                          for v in voices]
    _note(f"{len(manifest['voices'])} voice(s) fetched.")


def main() -> int:
    args = _parse_args()

    v = _app_version()
    _note(f"Reaper Dubbing App{' v' + v if v else ''} (contract v0.13)")

    if args.app_dir:
        _note("WARNING: --app-dir is deprecated as of v0.3 and IGNORED — "
              "the engine is standalone (settings come from config/, "
              "prompts from prompts/).")

    if args.selfcheck:
        return _selfcheck(args)

    if args.learn_final:
        keys, runner, ok_status = LEARNF_MANIFEST_KEYS, _run_learn_final, "ok"
    elif args.recommend_voice:
        keys, runner, ok_status = RECO_MANIFEST_KEYS, _run_recommend_voice, "ok"
    elif args.review_assist:
        keys, runner, ok_status = (ASSIST_MANIFEST_KEYS, _run_review_assist,
                                   "ok")
    elif args.suggest_fit:
        keys, runner, ok_status = SUGGEST_MANIFEST_KEYS, _run_suggest_fit, "ok"
    elif args.test_llm:
        keys, runner, ok_status = TEST_LLM_MANIFEST_KEYS, _run_test_llm, "ok"
    elif args.list_voices:
        keys, runner, ok_status = VOICES_MANIFEST_KEYS, _run_list_voices, "ok"
    elif args.voice_change:
        keys, runner, ok_status = (VOICE_CHANGE_MANIFEST_KEYS,
                                   _run_voice_change, "ok")
    elif args.regen_chunk:
        keys, runner, ok_status = REGEN_MANIFEST_KEYS, _run_regen, "ok"
    elif args.steps == "plan":
        keys, runner, ok_status = PLAN_MANIFEST_KEYS, _run_plan, "plan"
    elif args.steps == "dubplan":
        keys, runner, ok_status = MANIFEST_KEYS, _run_dubplan, "ok"
    elif args.steps == "translate":
        keys, runner, ok_status = (REVIEW_MANIFEST_KEYS, _run_translate,
                                   "review")
    elif args.steps == "dub":
        keys, runner, ok_status = MANIFEST_KEYS, _run_dub, "ok"
    else:
        keys, runner, ok_status = MANIFEST_KEYS, _run_full, "ok"

    manifest = {k: "" for k in keys}
    manifest["status"] = "error"     # flipped only on full success
    try:
        runner(args, manifest)
        manifest["status"] = ok_status
        manifest["error"] = ""
        _write_manifest(manifest, manifest.get("out_dir") or "", keys)
        if ok_status == "review":
            _note("Translate stage complete — review manifest written.")
        else:
            _note("Run complete — manifest written.")
        return 0
    except (Exception, SystemExit):
        # SystemExit is caught too (it is a BaseException, NOT an Exception)
        # as a belt-and-braces measure: the contract requires an error
        # manifest on EVERY failure, so convert it instead of letting the
        # process die manifest-less.
        tb = traceback.format_exc()
        print(tb, flush=True)
        manifest["status"] = "error"
        manifest["error"] = tb[-2000:]
        _write_manifest(manifest, manifest.get("out_dir") or "", keys)
        _note("Run FAILED — error manifest written.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
