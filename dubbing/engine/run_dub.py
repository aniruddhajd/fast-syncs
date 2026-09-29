#!/usr/bin/env python3
"""
Launcher for the Reaper dubbing engine (contract v0.3).

Adapted from the proven "fast syncs" run_sync.py launcher pattern. REAPER's
Lua panel starts this script in the background (macOS `... &`, Windows
`ExecProcess(cmd, -2)`), and this launcher does everything that cannot be
done portably from Lua:

  1. Clean and recreate the engine status directory.
  2. Spawn dub_engine.py with the SAME interpreter via subprocess.
  3. Tee the worker's combined stdout+stderr to status/engine_log.txt
     (utf-8, line-buffered) so the Lua poller can tail it live.
  4. Publish the CHILD pid to status/engine_pid.txt (for the Cancel button).
  5. On child exit, write the exit code to status/engine_done.txt LAST so
     the poller never sees a done marker before the log/manifest are final.

Do not remove this file or fold it into dub_engine.py: the panel launches
THIS script (RUN_DUB_PY in Dub_Pipeline_Panel.lua), never dub_engine.py,
which is only invoked directly for --selfcheck (setup_mac.command). Both
files are required.

Standard library only. No shell=True. Secrets never travel on the command
line — API keys are read from this repo's gitignored config/ directory by
the pipeline itself (config/llm_settings.json, config/tts_settings.json
and the key files they point at).

Every mode goes through this launcher, so the status-dir / log / pid /
done.txt behaviour is identical for full runs, staged runs, chunk
regeneration, LLM connection tests and voice-list fetches.

--app-dir is DEPRECATED as of v0.3 (the engine is standalone): it is still
accepted and forwarded for backward compatibility, and the engine logs a
warning and ignores it.

Usage:
    # one-shot (v0.1 behaviour)
    "<python>" run_dub.py --audio "<audio path>" --language <Language> \
        [--voice-id <ELid>] [--el-model <model>] [--steps full] [--no-emotion]

    # staged: stop after translation for script review
    "<python>" run_dub.py --audio "<audio path>" --language <Language> \
        --steps translate

    # staged: resume with the reviewed/edited translation text file
    "<python>" run_dub.py --audio "<audio path>" --language <Language> \
        --steps dub --script "<abs .txt>" [--no-emotion]

    # regenerate one chunk (text read from a file, never from argv)
    "<python>" run_dub.py --regen-chunk --language <Language> \
        --text-file "<abs .txt>" --out-wav "<abs .wav>"

    # test the configured LLM provider (one tiny call)
    "<python>" run_dub.py --test-llm

    # fetch the ElevenLabs voice catalogue for a language
    "<python>" run_dub.py --list-voices --language <Language>
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_DIR = os.path.join(ENGINE_DIR, "status")


def _worker_python():
    """The interpreter to run dub_engine.py with.

    The panel starts this launcher with pythonw.exe so REAPER cannot give it a
    console window. pythonw.exe would work for the worker too (it is spawned
    with CREATE_NO_WINDOW either way), but the worker is a plain console
    program with a piped stdout, so hand it python.exe and keep its
    environment exactly as it has always been.
    """
    exe = sys.executable
    if os.name == "nt" and exe:
        head, tail = os.path.split(exe)
        low = tail.lower()
        if low.endswith("w.exe"):
            cand = os.path.join(head, tail[:-len("w.exe")] + ".exe")
            if os.path.exists(cand):
                return cand
    return exe


# The 12 target languages supported by the pipeline (display names), plus
# any the user added in the panel (v0.7). This launcher validates --language
# before spawning the worker, so the two lists must agree — dub_engine.py
# extends its own copy from the same file, with the same stdlib-only read.
LANGUAGES = ["Bengali", "Hindi", "Kannada", "Malayalam", "Tamil", "Telugu",
             "Gujarati", "Marathi", "Punjabi", "Assamese", "Odia", "Nepali"]

# Which hand-edited names are usable. custom_languages.json is read FOUR
# times in total — here, in dub_engine.py, in pipeline/config.py, and by the
# REAPER panel — each with its own stdlib-only reader so that argparse choices
# exist before any heavy import. All four must agree on which names are valid,
# or an entry accepted by one and dropped by another produces an "unknown
# language" failure the user cannot explain.
#
# KEEP IN SYNC with:
#     engine/dub_engine.py                 _LANG_NAME_OK
#     engine/pipeline/config.py            _LANG_NAME_OK
#     dubbing/reaper/Dub_Pipeline_Panel.lua  V5._is_safe_lang_name
#
# Letters, digits, space, - _ . ( ) and any non-ASCII character (so native
# autonyms work). Shell metacharacters are all ASCII and all excluded; tabs
# and newlines are excluded too. Names are validated, never rewritten — a
# rewritten name would be a second spelling of the same entry.
# NOTE the explicit \u0080 lower bound on the non-ASCII range: writing
# "()-\U0010FFFF" makes ')' the START of a range running to the top of
# Unicode, which quietly re-admits ';', '|' and other metacharacters.
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

try:
    with open(os.path.join(ENGINE_DIR, os.pardir, "config",
                           "custom_languages.json"), "r",
              encoding="utf-8") as _f:
        for _e in (json.load(_f).get("languages") or []):
            _n = str((_e or {}).get("name") or "")
            if _lang_name_ok(_n) and _n not in LANGUAGES:
                LANGUAGES.append(_n)
except Exception:
    pass
LANGUAGES.sort()          # --language choices read alphabetically in --help


class _LaunchArgError(Exception):
    """argparse rejected the command line; str() is usage + error text."""


class _ArgParser(argparse.ArgumentParser):
    """ArgumentParser whose error() raises instead of exiting.

    The panel starts this launcher detached under pythonw.exe, so the stock
    behaviour (usage on stderr, exit 2) is invisible and the user only ever
    sees the panel's 90 s no-output watchdog. Raising lets main() write the
    message to engine_log.txt and finish with the normal done markers.
    --help still exits via SystemExit(0) as usual.
    """

    def error(self, message):
        raise _LaunchArgError(f"{self.format_usage()}{self.prog}: error: "
                              f"{message}\n")


def _resolve_status_dir(value):
    """Validated absolute status dir for a --status-dir value, else None.

    A per-run override must stay inside engine/status/ — this launcher
    rmtree's the target, so an arbitrary path would be a foot-gun.
    None/empty means the shared default (engine/status itself).
    """
    root = os.path.abspath(STATUS_DIR)
    if not value:
        return root
    cand = os.path.abspath(value)
    if cand != root and not cand.startswith(root + os.sep):
        return None
    return cand


def _prescan_status_dir(argv):
    """Find --status-dir in raw argv without argparse (which may have failed).

    Mirrors argparse: '--status-dir X', '--status-dir=X' and unambiguous
    prefixes ('--sta' and longer); the last occurrence wins. Returns the
    validated dir, or None when the value is missing or outside
    engine/status/.
    """
    flag = "--status-dir"
    found, value = False, None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            break
        name, eq, val = tok.partition("=")
        if len(name) >= 5 and flag.startswith(name):   # '--sta' is unique
            found = True
            if eq:
                value = val
            elif i + 1 < len(argv):
                value = argv[i + 1]
                i += 1
            else:
                value = None
        i += 1
    if found and not value:
        return None
    return _resolve_status_dir(value)


def _clean_status_dir(status_dir):
    """Clean *status_dir* so the Lua poller only sees files from THIS run.

    A stale engine_done.txt would end the poll early. A per-project subdir
    is cleaned wholesale; the shared root is cleaned file-by-file so a
    legacy (no --status-dir) launch can never wipe a sibling project's live
    run underneath it.
    """
    if status_dir != os.path.abspath(STATUS_DIR):
        shutil.rmtree(status_dir, ignore_errors=True)
    else:
        for name in ("engine_log.txt", "engine_pid.txt",
                     "engine_done.txt", "engine_done.json"):
            try:
                os.remove(os.path.join(status_dir, name))
            except OSError:
                pass
    os.makedirs(status_dir, exist_ok=True)


# Full-run manifest shape (dub_engine.py MANIFEST_KEYS) so every panel
# reader finds the keys it expects; all "" except status/error.
_FAIL_MANIFEST_KEYS = ["status", "error", "audio", "language", "out_dir",
                       "en_audio", "en_srt", "tts_wav", "timestamps_txt",
                       "synced_wav", "synced_srt", "sync_texts",
                       "synced_count", "unsynced_count"]


def _report_arg_error(argv, message, exit_code=2):
    """Make a rejected command line visible to the panel.

    Writes the argparse message to <status-dir>/engine_log.txt and an
    error engine_done.json, then engine_done.txt LAST (same ordering as a
    normal run), so the panel's failure phase shows the real reason instead
    of the 90 s watchdog. Also echoes to stderr for console launches.
    """
    if sys.stderr is not None:
        try:
            sys.stderr.write(message)
            sys.stderr.flush()
        except Exception:
            pass
    status_dir = _prescan_status_dir(argv)
    if status_dir is None:
        return exit_code        # nowhere safe to write; stderr only
    text = ("[run_dub] ERROR: launch rejected — invalid command line "
            f"(exit {exit_code}).\n"
            "[run_dub] The panel sent arguments this run_dub.py does not "
            "accept. If the panel and engine come from different versions, "
            "run Update… so both match.\n\n" + message)
    try:
        _clean_status_dir(status_dir)
        with open(os.path.join(status_dir, "engine_log.txt"), "w",
                  encoding="utf-8", errors="replace") as lf:
            lf.write(text)
        manifest = {k: "" for k in _FAIL_MANIFEST_KEYS}
        manifest["status"] = "error"
        manifest["error"] = text
        with open(os.path.join(status_dir, "engine_done.json"), "w",
                  encoding="utf-8") as jf:
            json.dump(manifest, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # The done marker is written LAST — the poller's only completion signal.
    try:
        with open(os.path.join(status_dir, "engine_done.txt"), "w",
                  encoding="utf-8") as df:
            df.write(str(exit_code))
    except Exception:
        pass
    return exit_code


def _parse_args(argv):
    """Parse + validate the launcher's argv. Returns (args, status_dir).

    Raises _LaunchArgError on any bad command line.
    """
    ap = _ArgParser(
        description="Launch the headless dubbing engine (run_dub.py owns "
                    "log/pid/done markers; dub_engine.py does the work).")
    ap.add_argument("--app-dir", default=None,
                    help="DEPRECATED (v0.3): accepted for backward "
                         "compatibility and ignored by the standalone "
                         "engine (a warning is logged)")
    ap.add_argument("--audio", default=None,
                    help="Path to the English source audio file (required "
                         "unless --regen-chunk/--test-llm/--list-voices)")
    ap.add_argument("--language", default=None, choices=LANGUAGES,
                    help="Target language display name (required unless "
                         "--test-llm)")
    ap.add_argument("--voice-id", default=None,
                    help="Optional ElevenLabs voice_id (auto-resolved from "
                         "the account's voice catalogue when omitted)")
    ap.add_argument("--el-model", default="eleven_v3",
                    help="ElevenLabs TTS model id (default: eleven_v3)")
    ap.add_argument("--steps", default="full",
                    choices=["full", "translate", "dub", "plan", "dubplan"],
                    help="Pipeline scope: 'full' = one shot, 'translate' = "
                         "stop after the translation for script review, "
                         "'dub' = resume from a reviewed script "
                         "(requires --script), 'plan' = pause-aware Preview "
                         "sync (free: no TTS, no LLM; requires "
                         "--provided-script or --plan), 'dubplan' = generate "
                         "from an approved plan (requires --plan)")
    ap.add_argument("--script", default=None,
                    help="Reviewed translation text file for --steps dub "
                         "(blank-line paragraph format)")
    ap.add_argument("--plan", dest="plan", default=None,
                    help="Sync plan file for --steps dubplan (generate) or "
                         "--steps plan (re-measure the corrected TR: lines). "
                         "Forwarded to dub_engine.py.")
    ap.add_argument("--provided-script", dest="provided_script", default=None,
                    help="User-provided translation text file: skips the "
                         "LLM translation chain (S2a-S2c) in --steps "
                         "translate/full runs")
    ap.add_argument("--script-source", dest="script_source",
                    default="prompt", choices=["prompt", "ai", "ai_test", "eleven"],
                    help="v0.18: 'ai' = the Lekhak agentic translator "
                         "instead of the Step1-3 prompt chain (and, with "
                         "--steps dub, learn from the reviewed script). "
                         "v0.22: 'eleven' = ElevenLabs Dubbing Studio "
                         "translates and voices, with a voice per speaker. "
                         "Forwarded to dub_engine.py.")
    ap.add_argument("--voice-change", dest="voice_change",
                    action="store_true",
                    help="Re-voice --in-wav with the ElevenLabs voice "
                         "changer (speech-to-speech) and write --out-wav")
    ap.add_argument("--in-wav", dest="in_wav", default=None,
                    help="Input audio file for --voice-change")
    ap.add_argument("--sts-model", dest="sts_model",
                    default="eleven_multilingual_sts_v2",
                    help="ElevenLabs speech-to-speech model for "
                         "--voice-change")
    ap.add_argument("--regen-chunk", dest="regen_chunk", action="store_true",
                    help="Regenerate ONE chunk: synthesize --text-file with "
                         "ElevenLabs and write --out-wav (no other stages)")
    ap.add_argument("--text-file", dest="text_file", default=None,
                    help="UTF-8 chunk text file for --regen-chunk (Indic "
                         "text never travels on argv)")
    ap.add_argument("--out-wav", dest="out_wav", default=None,
                    help="Output WAV path for --regen-chunk")
    ap.add_argument("--sync-mode", dest="sync_mode", default=None,
                    choices=["match", "legacy"],
                    help="v0.7 chunk-placement mode: 'match' (default) = "
                         "Gemini section matching + Auto-Sync-style "
                         "placement with Un sync statuses; 'legacy' = the "
                         "old whole-script TTS + re-transcription path")
    ap.add_argument("--chunk-mode", dest="chunk_mode", default=None,
                    choices=["clause", "sentence", "section"],
                    help="Piece size for match mode: 'clause' (default) = "
                         "sentences, long ones subdivided at ; : , or a "
                         "dash; 'sentence' = one piece per sentence; "
                         "'section' = one piece per matched thought")
    ap.add_argument("--emotion", dest="emotion", action="store_true",
                    default=None,
                    help="Force Step-4 emotion enrichment ON before TTS")
    ap.add_argument("--no-emotion", dest="emotion", action="store_false",
                    help="Skip Step-4 emotion enrichment before TTS")
    ap.add_argument("--test-llm", dest="test_llm", action="store_true",
                    help="Make one tiny LLM call on the configured provider "
                         "and write a {status, provider, model, reply} "
                         "manifest")
    ap.add_argument("--list-voices", dest="list_voices", action="store_true",
                    help="Fetch the ElevenLabs voice catalogue for "
                         "--language and write a {status, voices} manifest")
    ap.add_argument("--recommend-voice", dest="recommend_voice",
                    action="store_true",
                    help="v0.19: recommend voices for one line (request in "
                         "--text-file). Forwarded to dub_engine.py.")
    ap.add_argument("--learn-final", dest="learn_final",
                    action="store_true",
                    help="v0.21: AI mode learns from the final dub (the "
                         "timeline after every regeneration; request in "
                         "--text-file, UTF-8). Forwarded to dub_engine.py.")
    ap.add_argument("--review-assist", dest="review_assist",
                    action="store_true",
                    help="v0.18.6: review-screen assistant for one line "
                         "(request in --text-file, UTF-8). Forwarded to "
                         "dub_engine.py.")
    ap.add_argument("--suggest-fit", dest="suggest_fit", action="store_true",
                    help="v0.18: shorter renderings for the review screen's "
                         "too-long rows (rows in --text-file, UTF-8). "
                         "Forwarded to dub_engine.py.")
    ap.add_argument("--status-dir", dest="status_dir", default=None,
                    help="Per-run status directory (log/pid/done/manifest). "
                         "Must live inside engine/status/. The panel passes "
                         "one per REAPER project so concurrent runs from "
                         "two REAPER instances never clobber each other. "
                         "Default: engine/status itself.")
    args = ap.parse_args(argv)

    # Mirror dub_engine.py's mode validation here so a bad launch dies with
    # a clear argparse message instead of deep inside the detached child.
    if args.review_assist or args.recommend_voice or args.learn_final:
        if not args.language or not args.text_file:
            ap.error("--review-assist/--recommend-voice/--learn-final "
                     "require --language and --text-file")
    elif args.suggest_fit:
        if args.test_llm or args.regen_chunk or args.list_voices \
           or args.voice_change:
            ap.error("--suggest-fit cannot be combined with other modes")
        if not args.language:
            ap.error("--suggest-fit requires --language")
        if not args.text_file:
            ap.error("--suggest-fit requires --text-file <utf-8 rows file>")
    elif args.test_llm:
        if args.regen_chunk or args.list_voices or args.voice_change:
            ap.error("--test-llm cannot be combined with other modes")
    elif args.list_voices:
        if args.regen_chunk or args.voice_change:
            ap.error("--list-voices cannot be combined with other modes")
        if not args.language:
            ap.error("--list-voices requires --language")
    elif args.voice_change:
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
    elif args.regen_chunk:
        if not args.language:
            ap.error("--regen-chunk requires --language")
        if not args.text_file:
            ap.error("--regen-chunk requires --text-file <utf-8 chunk text>")
        if not args.out_wav:
            ap.error("--regen-chunk requires --out-wav <output wav path>")
        if args.script or args.provided_script:
            ap.error("--script/--provided-script are not valid with "
                     "--regen-chunk")
    else:
        if not args.audio:
            ap.error("--audio is required unless "
                     "--regen-chunk/--test-llm/--list-voices/--voice-change")
        if not args.language:
            ap.error("--language is required unless --test-llm")
        if args.steps == "dub" and not args.script:
            ap.error("--steps dub requires --script <translation text "
                     "file> — run '--steps translate' first, review/edit "
                     "its translation_text file, then pass that file here")
        if args.script and args.steps != "dub":
            ap.error("--script is only valid with --steps dub")
        if args.provided_script and args.steps == "dub":
            ap.error("--provided-script is only valid with --steps "
                     "translate/full/plan")
        # Pause-aware Preview sync — same rules as dub_engine._parse_args.
        if args.steps == "plan" and not (args.provided_script or args.plan):
            ap.error("--steps plan requires --provided-script <utf-8 target "
                     "script> or --plan <sync plan file>")
        if args.steps == "dubplan" and not args.plan:
            ap.error("--steps dubplan requires --plan <sync plan file> — run "
                     "'--steps plan' first")
        if args.plan and args.steps not in ("plan", "dubplan"):
            ap.error("--plan is only valid with --steps plan/dubplan")
        if args.provided_script and args.steps == "dubplan":
            ap.error("--provided-script is not valid with --steps dubplan "
                     "(the approved plan already carries the target text)")
        if args.text_file or args.out_wav or args.in_wav:
            ap.error("--text-file/--out-wav/--in-wav are only valid with "
                     "--regen-chunk / --voice-change")

    # Resolve the status directory (must stay inside engine/status/).
    status_dir = _resolve_status_dir(args.status_dir)
    if status_dir is None:
        ap.error(f"--status-dir must be {os.path.abspath(STATUS_DIR)} or a "
                 f"subdirectory of it (got: {os.path.abspath(args.status_dir)})")
    return args, status_dir


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    # Launcher-level argument errors must reach the panel: under pythonw
    # stderr is gone, so write them to the status dir with done markers.
    try:
        args, status_dir = _parse_args(argv)
    except _LaunchArgError as e:
        return _report_arg_error(argv, str(e))

    _clean_status_dir(status_dir)

    log_path = os.path.join(status_dir, "engine_log.txt")
    pid_path = os.path.join(status_dir, "engine_pid.txt")
    done_path = os.path.join(status_dir, "engine_done.txt")

    # Force UTF-8 in the worker: the pipeline prints Indic text and unicode
    # symbols, and on Windows a redirected stdout defaults to the legacy ANSI
    # code page, which would raise UnicodeEncodeError and kill the run.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Tell dub_engine.py where to write its engine_done.json manifest — it
    # must land in the same per-run dir as this launcher's log/pid/done.
    env["DUB_STATUS_DIR"] = status_dir

    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    exit_code = 1
    proc = None
    try:
        engine_script = os.path.join(ENGINE_DIR, "dub_engine.py")
        if not os.path.exists(engine_script):
            log_file.write(f"[run_dub] ERROR: dub_engine.py not found at "
                           f"{engine_script}\n")
            log_file.flush()
            return 1

        cmd = [_worker_python(), "-u", engine_script,
               "--el-model", args.el_model]
        if args.language:
            cmd += ["--language", args.language]
        if args.app_dir:
            # Deprecated: forwarded so the engine logs its own warning and
            # legacy panel launch commands keep working unchanged.
            log_file.write("[run_dub] note: --app-dir is deprecated (v0.3) "
                           "and ignored by the standalone engine.\n")
            cmd += ["--app-dir", args.app_dir]
        if args.learn_final:
            cmd += ["--learn-final", "--text-file", args.text_file]
        elif args.recommend_voice:
            cmd += ["--recommend-voice", "--text-file", args.text_file]
        elif args.review_assist:
            cmd += ["--review-assist", "--text-file", args.text_file]
            if args.script_source == "ai_test":     # v0.24: house rules
                cmd += ["--script-source", "ai_test"]
        elif args.suggest_fit:
            cmd += ["--suggest-fit", "--text-file", args.text_file]
            if args.script_source == "ai_test":     # v0.24: house rules
                cmd += ["--script-source", "ai_test"]
        elif args.test_llm:
            cmd.append("--test-llm")
        elif args.list_voices:
            cmd.append("--list-voices")
        elif args.voice_change:
            cmd += ["--voice-change",
                    "--in-wav", args.in_wav,
                    "--out-wav", args.out_wav,
                    "--sts-model", args.sts_model]
        elif args.regen_chunk:
            cmd += ["--regen-chunk",
                    "--text-file", args.text_file,
                    "--out-wav", args.out_wav]
        else:
            cmd += ["--audio", args.audio, "--steps", args.steps]
            if args.steps == "dub":
                cmd += ["--script", args.script]
            if args.provided_script:
                cmd += ["--provided-script", args.provided_script]
            if args.plan:
                cmd += ["--plan", args.plan]
            if args.script_source in ("ai", "ai_test", "eleven"):
                cmd += ["--script-source", args.script_source]
            if args.sync_mode:
                cmd += ["--sync-mode", args.sync_mode]
            if args.chunk_mode:
                cmd += ["--chunk-mode", args.chunk_mode]
        if args.voice_id and args.voice_id.strip():
            cmd += ["--voice-id", args.voice_id.strip()]
        # Tri-state emotion pass-through: only forward an explicit choice so
        # the engine's own default resolution (settings file, then ON) holds.
        if args.emotion is True:
            cmd.append("--emotion")
        elif args.emotion is False:
            cmd.append("--no-emotion")

        log_file.write("[run_dub] launching: "
                       + " ".join(f'"{c}"' if " " in c else c for c in cmd)
                       + "\n")
        log_file.flush()

        popen_kwargs = {}
        if os.name == "nt":
            # No visible console window when launched detached from REAPER.
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            # New session so Cancel can signal the whole worker tree.
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(
            cmd,
            cwd=ENGINE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,                      # line-buffered text pipe
            **popen_kwargs,
        )

        # Publish the CHILD pid immediately (the launcher's own pid is
        # useless for Cancel — killing it would orphan the worker).
        try:
            with open(pid_path, "w", encoding="utf-8") as pf:
                pf.write(str(proc.pid))
        except Exception:
            pass

        # Tee: every worker line goes to the log file (flushed per line so
        # the Lua poller sees it live) and, best-effort, to our own stdout.
        # stdout may be missing entirely (sys.stdout is None under pythonw.exe,
        # which the panel uses so no console window opens) or an invalid handle
        # when REAPER starts us detached, so the echo must never be allowed to
        # crash the tee loop.
        echo = sys.stdout
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            if echo is not None:
                try:
                    echo.write(line)
                    echo.flush()
                except Exception:
                    echo = None      # broken once, broken for the whole run

        proc.wait()
        exit_code = proc.returncode
    except Exception:
        try:
            log_file.write("\n[run_dub] launcher crashed:\n")
            log_file.write(traceback.format_exc())
            log_file.flush()
        except Exception:
            pass
        exit_code = 1
    finally:
        try:
            log_file.close()
        except Exception:
            pass
        # Worker is gone — drop the stale pid file so a later Cancel cannot
        # signal an unrelated, recycled PID.
        try:
            os.remove(pid_path)
        except OSError:
            pass
        # The done marker is written LAST — it is the poller's only signal
        # that log + manifest are complete.
        try:
            with open(done_path, "w", encoding="utf-8") as df:
                df.write(str(exit_code))
        except Exception:
            pass

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
