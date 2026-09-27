#!/usr/bin/env python3
"""
Offline sync-accuracy eval for AI-mode anchor sync (v0.20)
==========================================================
Replays stored real runs through the ENGINE'S OWN anchor-sync stage
(dub_engine._stage_dub / _stage_translate) with every paid call replaced:

  * Scribe      -> the cached transcript in dubbing/data/stt_cache (matched
                   to the run by its words; reading the cache is free)
  * ElevenLabs  -> fake TTS: one tone per unit, sized by
                   config.estimate_duration (the engine's own speaking rate)
  * LLM         -> never called for the dub; the "sim" variants use a fake
                   translator that returns the run's APPROVED paragraphs
                   (with or without simulated ' | ' pause markers)

Nothing is written next to the run: inputs are copied to a temp folder, and
AI_LEARNING_DIR / TRANSLATION_MEMORY_DB / DUB_STATUS_DIR point there too.

Variants (each one only when the engine has the code for it):
  baseline  dub resume on the run's own files: English cues = <base>_sync_en.srt
            (loudness regions), rows = the saved AI draft.
  ab        A+B: the same dub resume, but the English cues are the v0.20
            word-timestamp phrases (+ spectral pauses) the translate stage now
            writes to <base>_ai_phrases.srt.
  ab_sim    translate + dub offline on the phrases, fake translator, NO markers
            (control for c_sim: same paragraphs and windows, no pause markers).
  c_sim     as ab_sim, but the fake translator puts ' | ' where the English
            phrases change — what the pause-aware prompt asks the model for.
            Marker positions are SIMULATED (char-share of the English phrase
            lengths, snapped to a word gap), so this measures the mechanism,
            not a real model's marker quality.

Metrics (synced pieces, after the engine's own sync check):
  phrases / max_s   English cues used and the longest one
  start mean/max    |placed start - English window start|, ms
  end mean          |placed end - English window end|, ms
  within300         % pieces whose start is within 300 ms
  iou               speech-overlap IoU: English speech (Scribe word spans,
                    gaps < 0.2 s merged) vs the placed dub timeline, 10 ms grid

Usage:
  python eval_sync.py [--run DIR ...] [--out DIR] [--variants baseline,ab,...]
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(os.path.dirname(ENGINE_DIR), "data")
STT_CACHE = os.path.join(DATA_DIR, "stt_cache")

DEFAULT_RUNS = [
    r"C:/Users/Aniruddha/OneDrive/Desktop/temp/ripper_test/1/All Dialogue",
    r"C:/Users/Aniruddha/OneDrive/Desktop/temp/ripper_test/3/New folder/All Dialogue",
]
ALL_VARIANTS = ["baseline", "ab", "ab_sim", "c_sim"]


# ─── helpers ────────────────────────────────────────────────────────────────

def _nk(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def _srt_plain(text):
    return " ".join(ln for ln in text.splitlines()
                    if ln.strip() and "-->" not in ln
                    and not ln.strip().isdigit())


def find_words(srt_text):
    """Cached Scribe payload whose words spell exactly this run's English."""
    target = _nk(_srt_plain(srt_text))
    for f in sorted(glob.glob(os.path.join(STT_CACHE, "*.json"))):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        ws = [w for w in d.get("words") or [] if w.get("type", "word") == "word"]
        if ws and _nk(" ".join(w.get("text", "") for w in ws)) == target:
            return f, d
    return None, None


def find_base(run):
    for p in sorted(glob.glob(os.path.join(run, "*_sync_en.srt"))):
        return os.path.basename(p)[:-len("_sync_en.srt")]
    raise RuntimeError(f"no *_sync_en.srt in {run}")


def _label(run):
    """Short run name for the table: the part after ripper_test/ when
    present (e.g. '1', '3/New folder'), else the parent folder's name."""
    norm = run.replace("\\", "/").rstrip("/")
    if "ripper_test/" in norm:
        return os.path.dirname(norm.split("ripper_test/", 1)[1]) or norm
    return os.path.basename(os.path.dirname(norm)) or norm


def find_wav(run, base):
    for d in (run, os.path.dirname(run)):
        p = os.path.join(d, base + ".wav")
        if os.path.isfile(p):
            return p
    return None


def speech_timeline(words, merge_gap=0.2):
    iv = sorted((float(w["start"]), float(w["end"])) for w in words
                if w.get("type", "word") == "word" and w.get("text", "").strip()
                and float(w.get("end", 0)) > float(w.get("start", 0)))
    out = []
    for s, e in iv:
        if out and s - out[-1][1] < merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def iou(a, b, step=0.01):
    if not a and not b:
        return 0.0
    end = max([e for _s, e in a] + [e for _s, e in b] + [0.0])
    n = int(end / step) + 2
    A = bytearray(n)
    B = bytearray(n)
    for s, e in a:
        for k in range(int(s / step), min(n, int(e / step))):
            A[k] = 1
    for s, e in b:
        for k in range(max(0, int(s / step)), min(n, int(e / step))):
            B[k] = 1
    inter = sum(1 for k in range(n) if A[k] and B[k])
    union = sum(1 for k in range(n) if A[k] or B[k])
    return inter / union if union else 0.0


# ─── offline engine ─────────────────────────────────────────────────────────

class Offline:
    """Imports the engine with every paid path replaced."""

    def __init__(self, root):
        self.root = root
        os.environ["AI_LEARNING_DIR"] = os.path.join(root, "ai_learning")
        os.environ["TRANSLATION_MEMORY_DB"] = os.path.join(root, "tm.db")
        os.environ["DUB_STATUS_DIR"] = os.path.join(root, "status")
        if ENGINE_DIR not in sys.path:
            sys.path.insert(0, ENGINE_DIR)
        import dub_engine as de                      # noqa: E402
        self.de = de
        self.pl = de._import_pipeline()
        from pipeline import llm, ai_translator, ai_agents
        self._mods = (llm, ai_translator, ai_agents)

        def _no_llm(*_a, **_k):
            raise RuntimeError("eval_sync: LLM call blocked (offline)")
        for m in self._mods:
            m._llm_generate = _no_llm
        self.pl._llm_generate = _no_llm

        # AI mode must never read a prompt file (CONTRACT v0.18.3).
        def _no_prompt(*_a, **_k):
            raise RuntimeError("eval_sync: AI mode read a prompt file")
        llm._load_lang_prompt = _no_prompt
        self.pl._load_lang_prompt = _no_prompt
        self.pl.fit_to_seconds = lambda *a, **k: ""
        self.pl.synthesize_sentences_elevenlabs = self._fake_tts
        self.language = "Marathi"

        def _no_match(*_a, **_k):
            raise RuntimeError("eval_sync: anchor sync fell back to match "
                               "sync, which needs an LLM")
        de._stage_dub_match = _no_match
        self.captured = None
        real_check = de._run_sync_check

        def _capture(pl, ctx, pieces, placed, durations):
            out = real_check(pl, ctx, pieces, placed, durations)
            self.captured = ([dict(p) for p in pieces], out, list(durations))
            return out
        de._run_sync_check = _capture

    def _fake_tts(self, sentences, output_path, api_key=None, voice_id=None,
                  model_id=None, status_cb=None, voices=None):
        from pydub import AudioSegment
        from pydub.generators import Sine
        gap = AudioSegment.silent(duration=60, frame_rate=22050)
        audio, spans = AudioSegment.silent(duration=0, frame_rate=22050), []
        for t in sentences:
            ms = max(200, int(1000 * self.pl.estimate_duration(t, self.language)))
            tone = Sine(220, sample_rate=22050).to_audio_segment(
                duration=ms, volume=-14).set_channels(1)
            if len(audio):
                audio += gap
            s = len(audio)
            audio += tone
            spans.append((s, len(audio)))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio.export(output_path, format="wav")
        return output_path, spans

    def settings(self, extra):
        """A private engine_settings.json: the user's piece size, and no
        proofer / audit (they would only call the blocked LLM)."""
        src = os.path.join(ENGINE_DIR, "engine_settings.json")
        data = {}
        try:
            data = json.loads(_read(src))
        except Exception:
            pass
        keep = {k: data[k] for k in ("chunk_mode", "max_atempo",
                                     "sync_tolerance_ms") if k in data}
        keep.update({"ai_proofer": 0, "ai_audit_rate": 0})
        keep.update(extra or {})
        p = os.path.join(self.root, "engine_settings.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(keep, f)
        self.de.ENGINE_SETTINGS_FILE = p


def _args(language):
    return types.SimpleNamespace(language=language, script_source="ai",
                                 chunk_mode=None, emotion=False,
                                 el_model="eleven_v3", sync_mode=None,
                                 provided_script=None)


def _fake_translator(paras, groups, markers):
    """An LLM stand-in for ai_translate_script: returns the approved
    paragraphs, each claiming its phrase cues. With *markers*, ' | ' goes at
    the word gap nearest each English phrase boundary (char share)."""
    def _mark(text, cue_texts):
        if not markers or len(cue_texts) < 2:
            return text
        tot = float(sum(len(c) for c in cue_texts)) or 1.0
        spaces = [i for i, ch in enumerate(text) if ch == " "]
        cuts, acc, last = [], 0.0, -1
        for c in cue_texts[:-1]:
            acc += len(c)
            want = acc / tot * len(text)
            cands = [i for i in spaces if i > last]
            if not cands:
                break
            best = min(cands, key=lambda i: abs(i - want)
                       - (6 if i > 0 and text[i - 1] in ",;:.!?।—" else 0))
            cuts.append(best)
            last = best
        out, prev = [], 0
        for i in cuts:
            out.append(text[prev:i])
            prev = i + 1
        out.append(text[prev:])
        return " | ".join(s for s in out if s)

    def _gen(prompt, model=None, **_k):
        passages = []
        for blk in re.split(r"### PASSAGE ", prompt)[1:]:
            n = int(re.match(r"(\d+)", blk).group(1))
            ids = [int(x) for x in re.findall(r"^\[(\d+)\] @", blk, re.M)]
            idset, out = set(ids), []
            for text, cids, ctexts in zip(paras, groups["ids"], groups["texts"]):
                if cids and cids[0] in idset:
                    out.append({"cues": [c for c in cids if c in idset],
                                "text": _mark(text, ctexts)})
            passages.append({"n": n, "paragraphs": out})
        return json.dumps({"passages": passages}, ensure_ascii=False)
    return _gen


# ─── one variant of one run ─────────────────────────────────────────────────

def run_variant(off, run, variant, out_root, overrides=None):
    de, pl = off.de, off.pl
    base_name = find_base(run)
    wav = find_wav(run, base_name)
    srt_text = _read(os.path.join(run, base_name + ".srt"))
    words_path, payload = find_words(srt_text)
    if payload is None:
        return {"error": "no matching Scribe transcript in stt_cache"}
    words = payload["words"]
    draft_p = os.path.join(run, base_name + "_ai_draft.json")
    draft = json.loads(_read(draft_p)) if os.path.isfile(draft_p) else {}
    language = draft.get("language") or "Marathi"
    off.language = language
    approved_p = next((p for p in (
        os.path.join(run, base_name + "_translation_edited.txt"),
        os.path.join(run, base_name + "_review_translation.txt"))
        if os.path.isfile(p)), None)
    script = _read(approved_p).strip()
    paras = pl._split_translation_paragraphs(script)

    work = os.path.join(out_root, re.sub(r"[^A-Za-z0-9]+", "_", _label(run))
                        + "_" + variant)
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    base = os.path.join(work, base_name)
    off.settings(overrides or {})
    args = _args(language)
    audio_path = wav or os.path.join(run, base_name + ".wav")
    ctx = {"audio_path": audio_path, "out_dir": work, "base": base,
           "script_text": script}
    if wav:
        y, sr = pl._load_audio_any(wav)
        ctx["en_audio_dur"] = len(y) / float(sr)
    else:
        y = sr = None
        ctx["en_audio_dur"] = max(float(w.get("end", 0)) for w in words) + 1.0
    note = "" if wav else "wav not found: no regions / spectral pauses"

    log = io.StringIO()
    off.captured = None
    t0 = time.time()
    with contextlib.redirect_stdout(log):
        if variant in ("baseline", "ab"):
            for suffix in ("_sync_en.srt", ".srt", "_ai_draft.json"):
                src = os.path.join(run, base_name + suffix)
                if os.path.isfile(src):
                    shutil.copy(src, base + suffix)
            ctx["en_srt_text"] = _read(base + "_sync_en.srt")
            if variant == "ab":
                if not hasattr(de, "_ai_phrase_cues"):
                    return {"skipped": "engine has no _ai_phrase_cues yet"}
                regions = (pl._detect_regions_from_audio(
                    y, sr, pl.DEFAULT_THR_DB, pl.DEFAULT_HYS_DB,
                    pl.DEFAULT_MIN_MS) if wav else [])
                cues = de._ai_phrase_cues(pl, words, regions, audio_path,
                                          y=y, sr=sr)
                de._write_text(base + de.AI_PHRASES_SUFFIX,
                               de._cues_to_srt(pl, cues))
                de._load_ai_phrases(ctx)          # what --steps dub does
            de._stage_dub(pl, args, "offline", {}, ctx, "offlinevoice0000")
        else:
            if not hasattr(de, "_ai_phrase_cues"):
                return {"skipped": "engine has no _ai_phrase_cues yet"}
            if not wav:
                return {"skipped": "sim variants need the wav (S1b)"}
            # Full offline translate: cached Scribe, temp out dir, fake LLM.
            pl._transcribe_audio = lambda *_a, **_k: payload
            pl._prepare_output_dir = lambda _p: work
            regions = (pl._detect_regions_from_audio(
                y, sr, pl.DEFAULT_THR_DB, pl.DEFAULT_HYS_DB,
                pl.DEFAULT_MIN_MS) if wav else [])
            cues = de._ai_phrase_cues(pl, words, regions, audio_path, y=y,
                                      sr=sr)
            groups = _para_groups(pl, draft, paras, cues)
            fake = _fake_translator(paras, groups, variant == "c_sim")
            for m in off._mods:
                if m.__name__.endswith("ai_translator"):
                    m._llm_generate = fake
            de._stage_translate(pl, args, "offline", {}, ctx)
            ctx["script_text"] = ctx["punc_result"]
            de._stage_dub(pl, args, "offline", {}, ctx, "offlinevoice0000")
            for m in off._mods:
                if m.__name__.endswith("ai_translator"):
                    m._llm_generate = pl._llm_generate
    elapsed = time.time() - t0
    with open(base + "_eval_log.txt", "w", encoding="utf-8") as f:
        f.write(log.getvalue())
    if off.captured is None:
        return {"error": "the stage placed nothing (see _eval_log.txt)"}
    pieces, placed, durs = off.captured
    en_srt = (ctx.get("ai_phrases_srt") or ctx.get("en_srt_text")
              or (_read(base + "_sync_en.srt")
                  if os.path.isfile(base + "_sync_en.srt") else ""))
    if variant in ("ab_sim", "c_sim") and os.path.isfile(
            base + de.AI_PHRASES_SUFFIX):
        en_srt = _read(base + de.AI_PHRASES_SUFFIX)
    cue_list = [c for c in pl._extract_srt_entries(en_srt) if c[2].strip()]
    starts, ends, dub_iv = [], [], []
    for p, pc, d in zip(pieces, placed, durs):
        if pc["status"] != "synced":
            continue
        dub_iv.append((pc["position"], pc["position"] + d))
        if p.get("win"):
            starts.append(abs(pc["position"] - p["win"][0]))
            ends.append(abs(pc["position"] + d - p["win"][1]))
    marked = sum(1 for ln in log.getvalue().splitlines()
                 if "pause marker" in ln)
    return {
        "phrases": len(cue_list),
        "max_phrase_s": round(max((e - s for s, e, _t in cue_list),
                                  default=0.0), 2),
        "pieces": len(pieces),
        "synced": len(dub_iv),
        "start_mean_ms": int(1000 * sum(starts) / len(starts)) if starts else 0,
        "start_max_ms": int(1000 * max(starts)) if starts else 0,
        "end_mean_ms": int(1000 * sum(ends) / len(ends)) if ends else 0,
        "within300_pct": round(100.0 * sum(1 for s in starts if s <= 0.3)
                               / len(starts), 1) if starts else 0.0,
        "iou": round(iou(speech_timeline(words), dub_iv), 3),
        "words_cache": os.path.basename(words_path),
        "wav": wav or "",
        "note": note,
        "marker_log_lines": marked,
        "seconds": round(elapsed, 1),
    }


def _para_groups(pl, draft, paras, cues):
    """Which phrase cues each approved paragraph covers: by the saved draft
    row windows when the paragraph count still matches (what the real
    translator grouped), else by char share (_pair_review_rows)."""
    rows = draft.get("rows") or []
    ids, texts = [[] for _ in paras], [[] for _ in paras]
    if rows and len(rows) == len(paras) and all(
            r.get("start") is not None for r in rows):
        for k, (s, e, t) in enumerate(cues, 1):
            mid = (s + e) / 2.0
            best = min(range(len(rows)), key=lambda i: (
                0 if rows[i]["start"] - 0.05 <= mid <= rows[i]["end"] + 0.05
                else 1, abs(mid - (rows[i]["start"] + rows[i]["end"]) / 2)))
            ids[best].append(k)
            texts[best].append(t)
        # Keep cue order monotonic across paragraphs.
        flat = [i for grp in ids for i in grp]
        if flat == sorted(flat) and all(ids):
            return {"ids": ids, "texts": texts}
        ids, texts = [[] for _ in paras], [[] for _ in paras]
    # _pair_review_rows groups consecutive cues; walk them back out by text.
    k = 0
    for i, (en, _tr, _s, _e) in enumerate(pl._pair_review_rows(cues, paras)):
        acc = ""
        while k < len(cues) and len(_nk(acc)) < len(_nk(en)):
            acc += cues[k][2]
            ids[i].append(k + 1)
            texts[i].append(cues[k][2])
            k += 1
    return {"ids": ids, "texts": texts}


# ─── main ───────────────────────────────────────────────────────────────────

COLS = [("phrases", "phr"), ("max_phrase_s", "max_s"), ("pieces", "pcs"),
        ("synced", "sync"), ("start_mean_ms", "st_mean"),
        ("start_max_ms", "st_max"), ("end_mean_ms", "end_mean"),
        ("within300_pct", "<=300%"), ("iou", "IoU")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="engine_settings.json override for the replay, e.g. "
                         "--set ai_phrase_max_s=4 (numbers only)")
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        k, _eq, v = kv.partition("=")
        overrides[k.strip()] = float(v) if "." in v else int(v)
    runs = a.run or DEFAULT_RUNS
    out_root = a.out or tempfile.mkdtemp(prefix="eval_sync_")
    os.makedirs(out_root, exist_ok=True)
    off = Offline(out_root)
    variants = [v for v in a.variants.split(",") if v in ALL_VARIANTS]
    results = {}
    for run in runs:
        if not os.path.isdir(run):
            print(f"SKIP {run}: not found")
            continue
        label = _label(run)
        results[label] = {}
        for v in variants:
            try:
                results[label][v] = run_variant(off, run, v, out_root,
                                                overrides)
            except Exception as e:                   # noqa: BLE001
                import traceback
                results[label][v] = {"error": f"{type(e).__name__}: {e}",
                                     "trace": traceback.format_exc()[-800:]}
    print(f"eval_sync — offline replay (fake TTS by estimate_duration); "
          f"work dir {out_root}")
    for label, res in results.items():
        print(f"\n== {label} ==")
        print("variant   " + " ".join(f"{h:>8}" for _k, h in COLS))
        for v, r in res.items():
            if "skipped" in r or "error" in r:
                print(f"{v:<9} " + (r.get("skipped") or ("ERROR " + r["error"])))
                continue
            print(f"{v:<9} " + " ".join(f"{r[k]:>8}" for k, _h in COLS))
        info = next((r for r in res.values() if "words_cache" in r), None)
        if info:
            print(f"  words: {info['words_cache']}  wav: "
                  f"{info['wav'] or 'NOT FOUND'}" + (f"  ({info['note']})"
                                                    if info["note"] else ""))
    with open(os.path.join(out_root, "eval_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
