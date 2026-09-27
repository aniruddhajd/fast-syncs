"""
ElevenLabs Dubbing Studio (v0.22) — --script-source eleven
==========================================================
ElevenLabs' own dubbing pipeline instead of ours: it transcribes, splits
speakers, translates and voices the English itself. Plain automatic dubbing
has NO voice parameter (it clones the original speaker), so this module always
creates the dub in STUDIO mode (dubbing_studio=true). Studio exposes the dub
as an editable "resource" — speakers and timed segments — and lets us:

  1. read every segment (English text + translated text + times),
  2. push the reviewer's edited text back per segment,
  3. set a voice per detected speaker (the user's cast),
  4. re-dub the segments and render one full-length, time-aligned audio file.

Two stages, matching the app's review pause:
  translate  create_dub -> wait_dub -> get_resource -> parse_resource
  dub        update_segment_text / update_speaker_voice -> redub ->
             wait_redub -> render -> wait_render -> download

Everything ElevenLabs-specific lives here. Dubbing Studio is ElevenLabs'
"Dubbing v1" product (maintenance mode), so the JSON is parsed defensively:
unknown shapes raise a clear RuntimeError naming what was missing rather than
producing a silently wrong dub.

Every request goes through config._urlopen (verified TLS first). The API key
is only ever sent to *.elevenlabs.io — render downloads come from signed
storage URLs and never see it.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

from .config import _urlopen
from .stt import _multipart_body

API_BASE = "https://api.elevenlabs.io"
_POLL_SECS = 8.0
_HEARTBEAT_SECS = 30.0

StatusCb = Optional[Callable[[str], None]]


def _say(cb: StatusCb, msg: str) -> None:
    if cb:
        try:
            cb(msg)
        except Exception:
            pass


def el_lang_code(lang_entry: dict) -> str:
    """ISO 639 code for ElevenLabs from a TTS_LANGUAGES entry: 'bn-IN' -> 'bn',
    'ne-NP' -> 'ne', 'kok-IN' -> 'kok'."""
    code = str((lang_entry or {}).get("code") or "").strip()
    return code.split("-", 1)[0].lower() if code else ""


# ─── HTTP ────────────────────────────────────────────────────────────────────

def _is_el_host(url: str) -> bool:
    host = urllib.parse.urlsplit(url).netloc.lower().split(":", 1)[0]
    return host == "elevenlabs.io" or host.endswith(".elevenlabs.io")


def _http(method: str, url: str, api_key: str, body: Optional[bytes] = None,
          content_type: Optional[str] = None, timeout: int = 120) -> bytes:
    headers = {"Accept": "application/json"}
    if _is_el_host(url):
        headers["xi-api-key"] = api_key
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body, headers=headers,
                                 method=method)
    try:
        with _urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:800]
        except Exception:
            pass
        path = urllib.parse.urlsplit(url).path
        raise RuntimeError(f"ElevenLabs {method} {path} failed: HTTP "
                           f"{e.code} {detail}".strip())


def _json_call(method: str, path: str, api_key: str,
               payload: Optional[dict] = None, timeout: int = 120):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    raw = _http(method, API_BASE + path, api_key, body,
                "application/json" if body is not None else None, timeout)
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError(f"ElevenLabs {method} {path}: reply is not JSON "
                           f"({raw[:200]!r})")


# ─── Stage 1: create + wait + read ───────────────────────────────────────────

def create_dub(audio_path: str, target_lang: str, api_key: str,
               num_speakers: int = 0, drop_background: bool = True,
               disable_cloning: bool = False, name: str = "",
               status_cb: StatusCb = None) -> Tuple[str, float]:
    """POST /v1/dubbing in studio mode. Returns (dubbing_id, expected_s)."""
    with open(audio_path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"
    fields = [("source_lang", "en"), ("target_lang", target_lang),
              ("num_speakers", str(int(num_speakers))),
              ("dubbing_studio", "true"), ("watermark", "false"),
              ("drop_background_audio",
               "true" if drop_background else "false"),
              ("name", name or os.path.basename(audio_path))]
    if disable_cloning:
        fields.append(("disable_voice_cloning", "true"))
    body, boundary = _multipart_body(
        fields, [("file", os.path.basename(audio_path), mime, data)])
    _say(status_cb, f"Uploading {os.path.basename(audio_path)} "
                    f"({len(data) / 1e6:.1f} MB) to ElevenLabs Dubbing "
                    f"(studio, target '{target_lang}')…")
    try:
        raw = _http("POST", API_BASE + "/v1/dubbing", api_key, body,
                    f"multipart/form-data; boundary={boundary}", timeout=900)
    except RuntimeError as e:
        # Seen live: Dubbing Studio refuses 'mr' although automatic dubbing
        # lists Marathi. Say it plainly — nothing was charged.
        if "unsupported_target_language" in str(e):
            raise RuntimeError(
                f"ElevenLabs Dubbing Studio does not support target language "
                f"'{target_lang}' (nothing was charged). Voice selection needs "
                "Studio, so this language cannot use ElevenLabs Dub — switch "
                "Script to 'Prompt chain' or 'AI · learns'. ElevenLabs said: "
                + str(e))
        raise
    try:
        reply = json.loads(raw.decode("utf-8"))
    except Exception:
        reply = {}
    dub_id = str(reply.get("dubbing_id") or "").strip()
    if not dub_id:
        raise RuntimeError(f"ElevenLabs Dubbing returned no dubbing_id: "
                           f"{raw[:300]!r}")
    return dub_id, float(reply.get("expected_duration_sec") or 0.0)


def get_dub(dub_id: str, api_key: str) -> dict:
    return _json_call("GET", f"/v1/dubbing/{dub_id}", api_key)


def wait_dub(dub_id: str, api_key: str, expected_s: float = 0.0,
             status_cb: StatusCb = None, what: str = "dubbing",
             sleep=time.sleep) -> dict:
    """Poll GET /v1/dubbing/{id} until 'dubbed' (ok) or 'failed' (raise)."""
    limit = max(600.0, 4.0 * float(expected_s or 0.0))
    t0 = time.monotonic()
    last_beat = -_HEARTBEAT_SECS
    while True:
        info = get_dub(dub_id, api_key)
        status = str(info.get("status") or "").lower()
        if status == "dubbed":
            return info
        if status == "failed":
            raise RuntimeError(f"ElevenLabs {what} failed: "
                               f"{info.get('error') or 'no reason given'}")
        waited = time.monotonic() - t0
        if waited > limit:
            raise RuntimeError(f"ElevenLabs {what} still '{status}' after "
                               f"{waited:.0f}s — gave up (dubbing_id "
                               f"{dub_id}; it may still finish on "
                               "elevenlabs.io).")
        if waited - last_beat >= _HEARTBEAT_SECS:
            last_beat = waited
            _say(status_cb, f"ElevenLabs {what}… {waited:.0f}s "
                            f"(status '{status or '?'}'"
                            + (f", expected ~{expected_s:.0f}s"
                               if expected_s else "") + ")")
        sleep(_POLL_SECS)


def get_resource(dub_id: str, api_key: str) -> dict:
    return _json_call("GET", f"/v1/dubbing/resource/{dub_id}", api_key)


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_resource(res: dict, lang: str) -> Tuple[List[dict], List[dict]]:
    """Studio resource -> (speakers, segments), segments sorted by time.

    speakers: [{"el_id", "name", "voice_id"}]
    segments: [{"id", "start", "end", "speaker", "en", "tr"}] — start/end in
              seconds on the SOURCE timeline (the dub's own times when the
              segment carries them for *lang*).
    """
    tracks = res.get("speaker_tracks") or {}
    segs = res.get("speaker_segments") or {}
    if not isinstance(tracks, dict) or not isinstance(segs, dict) or not segs:
        raise RuntimeError("ElevenLabs dubbing resource has no speaker "
                           "segments (keys: " + ", ".join(sorted(res)) + ")")
    speakers, seg_speaker = [], {}
    for n, (tid, t) in enumerate(tracks.items(), 1):
        t = t or {}
        el_id = str(t.get("id") or tid)
        voices = t.get("voices") or {}
        speakers.append({"el_id": el_id,
                         "name": str(t.get("speaker_name") or f"Speaker {n}"),
                         "voice_id": str(voices.get(lang) or "")
                         if isinstance(voices, dict) else ""})
        for sid in t.get("segments") or []:
            seg_speaker[str(sid)] = el_id
    out = []
    for sid, s in segs.items():
        s = s or {}
        sid = str(s.get("id") or sid)
        dub = (s.get("dubs") or {}).get(lang) or {}
        start = _f(dub.get("start_time"))
        end = _f(dub.get("end_time"))
        if start is None or end is None:
            start, end = _f(s.get("start_time")), _f(s.get("end_time"))
        if start is None or end is None:
            raise RuntimeError(f"ElevenLabs segment {sid} has no times")
        out.append({"id": sid, "start": start, "end": max(start, end),
                    "speaker": seg_speaker.get(sid, str(s.get("speaker_id")
                                                        or "")),
                    "en": " ".join(str(s.get("text") or "").split()),
                    "tr": " ".join(str(dub.get("text") or "").split())})
    out.sort(key=lambda r: (r["start"], r["end"]))
    return speakers, out


# ─── Stage 2: edit + voices + re-dub + render ────────────────────────────────

def update_segment_text(dub_id: str, seg_id: str, lang: str, text: str,
                        api_key: str) -> None:
    _json_call("PATCH", f"/v1/dubbing/resource/{dub_id}/segment/{seg_id}/"
                        f"{lang}", api_key, {"text": text})


def update_speaker_voice(dub_id: str, speaker_id: str, voice_id: str,
                         lang: str, api_key: str) -> None:
    _json_call("PATCH", f"/v1/dubbing/resource/{dub_id}/speaker/{speaker_id}",
               api_key, {"voice_id": voice_id, "languages": [lang]})


def redub(dub_id: str, seg_ids: List[str], lang: str, api_key: str) -> None:
    _json_call("POST", f"/v1/dubbing/resource/{dub_id}/dub", api_key,
               {"segments": list(seg_ids), "languages": [lang]})


def wait_redub(dub_id: str, seg_ids: List[str], lang: str, api_key: str,
               status_cb: StatusCb = None, sleep=time.sleep,
               limit_s: float = 1800.0) -> dict:
    """Poll the resource until none of *seg_ids* has stale audio for *lang*."""
    want = set(seg_ids)
    t0 = time.monotonic()
    last_beat = -_HEARTBEAT_SECS
    while True:
        res = get_resource(dub_id, api_key)
        segs = res.get("speaker_segments") or {}
        stale = 0
        for sid, s in segs.items():
            if str((s or {}).get("id") or sid) not in want:
                continue
            dub = ((s or {}).get("dubs") or {}).get(lang) or {}
            if dub.get("audio_stale", False):
                stale += 1
        if not stale:
            return res
        waited = time.monotonic() - t0
        if waited > limit_s:
            raise RuntimeError(f"ElevenLabs re-dub: {stale} segment(s) still "
                               f"not voiced after {waited:.0f}s")
        if waited - last_beat >= _HEARTBEAT_SECS:
            last_beat = waited
            _say(status_cb, f"ElevenLabs re-voicing… {waited:.0f}s "
                            f"({len(want) - stale}/{len(want)} segment(s) "
                            "done)")
        sleep(_POLL_SECS)


def render(dub_id: str, lang: str, api_key: str,
           render_type: str = "wav") -> str:
    """POST .../render/{lang}. Returns the render_id ('' if not given)."""
    reply = _json_call("POST", f"/v1/dubbing/resource/{dub_id}/render/{lang}",
                       api_key, {"render_type": render_type})
    return str(reply.get("render_id") or "")


def wait_render(dub_id: str, render_id: str, lang: str, api_key: str,
                status_cb: StatusCb = None, sleep=time.sleep,
                limit_s: float = 1800.0) -> str:
    """Poll the resource's renders until this render is complete. Returns its
    download URL ('' when the resource never names one — the caller then
    falls back to GET /v1/dubbing/{id}/audio/{lang})."""
    t0 = time.monotonic()
    last_beat = -_HEARTBEAT_SECS
    while True:
        res = get_resource(dub_id, api_key)
        renders = res.get("renders") or {}
        r = renders.get(render_id) if render_id else None
        if r is None and not render_id:
            # No id returned: take the newest render for this language.
            cands = [v for v in renders.values()
                     if (v or {}).get("language") == lang]
            r = cands[-1] if cands else None
        status = str((r or {}).get("status") or "").lower()
        if status in ("complete", "completed", "done", "ready"):
            ref = (r or {}).get("media_ref") or {}
            return str(ref.get("url") or "")
        if status in ("failed", "error"):
            raise RuntimeError(f"ElevenLabs render failed ({render_id})")
        waited = time.monotonic() - t0
        if waited > limit_s:
            raise RuntimeError(f"ElevenLabs render still '{status}' after "
                               f"{waited:.0f}s")
        if waited - last_beat >= _HEARTBEAT_SECS:
            last_beat = waited
            _say(status_cb, f"ElevenLabs rendering… {waited:.0f}s "
                            f"(status '{status or '?'}')")
        sleep(_POLL_SECS)


def download(url: str, out_path: str, api_key: str) -> str:
    data = _http("GET", url, api_key, timeout=900)
    if not data:
        raise RuntimeError("ElevenLabs download returned no data")
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def download_dubbed_audio(dub_id: str, lang: str, out_path: str,
                          api_key: str) -> str:
    """Fallback: GET /v1/dubbing/{id}/audio/{lang} (mp3 or mp4 bytes)."""
    return download(f"{API_BASE}/v1/dubbing/{dub_id}/audio/{lang}", out_path,
                    api_key)


# ─── Cast -> speaker voices ──────────────────────────────────────────────────

def speaker_voices(segments: List[dict], para_voice: Dict[int, str],
                   default_voice: str) -> Tuple[Dict[str, str], List[str]]:
    """{el_speaker_id: voice_id} from the paragraph cast (1-based paragraph =
    segment order). A speaker takes the voice most of its segments were cast
    to; ties go to the earliest segment. Returns (map, warnings)."""
    tally: Dict[str, Dict[str, int]] = {}
    first: Dict[str, Dict[str, int]] = {}
    for n, seg in enumerate(segments, 1):
        spk = seg.get("speaker") or ""
        v = para_voice.get(n) or default_voice
        if not spk or not v:
            continue
        tally.setdefault(spk, {})
        tally[spk][v] = tally[spk].get(v, 0) + 1
        first.setdefault(spk, {}).setdefault(v, n)
    out, warns = {}, []
    for spk, counts in tally.items():
        best = sorted(counts, key=lambda v: (-counts[v], first[spk][v]))[0]
        out[spk] = best
        if len(counts) > 1:
            warns.append(f"speaker {spk}: segments were cast to {len(counts)} "
                         f"voices — ElevenLabs sets ONE voice per speaker, "
                         f"using the majority ({best})")
    return out, warns
