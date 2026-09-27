# Reaper Dubbing App — Interface Contract

Goal: run the dubbing pipeline of the existing Tkinter app FROM INSIDE REAPER.
The original app is READ-ONLY — never modify or write anything inside
`APP_DIR`. As of v0.3 the project is STANDALONE (see v0.3 section): APP_DIR
is only a one-time extraction source and optional key-migration source, never
a runtime dependency.

## VERSION 0.17.1 — Launcher argument errors reach the panel

`run_dub.py` used to die in argparse (exit 2, usage on stderr) BEFORE it
created the status dir. Under pythonw stderr is lost, so a flag the launcher
did not know showed up only as the panel's 90 s "Python produced no output"
watchdog. Now any launcher-level argument error (unknown flag, bad choice,
mode validation) is written to the run's status dir:

- `--status-dir` is pre-scanned from raw argv (same rule: must be
  `engine/status` or inside it; `--status-dir X`, `--status-dir=X` and
  prefixes like `--sta` are recognised; absent = the shared root). An
  invalid or missing value writes nothing — stderr only, as before.
- The dir is cleaned exactly as a normal launch cleans it, then
  `engine_log.txt` (the argparse usage + error, UTF-8), then
  `engine_done.json` (`status: "error"`, `error:` the same text, every other
  full-run key `""`), then `engine_done.txt` = `2` **last**.
- No `engine_pid.txt` is written (nothing was spawned). `--help` is
  unchanged (exit 0, no files). The panel needs no change: its normal
  failure phase shows the manifest `error`.

## v0.18 — AI mode: a script writer that learns from your reviews

A third **Script** source next to "Prompt chain" (was "Translate with AI")
and "I have a script": **AI · learns**. It replaces only the script
generation. Every sync stage (S2d..S3e, match or legacy) is untouched.

**What it does instead of Step1 → Step2 → Step3.** It is a port of the
**Lekhak** agentic architecture (`D:\isha automation\lekhak`,
`src/server/translate.ts` and friends). Each Lekhak file maps to one
module: `ai_memory` = `tm.ts`, `ai_checks` = `checks.ts`, `ai_agents` =
`critics.ts` + `proofer.ts`, `ai_lang` = `languages.ts`, and
`ai_translator` = `translate.ts` plus the learning.

1. **Translate** (`ai_translate_script`, the orchestrator). The English cues
   are packed into pages of about 1800 chars. Each page then goes through
   these stages:
   1. **Memory** (`AiMemory`). Each sentence of the page is matched against
      the translation memory's approved pairs, exact first and then fuzzy
      (0.4 token-set + 0.6 char similarity, over at most 60 index
      candidates).
      - Tier 1 (all sentences exact) is reused with no model call.
      - Tiers 2–4 go to the model with **anchors** ("reproduce exactly" or
        "change only the words the English changed") and up to 5 **worked
        examples**.
      - Deviation from Lekhak: a tier-2 fuzzy draft is NOT shipped
        unmodelled. A dub can run with no human, and the draft has no
        timing fit.
   2. **Translator.** 6 pages go in one call; batches run in parallel
      (`ai_concurrency`, default 3); temperature 0.3.
      - The static prefix is the same for every batch, so it is
        prompt-cached. It holds Lekhak's six principles, a timing
        principle, the per-language directives (honorific, positive
        vocabulary / avoid-term, transliteration), the `guidelines`, the
        learned profile (compact JSON) and the 8 newest corrections.
      - Each passage block holds the required glossary terms, anchors,
        examples and timed cues (`[id] @start (dur, pause after)`).
      - The reply is `{"passages":[{"n","paragraphs":[{"cues","text"}]}]}`.
        A dropped passage is retried alone once.
      - Cue coverage is repaired, never trusted. If the model's claims
        overlap, the paragraphs are merged, so English and translation
        always cover the same content.
   3. **Checks** (`run_checks`). These are free and run on every page:
      EMPTY, PASSTHROUGH, SCRIPT (at least 90% of letters in the target
      script), DIGITS, **TIMING**, TRUNCATED, COMMENTARY, GLOSSARY, plus
      COVERAGE as a warning.
      - TIMING replaces Lekhak's LENGTH. A paragraph fails when its
        estimated speech is more than 1.5× its cue window plus 0.6 s.
      - ENCODING is dropped, because the source is a transcript.
   4. **Repair.** Pages that fail a check get one more pass, with the check
      details as the "PREVIOUS ATTEMPT WAS REJECTED" notes. The repair is
      kept only if it fails no more checks than before.
   5. **Critic panel** (`run_critic_panel`). It runs on pages still failing
      after repair, plus a random `ai_audit_rate` sample (default 0.02).
      - Three judges run in parallel at temperature 0: Grammar &
        Honorifics, Glossary & Mappings, and Tone & Format.
      - A judge that errors approves nothing: the page is marked
        "Not verified". There is no loop back to generation; the critique
        goes into the report.
   6. **Proofer** (`proof_pages`). It covers every page except tier 1, in
      batches of 6, at temperature 0.2. It is advisory: its findings are
      "review" or "polish", each with drop-in suggestions, and they are
      checked (exact substring, at most 6 per page).
   7. **Risk.** Lekhak's `riskScore` is computed per page, plus up to 0.18
      for "review" findings from the proofer.
   - A page with no translation at the end fails the run. The other gates
     only flag pages for review.
   - Model roles: `translate` (`model_translate`), `critic`
     (`model_critic`), `proofer` (`model_proofer`). The last two are new
     in `llm_settings.json` and appear in the panel's per-stage model list.
     A blank value means the main model.
   - `_llm_generate` gained an optional `temperature`. It is sent only when
     given, so every existing request is unchanged.
   - `engine_settings.json` knobs, all optional: `ai_audit_rate`,
     `ai_proofer` (0/1), `ai_concurrency` (1–12).
2. **Learn** (`learn_from_review`), on `--steps dub` of an AI run:
   - The approved paragraphs are paired 1:1 with the AI draft. If the
     count changed or there is no draft, char-share pairing is used and no
     corrections are recorded.
   - Pairs go into the translation memory (`store_pairs` / `store_full`).
     This is the first caller of the capture half.
   - Changed rows become corrections.
   - One LLM call MERGES the evidence into the stored profile. The
     translator rebuilt its profile from scratch; this merge is bounded:
     at most 24 pairs and 14k chars of evidence per call, 80 terms and 40
     corrections kept.
   - Learning is fail-open: an error is logged as `[engine] WARNING`, and
     the dub continues.

3. **Anchor sync** (`_stage_dub_anchor`) replaces the sync prompts in AI
   mode. It needs no matching call, no `SyncingPrompt` and no second
   transcription.
   - **When it runs:** it is chosen automatically when
     `--script-source ai` and the approved script has the same number of
     paragraphs as the AI draft. The review screen keeps rows 1:1, so the
     counts normally match.
   - **When it doesn't:** if the counts differ, or there are no draft rows
     (a memory hit), the configured `sync_mode` runs instead, exactly as
     before. `ai_anchor_sync: 0` turns anchoring off entirely.
   - **Pieces:** each approved row keeps its English window. The row is
     split by `chunk_mode`, and the window is shared out by length (the same
     `build_pieces` logic). Emptied rows are skipped.
   - **Speech:** `synthesize_sentences_elevenlabs` produces each piece and
     reports its measured length.
   - **Fitting, cheapest step first.** A piece's slot is its window plus
     the pause before the next piece.
     1. If the piece is still too long even at `max_atempo`, **one targeted
        agent retry** runs (`fit_to_seconds`: "fit in N s, keep the
        meaning", in the learned voice). Only those lines are re-voiced,
        capped at `ai_fit_retry_max` (12). `ai_fit_retry: 0` turns it off.
     2. Each piece is then time-stretched up to `max_atempo`
        (`_stretch_segment`).
     3. Pieces are placed by `place_pieces`: the deterministic spring
        layout, then the order sweep with bounded borrowing. Anything
        still left over goes to Un sync.
   - **Changed wording:** any line shortened after approval is listed in
     `<base>_ai_fit_changes.txt`, as APPROVED vs SPOKEN, and noted in the
     log.
   - **Outputs:** the same artifacts and manifest keys as match mode
     (`_write_sync_outputs`), so the importers don't change. The stage tags
     S2d–S3e are unchanged too.

4. **Fit suggestions on the review screen** (`--suggest-fit`). This works
   in any script mode.
   - **Buttons:** the inspector shows "Suggest shorter", plus "All N
     overflows", on OVERFLOWS and eats-pause rows. They launch a QUIET
     utility job, so the review screen stays open.
   - **Request:** the panel writes
     `<out_dir>/_fit_suggest.txt` as UTF-8. Each row is
     `@@ <index> <speech_s> <hard_s>` followed by an `EN:` line and a
     `TR:` line.
   - **Engine:** `suggest_fits` asks for 2 shorter renderings in the
     learned house voice, at temperature 0.5. They are sized by the
     language's speaking rate and ranked with those that fit the hard slot
     first.
   - **Answer:** the engine writes `<request>.out` (`@@ <index>`, then one
     option per line). The manifest is
     `{status, error, suggest_txt, suggest_count}`.
   - **In the panel:** each option shows its own verdict and estimate.
     "Use" swaps it in, and the fit bar re-measures live. This costs no
     voice credits: it is a text-only call, and `run_dub.py` forwards the
     flag.

25. **v0.23.0: six text corrections** (review screen and Regenerate tab,
    shared `V5.ASSIST_QUICK`), in three pairs:
    - **Speed up / Slow down**: speed only, the script never changes. The
      instruction starts with `SPEED ONLY`; `review_assist` then returns no
      alternatives (enforced in code, not only by the prompt) and keeps
      `speed`, shown as 'Apply assistant speed'.
    - **Longer (add words)** (replaces Longer + Fix underfit) / **Longer
      (add pauses)** (same words, natural pauses).
    - **Shorter (fewer words)** / **Shorter (skip pauses)** (was Remove
      pauses: same words, one continuous flow).
    - Alternatives unchanged. Test: `engine/tests/test_assist_speed_only.py`.

24. **v0.22.1: "Increase length (pauses)" quick ask** (review screen and
    Regenerate tab, shared `V5.ASSIST_QUICK`). The mirror of v0.21.1's
    *Remove pauses*: an underfit line keeps the same words, meaning, names
    and numbers and gets longer by natural pauses (commas at clause
    breaks, a dash or `...` before a key phrase, a full stop between
    thoughts), never inside a name or number. 2-3 alternatives with
    different amounts of pausing, plus a slower speed if pauses alone
    cannot fill the slot. Panel only: the instruction goes through the
    existing `--review-assist` request; no engine or file-format change.

23. **v0.22.0: ElevenLabs Dub — `--script-source eleven`.** ElevenLabs
    Dubbing Studio does the job: it transcribes, finds the speakers,
    translates and voices. We keep the review pause and add voice selection.
    Plain automatic dubbing has no voice parameter (it clones the speaker),
    so the dub is always created with `dubbing_studio=true`.
    - **Translate** (`_stage_translate_eleven`, `pipeline/el_dubbing.py`):
      `POST /v1/dubbing` (source `en`, target = the `TTS_LANGUAGES` code
      prefix, e.g. `bn-IN` -> `bn`), then poll `GET /v1/dubbing/{id}` until
      `dubbed`, then `GET /v1/dubbing/resource/{id}`. No Scribe call, no
      LLM, no prompt files. **Dubbing credits are charged here, before the
      review.**
      - **`<base>_el_dub.json`** (new, optional; deleted at the start of
        every translate): `{dubbing_id, language, iso, created:
        "YYYY-MM-DD HH:MM:SS", speakers: [{el_id, name, voice_id, key}],
        segments: [{id, start, end, speaker, en, tr}]}`. It is first written
        with just the id, as soon as the dub exists, so a crash never loses
        a paid project.
      - Review rows are one paragraph per segment. `<base>.srt` and
        `<base>_sync_en.srt` are built from the segments' English.
      - With 2+ speakers, `<base>_speakers.json` is pre-filled (keys
        `s1..sN`, paragraph -> the segment's speaker), so the review
        screen's Cast lists them and the reviewer picks a voice for each.
    - **Dub** (`_stage_dub_eleven`, dispatched first in `_stage_dub`):
      - The approved paragraph count must equal the segment count;
        otherwise it raises a clear error. Lines can be edited, not
        added, removed or merged.
      - Only edited lines get `PATCH .../resource/{id}/segment/{seg}/{lang}`.
      - Each ElevenLabs speaker gets ONE voice: the majority of its
        paragraphs' cast voices, else `--voice-id`. A conflict is logged as
        a WARNING. Set with `PATCH .../resource/{id}/speaker/{spk}`.
      - Then `POST .../dub` (all segments), poll until no `audio_stale`,
        `POST .../render/{lang}` (`wav`), poll the render, and download it.
        The fallback download is `GET /v1/dubbing/{id}/audio/{lang}`.
    - **Outputs are match mode's:** `tts_wav` = the full-length render
      (already aligned to the English). Each segment is one piece, cut at its
      own time and placed at the same time. The timestamps, texts and synced
      SRT files, the synced render and the manifest keys are unchanged, so
      the importers need no change.
    - The API key is only sent to `*.elevenlabs.io`; render downloads come
      from signed storage URLs.
    - **engine_settings.json** (optional): `el_num_speakers` (0 = auto),
      `el_drop_background` (1), `el_disable_cloning` (0; 1 asks for a
      similar library voice instead of a clone, which needs the workspace
      permission).
    - **Panel:** Script gains "ElevenLabs Dub" (`script_mode: "eleven"` in
      the settings file). Its hint says what can be edited and when credits
      are charged. The Sync summary reads "ElevenLabs (per segment)".
    - **Caveats:** Dubbing Studio is ElevenLabs' v1 product (maintenance
      mode), so the resource JSON is parsed defensively and an unknown
      shape raises. Check language support per account (Bengali and Odia
      are unconfirmed).
    - **Tests:** `engine/tests/test_el_dubbing.py` (8 tests, fake HTTP, no
      credits).

22. **v0.21.0: AI mode learns from the final dub, not the review.** Lines
    fixed on the Regenerate tab after the review were never learned. Now
    learning waits for the final timeline.
    - **Review-time learning is off by default.** `--steps dub` runs
      `learn_from_review` only when `engine_settings.json` has
      `ai_learn_at: "review"` (default `"final"`). Otherwise it logs "AI
      learning waits for 'Learn from final dub' (Regenerate tab)". The rest
      of `--steps dub` is unchanged.
    - **Regen log (panel):** every successful swap in `apply_regen_result`
      appends one line to **`<base>_regen_edits.jsonl`** (new, optional;
      `<base>` = `<out_dir>/<name>`, as Export edited script uses):
      `{"t": start_s, "len": new_len_s, "old": text, "new": text,
      "merged": n, "at": "YYYY-MM-DD HH:MM:SS"}`. On a merge, `old` is all
      merged chunks' text joined.
    - **"✦ Learn from final dub"** chip on the Regenerate tab (next to
      Export edited script) and on the success screen after Import. It
      gathers the same clips as Export edited script (`V5.final_dub_rows`)
      and writes `<out_dir>/_learn_final.txt` (UTF-8): `BASE: <base>` and
      one `C: <start_s>|<len_s>|<text>` per clip, timeline order. It runs
      as a QUIET job; the banner reads "learned N pairs, M corrections, K
      held back".
    - **Engine `--learn-final --language L --text-file F`** (`run_dub.py`
      forwards it). Manifest `{status, error, learn_txt}`. `F.out` has
      `SKIPPED:`, `PAIRS:`, `CORRECTIONS:`, `HELD:`, `REGEN_EDITS:`,
      `TM_STORED:` and `NOTE:` lines. The only model call is the text-only
      profile merge. No audio, TTS or prompt files are used.
      - **English** for a time window: cues from `<base>_ai_phrases.srt`,
        else `<base>_sync_en.srt`, else `<base>.srt`, whose midpoint lies in
        the window. The English source is assumed to start at project start.
      - **Corrections:**
        - (a) every regen edit, with chains collapsed (A→B, then B→C at the
          same spot, becomes A→C). An edit whose new text is no longer on
          the timeline (undone) is dropped.
        - (b) an AI-draft paragraph → the joined text of the clips in its
          window. This is conservative, and is skipped when the row has no
          cues, a regenerated clip lies in its window, a clip is the
          engine's own fit shortening (`_ai_fit_changes.txt` SPOKEN), clips
          look like another paragraph's text, or the similarity is < 0.6.
      - **Pairs:** (draft English, final paragraph) for accepted paragraphs,
        else (English under the clip, clip text).
      - **Learner:** `learn_from_final` shares `_absorb_evidence` with
        `learn_from_review`: TM pairs, corrections (newest 40), one profile
        merge, save.
        - The `meaning_kept` guard **holds back** any correction whose
          final lost a number or half its text, and any pair carrying it.
        - The whole document goes into memory only when nothing was held
          back and every English cue has a clip within 2 s.
        - It never raises.
      - **Idempotent:** **`<base>_ai_learned_final.json`** (new) keeps
        `{sha256, learned_at, pairs, corrections, held_back, profile,
        learned_keys}`. The same final timeline is skipped. Pressing again
        after more fixes learns only pairs and corrections not in
        `learned_keys`.
    - **Tests:** `engine/tests/test_learn_final.py` (14 tests, fake LLM,
      temp dirs). Layout harness: the Regenerate tab chip row and the
      success screen after import.

21. **v0.20.1: no choppy fragments.** A real 13-minute run showed that splitting
    Marathi at the translator's pause markers (one unit per English phrase) chopped
    sentences into 27 pieces under 1 s, placed with gaps: rated worse than v0.19.
    - Marker splitting is now OFF by default (`ai_marker_split: 1` turns it on); the
      markers still guide translation length.
    - Pieces with under `ai_min_piece_s` (1.2 s) of speech are merged into a neighbour
      in the same paragraph (`_merge_short_groups`), never across paragraphs.
    - Fine English phrases and the Fourier pause detector stay on.

20. **v0.20.0: AI-mode sync accuracy — fine English phrases, a spectral
    pause detector, pause-aware translation.** AI mode only: prompt-chain,
    match/legacy sync and plan mode are unchanged (a prompt-mode translate
    replayed offline writes byte-identical `<base>.srt` and
    `<base>_sync_en.srt`).
    - **The problem:** anchor sync aligned against the sync SRT's cues,
      which are loudness regions (absolute -42 dB gate) split by spaCy —
      and spaCy is not installed. A fluent 76 s talk gave 14 phrases, one
      17.6 s long, so every clause in that stretch shared one window.
    - **A. Phrases from word timestamps** (`anchor_align.phrase_cues`).
      The Scribe words (`type == "word"`, bucketed into the speech regions
      by `_build_subtitle_srt`'s rule, so none is lost or doubled) are cut
      at a word gap ≥ 0.30 s, at a spectral pause (B), after `. ? !`, and
      split at their best word boundary past 5 s. Phrases under 0.8 s join
      a neighbour across a gap < 0.6 s. A region's first word starts at the
      region (Scribe starts soft onsets late), never before the previous
      word ends.
      - **Your chunks win:** when the English was cut into items on the
        timeline, those items are the phrases (`cues_from_items`; an item
        over 5 s is re-split). The panel's `<wav>.dubregion.json` gains an
        additive `"items": [{"start", "len"}]` (seconds from the rendered
        wav's 0:00, `V5.region_items`); 2+ items are used, older sidecars
        and plain files behave as before.
      - **Where they are used:** the translate stage feeds them to
        `ai_translate_script` and to the memory-hit re-pairing
        (`_memory_rows`), and saves them as **`<base>_ai_phrases.srt`**
        (new, optional; deleted at the start of every translate like the
        AI draft). `--steps dub` loads it (`_load_ai_phrases`) and anchor
        sync prefers it over `<base>_sync_en.srt`. `manifest["en_srt"]`
        stays `<base>.srt`: the panel derives the run's base name and the
        review timing from it, and both still work because the phrases
        spell exactly the same words.
      - Knobs (optional, `engine_settings.json`): `ai_phrase_gap_ms`
        (300), `ai_phrase_max_s` (5.0), `ai_phrase_min_s` (0.8),
        `ai_spectral_vad` (1). Any failure logs a WARNING and falls back
        to the region SRT.
    - **B. Spectral pause detector** (`pipeline/spectral_vad.py`, numpy
      only). STFT with 25 ms Hann windows and a 10 ms hop; energy in the
      300–3400 Hz speech band plus spectral flatness; thresholds relative
      to the file's own noise floor (10th percentile) and speech level
      (90th percentile), with hysteresis; pauses ≥ 120 ms between speech.
      `spectral_pauses(y, sr)` gives phrase cut points; `onset_after(y,
      sr, t)` is built for the deferred onset-snapping step.
    - **C. Pause-aware translation** (isochrony-aware NMT). Each English
      cue in the passage block now reads `[id] @start (dur, pause after
      gap, ~N chars)`: N = seconds × the language's speaking rate
      (`LANG_CHARS_PER_SEC`). The output rules ask for ` | ` inside a
      paragraph wherever the English moves to its next cue.
      - **Markers never reach a person or the voice:**
        `split_pause_markers` removes them where the reply is read
        (`_rows_from_reply`, salvage included). Draft rows gain an additive
        **`"pauses": [int]`**: char offsets in the clean, whitespace-
        normalised `tr` where each marked stretch starts. Review files,
        FinalScript, TTS text, memory and corrections see clean text;
        `meaning_kept`, TIMING and UNDERSHOOT ignore any `|`.
      - **Anchor sync:** `_anchor_rows` keeps a row's `pauses` only while
        the approved paragraph equals the draft text (whitespace aside); an
        edited row splits at clauses as before. `clause_units(text,
        pauses=)` cuts marked rows at the markers first, one unit per
        stretch (over-long ones still halved). The DP is unchanged: forcing
        strict 1-to-1 groups was measured and was worse.
      - **`PAUSES` check (warn):** marker count far from the rows' cue
        boundaries (more than max(2, 50 %) off).
    - **Eval** (`engine/tests/eval_sync.py`, offline: cached Scribe words,
      fake TTS sized by `estimate_duration`, every LLM/ElevenLabs call
      blocked, prompt files blocked). 76 s talk (ripper_test/3):
      phrases 14 → 31, longest 17.6 s → 4.0 s, mean start offset 241 →
      13 ms, max 2069 → 147 ms, mean end offset 912 → 542 ms (348 ms with
      simulated pause markers), within 300 ms 89 % → 100 %, speech-overlap
      IoU 0.58 → 0.65. The 72 s talk (ripper_test/1), already finely
      phrased, holds (IoU 0.80 → 0.80; 0.83 with markers). Marker positions
      in that replay are simulated, so a real model's markers still need
      one live run to confirm.
    - **Tests:** `engine/tests/test_sync_accuracy.py` (unittest; 28 tests,
      no network).
19. **v0.19.0: Change voice on the Regenerate tab, with AI
    recommendations.**
    - **CHANGE VOICE section:** now always visible. It holds the shared
      voice dropdown with its 🔊 Test voice button, which speaks this
      chunk's text.
    - **Recommend voices (`--recommend-voice`):**
      - The panel writes `_voice_reco.txt` with `LANG:`, `CURRENT:`,
        `SPEAKER:`, `TR:` and `EN:` lines, plus `V: <id>|<name>` for every
        fetched voice and bookmark.
      - `recommend_voices` picks 3 using the voice names and labels, the
        speaker, the line's tone and continuity with the current voice.
        Invented ids are dropped.
      - The answer is `.out` with `PICK: <id>|<why>` lines; manifest
        `{status, error, reco_txt}`. It runs as a QUIET job.
    - **Each pick** shows 🔊 Test (the existing voice preview) and Use
      (sets the voice for this line; Regenerate then voices it).
    - The manual voice id and "use the default" moved under
      "Voice id / reset".
18. **v0.18.9: Regenerate works on several selected chunks as one line.**
    - **Selection:** the chunks selected on the first chunk's track are
      taken in timeline order (`V5.regen_selection`).
    - **What the tab shows:** the text box holds their joined text, and the
      header reads "chunks a-b of N (k selected, one line)". The fit bar
      measures against the combined span (first start → last end). The
      assistant sees the combined line, the chunk before the first and the
      chunk after the last; its state is keyed by the selection.
    - **Speed:** −/+ applies to every selected chunk.
    - **"Regenerate k chunks as one line":** voices the combined text into
      the first chunk (the existing take-swap). `apply_regen_result` deletes
      the other selected chunks in the SAME undo block, so one Ctrl+Z
      restores them all. "Put the old one back" is replaced by that Ctrl+Z
      hint after a merge.
17. **v0.18.8: "Fix underfit" quick ask** (Review and Regenerate
    assistants). For a line too short for its slot (fit column "short"),
    the assistant returns 2–3 slightly fuller renderings sized to the
    speech slot, using connecting words and never new meaning. If fuller
    wording would sound padded, it also returns a slower speed ("Apply
    assistant speed ×N").
16. **v0.18.7: assistant and speed on the Regenerate tab.**
    - **Shared launcher:** the assistant is launched by `V5.assist_launch`,
      shared with the review screen. The answer is routed back by
      `V5.assist_target` (`review` or `regen`).
    - **Per chunk:** for the selected dub chunk, the tab has the same quick
      asks (Shorter / Longer / Speed up / Slow down / Alternatives), a free
      instruction box, and a conversation kept per chunk (by GUID).
    - **Context sent:** the chunk's text, the previous and next chunks'
      text, its slot (item length, plus the gap to the next chunk as the
      hard slot), and the English under it (from the run's
      `<base>_sync_en.srt`, assuming the English source starts at project
      start).
    - **Alternatives:** Use puts a line into the edit box; the existing
      Regenerate button then voices it.
    - **Speed is instant here:** the take's `D_PLAYRATE` with pitch kept
      (`B_PPITCH`), and the item length follows. It is one undo step with
      no engine run and no credits. The controls are −/+/reset and "Apply
      assistant speed ×N", clamped to 0.80–1.25. A newly regenerated take
      starts at ×1.00.
    - **Engine:** unchanged (it reuses `--review-assist`).
15. **v0.18.6: review assistant and per-row speed.**
    - **Assistant (`--review-assist`):** each row's inspector has an
      ASSISTANT box.
      - Quick asks: Shorter, Longer, Speed up, Slow down, Alternatives.
      - A free-text instruction box, for example "make it 2 seconds",
        "more formal", "speed x1.1".
      - The conversation for that row.
    - **Engine side:** `review_assist` makes one model call in the learned
      house voice with no prompt files. It sees the English, the current
      line, the previous and next lines, the slot, the speed and the
      conversation history.
    - **Request and answer files:** the request goes to
      `<out_dir>/_review_assist.txt` (`@@`, `EN:`, `TR:`, `PREV:`, `NEXT:`,
      `ASK:` and `H: role|text` lines) and the answer comes back as
      `<request>.out` (`REPLY:`, `SPEED:`, `ALT:` lines). Manifest keys:
      `{status, error, assist_txt}`. The panel runs it as a QUIET job.
    - **Alternatives** appear in SUGGESTIONS with Use.
    - **Suggested speeds** appear as "Apply assistant speed ×N", clamped to
      0.80–1.25.
    - **Per-row speed:** there are also −/+/reset buttons, and the fit bar
      counts the speed.
    - **Speed file:** on approval, the speeds are saved to
      `<base>_row_speed.json` as
      `{"version":1,"speeds":{"<paragraph>":factor}}`, next to
      `_speakers.json`. Anchor sync time-stretches that paragraph's pieces
      before fitting. Speeds apply in AI mode (anchor sync) only.
14. **v0.18.5: dropped content is caught, never salvaged silently.**
    - **The failure:** a 76 s talk came back as 3 paragraphs. The last one
      held 667 English chars against 271 Marathi, so everything after "If
      you start paying attention…" was missing. Anchor sync then spread the
      short text over a 55–73 s window, so the dub was off-sync with gaps.
    - **`UNDERSHOOT` check (fail):** this is Lekhak's LENGTH check, too-short
      side. A paragraph with at least 60 English chars fails when its
      translation is under expected ratio ÷ 1.6. The expected ratio is
      learned from 30+ approved pairs, and is 0.8 by default. The failure
      goes into the repair pass as "translate ALL of the English".
    - **`_complete_salvage`:** a salvaged passage is kept only if its
      paragraphs claim every cue of the page. Otherwise it is discarded and
      the page goes to the solo retry. Before this, an unreadable paragraph
      was silently dropped.
    - **Anchor sync warning:** it logs a WARNING when the voiced script
      covers under 60% of the English speech time.
13. **v0.18.4: quoted speech no longer fails a page.**
    - **The failure:** a translator reply with an unescaped `"` inside the
      Indic text (मी म्हणालो, "ही फांदी…") was invalid JSON. The page, its
      solo retry and then the run all failed.
    - **Output rules:** they now forbid `"` inside text values and ask for
      ‘ ’ or « » for quotations.
    - **Salvage parser:** if a reply is still invalid, `_salvage_passages`
      pulls each `{"cues":[…],"text":"…"}` object out directly (the text
      runs to its closing `"}`, so inner quotes survive) and assigns it to
      the nearest preceding `"n":` marker. This costs no extra call.
    - **Retry:** only if salvage finds nothing is the batch re-asked once,
      with the reason stated.
12. **v0.18.3: AI mode never reads a prompt file.**
    - `_required_prompts` returns nothing for `--script-source ai`: there is
      no Step1–4 or SyncingPrompt preflight.
    - When the draft rows can't be used (paragraph count changed, no draft,
      a row without a window), `_anchor_rows` now returns the approved
      paragraphs WITHOUT windows. Anchor sync then aligns them to the
      English phrases by duration alone, instead of falling back to the
      configured sync mode.
    - If there are no row windows AND no English phrases, AI mode uses
      match sync, whose matcher prompt is built inline in code.
    - `ai_anchor_sync: 0` also leads to match sync. AI mode never uses
      legacy (SyncingPrompt / Step4_Emotion).
    - This was verified with `_load_lang_prompt` patched to raise, in a
      run configured for legacy sync with emotion on.
    - Settings → Prompts now says these files are used by "Prompt chain"
      only.
11. **v0.18.3: no more truncated lines.**
    - **The chain:**
      1. Memory paragraphs were paired to English by length, which split a
         sentence across rows.
      2. The half-sentence row showed OVERFLOWS.
      3. The fit suggestion, built from that half English, "fit" by dropping
         "is 108 times the diameter of the moon".
      4. Learning then memorised the fragment (pair and full script) and
         taught it as a correction.
    - **`meaning_kept(original, shorter)`:** a shorter line must keep every
      number of the original (Indic digits folded to ASCII) and at least
      half its length. It gates `suggest_fits`, `fit_to_seconds`, and the
      pairs and corrections in `learn_from_review`. Both suggestion prompts
      now say the English may be a fragment and the rendering must keep
      everything the current line says.
    - **Learning:** a held-back row is logged. When any row is held back,
      the whole script is not stored as a full-doc memory. A row whose
      English was only estimated (memory row, empty `cues`) is never stored
      as a memory pair.
    - **`_memory_rows`:** memory paragraphs are paired first by their
      original English, looked up in memory by translation and consumed
      cue by cue. If that fails, length-share pairing is used with every
      boundary snapped to the nearest English sentence end.
10. **v0.18.2: no clipped words at piece cuts** (`_snap_cuts`).
    - **The problem:** piece audio is cut out of one continuous synthesis
      at ElevenLabs' character timings, from the first character's start to
      the last character's end. Inside fluent speech, which is where
      clause-level pieces cut, those timings run 100–300 ms off, and the
      audio between two spans belonged to neither piece. Word onsets and
      endings were lost.
    - **The fix:** every cut between neighbouring pieces moves to the
      quietest 10 ms frame within ±250 ms of the boundary (the gap between
      two words), and both pieces share that cut, so nothing between them
      is dropped.
    - Pieces more than 1.2 s apart (a request gap, or real silence) are left
      alone. The log line reads
      "N piece cut(s) moved to the nearest word gap".
9. **v0.18.1: anchor sync works at phrase level** (`pipeline/anchor_align.py`).
   - **Why:** whole-paragraph pieces drifted. An 11 s English paragraph
     holds several phrases with pauses, and one continuous 7 s block of
     Marathi runs past them all. Memory rows paired by length also crossed
     sentence boundaries.
   - **Units:** the approved paragraphs are split into clause units
     (`clause_units`: at `, ; : । . ! ? —`, fragments under 12 chars merged,
     clauses over 60 chars split once at the middle word gap). Every unit
     is voiced, so its real length is known.
   - **Alignment:** `align_units` is a monotonic dynamic-programming match
     on seconds. It groups consecutive units, 1–4 at a time, with
     consecutive English sync-SRT phrases, also 1–4 at a time, so their
     durations match. An English phrase can be left undubbed (fillers).
     A group never crosses a paragraph, so the cast is kept.
   - **Row windows** are soft hints: weight 3 for translator rows (their
     cues are exact), 1 for memory rows.
   - **Pieces:** each group is one piece whose window is its phrases'
     span.
   - **Piece size:** `section` keeps the old one-piece-per-paragraph
     behaviour.
   - **Sync check** now reports END offset as well (dub end vs English
     phrase end), so drift inside a piece is no longer reported as "on
     time".
   - **Measured on a real 72 s Marathi run** (speech estimated from text
     length): mean end offset went from 629 ms to 264 ms, and maximum start
     offset from 292 ms to 213 ms after the check.
6. **Sync check agent** (`pipeline/sync_check.py`, `_run_sync_check`).
   - **When it runs:** after placement in anchor sync, and in match sync at
     clause/sentence piece size. It is deterministic.
   - **Offset:** measured as the placed start minus the English speech
     start.
   - **Tightening:** a piece more than `sync_tolerance_ms` late (default
     300) is pulled earlier, and one that early is pushed later, as far as
     the neighbours allow. A move never reorders pieces or overlaps them.
   - **Rescue:** an Un sync piece goes back on the timeline when its window
     has room within 2× the tolerance.
   - **Report:** `<base>_sync_check.txt` gives on-time count, mean and max
     offset, tightened, rescued, overlaps, and pieces still off. There is
     also one `[S3d]` log line.
   - **Switch:** `sync_check: 0` in `engine_settings.json` turns it off.
     Any error keeps the placement as it was.
7. **Memory hits keep anchor sync.** When an AI-mode translate reuses a
   whole learned script from memory, its paragraphs are now timed against
   the English (`_pair_review_rows`) and written as draft rows
   (`"source": "memory"`). Before this, anchor sync fell back to the
   configured prompt sync on every re-run of a learned script.
8. **Fit-suggestion fixes (panel).**
   - `suggest_txt` and `suggest_count` were missing from the panel's
     `MANIFEST_KEYS`, so the answer path was dropped and every reply read
     as empty.
   - New **Apply best fits (N)**: for each row that has suggestions, it
     swaps in the first option that fits the speech slot, or failing that
     the one that fits the hard slot, so the red boxes turn green in one
     click.
5. **Fix: Preview sync launches again.** `run_dub.py` had been left behind
   by v0.13:
   - Its `--steps` only allowed `full`, `translate` and `dub`, and it had
     no `--plan` argument. So every Preview sync launch (`--steps plan`
     with a pasted script or `--plan`, and `--steps dubplan --plan`) died
     in the launcher's argparse before the engine ever started.
   - The launcher now accepts `plan` and `dubplan`, forwards `--plan`,
     and mirrors `dub_engine.py`'s plan rules. `--steps plan` needs either
     `--provided-script` or `--plan`; `dubplan` needs `--plan`; `--plan`
     is valid only with plan/dubplan; `--provided-script` is not valid
     with dubplan.

**CLI.** `--script-source prompt|ai`, default `prompt`.
- Valid with `--steps full/translate/dub`.
- Rejected with `--provided-script` and with plan/dubplan.
- `_required_prompts` skips the Step1-3 files for AI runs. The sync prompts
  are still required as before.

**Files** (all new, all optional; old runs unaffected):
- `<base>_ai_draft.json` is written by an AI translate. It is deleted at the
  start of every translate so a stale draft can't pose as this run's.
  Shape: `{"language", "created": "YYYY-MM-DD HH:MM:SS", "rows": [{"en",
  "tr", "start", "end", "cues": [int]}], "pages": [{"n", "cues", "tier",
  "tm_score", "risk", "verified", "attempts", "checks", "critique",
  "proof"}]}`.
- `<base>_ai_report.txt` is the review report, pages sorted by risk. It
  lists each page's checks, the critic feedback, and the proofer findings
  with their suggested replacements. It is deleted at the start of every
  translate, like the draft. The log also names the 3 riskiest pages.
- The profile gains `"glossary": {english: target}`. This is forced
  terminology (Lekhak's glossary), verified by the GLOSSARY check and
  edited by hand.
- `<base>_ai_learned.json` holds `{"sha256", "learned_at", "pairs",
  "corrections", "profile"}`. It makes learning idempotent: re-dubbing the
  same approved script learns nothing twice.
- `data/ai_learning/<language>.json` is the profile (env `AI_LEARNING_DIR`
  overrides): `{"version":1, "language", "guidelines", "toneDescription",
  "grammarRules", "vocabularyMappings":{en:term}, "corrections":[{"en",
  "draft", "final", "at"}], "samples", "runs", "updated"}`.
  - `guidelines` is hand-edited house rules; learning never overwrites it.
  - `data/` is gitignored and survives updates.

**Stage tags** are unchanged. S2a/S2b/S2c are still printed, with AI
wording. Learning prints untagged `[engine]` lines, so the panel's step
list is unaffected.

**Panel.**
- `script_mode` in `dub_panel_settings.json` gains the value `"ai"`. Older
  panels read an unknown value as `auto`.
- `build_engine_cmd` emits `--script-source ai` for both the translate and
  the dub-continue launch.
- Run history records `mode: "ai"`, which is shown as "dubbed".
- AI mode learns only from reviewed runs. A straight-through run uses the
  profile but teaches it nothing, and the panel says so.

## v0.17 — The region: the run covers what the timeline shows

Trimming an item is how you say what the dub is of, and the panel only half
heard it. A track holding one clean item handed its **source file** to the
engine — the whole file, however short the item had been trimmed to — and
anything else was rendered from project **0:00**, so a two-minute excerpt of
an hour-long talk was transcribed, translated and *spoken* in full. The
result then landed at 0:00, nowhere near the item it came from.

The engine is unchanged: it still dubs the wav it is given. What changed is
which wav that is, and where the answer comes back to.

### What gets taken (`V5.timeline_region`)

Resolved every frame, so the source row is live while you trim. Most
specific statement first:

1. **the time selection** — you dragged over the part you mean,
2. **the selected item(s)** — you clicked the piece you mean; that also says
   which track, so the picker may stay on "(from track)",
3. **everything the chosen track holds**.

A time selection is **clamped to the track's items**: one dragged past the
end of the talk would otherwise render, and transcribe, minutes of silence.
A selection spanning two tracks is narrowed to one — "dub this item" never
means "mix these two".

### What is handed over (`audio_for_region`)

- `V5.region_is_whole_file(plan)` is the only case that needs no render: ONE
  item, untrimmed (`D_STARTOFFS` 0, length == source length), unstretched
  (`D_PLAYRATE` 1), starting at 0:00, and the region is all of it. This is
  the v0.4.1 fast path plus the two conditions it was missing — the item has
  to *start* at zero, and the region has to be the whole of it.
- Anything else renders that exact span. `render_track_stem` took `from_s` /
  `to_s` for this; passing neither still means 0:00 → last item end, which is
  what the voice changer wants (it re-voices a whole track in place).
- The span is in the file name as well as the sidecar
  (`Sadhguru EN_2m00s-3m35s_20260820_101500.wav`): a `DubSource/` folder is
  otherwise four identical-looking wavs of the same talk.

### `<wav>.dubregion.json` — the span travels with the audio

```json
{"version": 1, "source": "D:\\p\\talk.wav", "project_pos": 120.5,
 "start": 120.5, "end": 215.5, "track": "Sadhguru EN",
 "why": "the track's items"}
```

Written **only beside audio the panel itself rendered** (in `DubSource/`), so
nothing is ever dropped next to the user's own files. A plain file has no
sidecar and `V5.region_offset` answers 0, which is exactly what every run did
before regions existed.

Two consumers, and both would otherwise be wrong by the length of the talk:

- **`import_to_timeline`** offsets every track it builds — EN Original, Dub
  Chunks, Un sync, the muted reference — by `project_pos`. Every time in the
  manifest is measured from the region's own 0:00, so placing them at the
  project's 0:00 puts the whole dub minutes away from the audio it was made
  for. The summary says `Placed at: 0:02:00.500` when it happens.
- **`V5.review_relink`** takes the review transport's zero from it. A region
  wav is never *on* the timeline — it IS a slice of something that is — so
  the item-by-filename search could only ever answer 0:00, and ▶ Play here
  would play the wrong part of the talk.

### The nudge

Pressing Run without pressing Use is the failure this feature exists to end,
so the source row's caption goes **amber** whenever the audio in the field is
not the span the timeline is showing: `press Use to dub 1:35 — 2:00 → 3:35
(the track's items) — not the whole file`. It reads `dubbing 1:35 — …` once
the two agree.

### Tests

`tests.lua` §9c, against a new minimal project model in `mock.lua`
(`T.set_project` — tracks, items, takes, sources, selection, time selection;
`CountTracks` used to answer 0, so the track picker had never been drawn in
this harness at all). It asserts the span for an untrimmed item, a trimmed
one, both time-selection cases and a selected item that supplies its track;
that a trimmed item is **never** handed over as its whole source file; the
sidecar round trip; the review transport's zero; and it draws the setup
screen with a project, which is where the caption line lives.

That last one caught the bug worth naming: `local ta, tb = f and f(...)`
adjusts the `and` expression to **one** value, so the time selection's end
was always nil and a time selection could never win.

## v0.16 — The cast: the voice is chosen on the screen that reads the script

The voice was one id in ⚙ Settings, chosen before the run and invisible on
the one screen where the script is actually read. A talk is not always one
voice — Sadhguru speaks most of it, a question comes from the audience, an
invocation belongs to someone else — and until now the engine's answer to a
saved per-paragraph voice map was to detect it, warn, and dub the whole
thing in one voice anyway.

The paragraph is where the question is obvious, and the review screen
already lists paragraphs. So the casting happens there, before anything is
spoken or billed, and the engine honours it.

### `<base>_speakers.json` — read AND written, and now obeyed

Same file the bulk app wrote and `pipeline/tts.py` already knew how to read
(`_speakers_load` / `_speakers_voice_map`). Written by the panel's review
screen on 💾 Save and before ▶ Continue:

```json
{
  "version": 1,
  "language": "Hindi",
  "default_voice_id": "<the main voice>",
  "speakers": [
    {"key": "s1", "name": "Sadhguru",   "voice_id": "…"},
    {"key": "s2", "name": "Questioner", "voice_id": "…"}
  ],
  "assignments": {
    "2": {"speaker": "s2", "voice_id": "…"}
  }
}
```

- The assignment key is the **1-based blank-line-separated paragraph number
  of the dub script** — which is exactly one row of the review screen, and
  exactly what `_speakers_voice_map` already returned.
- `speakers` is the panel's own record (names, so a resumed review comes
  back with the same cast); the engine reads `assignments` only.
- A paragraph with no assignment is spoken by the run's main voice, which
  the panel also passes as `--voice-id`. **A single-voice run writes no file
  and deletes a stale one** — an empty map means "one voice", never "the
  cast went missing".

### Engine: both sync modes speak the cast

The rule everywhere is *one request is one voice* — there is no way to ask
ElevenLabs for two voices in one call, and a request is the unit of prosody.

- `match` mode. When a cast exists the script is split by
  `_split_script_into_units_with_paras` /
  `_split_script_into_sentences_with_paras`, which return the paragraph
  number of every unit alongside the unit. `agentic_split_match` rewrites
  sentences in place and never renumbers them, so that list stays valid
  through matching; a piece takes the voice of the paragraph its first
  sentence came from. `synthesize_sentences_elevenlabs` /
  `synthesize_sections_elevenlabs` take an optional parallel `voices` list,
  and `_pack_sentences(…, keys=voices)` breaks a request on a voice change
  as well as on the size cap.
- `legacy` mode. Consecutive paragraphs that share a voice become one run;
  the runs go through `synthesize_sections_elevenlabs` and concatenate into
  the one TTS wav the rest of the stage expects. S3b re-transcribes that
  wav, so nothing downstream needs to know how many voices made it.
- **Request stitching stops at a voice change.** `previous_text` /
  `next_text` exist to keep prosody continuous; feeding the previous
  speaker's words across the boundary is how a new voice inherits the old
  one's cadence.
- **A tiny clause fragment is never merged across a paragraph.** The plain
  `_merge_tiny_units` folds "Questioner:" (11 chars) into whatever precedes
  it — which, with a cast, hands the start of one speaker's line to the
  previous speaker's voice. The `_with_paras` splitter refuses that merge;
  single-voice runs still take the plain path and are unaffected.
- **Emotion enrichment is checked, not trusted.** It rewrites the script
  (legacy mode only), and the cast is keyed by paragraph NUMBER. If the
  enriched text comes back with a different paragraph count the cast is
  reported as unusable and the run is single-voice, because guessing which
  line moved is how a talk gets the wrong voice. `--no-emotion` keeps the
  cast.
- **`section` piece size warns.** A section is a *thought*, and Gemini can
  group one across a paragraph — so a section can span two speakers. It is
  spoken by its first paragraph's voice, and the log says how many did that;
  `clause` or `sentence` keeps every speaker separate.
- Voice ids are sanitized per entry (`_resolve_voice_list`); a blank or
  unparseable one falls back to the main voice rather than failing a run
  whose translation has already been paid for and reviewed.

### Panel: 🎙 Cast on the review screen

- The strip rides on the **view row** (List/Grid, S/M/L) and **never
  wraps**. The review screen has no vertical slack left at 780 px — the
  layout harness measures 0 px of it in the degenerate text-measurement
  regimes — so one more wrapping row is the difference between a table and a
  1-pixel sliver of one. Speakers that do not fit collapse into a `+N` chip
  that opens the editor.
- The editor is drawn over the table and takes only what the table can
  spare; below ~90 px it does not draw at all rather than squeeze the script
  off the screen (the strip's chips and the inspector still cast lines).
- Casting a line: press a speaker chip to make it *active*, then click a
  paragraph's coloured chip. Clicking a paragraph already on the active
  speaker puts it back on the main voice. The inspector lists the whole cast
  for the selected paragraph.
- **🔎 Detect speakers** reads `Name:` labels off the English column and
  casts those paragraphs. It never edits the script — including the labels,
  which ElevenLabs would otherwise read aloud.
- **⇧ main** swaps a speaker with the main voice, naming the implicit
  assignments first so nobody's lines change voice.
- ▶ Continue and Skip edit refuse while a speaker that has paragraphs has no
  voice. The main voice may be left empty — that is the engine's documented
  auto-resolve, and the strip says `main voice: auto` so it is a choice
  rather than a surprise.
- Picking the main voice writes it to `config/tts_settings.json`, so Regen
  Audio, Track Voice and the next run all agree with what was cast here.

### Tests

- `dubbing/engine/test_cast.py` — offline, no API: the file round-trip, the
  paragraph numbers, the piece→voice rule, request packing, the voice-list
  validation and the legacy run grouping.
- `dubbing/reaper/layout-harness/tests.lua` §7c — the whole screen drawn
  with a real multi-speaker cast (strip, editor open, row chips in both
  layouts, the List's voice column, the eight-speaker `+N` overflow), plus
  the model assertions for assign / remove / promote / detect / load.

## v0.15 — The review page is where the script is corrected

v0.13 shipped two gates for one decision. `review` asked "is the wording
right?" over blank-line PARAGRAPHS with no timecode, no slot and no verdict;
`plan` asked "does it fit?" over pause-detected CHUNKS with the wording
already frozen into a file. Different units, so nothing lined up, and the fit
question could only be answered after the wording question had been answered
blind.

Worse, the loop both surfaces advertised did not exist: **`⟲ Reload` re-ran
`--steps plan` against the original pasted script**, which re-spread it across
the chunks and threw away every `TR:` edit. The panel hint and the engine's
own closing note each promised "edit the TR: line → save → Reload"; doing
that silently reverted the edit.

### `--steps plan` takes its text from either source

`--provided-script` (the flowing pasted script, spread by duration share) is
the FIRST look at a run. `--plan <file>` re-measures the `TR:` lines that are
already assigned per chunk, against freshly detected pauses — the same
"timings from the audio, text from the file" rule `dubplan` has always
followed. Passing neither is still an error; `--plan` is now valid with
`plan` and `dubplan` only.

`V5.plan_reload` passes the plan file, falling back to the script only if the
plan has gone missing. Corrections now survive a reload, and re-measuring
stays free (the S1a transcription is disk-cached).

### `preview_html.py` — readable became editable

The page keeps its two lanes and gains one editable row per chunk: a
`contenteditable` target cell, and a live re-estimate on every keystroke using
the same arithmetic as `estimate_fit` / `estimate_duration` (strip `[tags]`,
count characters, divide by the language rate), so the fit bar, the verdict,
the tally pills and the chunk's bar in the timeline all move as words are cut.
"Only problems" collapses the list to the chunks that are not `fits`; clicking
any bar jumps to its row.

The page **cannot write to disk** — it is opened as `file://` with no server —
so saving means `Copy corrected plan` (clipboard, with an `execCommand`
fallback because `file://` is not a secure context) or `Download plan file`.
Both emit a COMPLETE plan file, byte-identical to `format_plan_text`, with the
verdict/estimate/atempo columns recomputed for consistency. That the round
trip is a deliberate human step is the point: nothing reaches a paid run
without passing through the panel's own button.

### Panel: the gate, plus the lane strip

`V5.ui_phase_plan` is now the review gate — 🌐 Open review page ·
📥 Paste corrections · ⟲ Re-measure · 📝 Edit plan file · 📁 Folder — over
`V5.plan_strip`, a two-lane draw-list strip (source above, estimated dub
below, hatched pauses, one time axis). Drawn as rectangles on purpose:
**rectangles need no text shaping**, so it is the one honest picture of a
whole Indic run the panel can render. Hover names a chunk, click selects it,
and the selection is shared with the table below (`P.sel` / `P.scroll_to`).

Hit testing is decided by the SOURCE slot (`start_s` to `start_s + dur +
pause`), because those tile the axis without overlaps — an overflowing dub bar
reaches into its neighbour's slot, so letting the longer of the two decide
hands clicks to the wrong row.

`V5.plan_paste_corrections` validates before it writes: the clipboard must
parse as plan rows with `TR:` lines AND carry exactly this run's chunk count
(a stale page from an older preview is refused by name), and the previous plan
is copied to `<plan>.bak` first, because the reload rewrites it.

## v0.13 — Pause-aware sync: the source's own silences are the chunk grid

Every mode up to here cuts chunks from the SCRIPT and then searches for
somewhere to put them — Gemini section matching, spring rounds, bleed-over,
the order sweep, and a demotion path to `Un sync` for whatever will not fit.
The drift only becomes visible after ElevenLabs has been billed. v0.13 adds a
second, independent path that inverts this: the source audio's pauses are the
chunk grid, so every chunk's position is fixed before any text is considered,
and a free dry run reports the fit before a credit is spent.

Both engine modes are additive. `full`, `translate`, `dub`, `--regen-chunk`,
`--voice-change`, `--test-llm` and `--list-voices` are byte-for-byte
unchanged, and **the paid mode emits the same five artifacts as match mode**
(`tts_wav`, `timestamps_txt`, `sync_texts`, `synced_srt`, `synced_wav`), so
`import_to_timeline`, `Import_Dub_Results.lua` and `parse_timestamps_file`
need no changes at all.

### `pipeline/pausechunk.py` (new)

Pure functions, seconds throughout, no I/O and no network.

- `pause_chunks_from_regions(regions, total_dur_s, pause_min_s)` — the
  primitive the repo never had: `_detect_regions_from_audio` returns SPEECH
  spans and nothing ever materialized the silences between them. Regions
  closer than `PAUSE_MIN_S` (0.20 s) are merged — a 150 ms breath is a region
  boundary but not a place a dubber would cut. Emits `index`, `start_s`,
  `end_s`, `dur_s`, `pause_after_s`; the trailing chunk measures its pause to
  `total_dur_s`, so a long tail of room tone is visible rather than silently
  becoming unlimited headroom. Chunks tile the timeline exactly:
  `end_s + pause_after_s == next start_s`.
- `source_text_for_chunks(chunks, words)` — Scribe words bucketed per chunk
  with the SAME rule as `_build_subtitle_srt` (first chunk whose end the word
  does not pass; non-`word` tokens skipped).
- `assign_script_to_chunks(script_text, chunks, language)` — the pasted script
  is in the TARGET language and the transcript is in the source language, so
  there is nothing to align on textually. Each chunk's share of total speech
  time becomes its share of the characters; sentences
  (`tts._split_script_into_sentences`, danda-aware, tag-safe) are never split
  and land in the chunk holding their character MIDPOINT. Midpoint, not
  greedy accumulation: a script with fewer sentences than chunks must spread
  out instead of piling into the first few and leaving the tail silent.
  Always returns exactly `len(chunks)` strings. This is a starting point, not
  an answer — the plan file exists so a wrong assignment is moved by hand and
  re-previewed for free.
- `estimate_fit(...)` — two slots per chunk. `speech_slot = dur_s` (how long
  the source speaker talked) and `hard_slot = dur_s + pause_after_s`
  (overrunning THAT means the dub is still talking when the next chunk must
  start). Verdicts: `fits` · `tight` (eats into the pause) · `over` (exceeds
  even speech + pause) · `short` (under `SHORT_RATIO` = 0.85 × speech slot) ·
  `empty`. `atempo` is the speed-up the paid run would apply, clamped to
  `MAX_ATEMPO`.
- `format_plan_text` / `parse_plan_text` / `plan_counts` / `summarize_plan`.

### The rate table moved to `config.py`

`LANG_CHARS_PER_SEC`, `DEFAULT_CHARS_PER_SEC` and `estimate_duration` lived in
`agent_splitter.py` — a module about LLM rephrasing — and the dry run needs
the same numbers. Two copies that drifted apart would mean the preview and the
shortener disagreed about what fits. `agent_splitter` re-exports them, so
every existing caller is unchanged. The bare `14.3` literal duplicated in
`tts.py` and `dub_engine.py` is now `config.CLAUSE_CHARS_PER_SEC`.

### Plan file — `<base>_sync_plan.txt` (the editable artifact)

Line-oriented, NOT JSON, and deliberately so: the panel's Lua reader
(`json_field`) is flat-scalar only and `V5.json_object_array` drops numeric
fields, so a nested JSON plan would arrive half-parsed. This shape parses with
one `string.match`, exactly like the timestamps sidecar.

```
# comment lines, regenerated every run
[1] [0ms] [4120ms] [4120ms] [680ms] [fits] [3980ms] [1.00]
EN: what the source said here
TR: the target text that has to fit
                                   <- blank line between rows
```
`[idx] [start] [end] [duration] [pause after] [verdict] [estimate] [atempo]`.

**Only the `TR:` lines are read back.** Every timing is re-derived from the
audio on every run, so a hand-edited (or stale) timestamp cannot desync a paid
run. `parse_plan_text` indexes by the row's own `[idx]`, so reordered or
renumbered rows still land on the chunk they name, and a `TR:` line wrapped
across several lines is rejoined. Readers handle CRLF and a UTF-8 BOM
(Notepad writes one).

### `--steps plan` — the free dry run

Requires `--provided-script` (the target text, the same file channel Paste
Translation uses — Indic text never travels on argv) — or, since v0.15,
`--plan` to re-measure text already assigned per chunk. Runs S1a (transcription,
already disk-cached by `_transcript_cache_path`, so Reload never re-uploads)
and S1b (pause detection), then chunks, assigns, estimates and writes
`<base>_sync_plan.txt` + `<base>_sync_plan.html`. **No TTS request and no LLM
call** — `_begin_run(..., need_llm=False)`, because requiring a working LLM
provider would block a free preview on a machine that only has an ElevenLabs
key, which is exactly the setup this serves.

Manifest `"status": "plan"` with `PLAN_MANIFEST_KEYS`: `plan_txt`,
`plan_html`, `chunk_count` and the five verdict tallies (strings, like every
other numeric manifest field). A failure to write the HTML is a warning — the
plan file is the contract, the HTML is the readable half.

### `--steps dubplan --plan <file>` — the only paid step

The whole matching/placement stack is bypassed: nothing can be demoted to
`Un sync`, because nothing has to compete for a position.

1. Plan read and validated BEFORE the voice is resolved or anything is
   transcribed — a mistyped path is a local file check and must not cost an
   API round trip.
2. Timings re-derived by re-running the S1a/S1b stage (`_stage_pause_plan`,
   shared with `plan`); only the `TR:` text comes from the file.
3. ONE stitched `synthesize_sections_elevenlabs` pass over the chunks that
   have text — per-chunk requests in isolation would lose `previous_text` /
   `next_text` prosody continuity and cost more. A span count that disagrees
   with the chunk count is a hard failure, never a silent mis-pairing.
4. Fit: `measured > hard_slot` → `tts.stretch_wav_atempo` at
   `min(measured / hard_slot, max_atempo)`. Ratios past the ceiling are **not
   squashed** — the chunk stays long and is named in the log, because the
   point of this feature is seeing drift, not hiding it. Chunks are
   reassembled with the same `SECTION_GAP_MS` cushion between them and the
   spans exclude it, for the same reason match mode does it.
5. Placement is not an algorithm: `synced_start_ms` is the source chunk's own
   `start_s`, every status is `synced`, and the silence padding falls out for
   free because `sync_audio_with_timestamps` overlays onto a silent canvas —
   the untouched gaps ARE the original pauses.

`tts.stretch_wav_atempo(in, out, ratio)` shells `FFMPEG_PATH` with an
`atempo` chain (`_atempo_chain` splits ratios outside ffmpeg's 0.5–2.0
per-instance range). It returns the INPUT path unchanged on any failure or a
missing ffmpeg, after a loud status line: an overlong chunk is a far better
outcome than a run that dies after the credits are spent.

### `pipeline/preview_html.py` (new)

`render_plan_html(...)` writes ONE self-contained file — inline CSS/JS, system
font stack, zero external requests (a CDN reference would render this blank
behind corporate TLS inspection, exactly when it matters). Two lanes on one
shared px/sec axis, both anchored at the chunk's real start time: a target bar
overhanging its source bar IS the drift. Pause gaps are hatched. This exists
because Dear ImGui has no complex-text shaping — Devanagari is not legible in
the panel, and a preview whose point is reading the target text has to leave
it.

### `engine_settings.json` keys (all optional, `_engine_setting`)

`pause_min_ms` (200) · `pause_thr_db` (−42.0) · `max_atempo` (1.25) ·
`plan_rate_override` ("") · `plan_split_mode` ("proportional", a hook for a
future Gemini splitter). An install that never touches the file gets the
tuned defaults.

### Panel (`Dub_Pipeline_Panel.lua`)

- New `_ui_phase` value **`plan`** — a sixth phase, mirroring `review`. It is
  NOT a util mode: it owns the phase and carries a manifest, like `translate`.
  Mode names are `plan` / `dubplan`; **`preview` was already taken** as a
  `UTIL_MODES` key by the v0.11 voice preview.
- `MODE_STAGES` gains `plan = {S1a, S1b}` and `dubplan`.
- `V5.parse_plan_file` (the Lua half of the plan format), `V5.enter_plan_phase`,
  `V5.ui_phase_plan` (the gate: 🌐 Open preview · 📝 Edit plan · ⟲ Reload ·
  📁 Folder, a verdict tally, the chunk table with the same capability-probed
  `BeginTable` / plain-rows fallback as the review editor, and a green
  ▶ Approve & Generate), `V5.start_plan_run`, `V5.start_dubplan_run`,
  `V5.plan_reload`. All on `V5` — the main chunk is at Lua's 200-local limit.
- `⟲ Reload` re-runs `--steps plan` against the SAME script file the first run
  wrote, so an edited plan is re-measured without re-pasting the script and
  without re-transcribing. **Superseded in v0.15** — re-spreading the script
  discarded every `TR:` edit; Reload now passes the plan file itself.
- Setup gains a `🔍 Preview sync` button beside Run, enabled only in the
  "I have a script" mode (it needs the pasted target text).
- `build_engine_cmd` gains `--plan`; `_finish_run` routes `status == "plan"`;
  a finished `dubplan` run records in the per-project history like any dub.

## v0.12 — Clause-sized pieces (sentence boundaries were still too coarse)

v0.10 cut per SENTENCE, which is coarse for these scripts: Indic
translations chain clauses with `;`, `,` and dashes, so one sentence ran
up to ~15 s and landed as a block. Measured on a real 3.4k-char Telugu
script: 53 sentence pieces, 7 of them over 8 s, worst 14.7 s. The
pre-v0.7 pipeline avoided this by cutting the TTS audio wherever it went
quiet (80 ms at −42 dB) — those silences ARE the clause marks.

- `tts._split_script_into_units(text, max_chars, min_chars)`: sentence
  split first, then any unit over `max_chars` subdivided at the hierarchy
  the Auto Sync matcher prompt already documents — `;`/`:`, then `,`,
  then a dash. Punctuation stays with the LEFT part and only whitespace is
  consumed, so re-joining the units reproduces the text exactly: the TTS
  request (and therefore prosody) is unchanged by how we cut.
  `_merge_tiny_units` folds fragments under `min_chars` (18) into a
  neighbour — a stray two-word piece desyncs easily and clutters Un sync.
  A sentence with no clause mark is left WHOLE; nothing is ever cut
  mid-phrase.
- `CLAUSE_MAX_CHARS = 60` (≈4 s at the ~14.3 chars/s these voices
  average). Same script: ~85 pieces, worst 5.7 s, none over 6 s. Below
  ~55 nothing improves — the marks run out. Overridable per install via
  `chunk_max_chars` in `engine_settings.json` (values < 20 ignored).
- `--chunk-mode` gains `clause` and it is the **new default**;
  `sentence` (v0.10) and `section` (v0.7) are kept verbatim as fallbacks.
  Panel: Settings → Advanced → "Dub piece size" offers all three.
- S2d logs the unit count and the longest unit in seconds, so a run that
  still produces blocks says so in its own log instead of only in the
  arrange view.
- Everything downstream is untouched: clause units flow through the same
  matcher, `build_pieces`/`place_pieces`, timestamps/texts/SRT/manifest
  and importers. Only the number of pieces changes.

## v0.10 — Sentence-sized pieces cut at ElevenLabs' own character times

Match mode's v0.7 pieces were one per matched THOUGHT (several sentences),
placed as blocks — reported as "coming in bunches" with drift inside each
block. Pieces are now one per SENTENCE by default, without giving up the
one-long-recording voice quality and without any re-transcription.

- `--chunk-mode sentence|section` on run_dub.py + dub_engine.py (also a
  `chunk_mode` key in `engine_settings.json`, written by the panel's
  Settings → Advanced → "Dub piece size" combo; CLI wins; default
  **sentence**). `section` is the v0.7 behaviour, kept verbatim.
- TTS: `tts.synthesize_sentences_elevenlabs` packs sentences into long
  requests (never splitting inside a sentence; an oversized single
  sentence gets its own request) and calls the
  `/v1/text-to-speech/{voice}/with-timestamps` endpoint —
  `_elevenlabs_tts_post_ts` — which returns the audio AND per-character
  times. Sentence spans are read from `alignment` (the ORIGINAL-text
  table, never `normalized_alignment`: our offsets index the text we
  sent). The last sentence of each stretch keeps the stretch's audio tail
  so the decay is never clipped. Stretches are joined with the same 240 ms
  gap as sections; spans exclude it. Defensive: a short/absent alignment
  degrades to a proportional estimate for that stretch, loudly.
- Matching is unchanged (same single Gemini sections call). `match.
  build_pieces` explodes each section into one piece per sentence; the
  section keeps the certainty, and its English window is sliced among its
  sentences proportionally by character share (last slice takes the
  remainder, slices tile the window exactly).
- `match.place_pieces`: the v0.7 rounds against each piece's own window
  slice, then the order sweep with ONE-level bounded borrowing — a piece
  that would overrun the next piece's target may shift that single
  neighbour later by ≤ `CASCADE_MAX_S` (1.5 s), only when the neighbour
  still ends before ITS successor's target. Anything else demotes to the
  Un sync chain exactly as before. Un sync items are now single sentences
  (drag once), not multi-sentence blobs.
- Downstream contract UNCHANGED: same timestamps txt (6th [status]
  field), texts sidecar, synced-only SRT + render
  (`extend_last=False`), manifest counts, importer behaviour. Sentence
  mode simply produces more, smaller entries.

## v0.7 — Auto-Sync-style matching for dub runs, Un sync track, version, history

### Sync mode (engine)

- `--sync-mode match|legacy` on run_dub.py + dub_engine.py (also a
  `sync_mode` key in `engine_settings.json`; CLI wins; default **match**).
  Applies to the dub half of `--steps full` and to `--steps dub`; regen /
  voice-change / test-llm / list-voices are unaffected.
- **match** (new default) replaces the S3b/S3c/S3d internals for BOTH the
  Full Pipeline and Paste Translation runs; **legacy** is the old
  whole-script-TTS + Scribe-re-transcription + SyncingPrompt path,
  kept verbatim in `_stage_dub_legacy`.
- Match flow (`pipeline/match.py` + `tts.synthesize_sections_elevenlabs`):
  1. The dub script (LLM translation or `--provided-script` text, possibly
     user-edited at review) is split into sentences
     (`tts._split_script_into_sentences` — danda-aware, tag-safe).
  2. ONE Gemini call (`call_match_sections`, transported through the
     provider-agnostic `_llm_generate`, so vertex/gemini/gateway all work)
     groups EN sync-SRT cue ids with script sentence ids into sections —
     the same section prompt idea as the fast-syncs `sync_matcher.py`.
     Retries once, then HARD-fails (no silent fallback, like Auto Sync).
     Unmentioned ids are mechanically filed as unmatched.
  3. `build_chunks`: one TTS chunk per matched section; consecutive
     unmatched sentences merge into unsync chunks; script order kept.
  4. Per-section ElevenLabs TTS with request stitching (previous_text /
     next_text = neighbouring script text) into ONE `tts_wav`, 240 ms
     silence between sections; exact per-section spans returned — S3b's
     second Scribe pass and S3c's SyncingPrompt call are GONE in this mode
     (S3a-S3c are printed as book-keeping lines so the panel checklist
     still advances in order).
  5. `place_chunks`: the Auto Sync placement — slot from the section's EN
     cues, rounds center/align_start/align_end ×2 iterations,
     align_start fallback (bleed-over), then the order-preserving sweep:
     a chunk that cannot end before the next chunk's spring position is
     demoted to **unsync** with a chain position (each unsync chunk sits
     right after the previous clip, chronological). No silence correction
     (TTS starts at speech; EN cue starts are word-refined already).
  6. Step-4 emotion enrichment is SKIPPED in match mode (logged at S2d):
     per-section enrichment would multiply LLM calls and whole-script
     enrichment would break the sentence-id mapping.
- Timestamps file gains an OPTIONAL 6th bracket `[synced]` / `[unsync]`
  (header gains `[Status]`). 5-field files stay byte-identical and every
  reader treats a missing status as synced — old runs import unchanged.
  `_parse_timestamps_text` returns it as `sync_status`.
- New sidecar `<base>_sync_texts.txt`: blank-line-separated blocks, block N
  = chunk text for timestamps index N (per-item text for BOTH tracks; the
  synced SRT covers only synced chunks, so the text needs its own channel).
- `_sync_synced.srt` = synced chunks only (it must not describe the
  Un sync chain). `_synced.wav` renders synced chunks only, via
  `sync_audio_with_timestamps(..., extend_last=False)` (spans are exact;
  extending the last segment would drag trailing unsync audio in).
- Manifest (full/dub) gains `sync_texts`, `synced_count`, `unsynced_count`
  (strings; "" from legacy mode — consumers skip empties, as always).

### Import (both importers)

- Entries parse the optional trailing `[synced]/[unsync]` bracket (letters
  only, so a 5-field line's `[1234ms]` tail can never match).
- Synced entries → `Dub Chunks` (fresh-suffix rule unchanged). Unsync
  entries → the **`Un sync`** track — find-by-exact-name or append, the
  SAME name + reuse rule as `auto_sync_pipeline.lua`, so Auto Sync and dub
  runs park leftovers on one track. Item take names: `unsync NN`.
- Per-item chunk text prefers the `sync_texts` sidecar (block by entry
  index) and falls back to the synced-SRT cue matching. Summary line reports
  "Synced chunks placed / Un sync chunks" when any unsync exist; the
  panel success phase shows `Chunks: N synced, M unsynced` from the
  manifest counts.

### Version (the "which build am I on" answer)

- Root `VERSION` file (started at 0.7.0, currently 0.9.0), shipped in the
  repo so git pull / ZIP overlay updates it. Shown: dub panel title bar and
  Settings tab (`V5.APP_VERSION`), Auto Sync standalone title bar, engine log
  banner (`[engine] Reaper Dubbing App vX (contract v0.7)`).
- **Every shipped change bumps `VERSION` in the same commit.** It is the only
  way a user can tell whether their update actually landed — a fix that ships
  under the old number reads as "the update did nothing". Minor bump for
  features and behaviour changes, patch bump for fixes.
- Both ImGui windows now carry a `###` ID suffix (`###dub_pipeline`,
  `###auto_sync_pipeline`) so the version text in the title never resets
  the saved window position again (the one rename to add the suffix does,
  once).

### A model per stage (engine + panel)

- `llm_settings.json` gains `model_translate`, `model_emotion`,
  `model_match`, `model_mapping`, `model_sync_match`. **Blank is the
  default and means "use the provider model"**, so an upgraded install
  behaves exactly as before until a field is filled in.
- `_llm_generate(prompt, model, static_prefix, role=None)` resolves the
  model through `_model_for(role, fallback)`: `model_<role>` → the
  provider-wide model (`openai_model` / `gemini_model`) → the caller's
  default. Roles are named at the call sites: `translate` (Step1-3),
  `emotion` (Step4), `match` (v0.7 section matching), `mapping` (legacy
  S3c). `--test-llm` deliberately passes no role — it tests the main model.
- `_llm_role_overrides_label()` is printed in the startup banner whenever
  anything is overridden, for the same reason `_llm_provider_label()` is:
  a run that used a different model than the Settings field shows must
  say so in its own log.
- `model_sync_match` is Auto Sync's, and Auto Sync reads its model from
  `sync_pipeline_settings.json` — so `save_sync_credentials()` writes that
  value into `gemini_model` there (falling back to the main model when
  blank). The engine never reads `model_sync_match`.

### User-added languages (engine + panel)

- `config/custom_languages.json`:
  `{"languages":[{"name","code","tag"}]}` (gitignored with the rest of
  `config/`). A language is a name plus five prompt files — nothing else
  in the engine is language-specific.
- `pipeline/config.py::_load_custom_languages()` merges each entry into
  `TTS_LANGUAGES` at import (marked `google_unavailable`; ElevenLabs
  detects the script from the text). A name that collides with a built-in
  is **ignored, not overwritten** — the shipped entries carry Google voice
  lists a hand-written one cannot.
- `dub_engine.py` and `run_dub.py` extend their `--language` choices from
  the SAME file with their own stdlib-only read: argparse builds choices
  before the pipeline import, and the launcher validates `--language`
  before spawning the worker, so both lists must agree.
- `--selfcheck` treats missing prompts for a user-added language as a
  WARNING naming the files, not a failure. A half-configured extra
  language must never block setup for the twelve built-ins.

### Prompt editing (panel)

- Settings → **Prompts**: language combo × stage combo → the file's text in
  an editable box, with Save / Reload / Open in editor.
  `dubbing/prompts/<Stage>_<Language>.txt`, the same files the engine
  loads through `_load_lang_prompt()`.
- Settings → **Languages** → *Add language* writes the JSON entry and
  seeds the five prompts from an existing language (`V5.prompts_seed`,
  which never overwrites an existing file). *Remove* drops the entry and
  leaves the prompt files on disk — deleting user-edited text on a
  mis-click is not recoverable.

### Voice bookmarks + search (panel)

- `reaper/voice_bookmarks.json` (gitignored, GLOBAL — not per project, not
  per language): `{"voices":[{"id","name"},…]}`, the SAME shape the
  `--list-voices` manifest uses, so `parse_voices_json` reads it unchanged.
- `V5.ui_voice_picker(ctx, key, cur, label)` is the one voice widget, used
  by ⚙ Settings, Track Voice and Text to Speech (`key` keeps the ImGui ids
  unique). It renders a search box (case-insensitive substring on name or
  id, `find(…, 1, true)` so a name with `(`/`-` is not read as a pattern),
  one combo listing bookmarks (★) before the fetched catalogue, and a
  star/unstar button for the current voice. Returns the chosen id; every
  host keeps its manual "Voice id" field, which still wins.
- Bookmarks survive panel restarts and language switches; the fetched
  catalogue (`_voices`) does not, which is the whole point.

### Text to Speech tab (panel)

- Paste text → synthesize → item on a `TTS` track at the EDIT CURSOR
  (find-or-create track, same rule as `Un sync`). Hidden item text = the
  spoken text; take name = the wav name.
- Runs the EXISTING `--regen-chunk` engine mode (text file in, wav out, no
  emotion, no other stages) — no new engine mode, no contract change on the
  Python side. Only the finish handling differs from Regen Audio: the wav
  is imported instead of replacing an item's take.
- `tts` is a UTIL_MODE: it never owns the phase state, never clears the
  last run's manifest or a pending review, and returns to the phase it was
  started from. It has no entry in `MODE_STAGES`, so no stage checklist is
  drawn (same as regen).
- Files land in `<project media path>/TTS/` as `TTS_<stamp>.txt` (UTF-8 —
  Indic text never travels on argv) and `TTS_<stamp>[_vN].wav`. An unsaved
  project is refused with a message, since there is no media path yet.
- Voice resolution: the tab's own voice → ⚙ Settings voice → engine
  auto-resolve. `--language` is still passed (it only matters for
  auto-resolution; eleven_v3 detects the language from the text).

### Per-project run history (panel)

- `engine/history/<project-slug>.json` (same slug as the status dirs;
  gitignored; NOT under engine/status/, which run_dub.py wipes per launch).
  Shape: `{"entries":[{"ts","mode","audio","language","out_dir","status"},…]}`
  newest first, deduped by out_dir (a finished dub replaces its review
  entry), capped at 20. Writers: `enter_review_phase` records "review",
  `_finish_run` records "ok" for full/dub runs. The history file only
  INDEXES runs — the authority stays each run's `<out_dir>/engine_done.json`
  (written there since v0.1 precisely because status dirs are transient).
- Setup phase (Full Pipeline AND Paste Translation tabs) shows a
  "Project history" section for the ACTIVE REAPER project (slug re-checked
  every frame, so switching project tabs swaps the list). Per entry:
  **Resume review** (reload the out_dir manifest → the normal review
  phase; transcription + translation are NOT redone), **Import to
  timeline** (finished runs; re-imports from the out_dir manifest),
  **Use audio + language** (prefill setup), **Folder**.
- Unsaved projects share the `unsaved` slug — their history is one bucket
  by design (same trade-off as the status dirs).

## v0.6 — One entry point, credentials that cannot be half-configured

### One documented ReaScript

- `auto_sync_pipeline.lua` (fast-syncs root) is the ONLY script a user loads.
  It already redirects to the dub panel (see the v0.5 embed section); v0.6
  makes the docs and both setup scripts say only that. `Sync_Item.lua` moved
  to `scripts_optional/` so the repo root lists exactly one `.lua`.
- `dubbing/reaper/*.lua` are INTERNAL: loaded for the user, never advertised.
  They stay where they are — `Dub_Pipeline_Panel.lua` derives `BASE_DIR` from
  its own parent directory and keeps `dub_panel_settings.json` beside itself,
  so moving or renaming it would break every engine/venv/config path and
  orphan existing panel settings.
- `Import_Dub_Results.lua` is the ONE exception a user may load directly, and
  only when ReaImGui is unavailable.

### Credentials (a blank key must never reach a paid run)

- `_validate_llm_config()` also requires `openai_api_key` for the OpenAI
  provider, UNLESS the base URL host is loopback / RFC-1918 / `*.local` (a
  keyless local Ollama, vLLM or LiteLLM is legitimate). Shared rule:
  `_gateway_needs_key()` in `pipeline/llm.py`, mirrored by
  `V5.gateway_needs_key()` in the panel — keep the two in step.
- A blank key means NO `Authorization` header, and gateways answer that with
  wording about the key being wrong ("No api key passed in."). So a 401/403
  raised while no key was sent must say the key is MISSING, quoting the
  gateway's own text only as trailing detail.
- `_openai_api_urls()` rejects a base URL ending in `/ui`: that is the gateway's
  admin console pasted in place of the API root. LiteLLM answers the resulting
  `/ui/v1/chat/completions` with `405 Method Not Allowed`, which names nothing,
  and the existing HTML-body guard cannot help because the 405 body is JSON.
  Seen in the wild — a panel configured from the browser address bar.
- The engine banner prints `_llm_provider_label()` — active provider, model,
  base URL and whether the credential is set. Never the
  `GEMINI_DEFAULT_MODEL` constant: a gateway run used to log "gemini-2.5-pro".
- `_load_llm_settings()` coerces non-string scalars and PRINTS a line for any
  other type. A silent drop to `""` turns a configured key into a 401.
- Panel saves are blank-preserving: an empty credential box keeps the value
  already on disk (`V5.keep_stored_credentials()`), because the save rewrites
  every field. A per-field `Clear` button is the only way to remove a key.
- `preflight_engine(need_llm)` refuses LLM runs (`full`/`translate`/`dub`)
  when `V5.llm_creds_error()` reports a gap. `--test-llm` and the LLM-free
  modes (`--regen`, `--voice-change`) are never gated.
- Saving Settings with complete credentials auto-runs `--test-llm`; with an
  incomplete set it shows a warn banner naming the gap. "Settings saved" is
  never the last word on a config that cannot run.

## v0.5 — Merged into fast-syncs: 7-tab panel, embedded Auto Sync, per-project status dirs, robust updater

### Repo

- The whole app now lives at `<fast syncs>/dubbing/` and is distributed by
  the fast-syncs updater (git pull / ZIP overlay — the overlay never
  deletes, so `dubbing/venv`, `dubbing/config`, `dubbing/data` survive).
  `update.sh` / `update.bat` additionally refresh `dubbing/venv` from
  `dubbing/requirements.txt` WHEN that venv exists (first-time setup stays
  manual/panel-offered — dubbing deps are heavy and sync-only users must
  not pull them).
- `auto_sync_pipeline.lua` gains a small `Dubbing...` button (setup-phase
  footer) that registers `dubbing/reaper/Dub_Pipeline_Panel.lua` via
  `AddRemoveReaScript(true, 0, path, true)` and runs it.
- All paths stay self-relative — nothing depends on the folder's location.

### Engine

- `run_dub.py --status-dir <abs>`: per-run status directory for
  log/pid/done/manifest. MUST be `engine/status` itself or a subdirectory
  (launcher refuses anything else — it cleans the target). Subdirs are
  cleaned with rmtree; the shared root is cleaned file-by-file so a legacy
  root-mode launch can never wipe a live sibling subdir. The dir is passed
  to `dub_engine.py` via the `DUB_STATUS_DIR` env var (manifest must land
  in the same per-run dir).

### Panel — one TabBar, seven tabs

`Full Pipeline` (all phases; staged/review default) · `Paste Translation`
(own audio+language+script form → runs with `--provided-script`) ·
`Auto Sync` (the embedded fast-syncs pipeline — §Auto Sync embed) ·
`Regen Audio` (the `ui_regen_section` utility) · `Track Voice`
(`ui_voice_change_section`) · `Logs` · `Settings` (LLM/TTS keys + Advanced
python override + fast-syncs `Update…`; locked while a dub run is active).

- The tab a run starts from decides the script source (Full Pipeline → LLM,
  Paste → provided) — the old "I already have the translation" checkbox is
  gone. Regen/Track Voice are no longer inline in setup/success — they are
  their own tabs, usable any time (their own launch still goes through
  `preflight_engine`, which blocks starting one over a live run).
- Per-project status dirs: `engine/status/<projname>_<djb2(projpath)>/`
  (`unsaved` for unsaved projects). Re-resolved at load and in preflight
  ONLY while `_ui_phase == "setup"`; a run keeps its launch-time paths.
  `--status-dir` is passed on EVERY engine invocation. The
  previous-run-alive probe is scoped to this status dir (`ps … | grep
  --status-dir <dir>`, not a bare `pgrep run_dub.py`). The startup review
  probe also checks the legacy shared root once (pre-v0.5 paused reviews
  stay resumable).
- Import: the launch-time project is remembered
  (`reaper.EnumProjects(-1,"")`), and Import switches back to it via
  `SelectProjectInstance` if the user changed project tabs mid-run.
- Missing venv at preflight → MB offer to run the platform setup script in
  a terminal.
- NOTE: every v0.5 symbol lives in the single `V5` table — the main chunk
  is at Lua's 200-local limit; add new file-scope state as `V5.*` fields.

### Auto Sync embed (dub panel ↔ auto_sync_pipeline.lua)

- `auto_sync_pipeline.lua` is DUAL-MODE. Standalone (run as an action) it
  behaves exactly as before. When `_G.__FASTSYNC_EMBED == true` at load, it
  skips its own `ClearConsole`/`CreateContext`/font-attach/`Begin`/`defer`
  and instead `return`s a module `{ render(ctx,on_close), poll(),
  is_running(), reload_settings() }`.
- The dub panel `dofile()`s it once at startup (`V5.load_sync`), setting the
  flag around the call. Because it is a separate `dofile` chunk it has its
  OWN fresh Lua-local budget AND its own copies of `_ui_phase`, poll offsets
  etc. — no collision with the dub panel's identically-named locals.
- `V5.SYNC.poll()` is called every dub frame (outside the tab bar) so a sync
  run progresses regardless of the active tab; `V5.SYNC.render(ctx,
  close_window)` draws the sync phases inside the `Auto Sync` tab, sharing
  the dub panel's ImGui context. In embed the sync uses the default font
  (no cross-context `Attach`); pasted Indic `script_text` may not shape
  in-panel (cosmetic).
- Sync paths stay rooted at the fast-syncs root (the sync file's own dir),
  so `run_sync.py`, the sync `venv`, `sync_pipeline_settings.json`,
  `sync_config.json`, `sync_results.json`, and `sync_python_{log,pid,done}`
  are all the fast-syncs ones — NOT the dubbing ones. The sync tab needs the
  fast-syncs root `venv` (root `setup.sh`/`.bat`), separate from
  `dubbing/venv`. The standalone sync `Dubbing...` button is hidden in embed
  (it would re-open the dub panel).
- If `auto_sync_pipeline.lua` is absent (dubbing installed standalone, not
  under fast-syncs), the `Auto Sync` tab shows `V5.sync_err` instead.

### Update button (both this panel and auto_sync_pipeline.lua)

- The `osascript -e 'tell application "Terminal" …'` launch was replaced:
  it needs macOS Automation permission (REAPER→Terminal) and fails SILENTLY
  when that was never granted — the "Update/Setup does nothing" reports.
  Now: macOS `open`s a generated `.command` wrapper in `$TMPDIR` (no
  Automation permission, no Gatekeeper quarantine on a locally-written
  file); Windows uses `start "" "<script>"` (empty title so a spaced path
  isn't taken as the title) instead of the fragile nested-quote `cmd /k`.
  A `.command` target is `open`ed directly. The dialog now also prints the
  exact shell command as a manual fallback.
- `update.sh`/`update.bat`: `git pull --ff-only` failure (diverged / locally
  modified tracked files) no longer aborts the update — it falls back to the
  ZIP overlay (add/overwrite, never delete; gitignored venv/settings safe).

## v0.4 — Provided translation, track voice change, log tab, Windows

### Engine changes (all previous modes/flags/manifests unchanged)

- `--provided-script "<abs .txt>"` (run_dub.py + dub_engine.py). Valid with
  `--steps translate` and `--steps full` only (dub already takes `--script`;
  invalid with regen/voice-change). The UTF-8 file replaces the S2a–S2c LLM
  translation chain: S1a/S1b still run (sync needs the transcription and
  regions), the TM lookup is skipped, and the file's text is used verbatim
  as tr/rev/punc result (S2a–S2c log "skipped" lines). Read + validated
  BEFORE any paid API call. Everything downstream (review files, FinalScript,
  manifests) behaves exactly as if the LLM had produced that text.
- `--voice-change --in-wav "<abs>" --out-wav "<abs>" --language <Lang>
  [--voice-id ID] [--sts-model M]` — ElevenLabs speech-to-speech re-voice
  of an existing audio file (new pipeline helper
  `tts.voice_change_elevenlabs`). Voice resolution follows the same rule as
  every other mode (explicit id wins, else first language-matched account
  voice). Long inputs are split into ≤ ~4-minute chunks at the quietest
  point near the boundary, converted per chunk
  (`POST /v1/speech-to-speech/<voice>`, model default
  `eleven_multilingual_sts_v2`), joined, and written as WAV to --out-wav.
  Manifest: `{"status":"ok","vc_wav":"<abs>"}` (status-dir copy only — no
  out_dir of its own). Goes through run_dub.py like every mode (same
  status/log/pid/done plumbing); no LLM required.

### Panel changes

- Tabs: one TabBar with " Pipeline " (all phases) and " Log " (full live
  log + autoscroll + open-in-editor). The running phase shows progress,
  stage line and checklist WITHOUT the log, then the setup UI read-only
  (BeginDisabled) below it, so settings are never hidden by log output.
  Poll runs outside the tabs (switching tabs cannot stall a run). Fallback
  for tab-less ReaImGui: old inline layout.
- Per-language fonts: fonts are created+attached lazily PER SCRIPT and
  follow the language combo live. Candidates: user-installed Noto Sans/Serif
  per script, then macOS `/System/Library/Fonts/Supplemental/<Script> Sangam
  MN.ttc` (+ MT/Kohinoor variants, Arial Unicode fallback), Windows
  `Nirmala.ttf` (covers all 12 languages) + segoeui fallback. Dear ImGui
  still has no complex shaping — documented as cosmetic; the clipboard
  round-trip below is the supported perfect-rendering path.
- Review phase toolbar: 📋 Copy script (whole translation →clipboard via
  ImGui_SetClipboardText), 📋 Copy English, 📥 Paste script (clipboard
  replaces the whole translation, blank-line paragraph re-split), Open in
  editor (saves then opens `<base>_translation_edited.txt`), ⟲ Reload file
  (re-reads edited file, else the engine's translation_text).
- Setup phase: "I already have the translation" checkbox (persisted as
  `script_mode`: auto|have) + multiline paste box (+ paste-from-clipboard /
  clear buttons, paragraph+char count). On Run the text is written to
  `<out_dir>/<base>_provided_translation.txt` (out_dir mirrors
  `_prepare_output_dir` convention) and passed via `--provided-script`.
  Staged runs still pause for review (pairing check); full runs go
  straight through.
- 🎤 Change track voice section (setup + success phases): track combo
  (any project track), voice from the fetched catalogue and/or manual id
  (persisted as `vc_voice_id`; falls back to the Settings voice). Flow:
  render the track to `<project media path>/VoiceChange/<name>_<ts>.wav`
  via the render API (RENDER_SETTINGS=3 selected-track stems, custom
  bounds 0..last item end, mono, "evaw" format, action 42230, every
  touched setting saved/restored) → engine `--voice-change` → on ok, new
  track "<name> (voice changed)" inserted directly below the original with
  the result wav at position 0; the original track is MUTED, never
  modified. voice_change is a utility mode (returns to its origin phase,
  keeps manifests).
- English audio from a project track (setup phase): "From track" combo +
  "Use track" button next to the file picker. Resolution rule
  (audio_from_track): a track with exactly ONE item that is untrimmed
  (STARTOFFS≈0), unstretched (playrate≈1) and full-length (item length ≈
  source length ±50 ms) → the item's source FILE is used directly (section/
  reverse wrappers unwrapped via GetMediaSourceParent); anything else →
  the track is rendered (same render_track_stem helper) to
  `<project media path>/DubSource/<name>_<ts>.wav` and that wav is used.
  render_track_stem also temporarily unmutes the target track during the
  render (muted tracks render silent) and restores the mute state after.
- OS-aware setup hints: every error/hint names `setup_windows.bat` on
  Windows, `setup_mac.command` elsewhere.

### Repo

- `setup_windows.bat`: Windows twin of setup_mac.command (py-launcher
  discovery rejecting the Store alias, local venv, requirements install,
  selfcheck, ffmpeg check with winget offer, REAPER install steps,
  re-run safe, CRLF).

## v0.3 — Standalone app (GitHub-shareable, zero bulk-app dependency)

### Architecture

`dub_engine.py` no longer imports `Translation_and_Syncing_App`. All needed
logic is EXTRACTED (copied verbatim where possible, minimally adapted:
UI/global refs removed, settings paths repointed) into:

```
engine/pipeline/
  __init__.py
  config.py      # languages table, constants, ffmpeg discovery, settings loaders
  stt.py         # ElevenLabs Scribe STT + voice catalogue helpers
  srt_tools.py   # region detection, SRT builders, SpaCy chunking (optional dep),
                 # LLM input formats, timestamps txt read/write
  llm.py         # provider layer (vertex | gemini | openai-compatible),
                 # prompt loading, 3-step chain, emotion, EN<->target mapping
  tts.py         # ElevenLabs TTS (+ Google Cloud TTS port), chunkers, stitching
  sync.py        # sync engine (springs/bleed-over/order sweep), audio
                 # reassembly, captions rechunk
  tm.py          # translation_memory port (SQLite, data/ dir inside this repo)
prompts/         # copied from APP_DIR/prompts (all languages, 5 stages)
config/          # ALL user secrets/settings — entire dir gitignored
  llm_settings.json   # same schema as bulk app's llm_settings.json
  tts_settings.json   # {"elevenlabs_api_key","el_model","voice_id",
                      #  "google_tts_key_path"}
  vertex_key.json     # optional, user-provided
  TTS_Key.json        # optional, user-provided
data/            # translation_memory.db lives here (gitignored)
```

Provenance note: each pipeline module starts with a comment naming the source
file + line ranges it was extracted from.

### Engine changes

- All v0.1/v0.2 CLI modes keep IDENTICAL flags/manifests — panel unaffected
  except new flags below. `--app-dir` becomes OPTIONAL and IGNORED at runtime
  (accepted for backward compat, warning logged).
- Keys/settings resolution: `config/llm_settings.json` + `config/tts_settings.json`
  ONLY. Clear actionable error when missing ("run setup / open panel Settings").
  `openai_base_url` is an API base (host, or host + path such as `/v1`), never
  the chat-UI address and never with `/chat/completions` appended; the panel
  strips those on save. Optional `http_user_agent` overrides the engine's HTTP
  agent string for gateway calls (blank → browser default, needed because
  Cloudflare-fronted gateways reject `Python-urllib/*` with 403 / code 1010);
  `DUB_HTTP_USER_AGENT` in the environment does the same. The panel round-trips
  the key so a Settings save cannot drop it.
- Credentials are entered in ONE place: the panel's Settings tab. Every save also
  mirrors the shared keys into `<fast-syncs>/sync_pipeline_settings.json`
  (Auto Sync's own file, read by `run_sync.py`), touching only those keys —
  tracks, language, match mode, script text and `python_cmd` stay as Auto Sync
  left them. Vocabulary differs by design: `provider` here vs `conn_mode` there
  (`gemini`→`studio`, `openai`→`gateway`), and Auto Sync keeps both the Gemini
  key and the gateway Bearer token in one `gemini_key` field, so the mirror hands
  over whichever matches the selected provider. `run_sync.py` back-fills any
  credential missing from the sync file from `config/llm_settings.json`.
- `provider` may also be `"Server proxy (Auto Sync only)"` (alias `server`) with
  `server_url` + `server_token`: Auto Sync routes all AI calls through the user's
  own server. The engine has no server path, so every LLM entry point raises an
  actionable error instead of falling back to another provider's key.
- New modes (all through run_dub.py, same status/log/pid/done plumbing):
  - `--test-llm` → manifest `{"status":"ok","provider":"…","model":"…","reply":"…"}`
    (one tiny LLM call) or status error.
  - `--list-voices --language <Lang>` → manifest `{"status":"ok","voices":
    [{"id","name"},…]}` from ElevenLabs, language-token sorted like the app.
- `--selfcheck` reworked: imports pipeline modules (not the app), asserts
  required symbols + prompt files exist, prints SELFCHECK OK.

### Panel changes

- New ⚙ Settings section (collapsible, setup phase):
  - LLM: provider combo (vertex | gemini | openai) + model text field
    (default gemini-2.5-pro) + per-provider fields (gemini API key masked,
    vertex key path, openai base URL + key masked) + "Test connection" button
    (runs --test-llm, shows reply/error).
  - TTS: ElevenLabs key (masked), model combo (4 models), voice: "Fetch
    voices" button (--list-voices) filling a combo, manual voice-id fallback.
  - Panel WRITES config/llm_settings.json + config/tts_settings.json directly
    (same JSON-escape discipline as dub_panel_settings.json). Engine reads them.
- Engine location unchanged (sibling engine/). app_dir setting removed from
  UI (kept harmless in old settings files).

### Repo prep (GitHub-ready)

- `git init` in project root. `.gitignore`: `config/`, `data/`, `venv/`,
  `__pycache__/`, `engine/status/`, `*.pyc`, `.DS_Store`, `dub_panel_settings.json`.
- `requirements.txt` (engine deps only: numpy, pydub, google-genai,
  google-cloud-texttospeech, google-auth, pyphen; spacy optional — document).
  NO tkinter/matplotlib/sounddevice/opencv — nothing UI-related.
- `setup_mac.command` rewritten: creates LOCAL `venv/` in project root,
  pip-installs requirements, offers one-time key MIGRATION by copying (never
  moving) from the bulk app install if found at the known path (api.txt →
  tts_settings, llm_settings.json, vertex_key.json, TTS_Key.json, el_model.txt),
  prints REAPER script-install steps. Re-run safe.
- Python discovery order (panel + docs): project `venv/` FIRST, then old
  app-venv fallback (transition), then system candidates.
- README rewritten for a public audience: what it is, requirements, setup,
  keys (bring-your-own, stored only in gitignored config/), usage, file map,
  credits note that it's a REAPER port of an internal dubbing pipeline.

### Out of scope v0.3 (documented as roadmap)

Batch processing, TTS-Studio-style per-sentence editor, history/re-dub
manager, multi-speaker per-paragraph voices, prompt editor UI (prompts are
plain files — edit in any editor), updater/feedback systems.



## Fixed paths & names

- `APP_DIR` (default): `/Users/ilp/Documents/Claude code/Akash anna Translation and Syncing App_All`
- App module: `Translation_and_Syncing_App.py` (import as module — entry point is
  guarded by `if __name__ == "__main__"`, so importing never opens the Tk UI).
- App venv python (has ALL deps): `APP_DIR/.venv/bin/python3` (mac). Windows:
  `APP_DIR\.venv\Scripts\python.exe`.
- This project layout:
  ```
  Reaper Dubbing App/
    CONTRACT.md
    README.md
    setup_mac.command
    engine/
      run_dub.py          # launcher (env, tee log, pid, done marker)
      dub_engine.py       # worker: imports app module, runs pipeline
      hidden_run.py       # Windows: re-launch a helper with no console window
      engine_settings.json# written by setup: {"app_dir": "..."}
      status/             # created at runtime by run_dub.py
        engine_log.txt    # live tee of worker stdout+stderr
        engine_pid.txt    # worker PID (for cancel)
        engine_done.txt   # written LAST: single line = exit code
        engine_done.json  # copy of result manifest (also written to out_dir)
    reaper/                    # internal: loaded by auto_sync_pipeline.lua
      Dub_Pipeline_Panel.lua   # ReaImGui panel (run + poll + import)
      Import_Dub_Results.lua   # standalone importer (no ReaImGui needed)
  ```

## Engine CLI (Lua → Python)

```
"<python>" "<.../engine/run_dub.py>" --app-dir "<APP_DIR>" --audio "<audio path>" \
    --language <Language> [--voice-id <ELid>] [--el-model <model>] [--steps full]
```

Setup-time self-check (the ONLY supported direct `dub_engine.py` invocation —
used by `setup_mac.command`; all real runs go through `run_dub.py`):

```
"<python>" "<.../engine/dub_engine.py>" --selfcheck --app-dir "<APP_DIR>"
```

- `--selfcheck`: imports the app module headlessly and verifies every
  required module-level symbol exists — no network, no audio processing,
  no status files, no manifest. Prints `SELFCHECK OK` and exits 0 on
  success; non-zero on failure. `--audio`/`--language` are not required
  with this flag.
- `--language`: display name, one of: Assamese Bengali Gujarati Hindi Kannada
  Malayalam Marathi Nepali Odia Punjabi Tamil Telugu
- `--el-model` default `eleven_v3`; `--steps` only `full` for v0.1.
- `run_dub.py`: stdlib only. Deletes old status files, spawns
  `dub_engine.py` via subprocess (same interpreter, except that a pythonw.exe
  launcher spawns the worker as python.exe — see the no-console note below),
  tees combined
  stdout/stderr to `status/engine_log.txt` (line-buffered, utf-8), writes
  `status/engine_pid.txt` (its OWN pid pre-fork is useless — write the CHILD
  pid), on child exit writes `status/engine_done.txt` with exit code.
  No shell=True. Secrets never on the command line (keys come from APP_DIR
  files read by the app module itself: api.txt, llm_settings.json,
  vertex_key.json, TTS_Key.json — engine passes nothing).
- Progress protocol: worker prints lines `[S1a] message`, `[S2a] …` etc. —
  stage tags mirror the app's stages: S1a transcribe, S1b regions/SRT,
  S2a translate, S2b review, S2c punctuation, S2d TTS, S3a EN sync SRT,
  S3b TTS regions/SRT, S3c mapping, S3d sync, S3e render. Panel shows the
  last tag as current stage.
- Diagnostic lines from the pipeline modules carry a module tag instead of a
  stage tag — `[stt]`, `[LLM]`, `[config]`, `[engine]`. They are log-only and
  MUST NOT move the panel's stage: `stt.py` prints `[stt]` progress during
  both S1a and S3b, so a `[S1a]` tag there would drag the checklist backwards
  mid-dub.

## Result manifest — `engine_done.json`

Written by `dub_engine.py` to BOTH `<out_dir>/engine_done.json` and
`engine/status/engine_done.json`. UTF-8 JSON:

```json
{
  "status": "ok",              // or "error"
  "error": "",                 // traceback tail when status=error
  "audio": "<input audio abs path>",
  "language": "Bengali",
  "out_dir": "<abs path>",
  "en_audio": "<abs path to copied original audio inside out_dir>",
  "en_srt": "<abs>",           // <base>.srt
  "tts_wav": "<abs>",          // full TTS wav (dub voice, pre-sync)
  "timestamps_txt": "<abs>",   // <base>_sync_timestamps.txt
  "synced_wav": "<abs>",       // final synced wav
  "synced_srt": "<abs>"        // <base>_sync_synced.srt
}
```

Missing/not-produced files = "" (empty string). Consumers must skip empties.

## Timestamps file format (unchanged from app, line 1902 area)

```
[<idx>] [<orig_start>ms] [<orig_end>ms] [<orig_dur>ms] [<synced_start>ms]
```
Chunk audio source = `tts_wav`; item: STARTOFFS=orig_start, LEN=orig_dur,
POSITION=synced_start (all seconds in Reaper, file is ms).

## Import layout (both importers build the same thing)

Tracks appended at end of project, in order:
1. `EN Original` — one item, `en_audio`, position 0.
2. `Dub Chunks` — one item per timestamps line (source `tts_wav`, offsets per
   above). Chunk text = matching cue text from `synced_srt` (match by order;
   if counts differ, match by nearest start ≤0.5s; else leave it empty).
3. `Dub Rendered (ref)` — one item, `synced_wav`, position 0, track MUTED.
v0.8: the chunk text is stored in the hidden item ext state
`P_EXT:fastsyncs_chunk_text`, NOT in `P_NOTES`, and NO project regions are
created — both painted over the arrange view and hid the waveforms. Writers
also clear `P_NOTES` so pre-v0.8 items stop drawing text; readers fall back
to `P_NOTES` when the ext state is empty. Wrap in one undo block.
All in Lua; SRT parser must handle CRLF, multi-line cue text (join with " "),
and UTF-8 Indic text (byte-safe string handling only).

## Panel behavior (`Dub_Pipeline_Panel.lua`)

Adapt proven patterns from `/Users/ilp/Documents/Claude code/fast syncs/auto_sync_pipeline.lua`
(READ-ONLY reference — do not modify that file):
- `imgui_available()` probe + ReaPack install guidance (copy pattern).
- Settings persist to `dub_panel_settings.json` next to the Lua script:
  app_dir, python_cmd override, language, voice_id, el_model, last_audio.
- Python discovery order: settings override → `app_dir/.venv` python →
  fast-syncs-style candidates (`probe_python` pattern).
- Launch non-blocking: Windows `reaper.ExecProcess(cmd, -2)`; macOS
  `os.execute(cmd .. ' >/dev/null 2>&1 &')` — run_dub.py owns log/pid/done.
- No console windows on Windows. `ExecProcess(cmd, -2)` detaches the child but
  does NOT suppress its console, and a console on top of REAPER shows only
  what the panel's own log pane already shows. So:
  * the engine is launched with **pythonw.exe** — the same interpreter built
    without a console (`V5.pythonw` swaps `python.exe` → `pythonw.exe` when it
    exists beside it). `run_dub.py` still spawns the WORKER as `python.exe`
    (`_worker_python()`), which stays no-window via `CREATE_NO_WINDOW`;
  * `cmd.exe`/`curl.exe`/`powershell` helpers (update check, connection
    probes, preview playback) go through `V5.win_hidden`, which writes the
    command to a scratch `.bat` and has pythonw.exe run `engine/hidden_run.py`
    on it with `CREATE_NO_WINDOW`. **A batch file eats a lone `%`** — the
    generated .bat doubles them, or curl's `-w "%{http_code}"` would arrive as
    `"{http_code}"` and the probe would never report a status.
  Both routes fall back to the old visible-window command when pythonw.exe or
  hidden_run.py is missing, so a partial install still works.
- Anything the user needs to see belongs in the panel, never in a console:
  the engine's stdout+stderr is teed to `engine_log.txt` and rendered in the
  log pane; helper output is read back from files by the pollers.
- Poll in `reaper.defer` loop: tail `engine_log.txt` (read from last size),
  show stage from `[Sxx]` tags; when `engine_done.txt` appears → read
  `status/engine_done.json` → success/failure phase.
- Cancel button: kill pid from `engine_pid.txt` (`kill -9` / `taskkill /F /T`).
- Success phase: "Import to timeline" button → same import routine as
  `Import_Dub_Results.lua` (shared code duplicated is fine for v0.1).
- Audio file picker: `reaper.GetUserFileNameForRead`. Language: Combo of the
  12 languages. Voice id: plain text field (optional).
- Panel locates the engine via its own path: `reaper/` → sibling `engine/`.
  `app_dir` default from `engine/engine_settings.json`, fallback to the
  CONTRACT default above.

## v0.2 — Review pause + chunk regeneration

### Staged runs (script review between translation and dubbing)

`--steps` gains two values (old `full` stays and behaves as before):

- `--steps translate` — run S1a…S2c only (transcribe, regions/SRT, translation
  chain incl. punctuation; NO emotion, NO TTS). Write the app-convention
  outputs (EN SRT, `<base>_FinalScript.txt`, …) into out_dir, then write a
  REVIEW manifest to `engine/status/engine_done.json` AND
  `<out_dir>/engine_done.json`:

  ```json
  {
    "status": "review",
    "error": "",
    "audio": "…", "language": "…", "out_dir": "…",
    "en_srt": "<abs>",
    "en_text": "<abs>",          // plain-text EN transcript, paragraph per SRT chunk
    "translation_text": "<abs>", // plain-text translation, SAME paragraph count/order
    "final_script": "<abs>"      // <base>_FinalScript.txt (app format)
  }
  ```
  `en_text` / `translation_text`: UTF-8, one paragraph per line-block separated
  by ONE blank line — the panel renders EN left / translation right.
  Exit code 0. engine_done.txt written LAST as usual.

- `--steps dub --script "<abs path>"` — resume in the SAME out_dir (derive it
  from --audio exactly like a full run; the translate stage must have run
  first). `--script` points to the (possibly user-edited) translation text
  file (same blank-line paragraph format). Engine reads it, runs emotion
  enrichment (unless --no-emotion) + TTS + S3a…S3e, writes the normal
  v0.1 `"status":"ok"` manifest. Stale review manifests must be overwritten.

Panel flow: Run button now defaults to staged mode →
`--steps translate` → poll → on `"status":"review"` enter REVIEW phase:
side-by-side panes, EN read-only left, translation EDITABLE right
(ImGui InputTextMultiline, Indic font attached, per-paragraph rows aligned
where feasible). Buttons: "💾 Save" (write edited text to
`<out_dir>/<base>_translation_edited.txt`), "▶ Continue to Dubbing"
(save, then launch `--steps dub --script <edited file>`), "Skip edit"
(continue with engine-produced file). A "Full run (no review)" checkbox in
setup phase preserves old one-shot `--steps full` behavior.

### Chunk regeneration (like the app's Compare-view regen, but non-destructive)

Engine mode:
```
"<python>" run_dub.py --app-dir "…" --regen-chunk --language <Lang> \
    [--voice-id ID] [--el-model M] --text-file "<abs .txt>" --out-wav "<abs .wav>"
```
- `--text-file`: UTF-8 chunk text (never pass Indic text on argv).
- Engine synthesizes that text alone with the app's ElevenLabs helpers
  (same voice-resolution rule as v0.1), converts to WAV, writes `--out-wav`.
- Goes through run_dub.py like any run: same status dir, log, pid, done.txt.
  Manifest: `{"status":"ok","regen_wav":"<abs>"}` (or status error).
- No emotion pass on regen (text is already final; document if deviating).

Panel regen UI (post-import, success phase — and available any time via a
"Regen selected item" section): read the selected media item on a
"Dub Chunks*" track, load its stored chunk text into an editable text box
(hidden ext state, `P_NOTES` fallback for pre-v0.8 projects), user edits,
"⟳ Regenerate" → write text to `<out_dir>/regen/chunk_<n>.txt`, out-wav
`<out_dir>/regen/chunk_<n>_v<K>.wav` (K increments, never overwrite), launch
engine, poll as usual; on ok: in one undo block set the item's take source to
the new wav (PCM_Source_CreateFromFile + SetMediaItemTake_Source),
D_STARTOFFS=0, D_LENGTH=new source length (BR_GetMediaSourceProperties not
required — use reaper.GetMediaSourceLength), store the edited text back in
the item's hidden ext state, UpdateArrange. Original synced wav and app files untouched.

v0.8: the regen section carries an optional voice override under the button
(`V5.regen_voice`, shared `V5.ui_voice_picker` with key `regen`, plus a manual
voice-id box). Non-empty → `--voice-id` on the `--regen-chunk` launch, so the
same text comes back in a different voice; empty → the ⚙ Settings voice, i.e.
the pre-v0.8 behaviour. The choice persists across selections within a panel
session (deliberate: chunks are usually re-voiced in batches) and is NOT
written to settings. No engine change — `_run_regen` already resolves
`--voice-id` through `_resolve_voice`.

## Hard rules

- NEVER write into APP_DIR. Engine must not update `projects.json`,
  `run_history.json`, or any app-owned state. Out_dir (next to the input
  audio) is app-convention output — allowed.
- dub_engine.py: import app module by adding APP_DIR to sys.path; call its
  MODULE-LEVEL functions; never instantiate `EndToEndApp`, never import
  tkinter-touching entry paths beyond what module import itself does.
  Set `matplotlib.use("Agg")` BEFORE importing the app module.
- All Lua: version-safe ReaImGui calls (pcall wrappers like the reference),
  two-arg `GetMediaSourceFileName(src, "")`, `AddProjectMarker2(0, true, ...)`.
- macOS primary target; keep Windows branches where the reference has them.
