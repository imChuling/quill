# Quill — Capture · Preserve · Transform

**A Nexus app that learns how a sound *sounds* and how it is *played* — as two
separate, swappable assets — and performs your Audiotool regions with them,
written back in place.**

Live app: **https://chuling-li-cs--quill-render-serve.modal.run/**

Built for Audiotool *Let's Build!* 2026. Backed by the research system behind
our ISMIR 2026 LBD submission *"Quill: Learning Timbre and Performance Behavior
as Separate, Swappable Assets."*

---

## What it does

Every expressive renderer has to learn two things: **timbre** (how an
instrument sounds) and **behavior** (how a player plays it — vibrato, slides,
energy arcs). Quill keeps them as two independent assets, so either can be
swapped without retraining:

- **INK a region.** Pick any note region in your Audiotool project. Quill
  reads it through Nexus (notes, `doesSlide`, velocities, tempo), performs it
  with real per-note behavior retrieved from recorded performances, and writes
  the audio **back into your project — aligned to the source region's bar,
  tempo-locked, one track per timbre** so you can solo-compare instantly.
- **Capture a new sound.** Drop a recording (or record from the mic): one
  phrase, a hum, a wine glass. Quill extracts **both assets** from it — a
  playable timbre snapshot *and* a style library of how it was played — and
  they immediately join your palette.
- **Swap behavior across timbres.** Render a violin with the behavior
  extracted from your own recording, or your captured sound with violin
  behavior. Recorded expression becomes as reusable as a sample.

## How to use

1. Open the app and **Login** with your Audiotool account (OAuth; the token
   never leaves your browser).
2. **SOURCE** — pick one of your projects; its note regions appear as cards
   with a piano-roll preview (gold = slide, bronze = vibrato, cream = plain).
3. **PERFORM** — check one or more timbres (violin / trumpet / guitar /
   glass / anything you captured), optionally pick a captured **behavior**,
   and press **INK THIS REGION**.
4. Each take lands in your project on its own track, aligned at the source
   region's bar. Audition inline under **TAKES**, then solo/mute-compare in
   Audiotool.
5. **CAPTURE** — drop audio or record the mic to teach Quill a new sound;
   both extracted assets appear within seconds.

## Technical highlights

- **Region-native Nexus integration.** Notes are grouped by their
  `NoteCollection` into regions; `doesSlide` maps to Quill's articulation
  labels. Write-back uses `insertSample` with `region.positionTicks` (bar
  alignment) and `sample.musicDurationTicks` (musical-time lock, so alignment
  survives later tempo changes).
- **Two-asset engine.** Timbres are lightweight DDSP models (~500K params)
  or spectral snapshots captured from a single recording; behavior is a bank
  of per-note pitch/energy trajectories with articulation labels and a 6-D
  context vector, retrieved by nearest-neighbor lookup at render time. The
  two interface through plain (f0, loudness) — no learned adapter.
- **Timeline-faithful cloud rendering.** The render service splits scores at
  gaps, renders phrases whole (preserving legato/crossfade continuity), and
  places them at exact onsets; chords and let-ring overlaps from real DAW
  regions are monophonized with the same rules as our MIDI importer.
- **One recording, both assets.** `/capture` runs pYIN-based note
  segmentation, automatic articulation labeling, and trajectory extraction
  (the exact pipeline evaluated in the paper) plus a harmonic+residual
  timbre snapshot — server-side, from any browser, no install.
- **Stateless-friendly deployment.** FastAPI on Modal (CPU), assets on a
  persistent volume, the web app served same-origin. Renders are idempotent
  by content hash.

## Run it yourself

```bash
# web app (this repo)
npm install
VITE_QUILL_API=<render-service-url> npx vite build   # or `npx vite` for dev

# render service (see deploy/modal_app.py in the engine repo)
modal deploy deploy/modal_app.py
```

The OAuth redirect URI must be registered at developer.audiotool.com for the
origin you serve from.

## Credits

Chuling Li — University of Nottingham. Engine, evaluation, and listening
study: see the ISMIR 2026 LBD paper. Sample material for the glass case
study: decent|SAMPLES.
