# Quill

Record a sound, play it back as an instrument.

Quill is a small desktop tool that takes a few seconds of almost any pitched sound — your voice, a wine glass, a bowed string, a synth pad — and turns it into something you can play across a MIDI keyboard.

It's not a sampler. Instead of looping the clip, it pulls the sound apart into its harmonics, its noise, its breath, and rebuilds it note by note. So you can play it at any pitch and hold it for as long as you want, and it still sounds like itself. It also picks up *how* the original was played — the slides, the legato, the way a phrase moves — and lets you put that phrasing on a different sound. A voice that slides like a guitar, if that's what you're after.

It's still a work in progress. A proper walkthrough video and a downloadable macOS build are coming soon.

## What works right now

- Capture any pitched sound in a few seconds and play it on the keyboard.
- Notes stay in the right register — no chipmunk effect when you play high up.
- Overlap keys for legato and slides; how hard you press sets how fast it slides; chords strum.
- Pull one instrument out of a full song and capture that.
- A built-in piano roll — record a phrase, swap the sound underneath it, keep the performance. Export to MIDI or audio.
- Works with any standard MIDI keyboard (pitch bend, mod wheel, sustain), and sits fine next to a DAW.
- There's a texture mode too, for unpitched things — rain, wind, machines.

All of it runs in real time on a laptop.

## Where it's headed

- A plug-in version, so it lives inside your DAW instead of on its own.
- Per-note expression (MPE) — independent bend and vibrato per finger.
- Better neural capture, to close the last bit of distance to the real thing.
- A way to share not just sounds, but ways of playing them.
- Real-time capture of plucked and percussive sounds, which it doesn't handle well yet.
