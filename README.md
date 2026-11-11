# Quill

Capture a few seconds of any pitched sound and get two separate, reusable assets: the
**timbre** and the **playing behavior**. Play the result from a MIDI keyboard in real time,
or swap either asset across instruments without retraining.

[![Demo](https://img.shields.io/badge/Demo-YouTube-daa520?style=flat&labelColor=100d08)](https://www.youtube.com/watch?v=EhepbJxW9SM)
[![ISMIR 2026](https://img.shields.io/badge/ISMIR_2026-Late--Breaking_Demo-daa520?style=flat&labelColor=241f14)](https://ismir2026program.ismir.net/lbd_613.html#details)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-c5962b?style=flat&labelColor=100d08)](https://creativecommons.org/licenses/by-nc/4.0/)

## How it works

A sampler stores a recording and replays it. Quill stores a recipe: the sound is
decomposed into harmonic and noise components and resynthesized at any pitch and
length, with formants preserved. Playing techniques (vibrato, slides, legato, strums) are
extracted from the same clip as note-level control trajectories, a *behavior library*, and
can drive any other timbre: a wine glass played with a violinist's vibrato.

```
L2  Neural expressivity   DDSP timbre fine-tuning · distilled behavior model
L1  Behavior engine       segmentation · articulation labels · style libraries · MIDI gesture interpreter
L0  Spectral core         harmonic + noise decomposition · resynthesis at any pitch · real-time MIDI
```

The real-time path depends on `numpy` and `numba` only; `torch` and `librosa` stay out of
the audio callback. Live playback and offline rendering share one code path.

## Quickstart

```bash
conda create -n quill python=3.10 && conda activate quill
pip install -r requirements.txt
python ui/app_web.py
```

In the app: **PRESS TO CAPTURE** a sound, play the keyboard, record into the piano roll,
then swap the timbre or the behavior and replay the same performance. **SAVE** stores the
tone and extracts its behavior library; **BAKE DDSP** trains a neural timbre from the take
on the Mac GPU. Class-compliant MIDI keyboards work out of the box.

## Checkpoints

Weights and libraries come from the [v1.0 release](https://github.com/imChuling/quill/releases/tag/v1.0):

```bash
curl -L https://github.com/imChuling/quill/releases/download/v1.0/quill-checkpoints-v1.0.tar.gz | tar xz --strip-components=1 -C checkpoints/
```

| File | SHA-256 prefix | Source |
|---|---|---|
| `ddsp_glass.pt`, `glass_style.npz` | `b718be51`, `198675c2` | self-recorded wine glass |
| `ddsp_guitarset.pt`, `guitarset_trajectories.npz` | `bdc15394`, `ac41f33d` | GuitarSet (CC BY-NC 4.0) |

The violin and trumpet models and libraries are derived from URMP, which has no
redistribution terms, so they are not published. To rebuild them, register for URMP and run
`tools/extract_trajectories_urmp.py` and `neural/train_ddsp.py`. Details and provenance:
`checkpoints/MODEL_CARDS.md`.

## Reproduce the paper

```bash
python tools/reproduce_lbd.py --check-only   # hashes checkpoints, compares stored results, under 1 s
python tools/reproduce_lbd.py                # re-runs every evaluation, needs a GPU, about 10 min
```

Numbers live in `checkpoints/eval/eval_results.json`. Scripts that read public datasets
look under `$QUILL_DATASETS`.

| Result | Value |
|---|---|
| Articulation classification, IDMT-SMT-Guitar (macro-F1) | 0.828 |
| Cross-rendering source attribution | 12/12 |
| Listening study, N=19: more human-like / more expressive | 19/19 · 18/19 |
| Real-time block render, 16 voices (median / p95) | 2.65 ms / 3.1 ms of a 5.8 ms budget |

## Layout

```
analysis/   capture: f0, harmonics, residual, snapshot, segmentation, articulation
synth/      numpy/numba engine: oscillator, voice, instrument, noise, render
runtime/    block engine, gesture interpreter, MIDI input
neural/     DDSP timbre model, training, behavior retrieval and distillation
ui/         desktop app (pywebview), render service
web_demo/   static listening page with prerendered clips
eval/ tools/ tests/   evaluation, scripts, pytest suite
docs/       behavior library and .tplib format specs, third-party licenses
```

```bash
ruff check . && python -m pytest -q    # tests needing checkpoints or a GPU skip
```

## Citation

```bibtex
@inproceedings{li2026quill,
  title={Quill: Learning Timbre and Performance Behavior as Separate, Swappable Assets},
  author={Li, Chuling},
  booktitle={Extended Abstracts for the Late-Breaking Demo Session of the 27th Int. Society for Music Information Retrieval Conf.},
  year={2026}
}
```

License: CC BY-NC 4.0 (`LICENSE`).
