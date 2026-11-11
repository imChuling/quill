# Quill

> Capture a few seconds of any pitched sound and get two separate, reusable assets: the
> **timbre** and the **playing behavior**. Play the result from a MIDI keyboard in real time,
> or swap either asset across instruments without retraining.

[![Demo](https://img.shields.io/badge/Demo-YouTube-daa520?style=flat&labelColor=100d08)](https://www.youtube.com/watch?v=EhepbJxW9SM)
[![ISMIR 2026](https://img.shields.io/badge/ISMIR_2026-Late--Breaking_Demo-daa520?style=flat&labelColor=241f14)](https://ismir2026program.ismir.net/lbd_613.html#details)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-c5962b?style=flat&labelColor=100d08)](https://creativecommons.org/licenses/by-nc/4.0/)

Paper page: [ISMIR 2026 LBD 613](https://ismir2026program.ismir.net/lbd_613.html#details) ·
Model cards: `checkpoints/MODEL_CARDS.md` · Behavior library format: `docs/behavior_library_format.md`

## Why Quill

Turning a recording into something you can play already exists, but every existing tool is
a player, not an instrument. A sampler stores the recorded audio itself: press a key and it
replays that fixed clip, pitch-shifted, with no technique it did not record and no genuine
note length. Commercial MIDI instruments (MODO Bass, Ample Guitar) solve articulation and
sustain beautifully, but their timbre is authored by the vendor.

**Quill stores a recipe, not a recording.** It decomposes a few seconds of any pitched sound
into harmonic and noise components and resynthesizes it sample by sample: any pitch, any
duration, any articulation. Playing techniques (slides, legato, vibrato, strums) are mined
few-shot from a single clip as retargetable control trajectories, decoupled from timbre and
recombinable across sources: a voice that slides like a guitar, a wine glass played with a
violinist's vibrato.

## Quickstart

```bash
conda create -n quill python=3.10 && conda activate quill
pip install -r requirements.txt      # requirements-lock.txt pins every transitive dependency
python ui/app_web.py                 # native desktop app (pywebview)
```

In the app: **PRESS TO CAPTURE** (record about 5 s of any pitched sound, or **FROM A SONG**
to extract an instrument from a mix), play the keyboard (overlapping keys give legato or
slide, chord bursts give strums), record into the piano roll, then swap the tone and replay
the same performance. Class-compliant MIDI keyboards work out of the box.

One recording, two assets. **SAVE** in the capture window stores the tone and, in the
background, extracts a same-named *behavior library* from the same take (it needs a few
notes, not one held pitch). **BAKE DDSP** fine-tunes a neural timbre from the take on the
Mac GPU (8 s or more of audio, about 4 minutes) and adds it to the piano roll.

Where the two assets meet, one pair, two lists:

- **TIMBRE** (left): captured tones (marked *live*: they play the keys and render), built-in
  DDSP timbres (violin, trumpet, wine glass) and baked timbres (*render*: piano roll only).
- **BEHAVIOR** (right): *rules · gesture* (the interpreter, live only) and every library:
  built-in, distilled, and the ones extracted from your recordings. The ⟷ mark shows a
  timbre and a library that came from the same take.
- The current **timbre × behavior** pair is shown at the top (NOW) and on the piano roll. It
  drives both the live keys and the roll's render. When a DDSP timbre is selected, the keys
  keep the current captured tone because the live engine is the spectral one.
- After **SAVE** finishes extracting, a notice offers the cross-use in one click: your new
  tone played with the violin library.
- Every knob on the panel acts on the sound (brightness, air/noise, intensity, strum, glide,
  pick, vibrato, ADSR). Captured tones get an automatic noise floor from their spectrum.

**Hear it without installing:** the [demo video](https://www.youtube.com/watch?v=EhepbJxW9SM),
or the `web_demo/` page, a prerendered timbre × style-library matrix.

## Architecture

```
L2  Neural expressivity   DDSP timbre fine-tuning · distilled behavior model · vibrato model
L1  Behavior engine       event segmentation · articulation labels · style libraries · MIDI gesture interpreter
L0  Spectral core         harmonic + noise decomposition · formant-preserving resynthesis at any pitch · real-time MIDI
```

The real-time path (`synth/`, `runtime/`) depends on `numpy` and `numba` only. Heavy
dependencies (`librosa`, `pesto`, `torch`) live in `analysis/` and `neural/` and never run
inside the audio callback. `Engine.process_block` is the single code path shared by the live
audio stream and by offline rendering, so unit tests, regression stimuli and listening-study
material come from exactly the code that plays.

A **behavior library** is note-level control trajectories extracted from a real performance:
48 frames of pitch deviation and energy per note, an articulation label from the pYIN contour
(plain, vibrato, slide) and a 6-D context vector. Rendering retrieves the nearest entry,
filtered by label, and time-stretches it to the note. A **timbre** is either a spectral
snapshot (L0, captured in seconds) or a small DDSP model (L2, harmonic plus noise, about 500K
parameters, 16 kHz at 250 fps) that takes the same f0 and loudness the library stores, so the
two assets interface without a learned adapter.

Negative result, reported honestly: RAVE-style neural codecs capture timbre near studio grade
but are not playable through external control (latents leave the training manifold). Playable
claims in Quill cover the spectral core and DDSP timbres only.

## Evaluation highlights

| What | Number |
|---|---|
| Articulation classification (IDMT-SMT-Guitar, 5164 events, 3-class macro-F1) | rule tree 0.545, random forest 0.774, MERT probe **0.828** |
| Formant preservation (+12 st, real voice) | formant drift **22 Hz** (249 Hz naive) |
| Real time, 16 voices at full load, per-block render | median **2.65 ms**, p95 **2.7 to 3.1 ms** on an idle M2 (5.8 ms budget at 256 samples, 44.1 kHz) |
| Cross-rendering source attribution (violin, trumpet, guitar) | 12/12 checks pass |
| Listening study (N=19, pre-registered) | retrieved behavior rated more human-like by 19/19, more expressive by 18/19; cross-instrument ABX median accuracy 100% |
| Paired counterfactual (vibrato vs plain, same note) | median ratio 7.01× (CI [4.79, 18.65]) |

## Download checkpoints

Model weights and behavior libraries are not in the git tree. The published set is a GitHub
Release asset; unpack it into `checkpoints/`:

```bash
curl -L https://github.com/imChuling/quill/releases/download/v1.0/quill-checkpoints-v1.0.tar.gz -o quill-checkpoints-v1.0.tar.gz
tar xzf quill-checkpoints-v1.0.tar.gz --strip-components=1 -C checkpoints/

# Verify (first 16 hex chars of SHA-256):
python -c "
import hashlib, pathlib
expected = {
    'ddsp_glass.pt':              'b718be519aee9e87',
    'ddsp_guitarset.pt':          'bdc153946fec299a',
    'glass_style.npz':            '198675c22031590b',
    'guitarset_trajectories.npz': 'ac41f33d253d86bb',
}
ckpt = pathlib.Path('checkpoints')
ok = 0
for name, prefix in expected.items():
    p = ckpt / name
    if not p.exists():
        print(f'MISSING  {name}')
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    status = 'OK' if h == prefix else f'MISMATCH (got {h})'
    print(f'{status:10s}  {name}')
    ok += h == prefix
print(f'\n{ok}/{len(expected)} files verified.')
"
```

| File | Size | SHA-256 prefix | In the release |
|---|---|---|---|
| `ddsp_glass.pt` | 1.9 MB | `b718be51` | yes |
| `ddsp_guitarset.pt` | 1.9 MB | `bdc15394` | yes (GuitarSet audio, CC BY-NC 4.0) |
| `glass_style.npz` | 36 KB | `198675c2` | yes |
| `guitarset_trajectories.npz` | 10 MB | `ac41f33d` | yes (derived from GuitarSet, CC BY-NC 4.0) |
| `ddsp_urmp_vn_12k.pt`, `ddsp_urmp_tpt_12k.pt` | 1.9 MB each | | no: trained on URMP audio |
| `urmp_vn_trajectories.npz`, `urmp_tpt_trajectories.npz` | 4.5 / 2.4 MB | | no: derived from URMP |
| `behavior_distilled.pt`, `vibrato_model.pt` | 6.4 MB / 16 KB | | no: trained on URMP-derived trajectories |

URMP ships no standard license, requires registration and citation, and does not state terms
for derived data, so nothing derived from it is redistributed. With the release set you get
the wine-glass and guitar timbres and libraries, which is what the desktop app and the web
page need. The violin and trumpet results in the paper need the URMP files: register for
URMP, rebuild the libraries with `tools/extract_trajectories_urmp.py` and train the two DDSP
models with `neural/train_ddsp.py` (see [Behavior libraries](#behavior-libraries)).

## Reproduce the paper

All numbers from the ISMIR 2026 LBD paper are stored in `checkpoints/eval/eval_results.json`.
Both commands need the checkpoint files listed in `checkpoints/MODEL_CARDS.md` in place under
`checkpoints/`; the check verifies their hashes first.

```bash
# Structural check (hashes the checkpoints, compares stored results, no GPU, under 1 s):
python tools/reproduce_lbd.py --check-only

# Full reproduction (re-runs every evaluation, needs checkpoints and a GPU, about 10 min):
python tools/reproduce_lbd.py
```

Expected output with the checkpoints present:

```
PASS: all paper numbers reproduced within tolerance.
```

Key numbers: paired counterfactual median ratio 7.01× (CI [4.79, 18.65]), decoupling 12/12,
glass 2×2 4/4, processing ablation none 1.55× / full 7.01×.

Evaluation scripts that read public datasets look for them under `$QUILL_DATASETS`
(IDMT-SMT-Guitar, TinySOL, GuitarSet, VocalSet). Set the variable to your dataset folder.

## Web demo

`web_demo/` is a single static page served by the render service: a 3 × 4 timbre × behavior
matrix of prerendered clips (two melodies; the guitar DDSP timbre is left out for quality,
the guitar behavior stays), plus "your melody": the desktop piano roll ported to the browser.
Record from the on-screen keys, the computer keyboard or a MIDI keyboard (Web MIDI), drag and
resize notes, right-click to mark vibrato or slide, load a MIDI file, then Play renders the
phrase on the server with any timbre × behavior pair. Capture, baking and real-time Quill
sound stay in the desktop app.

```bash
python tools/build_web_demo_matrix.py   # regenerate web_demo/audio + manifest.json
QUILL_WEB_DIR=$PWD/web_demo python -m uvicorn ui.render_service:app --port 8726
```

## Behavior libraries

A behavior library is an `.npz` of note-level performance trajectories (pitch, energy,
brightness, noise; 4 × 48 frames per note) plus retrieval features. Format spec:
`docs/behavior_library_format.md`. A torch-free player that turns a score plus a library into
250 fps control signals: `tools/behavior_player.py`.

| Library | Source | License | Distributed |
|---|---|---|---|
| `glass_style.npz` | self-recorded wine glass | CC BY-NC 4.0 | yes (GitHub Release v1.0) |
| `guitarset_trajectories.npz` | GuitarSet | derived, source CC BY-NC 4.0 | yes (GitHub Release v1.0) |
| `urmp_vn_trajectories.npz` | URMP violin | derived; URMP has no standard license | **no** |
| `urmp_tpt_trajectories.npz` | URMP trumpet | derived; URMP has no standard license | **no** |

To rebuild the URMP libraries locally, register for URMP, then:

```bash
python tools/extract_trajectories_urmp.py --inst vn  --urmp /path/to/URMP -o checkpoints/urmp_vn_trajectories.npz
python tools/extract_trajectories_urmp.py --inst tpt --urmp /path/to/URMP -o checkpoints/urmp_tpt_trajectories.npz
```

This is the script that produced the libraries used in the paper.
`tests/test_urmp_extraction_rules.py` verifies that its labeling and context rules reproduce
every stored label and feature.

## Repository layout

```
quill/
├─ analysis/     capture: io, f0, harmonics, residual, timbre snapshot, segmentation, articulation, .tplib
├─ synth/        numpy/numba engine: oscillator kernel, voice, instrument, noise, offline render
├─ runtime/      block engine (live + offline share one code path), gesture interpreter, MIDI input
├─ neural/       DDSP timbre model, training and baking, behavior retrieval and distillation
├─ ui/           desktop app (pywebview) and its bridge, render service for the web page
├─ web_demo/     static web page + prerendered clips
├─ eval/         metrics, acceptance scripts, listening-study material and analysis
├─ tools/        extraction, training, evaluation and reproduction scripts
├─ tests/        pytest suite (runs in CI without checkpoints; checkpoint-dependent tests skip)
├─ checkpoints/  MODEL_CARDS.md and stored evaluation results (weights come from the Release)
└─ docs/         behavior library and .tplib format specs, third-party license notes
```

## Development

```bash
pip install -r requirements-ci.txt
ruff check .
python -m pytest -q                 # about 280 tests; those needing checkpoints or a GPU skip
```

`tests/test_latency.py` gates the real-time kernel on the median and p95 block time. It is
sensitive to machine load; run it on an idle machine.

## Data and third-party components

- Datasets used for evaluation only: IDMT-SMT-Guitar, TinySOL, NSynth, GuitarSet, VocalSet,
  URMP. None of their audio is in this repository. See `docs/third_party_licenses.md`.
- Pretrained components used as dependencies: PESTO (pitch), Basic Pitch (transcription),
  MERT (research probe, offline only), Demucs (optional stem separation; its weights carry
  non-commercial terms).
- The desktop UI vendors React 18.3.1 (MIT) and four typefaces under the SIL Open Font
  License; the license texts sit next to the files in `ui/web/vendor/`.

## Citing

```bibtex
@inproceedings{li2026quill,
  title={Quill: Learning Timbre and Performance Behavior as Separate, Swappable Assets},
  author={Li, Chuling},
  booktitle={Extended Abstracts for the Late-Breaking Demo Session of the 27th Int. Society for Music Information Retrieval Conf.},
  year={2026}
}
```

`CITATION.cff` carries the same entry in machine-readable form.

## License

CC BY-NC 4.0 for the code, the self-recorded assets and the released checkpoints. See `LICENSE`.
