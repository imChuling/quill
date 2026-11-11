# Third-party dependencies and licenses (Sprint 0 · 2026-07-29)

> Compliance item §13 of the sprint notebook. Scope = all of requirements.txt + datasets + pretrained models.
> Conclusion first: **the runtime core chain is entirely under permissive licenses (BSD/MIT/Apache/ISC), with no GPL contagion;
> the compliance points to watch are concentrated in two places: "datasets restricted to research use" and "Demucs pretrained weights".**

## 1. Runtime core (distributed with the product)

| Dependency | License | Risk |
|---|---|---|
| numpy / scipy | BSD-3 | none |
| numba | BSD-2 | none |
| librosa | ISC | none |
| soundfile | BSD-3 (bundled libsndfile LGPL-2.1, dynamically linked) | dynamic linking is compliant, do not bundle statically |
| sounddevice | MIT (PortAudio MIT) | none |
| mido / python-rtmidi | MIT (RtMidi MIT-like) | none |
| pywebview | BSD-3 | none |
| PyYAML | MIT | none |
| React / ReactDOM 18.3.1 (`ui/web/vendor/`, bundled with the desktop UI, license text in the same directory) | MIT | none |
| Cormorant Garamond / Cormorant SC / Italianno / Spline Sans Mono (`ui/web/vendor/fonts/`, latin subset woff2, bundled with the desktop UI, OFL.txt in the same directory) | SIL OFL 1.1 | none |

## 2. Pretrained models (key items)

| Component | Code license | Weights/data notes |
|---|---|---|
| **PESTO** (f0 front end, main chain) | LGPL-3.0 (code) | used as a dynamically imported pip package = compliant; **do not** copy its code into this repository; external wording: "uses the pretrained PESTO pitch estimator" |
| **Demucs** htdemucs_6s (Mode 3, optional) | code MIT | **non-commercial risk for the htdemucs weights**: training data includes research-only data such as MUSDB; Let's Build is a non-commercial event = OK, but the README/ModelCard must state "research/non-commercial use" |
| **basic-pitch** (FR-5 transcription) | Apache-2.0 | none (official Spotify open source) |
| **MERT-v1-95M** (research probe, not in the product) | weights CC-BY-NC-4.0 | **non-commercial**; used only for offline evaluation, not distributed with the product, citation in the paper is enough |
| torch / transformers / nnAudio | BSD/Apache/MIT | none |

## 3. Datasets (evaluation only, not distributed with the product)

| Dataset | License | Permitted use |
|---|---|---|
| TinySOL | CC-BY-4.0 | evaluation + paper ✓, with attribution |
| URMP (violin/trumpet DDSP training + behaviour library source) | **no standard license**, requires registration and citation, no terms on derivatives | evaluation + paper ✓; derived behaviour libraries are **not published** (decided 2026-10-27, only for reproduction inside the repository) |
| GuitarSet | MIT (annotations) / CC-BY-4.0 (audio) | evaluation + paper ✓ |
| IDMT-SMT-Guitar / Bass | **non-commercial research only** | evaluation + paper ✓; **do not** use the audio as material in demo videos |
| NSynth | CC-BY-4.0 | evaluation + prior training ✓ |
| Own recordings (voice/guitar/wineglass etc.) | own | needs a publicity release from the performer (the guitarist, sprint notebook §13) |

## 4. Red lines for external wording (in the claim ledger, SCOPE-01)

- Allowed: "the capture-to-performance path requires **no training**; **no per-user/per-timbre training**"
- Forbidden: "no pretrained models" (PESTO/Demucs/basic-pitch/MERT are all pretrained components)
- If the Let's Build submission includes a demo of the Demucs feature, the page must carry a one-line non-commercial statement;
- The acknowledgements of the ISMIR LBD abstract cite the sources of TinySOL/GuitarSet/NSynth.

## 5. To do

- [ ] Audiotool Nexus SDK (@audiotool/nexus) license and T&C: to be confirmed after registration (before Sprint 1);
- [ ] If the Atlas passes its gate: double-check NSynth/TinySOL for training use (CC-BY allows it, attribution is enough).
