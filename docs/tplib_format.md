# `.tplib`: articulation library format specification v1

> Quill's open exchange format. **An articulation library is a set of retargetable control curves, not audio slices.** This is the
> fundamental difference from a sampler: a slide is stored as a normalised curve that can be retargeted to any interval/duration, and a strum is stored as
> string-order offsets that can be applied to any chord. Libraries can be shared, exchanged and stacked by the community (GTM: community seeding).

## Container

`.tplib` = **ZIP** with two members:
```
manifest.json   # structure + scalars + strings (human-readable)
arrays.npz      # all float32 arrays (numpy compressed)
```
The manifest references array keys in `arrays.npz` with `"key"` strings, which separates structure from data and makes read/write round trips lossless.

## Data contract (Part B)

```python
ArticulationLibrary = {
  "pluck":  [ {"rms_bucket": 0|1|2, "grain": float32[2646]} ],     # 60ms@44.1k transient residual, three velocity buckets
  "slide":  [ {"interval_st": float, "dur_s": float,
               "curve_ctrl": float32[8]} ],                        # normalised spline g(τ)∈[0,1] (normalised by interval × duration)
  "legato": [ {"amp_curve": float32[25], "tilt_curve": float32[25]} ],  # ±50ms@250Hz transition curves
  "strum":  [ {"offsets_ms": float32[N], "direction": "up"|"down"} ],   # string-order offset vector + direction
  "sustain": TimbreSnapshot,                                       # timbre snapshot reference (embedded, self-contained)
  "meta":   {"instrument_guess": str, "source": str,
             "stats": {pluck,slide,legato,strum counts}, "format_version": 1}
}
```

### Field semantics
- **pluck.grain**: residual grain of the pluck transient, played back granularly (Roads 2001); `rms_bucket` selects the bucket by velocity.
- **slide.curve_ctrl**: the **normalised shape** g(τ)∈[0,1] of a slide, τ = time fraction. The synthesis side retargets it with
  `f0 = f0_start · 2^(g(t/T)·interval/12)` to **any** start/end pitch and duration (velocity -> slide speed).
- **legato.amp_curve / tilt_curve**: amplitude and brightness transitions within ±50ms of a legato join (250Hz, 25 points).
- **strum.offsets_ms**: onset offset of each string relative to the first string (milliseconds); `direction` is an up/down strum. Applies to any chord.
- **sustain**: the timbre side (`TimbreSnapshot`: f0_ref + H_env[60,T] + N_env[40,T] + attack_res[2646]
  + partial_freqs|None + vibrato|None). **Decoupled from articulation** -> can be cross-recombined with the articulations of any other library (FR-9).

## Reading and writing
```python
from analysis.tplib import build_library, save_tplib, load_tplib

lib = build_library(labeled_events, y, sr, sustain_snapshot=snap)
save_tplib("guitar.tplib", lib)  # lossless round trip
lib2 = load_tplib("guitar.tplib")
```
The loader is **tolerant**: new fields get defaults and unknown keys are ignored (forward/backward compatible). `format_version` marks the version.

## Compatibility conventions
- Arrays are always `float32`; scalars/strings go into manifest.json (UTF-8).
- Time units: seconds (dur_s) / milliseconds (offsets_ms); intervals: semitones (interval_st).
- Default sample rate 44100; the meta inside sustain carries its own sr.
- Cross recombination only requires sustain and the articulations to be individually valid; timbre and articulation **may come from different `.tplib` files**.
