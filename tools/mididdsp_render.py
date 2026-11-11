"""E1 MIDI-DDSP rendering: produce audio for QB×MD and MB×MD conditions.

Runs in the `mididdsp` conda environment (tensorflow-macos 2.9 + midi-ddsp 0.2.6).
Reads teacher exports from checkpoints/eval/mididdsp_exports/*.npz, re-synthesizes
with real DDSP processor group, optionally swapping f0 and loudness from Quill.

Usage:
    conda run -n mididdsp python tools/mididdsp_render.py \
        --quill-controls checkpoints/eval/e1_quill_controls/
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

QUILL = Path(__file__).resolve().parent.parent
EXPORTS = QUILL / "checkpoints" / "eval" / "mididdsp_exports"
OUT = QUILL / "checkpoints" / "eval" / "e1_renders"
SR = 16000
FPS = 250

MELODIES_ORDER = ["lyrical", "scale_run", "leaps", "folk"]


def strip_slides(scores):
    """Replace art=2 (slide) with art=0 (plain)."""
    return {k: [(m, d, 0 if a == 2 else a) for m, d, a in v]
            for k, v in scores.items()}


SCORES = {
    "lyrical": [
        (62, 0.8, 1), (65, 0.4, 0), (67, 1.2, 1), (65, 0.4, 0),
        (69, 0.4, 2), (72, 1.6, 1), (67, 0.8, 0), (62, 1.6, 1),
    ],
    "scale_run": [
        (62, 0.2, 0), (64, 0.2, 0), (65, 0.2, 0), (67, 0.2, 0),
        (69, 0.2, 0), (71, 0.2, 0), (72, 0.2, 0), (74, 0.2, 0),
        (76, 0.4, 0), (74, 0.4, 0), (72, 1.6, 1),
    ],
    "leaps": [
        (57, 0.4, 0), (64, 0.4, 0), (62, 0.4, 2), (69, 0.8, 1),
        (65, 0.4, 0), (72, 0.4, 2), (71, 0.8, 0), (64, 1.2, 1),
    ],
    "folk": [
        (69, 0.4, 0), (71, 0.2, 0), (72, 0.4, 0), (69, 0.4, 0),
        (67, 0.8, 1), (64, 0.4, 0), (67, 0.4, 2), (69, 1.2, 1),
        (67, 0.4, 0), (64, 0.4, 0), (62, 1.6, 1),
    ],
}


def voiced_ratio(audio):
    import librosa
    f0, _, _ = librosa.pyin(audio, fmin=80, fmax=2000, sr=SR,
                            frame_length=1024, hop_length=160)
    return float(np.sum(~np.isnan(f0)) / max(len(f0), 1))


def quill_note_frames(score, fps=250, cf_frames=5):
    """Per-note (start_frame, end_frame) with Quill's crossfade overlap."""
    frames = []
    t = 0
    prev_n = None
    for _midi, dur, _art in score:
        n = max(4, int(dur * fps))
        if prev_n is not None:
            t -= min(cf_frames, prev_n // 2, n // 2)
        frames.append((t, t + n))
        t += n
        prev_n = n
    return frames


def midi_note_frames(npz_path):
    """Per-note (onset_frame, offset_frame) from MIDI-DDSP npz note_table."""
    d = np.load(str(npz_path), allow_pickle=True)
    nt = d["note_table"]
    nk = list(d["note_keys"])
    oi = nk.index("onset")
    fi = nk.index("offset")
    return [(int(row[oi]), int(row[fi])) for row in nt]


def resample_per_note(signal, src_frames, dst_frames, total_dst_len):
    """Resample a 1-D signal per-note from src timing to dst timing."""
    assert len(src_frames) == len(dst_frames), \
        f"note count mismatch: {len(src_frames)} vs {len(dst_frames)}"
    out = np.zeros(total_dst_len, dtype=np.float32)
    for (s0, s1), (d0, d1) in zip(src_frames, dst_frames):
        src_seg = signal[s0:min(s1, len(signal))]
        dst_len = d1 - d0
        if len(src_seg) == 0 or dst_len == 0:
            continue
        if len(src_seg) == dst_len:
            out[d0:d1] = src_seg
        else:
            x_old = np.linspace(0, 1, len(src_seg))
            x_new = np.linspace(0, 1, dst_len)
            out[d0:d1] = np.interp(x_new, x_old, src_seg)
    return out


def inverse_exp_sigmoid(y, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Inverse of DDSP exp_sigmoid: linear amplitude -> logit."""
    y = np.clip(y, threshold * 2, max_value - 1e-6)
    s = ((y - threshold) / max_value) ** (1.0 / np.log(exponent))
    s = np.clip(s, 1e-12, 1.0 - 1e-12)
    return np.log(s / (1.0 - s)).astype(np.float32)


def quill_loudness_to_logit(quill_loud, npz_path):
    """Map Quill z-scored log-RMS loudness to MIDI-DDSP synth-param logits.

    Chain: z-scored -> dB (using MIDI-DDSP voiced-frame stats)
           -> linear (10^(dB/20)) -> logit (inverse exp_sigmoid).
    """
    d = np.load(str(npz_path), allow_pickle=True)
    md_dB = np.squeeze(d["amplitudes"]).astype(np.float32)
    voiced = md_dB > -75
    if voiced.sum() < 10:
        voiced = np.ones(len(md_dB), dtype=bool)
    mean_dB = float(md_dB[voiced].mean())
    std_dB = float(md_dB[voiced].std())
    print(f"    dB stats: mean={mean_dB:.1f}, std={std_dB:.1f} "
          f"(from {int(voiced.sum())} voiced frames)")

    target_dB = quill_loud * std_dB + mean_dB
    target_dB = np.clip(target_dB, -90, -20)

    linear = 10.0 ** (target_dB / 20.0)
    logit = inverse_exp_sigmoid(linear)

    print(f"    mapped dB range: [{float(target_dB.min()):.1f}, {float(target_dB.max()):.1f}]")
    print(f"    logit range: [{float(logit.min()):.2f}, {float(logit.max()):.2f}]")
    return logit


def load_export(melody):
    path = EXPORTS / f"{melody}.npz"
    d = np.load(str(path), allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return {"meta": meta, "f0_hz": d["f0_hz"]}


def synthesize_with_mididdsp(midi_path, seed):
    """Re-run MIDI-DDSP synthesis. Model loaded FIRST, then seed set (matching export order)."""
    import tensorflow as tf
    from midi_ddsp.midi_ddsp_synthesize import load_pretrained_model
    from midi_ddsp.utils.midi_synthesis_utils import synthesize_mono_midi

    synthesis_generator, expression_generator = load_pretrained_model()

    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    midi_audio, midi_control_params, midi_synth_params, conditioning = \
        synthesize_mono_midi(
            synthesis_generator,
            expression_generator,
            str(midi_path),
            instrument_id=0,
            output_dir=None,
        )

    return midi_audio, midi_control_params, midi_synth_params, synthesis_generator


def render_from_synth_params(synthesis_generator, midi_synth_params,
                             f0_override=None, amps_logit_override=None):
    """Render audio from raw synth params, optionally overriding f0 and amplitudes (logit domain)."""
    import tensorflow as tf

    processor_group = synthesis_generator.processor_group

    inputs = {}
    for k, v in midi_synth_params["inputs"].items():
        inputs[k] = tf.identity(v)

    T_synth = int(inputs["f0_hz"].shape[1])

    if f0_override is not None:
        T_use = min(len(f0_override), T_synth)
        f0_buf = inputs["f0_hz"].numpy().copy()
        f0_buf[0, :T_use, 0] = f0_override[:T_use]
        inputs["f0_hz"] = tf.constant(f0_buf, dtype=tf.float32)
        print(f"    f0: {T_use}/{T_synth} frames, "
              f"range=[{f0_override[:T_use].min():.1f}, {f0_override[:T_use].max():.1f}] Hz")

    if amps_logit_override is not None:
        T_use = min(len(amps_logit_override), T_synth)
        amps_buf = inputs["amplitudes"].numpy().copy()
        amps_buf[0, :T_use, 0] = amps_logit_override[:T_use]
        inputs["amplitudes"] = tf.constant(amps_buf, dtype=tf.float32)
        print(f"    amps: {T_use}/{T_synth} frames, "
              f"logit range=[{amps_logit_override[:T_use].min():.2f}, "
              f"{amps_logit_override[:T_use].max():.2f}]")

    controls = processor_group.get_controls(inputs)
    audio = processor_group.get_signal(controls)
    return audio.numpy().squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quill-controls", type=str, default=None,
                        help="Directory with {melody}_f0.npy and {melody}_loudness.npy")
    parser.add_argument("--melodies", nargs="+", default=MELODIES_ORDER)
    parser.add_argument("--no-slide", action="store_true",
                        help="Replace slide (art=2) with plain (art=0) in scores")
    args = parser.parse_args()

    scores = strip_slides(SCORES) if args.no_slide else SCORES
    if args.no_slide:
        print("--no-slide: art=2 → art=0")

    OUT.mkdir(parents=True, exist_ok=True)

    for mname in args.melodies:
        print(f"\n=== {mname} ===")
        export = load_export(mname)
        meta = export["meta"]
        midi_path = Path(meta["source_midi"])
        seed = meta["source_seed"]
        npz_path = EXPORTS / f"{mname}.npz"

        if not midi_path.exists():
            midi_path = QUILL / "checkpoints" / "eval" / "midis" / f"{mname}.mid"

        print(f"  Re-synthesizing from {midi_path.name} (seed={seed})...")
        midi_audio, midi_ctrl, midi_synth, synth_gen = \
            synthesize_with_mididdsp(midi_path, seed)

        # --- MB×MD: MIDI-DDSP full stack, no overrides ---
        y_mbmd = midi_audio.numpy().squeeze() if hasattr(midi_audio, 'numpy') \
            else np.array(midi_audio).squeeze()
        peak = max(np.abs(y_mbmd).max(), 1e-9)
        y_mbmd = (y_mbmd / peak * 0.9).astype(np.float32)
        sf.write(str(OUT / f"{mname}_MB_MD.wav"), y_mbmd, SR)
        vr = voiced_ratio(y_mbmd)
        print(f"  MB×MD: {len(y_mbmd)/SR:.1f}s, voiced={vr:.0%}")

        # Gate: re-synthesis must match stored audio (same model load order + seed)
        ref_path = EXPORTS / "audio" / f"{mname}.wav"
        if ref_path.exists():
            ref, _ = sf.read(str(ref_path))
            T_min = min(len(ref), len(y_mbmd))
            corr = float(np.corrcoef(ref[:T_min], y_mbmd[:T_min])[0, 1])
            print(f"  vs stored audio: corr={corr:.6f}")
            if corr < 0.999:
                print(f"  FATAL: corr {corr:.6f} < 0.999. "
                      f"Seed not reproducing — check model load / seed order.")
                sys.exit(1)
        else:
            print(f"  WARNING: no stored audio at {ref_path}")

        # --- QB×MD: Quill behavior -> MIDI-DDSP decoder ---
        if args.quill_controls:
            qdir = Path(args.quill_controls)
            f0_path = qdir / f"{mname}_f0.npy"
            loud_path = qdir / f"{mname}_loudness.npy"
            if f0_path.exists() and loud_path.exists():
                f0_q = np.load(str(f0_path))
                loud_q = np.load(str(loud_path))

                score = scores[mname]
                q_frames = quill_note_frames(score)
                m_frames = midi_note_frames(npz_path)
                T_synth = int(midi_synth["inputs"]["f0_hz"].shape[1])

                print(f"  QB×MD: per-note resampling "
                      f"Quill {len(f0_q)} frames ({len(q_frames)} notes) "
                      f"-> MIDI grid {T_synth} frames ({len(m_frames)} notes)")

                f0_resampled = resample_per_note(f0_q, q_frames, m_frames, T_synth)
                loud_resampled = resample_per_note(loud_q, q_frames, m_frames, T_synth)

                amps_logit = quill_loudness_to_logit(loud_resampled, npz_path)

                y_qbmd = render_from_synth_params(
                    synth_gen, midi_synth,
                    f0_override=f0_resampled,
                    amps_logit_override=amps_logit,
                )
                peak = max(np.abs(y_qbmd).max(), 1e-9)
                y_qbmd = (y_qbmd / peak * 0.9).astype(np.float32)
                sf.write(str(OUT / f"{mname}_QB_MD.wav"), y_qbmd, SR)
                vr = voiced_ratio(y_qbmd)
                print(f"  QB×MD: {len(y_qbmd)/SR:.1f}s, voiced={vr:.0%}")
                if vr < 0.5:
                    print(f"  FATAL: QB×MD voiced ratio {vr:.0%} < 50%!")
                    sys.exit(1)
            else:
                print(f"  SKIP QB×MD: Quill controls not found at {qdir}")
        else:
            print("  SKIP QB×MD: --quill-controls not provided")

    print(f"\nOutputs in {OUT}/")


if __name__ == "__main__":
    main()
