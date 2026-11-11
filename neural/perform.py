"""Articulation × neural timbre: turn a phrase score into a 250 Hz (f0, loudness) control trajectory and feed it to a DDSP timbre.

control_from_behavior is the main render path (called by neural_take): micro-dynamics come from 4×48 trajectories retrieved from the behaviour library.
control_from_score / render_midi / render_strum / control_from_cooked / render_performance
are the hand-written rule versions of control trajectories and rendering, used by eval scripts and this module's demo.

  python -W ignore -m neural.perform
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural.ddsp import DDSP, HOP, SR  # noqa: E402

FPS = SR / HOP  # 250 Hz frame rate


def _m2f(midi):
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _is_new_arch(path):
    """Whether the checkpoint matches the current DDSPDecoder structure (inp.4 is the second Linear); mismatched weights cannot be loaded."""
    if not Path(path).exists():
        return False
    try:
        w = torch.load(path, map_location="cpu").get("dec.inp.4.weight")
        return w is not None and w.dim() == 2  # Linear weights are 2D; in the incompatible structure this is LayerNorm (1D)
    except Exception:
        return False


def note_template(real_note_path, sr_in=44100, pre=3):
    """Extract a loudness envelope template from one clean real instrument note, cropped to the main attack, removing the pre-triggered "double head".

    Every MIDI note uses this real envelope, so heads/decays are realistic with no pre-triggered double attack.
    """
    from scipy.signal import find_peaks

    from analysis.io import load
    from neural.ddsp import extract_features

    y, _ = load(real_note_path, sr=sr_in)
    _, loud, _ = extract_features(y, sr_in)
    t = loud[0, :, 0].numpy()
    pks, _ = find_peaks(t, prominence=0.8, distance=20)
    if len(pks):
        t = t[max(0, int(pks[0]) - pre) :]  # crop to the main attack = one pluck
    return t


def control_from_score(score, vib_rate=5.5, template=None):
    """score: [{midi, dur, slide?, vibrato?(cents), decay?}] -> (f0[1,T,1], loud[1,T,1], onsets).

    slide=True: slide from the previous note into this one (first 40%); vibrato=cents: sinusoidal vibrato on this note;
    loudness: per-note pluck envelope (instant attack + exponential decay), log then standardised per note (matches training).
    """
    f0_segs, loud_segs, onsets, prev, cum = [], [], [], None, 0
    for seg in score:
        n = max(2, int(seg["dur"] * FPS))
        onsets.append((cum * HOP, not seg.get("slide")))  # (sample position, whether it is a plucked attack)
        cum += n
        f0_t = _m2f(seg["midi"])
        f0 = np.full(n, f0_t, dtype=np.float64)
        if seg.get("slide") and prev is not None:
            sl = max(2, int(n * 0.4))
            f0[:sl] = _m2f(prev) * (f0_t / _m2f(prev)) ** np.linspace(0, 1, sl)
        if seg.get("vibrato"):
            t = np.arange(n) / FPS
            u = np.clip((t - 0.11) / 0.22, 0.0, 1.0)  # vibrato bloom: 110ms delay + 220ms fade-in
            venv = u * u * (3.0 - 2.0 * u)  # smoothstep fade-in
            f0 = f0 * 2.0 ** (venv * seg["vibrato"] * np.sin(2 * np.pi * vib_rate * t) / 1200.0)
        if template is not None:  # real envelope template: realistic dynamics, no double head
            loge = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(template)), template)
        else:  # fallback: synthetic pluck envelope
            t = np.arange(n) / FPS
            env = np.exp(-t / seg.get("tau", 0.45))
            a = max(1, int(0.010 * FPS))
            env[:a] *= np.linspace(0.0, 1.0, a)
            loge = np.log(env + 1e-6)
            loge = (loge - loge.mean()) / (loge.std() + 1e-5)
        f0_segs.append(f0)
        loud_segs.append(loge)
        prev = seg["midi"]
    return (
        torch.tensor(np.concatenate(f0_segs))[None, :, None].float(),
        torch.tensor(np.concatenate(loud_segs))[None, :, None].float(),
        onsets,
    )


def control_from_behavior(
    score,
    behavior_model,
    crossfade_ms=20.0,
    pitch_glide=True,
    envelope="pluck",
    traj_gain=1.0,
    vibrato_source="retrieval",
    vibrato_predictor=None,
    instrument_idx=0,
    randomness=0.3,
    label_processing="full",
    concat_lambda=0.0,
    contiguity=0.0,
    extra_dims=False,
):
    """score + behaviour model -> (f0[1,T,1], loud[1,T,1]) control stream.

    Micro-dynamics (pitch vibrato/bend, energy envelope) come from 4×48 trajectories predicted by the behaviour model,
    not from hand-written rules.

    score: [{midi, dur, art?}]  art: 0=plain 1=vibrato 2=slide, default 0
    behavior_model: object with .predict(feat) -> (4,48); if its use_art=True,
                    feat carries a 7th articulation-label dimension (conditioned retrieval).
    pitch_glide: if True, f0 crossfades at note boundaries (gives a sliding feel);
                 if False, f0 switches hard at the midpoint of the overlap and only loudness transitions smoothly.
    envelope: "pluck" (exponential decay, plucked strings) / "sustain" (attack-hold-release, bowed strings/winds)
    traj_gain: trajectory strength (1.0=unchanged; >1 exaggerates micro-dynamics, making cross-instrument transfer more audible)
    vibrato_source: "retrieval" (default, pitch entirely from retrieved trajectories) /
                    "model" (pitch of vibrato notes synthesised by the learned model, plain/slide unchanged)
    vibrato_predictor: VibratoPredictor instance (required when vibrato_source="model")
    instrument_idx: instrument index (0=violin 1=cello 2=guitar), passed to the vibrato model
    randomness: vibrato randomness 0–1 (only with vibrato_source="model")
    label_processing: "full" (default, label-based post-processing: plain lowpass/clamp, vibrato detrend/clamp/AM) /
                      "none" (retrieved trajectories as is, no label post-processing; for ablation) /
                      "uniform" (every note detrended + clamped to ±50c regardless of label; for ablation)
    concat_lambda / contiguity: concatenation-cost reranking and continuous retrieval bonus (see ConditionedNN.predict)
    extra_dims: if True, also return two numpy trajectories (bright[T], noise[T]) (the library's 3rd/4th dims:
                log2 centroid residual, log10 flatness residual), scaled by traj_gain and crossfaded the same way,
                for decoders that can use them (L0 control-driven).
    """
    cf = max(1, int(crossfade_ms / 1000.0 * FPS))
    use_art = bool(getattr(behavior_model, "use_art", False))
    segs_f0, segs_loud = [], []
    segs_bright, segs_noise = [], []
    if hasattr(behavior_model, "last_idx"):
        behavior_model.last_idx = None  # new phrase: continuous retrieval starts over

    # accumulate each note's onset in advance, so the gap computation is O(1) per note
    _onsets = []
    _cum = 0.0
    for s in score:
        _onsets.append(s.get("onset", _cum))
        _cum = _onsets[-1] + s["dur"]

    prev_traj = None
    for i, note in enumerate(score):
        midi, dur = note["midi"], note["dur"]
        n = max(4, int(dur * FPS))

        if i > 0:
            prev_gap = max(0.0, float(_onsets[i] - (_onsets[i - 1] + score[i - 1]["dur"])))
        else:
            prev_gap = 1.0
        if i < len(score) - 1:
            next_gap = max(0.0, float(_onsets[i + 1] - (_onsets[i] + dur)))
        else:
            next_gap = 1.0
        base_feat = [
            float(midi),
            dur,
            prev_gap,
            next_gap,
            1.0 if i == 0 else 0.0,
            1.0 if i == len(score) - 1 else 0.0,
        ]
        if use_art:
            base_feat.append(float(note.get("art", 0)))
        feat = np.array(base_feat, np.float32)

        if (
            (concat_lambda > 0 or contiguity > 0)
            and prev_traj is not None
            and getattr(behavior_model, "supports_concat", False)
        ):
            traj = behavior_model.predict(
                feat, prev_traj=prev_traj, concat_lambda=concat_lambda, contiguity=contiguity
            )
        else:
            traj = behavior_model.predict(feat)
        pitch_d = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[0])
        energy_d = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[1]) * traj_gain
        if extra_dims:
            segs_bright.append(
                np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[2]) * traj_gain
            )
            segs_noise.append(
                np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[3]) * traj_gain
            )
        # intonation anchoring: real trajectories carry the performer's intonation error (mean≠0), and gain would amplify being out of tune.
        # plain/vibrato subtract the whole-trajectory mean (keep shape, remove bias); slide anchors the tail (landing on the target note after the slide).
        art_i = int(note.get("art", 0))
        anchor = pitch_d[-max(1, n // 4) :].mean() if art_i == 2 else pitch_d.mean()
        pitch_d = (pitch_d - anchor) * traj_gain

        # slide interval alignment: the retrieval features have no interval direction, so the direction/size of a retrieved slide trajectory may not match the
        # actual melodic interval. Decomposition: the slide timing (when and how fast) comes from the displacement progress curve of the retrieved trajectory,
        # while size and direction come from the real melodic interval, starting from the previous note and landing exactly on this one.
        if art_i == 2 and i > 0:
            interval = (float(score[i - 1]["midi"]) - float(midi)) * 100.0
            if abs(interval) > 1.0:
                # slide progress = net displacement fraction toward the end point (made monotonic + clamped). Not cumulative |displacement|: small release
                # jitter at the trajectory tail dilutes progress, plateauing at 85% and leaving the pitch hanging by interval×15%.
                s = pitch_d - float(np.median(pitch_d[:3]))
                # the denominator uses the median of the hold section (50%–85%) rather than the last frame: some library trajectories sit 50 c off for the whole hold
                # and only correct at release; normalising by the last frame would reproduce that off-pitch plateau, normalising by the hold section lands in tune.
                hold = float(np.median(s[int(0.5 * n) : max(int(0.5 * n) + 2, int(0.85 * n))]))
                u = None
                if abs(hold) > 5.0:
                    u = np.clip(np.maximum.accumulate(s / hold), 0.0, 1.0)
                    # sliding too early (done within the first 15% = a hard jump) or starting too late (not started at 60% =
                    # held-then-jump) are fake slide trajectories in the library, discard them
                    t95 = int(np.argmax(u >= 0.95))
                    t05 = int(np.argmax(u >= 0.05))
                    if t95 < max(2, int(0.15 * n)) or t05 > int(0.6 * n):
                        u = None
                if u is None:
                    # fallback: a fast shift-style slide (a violin shift is about 50–120 ms, not a uniform portamento)
                    sl = max(2, min(int(0.12 * FPS), int(n * 0.4)))
                    u = np.ones(n)
                    t01 = np.linspace(0.0, 1.0, sl)
                    u[:sl] = t01 * t01 * (3.0 - 2.0 * t01)
                    resid = np.zeros(n)  # the texture of a fake trajectory contains the jump itself, cannot be added back
                else:
                    # residual = the micro texture left after removing the retrieved trajectory's own macro slide line
                    # (overshoot and settle, vibrato after the slide, tail jitter), defined by the trajectory's own timing.
                    resid_raw = s - hold * u
                    # the macro slide line does not use the retrieved progress curve directly: a jittery trajectory becomes a staircase after being made monotonic (each
                    # back-jitter adds a plateau), and whole-tone slides become smeary. Only the slide duration is taken (t95, clamped to the shift range
                    # 50–200 ms), with a smoothstep shape: one clean accelerating slide into place, which sounds like a string shift.
                    sl = int(
                        np.clip(t95, max(2, int(0.05 * FPS)), min(int(0.2 * FPS), max(2, n // 2)))
                    )
                    u = np.ones(n)
                    t01 = np.linspace(0.0, 1.0, sl)
                    u[:sl] = t01 * t01 * (3.0 - 2.0 * t01)
                    t95 = sl
                    # re-anchor to zero after landing + linear detrend (vocal trajectories keep drifting up after landing, hanging +30 c at the tail;
                    # detrending keeps the jitter and removes the drift), clamp to ±40 c to prevent macro leakage, fade in with progress (no interference
                    # with intonation during the slide).
                    land = resid_raw[t95:]
                    if len(land) >= 4:
                        x01 = np.arange(len(land), dtype=np.float64)
                        resid_raw[t95:] = land - np.polyval(np.polyfit(x01, land, 1), x01)
                    resid_raw = resid_raw - float(np.median(resid_raw[t95:]))
                    resid = np.clip(resid_raw, -40.0, 40.0) * u
                pitch_d = interval * (1.0 - u) + resid

        # attack stabilisation: the first few frames of a trajectory may carry the performer's pitch instability (especially the moment a bow bites), so over the first 30 ms
        # pitch_d fades in from 0 to keep the attack in tune. Slides excepted: the head offset of a slide is exactly "starting from the previous
        # note" and must not be erased.
        onset_frames = max(1, int(0.030 * FPS))
        if art_i != 2 and onset_frames < n:
            fade = np.linspace(0.0, 1.0, onset_frames)
            pitch_d[:onset_frames] *= fade

        # plain-note vibrato suppression: trajectories in the URMP "plain" pool often contain natural vibrato; without suppression plain/vibrato have no contrast.
        # Lowpass removes pitch oscillation >3 Hz (keeps slow drift, removes the vibrato band).
        if label_processing == "full" and art_i == 0 and n >= 8:
            from scipy.signal import butter, sosfiltfilt

            cutoff = 3.0  # Hz
            nyq = FPS / 2.0
            if cutoff < nyq:
                sos = butter(2, cutoff / nyq, btype="low", output="sos")
                pitch_d = sosfiltfilt(sos, pitch_d)
            # the attack scoop of the vocal library can reach ±400 c, so the first third of fast notes is out of tune.
            # Clamp to ±50 c: keeps slow drift, cuts large attack scoops.
            pitch_d = np.clip(pitch_d, -50.0, 50.0)

        # vibrato detrend + amplitude clamp: vibrato is an oscillation around a stable centre; library trajectories (especially vocal) often carry <2 Hz
        # slow drift; subtracting it keeps the pitch centre, the 5–7 Hz oscillation is kept as is; then clamp to ±35 c to prevent overly wide vibrato.
        if label_processing == "full" and art_i == 1:
            if n >= 8:
                from scipy.signal import butter, sosfiltfilt

                nyq = FPS / 2.0
                sos_d = butter(2, 2.0 / nyq, btype="low", output="sos")
                pitch_d = pitch_d - sosfiltfilt(sos_d, pitch_d)
            pitch_d = np.clip(pitch_d, -35.0, 35.0)

        # uniform mode: every note detrended + clamped to ±50 c regardless of label (for ablation)
        if label_processing == "uniform" and art_i != 2:
            if n >= 8:
                from scipy.signal import butter, sosfiltfilt

                nyq = FPS / 2.0
                sos_u = butter(2, 2.0 / nyq, btype="low", output="sos")
                pitch_d = pitch_d - sosfiltfilt(sos_u, pitch_d)
            pitch_d = np.clip(pitch_d, -50.0, 50.0)

        # vibrato model branch: for art=1 the learned model replaces the retrieved pitch trajectory
        if vibrato_source == "model" and art_i == 1 and vibrato_predictor is not None:
            prev_gap_v = prev_gap
            next_gap_v = next_gap
            vib_cents, _ = vibrato_predictor.synth(
                midi,
                dur,
                prev_gap_v,
                next_gap_v,
                1.0 if i == 0 else 0.0,
                1.0 if i == len(score) - 1 else 0.0,
                art_i,
                instrument_idx=instrument_idx,
                n_frames=n,
                randomness=randomness,
            )
            pitch_d = vib_cents

        f0_seg = _m2f(midi) * 2.0 ** (pitch_d / 1200.0)

        t = np.arange(n) / FPS
        if envelope == "sustain":
            # bowed-string envelope assigns roles by position in the phrase: notes inside a phrase are slurred and must not each re-attack
            # (otherwise every boundary has a -13 dB pit and sounds broken/stuttering; pits inside slides are the most grating).
            base = np.ones(n)
            if i == 0:
                a = max(1, int(0.03 * FPS))
                base[:a] = np.linspace(0.05, 1.0, a)  # phrase start: a real attack
            elif art_i == 2:
                a = max(1, int(0.02 * FPS))
                base[:a] = np.linspace(0.9, 1.0, a)  # slide: slurred slide-in, nearly seamless
            else:
                a = max(1, int(0.02 * FPS))
                base[:a] = np.linspace(0.6, 1.0, a)  # note change within the phrase: light attack
            if i == len(score) - 1:
                r = max(1, int(min(0.06, dur * 0.2) * FPS))
                base[-r:] *= np.linspace(1.0, 0.3, r)  # phrase end: bow release
            else:
                nxt_slide = int(score[i + 1].get("art", 0)) == 2
                r = max(1, int(min(0.04, dur * 0.15) * FPS))
                base[-r:] *= np.linspace(1.0, 0.95 if nxt_slide else 0.8, r)
        else:
            base = np.exp(-t / max(dur * 0.6, 0.2))
            a = max(1, int(0.008 * FPS))
            base[:a] *= np.linspace(0.01, 1.0, a)
        log_e = np.log(base + 1e-7)
        log_e = (log_e - log_e.mean()) / (log_e.std() + 1e-5)
        # after z-scoring notes within the phrase, add a floor at the troughs: per-note standardisation keeps each note's dynamic arc, but boundaries inside the phrase
        # must not drop into deep pits (broken/stuttering feel); the phrase-start attack and phrase-end release keep full depth.
        if 0 < i < len(score) - 1:
            log_e = np.maximum(log_e, -1.0)
        loud_seg = log_e + energy_d * 0.5

        # vibrato amplitude modulation: in real bowed vibrato the loudness rises and falls at the vibrato rate (harmonics moving in and out of body resonances).
        # Take the oscillation from pitch_d and couple about 15% into loudness.
        if label_processing == "full" and art_i == 1 and n >= 8:
            from scipy.signal import detrend as _detrend

            pitch_osc = _detrend(pitch_d)
            peak = np.abs(pitch_osc).max() + 1e-6
            am = 0.15 * (pitch_osc / peak)  # ±15% loudness modulation
            loud_seg = loud_seg + am

        segs_f0.append(f0_seg.astype(np.float64))
        segs_loud.append(loud_seg.astype(np.float64))
        prev_traj = traj

    total = sum(len(s) for s in segs_f0)
    f0_out = np.zeros(total, np.float64)
    loud_out = np.zeros(total, np.float64)
    w_f0 = np.zeros(total, np.float64)
    w_loud = np.zeros(total, np.float64)
    br_out = np.zeros(total, np.float64)
    nz_out = np.zeros(total, np.float64)
    pos = 0
    for i in range(len(segs_f0)):
        n = len(segs_f0[i])
        env = np.ones(n)
        fi = min(cf, n // 2) if i > 0 else 0
        fo = min(cf, n // 2) if i < len(segs_f0) - 1 else 0
        if fi:
            env[:fi] *= np.linspace(0, 1, fi)
        if fo:
            env[-fo:] *= np.linspace(1, 0, fo)

        if pitch_glide:
            env_f0 = env
        else:
            # f0 is not crossfaded: the first half of the overlap belongs to the previous note, the second half to the current note, no interpolated slide
            env_f0 = np.ones(n)
            if fi:
                env_f0[: fi // 2] = 0.0
            if fo:
                keep = fo - fo // 2  # the first half of the tail overlap still belongs to this note
                env_f0[n - fo + keep :] = 0.0

        if i > 0:
            pos -= min(cf, len(segs_f0[i - 1]) // 2, n // 2)
        end = pos + n
        f0_out[pos:end] += segs_f0[i] * env_f0
        loud_out[pos:end] += segs_loud[i] * env
        if extra_dims:
            br_out[pos:end] += segs_bright[i] * env
            nz_out[pos:end] += segs_noise[i] * env
        w_f0[pos:end] += env_f0
        w_loud[pos:end] += env
        pos = end
    T = pos
    # frames where the f0 weight is 0 (gaps at hard-switch boundaries) are filled with the previous value
    sw_f0 = w_f0[:T].copy()
    f0_n = f0_out[:T] / np.maximum(sw_f0, 1e-8)
    hole = sw_f0 < 1e-8
    if hole.any():
        idx = np.where(~hole)[0]
        f0_n = np.interp(np.arange(T), idx, f0_n[idx])
    sw_l = np.maximum(w_loud[:T], 1e-8)
    f0_t = torch.tensor(f0_n)[None, :, None].float()
    loud_t = torch.tensor(loud_out[:T] / sw_l)[None, :, None].float()
    if extra_dims:
        return f0_t, loud_t, br_out[:T] / sw_l, nz_out[:T] / sw_l
    return f0_t, loud_t


@torch.no_grad()
def render(model_path, f0, loud, device=None, noise_gain=1.0):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = DDSP(dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    y = model(f0.to(device), loud.to(device), noise_gain=noise_gain).cpu().numpy()[0]
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)


def _mute_template(template, tau=0.15):
    """Palm mute: overlay a fast decay on the real template -> a short muted sound."""
    t = np.arange(len(template)) / FPS
    lin = np.exp(template) * np.exp(-t / tau)
    lg = np.log(lin + 1e-6)
    return (lg - lg.mean()) / (lg.std() + 1e-5)


def render_midi(model_path, score, template, device=None, vib_rate=5.5):
    """MIDI monophonic phrase -> neural timbre. Control-curve articulations (f0/loudness trajectories):

    f0 type: pluck/legato/slide/bend/vibrato/trill (square-wave alternation of two notes)/gliss (long glissando);
    loudness type: mute (muted)/tremolo (fast repeated attacks)/swell (volume fade-in, no pluck head).
    Attack types each place a loudness template; legato/slide do not re-attack (continue the decay, only slide f0).
    """
    frames = [max(2, int(s["dur"] * FPS)) for s in score]
    total = sum(frames)
    f0 = np.zeros(total, np.float64)
    loud = np.full(total, float(template[-1]), np.float64)  # default = template tail (continued decay)
    cum, prev = 0, None
    for seg, n in zip(score, frames):
        midi = seg["midi"]
        art = seg.get("art", "pluck")
        tt = np.arange(n) / FPS
        sf0 = np.full(n, _m2f(midi))
        if art in ("legato", "slide") and prev is not None:  # continuation: slide f0, no re-attack
            sl = max(2, int(n * (0.22 if art == "legato" else 0.4)))
            sf0[:sl] = _m2f(prev) * (_m2f(midi) / _m2f(prev)) ** np.linspace(0, 1, sl)
        elif art == "trill":  # trill: square-wave alternation between this note and this note+iv
            iv = seg.get("trill", 2)
            rate = seg.get("trill_rate", 8.0)
            sf0 = _m2f(midi) * 2 ** ((np.sin(2 * np.pi * rate * tt) > 0) * iv / 12.0)
        elif art == "gliss":  # long glissando: slide the whole note to the target
            tgt = seg.get("gliss_to", midi + 7)
            sf0 = _m2f(midi) * (_m2f(tgt) / _m2f(midi)) ** np.linspace(0, 1, n)
        if art == "bend":  # string bend: rise quickly to +bend semitones within ~bend_ms, then hold
            b = seg.get("bend", 2.0)
            h = max(2, min(n - 1, int(seg.get("bend_ms", 170) / 1000.0 * FPS)))
            sf0[:h] *= 2 ** (np.linspace(0, b, h) / 12.0)
            sf0[h:] *= 2 ** (b / 12.0)
        if seg.get("vibrato"):
            vu = np.clip((tt - 0.07) / 0.16, 0.0, 1.0)  # vibrato bloom (solo: earlier onset, faster fade-in)
            ve = vu * vu * (3.0 - 2.0 * vu)
            sf0 *= 2 ** (ve * seg["vibrato"] * np.sin(2 * np.pi * vib_rate * tt) / 1200.0)
        f0[cum : cum + n] = sf0
        # ---- Loudness ----
        if art in ("legato", "slide"):  # continued decay, loud untouched
            pass
        elif art == "tremolo":  # tremolo: place the template repeatedly and fast
            period = max(2, int(FPS / seg.get("trem_rate", 10.0)))
            for st in range(0, n, period):
                m = min(len(template), n - st)
                loud[cum + st : cum + st + m] = template[:m]
        elif art == "swell":  # swell: raised-cosine fade-in, no pluck head
            r = min(n, max(2, int(n * seg.get("swell_frac", 0.5))))
            base, peak = float(template[-1]), float(np.percentile(template, 75))
            seg_loud = np.full(n, peak)
            seg_loud[:r] = base + (peak - base) * 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))
            loud[cum : cum + n] = seg_loud
        else:  # pluck/bend/trill/gliss/mute: single attack
            tpl = _mute_template(template) if art == "mute" else template
            m = min(len(tpl), total - cum)
            loud[cum : cum + m] = tpl[:m]
        cum += n
        prev = midi
    f0t = torch.tensor(f0)[None, :, None].float()
    lot = torch.tensor(loud)[None, :, None].float()
    return render(model_path, f0t, lot, device)


def render_strum(model_path, chord, template, dur=1.6, spread_ms=110, device=None):
    """Strum: chord notes triggered spread_ms apart (the neural timbre is monophonic, so staggered layers are summed at audio level).

    spread_ms is the strum speed: small = fast tight strum, large = slow string-by-string.
    """
    offs = np.linspace(0, spread_ms, len(chord))
    parts = []
    for m, off in zip(chord, offs):
        y = render_midi(model_path, [{"midi": int(m), "dur": dur}], template, device)
        parts.append((int(SR * off / 1000.0), np.asarray(y)))
    total = max(s + len(y) for s, y in parts)
    buf = np.zeros(total, np.float64)
    for s, y in parts:
        buf[s : s + len(y)] += y
    return (buf / (np.max(np.abs(buf)) + 1e-9) * 0.9).astype(np.float32)


def control_from_cooked(
    cooked, template, tail_s=1.0, vib_rate=5.5, vib_max_cents=50.0, bend_is_semitones=True
):
    """Cooked events from the gesture engine -> 250 Hz (f0, loudness) control track (ControlFrame).

    Monophonic: on=pluck (place template), legato/slide=continued decay with only f0 sliding, bend/vibrato=time-based f0 modulation.
    (Chord strums are polyphonic and go through render_strum; here a monophonic control is built along the timeline.)
    """
    events = sorted(cooked, key=lambda e: e[0])
    onsets = [
        (int(round(e[0] * FPS)), e[1], e[2], e[5])
        for e in events
        if e[1] in ("on", "legato", "slide")
    ]
    if not onsets:
        z = np.zeros(int(FPS))
        return z, np.full_like(z, float(template[-1]))
    n = int(round((events[-1][0] + tail_s) * FPS)) + 1
    f0 = np.zeros(n, np.float64)
    loud = np.full(n, float(template[-1]), np.float64)
    first = onsets[0][0]
    if first > 0:
        loud[:first] = float(template[-1]) - 6.0  # attenuate before the first note (near silence)
    for k, (fr, art, note, frm) in enumerate(onsets):
        nxt = onsets[k + 1][0] if k + 1 < len(onsets) else n
        nxt = max(nxt, fr + 2)
        seg = np.full(nxt - fr, _m2f(note))
        if art in ("legato", "slide") and frm is not None and frm >= 0:
            r = min(nxt - fr, max(2, int((nxt - fr) * (0.22 if art == "legato" else 0.4))))
            seg[:r] = _m2f(frm) * (_m2f(note) / _m2f(frm)) ** np.linspace(0, 1, r)
        f0[fr:nxt] = seg
        if art == "on":  # pluck: place the template (continued decay left to later)
            m = min(len(template), n - fr)
            loud[fr : fr + m] = template[:m]
    # bend / vibrato: start at the event time, last until the next event of the same kind (stream uses 0 to switch off)
    bend = np.zeros(n)
    vib = np.zeros(n)
    for e in events:
        fr = min(n - 1, int(round(e[0] * FPS)))
        if e[1] == "bend":
            bend[fr:] = float(e[4]) if bend_is_semitones else float(e[4]) / 8192.0 * 2.0
        elif e[1] == "vibrato":
            vib[fr:] = float(e[4]) * vib_max_cents
    tt = np.arange(n) / FPS
    f0 = f0 * 2 ** (bend / 12.0) * 2 ** (vib * np.sin(2 * np.pi * vib_rate * tt) / 1200.0)
    return f0, loud


def render_performance(model_path, raw_events, template, library=None, device=None):
    """Real MIDI performance -> neural timbre. raw: [(t,kind,note,vel[,val])] is classified into articulations by the gesture engine, then rendered.

    Closed loop: play MIDI (notes + gestures) -> GestureInterpreter classifies pluck/legato/slide/bend/vibrato
    -> control_from_cooked builds (f0,loudness) -> the DDSP neural timbre sounds.
    """
    from runtime.gesture import GestureInterpreter

    cooked = GestureInterpreter(library=library).process(raw_events)
    f0, loud = control_from_cooked(cooked, template)
    f0t = torch.tensor(f0)[None, :, None].float()
    lot = torch.tensor(loud)[None, :, None].float()
    return render(model_path, f0t, lot, device)


def main(argv=None):
    import soundfile as sf

    out = Path("assets/demos")
    out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    # guitar phrase: a few plucked notes -> slide -> vibrato on the last note
    guitar_score = [
        {"midi": 60, "dur": 0.5},
        {"midi": 64, "dur": 0.5},
        {"midi": 67, "dur": 0.5},
        {"midi": 69, "dur": 0.6, "slide": True},
        {"midi": 67, "dur": 0.5},
        {"midi": 64, "dur": 1.2, "vibrato": 30},
    ]
    f0, lo, onsets = control_from_score(guitar_score)
    # prefer the acoustic guitar model if available, otherwise fall back to the Fender
    g_model = (
        "assets/demos/ddsp_acoustic_guitar.pt"
        if Path("assets/demos/ddsp_acoustic_guitar.pt").exists()
        and _is_new_arch("assets/demos/ddsp_acoustic_guitar.pt")
        else "assets/demos/ddsp_fender_strat.pt"
    )
    try:
        y = render(g_model, f0, lo, dev)
        sf.write(out / "perform_guitar_neural.wav", y, SR)
        print(f"articulation × neural guitar ({Path(g_model).stem}) -> perform_guitar_neural.wav")
    except Exception as e:
        print(f"  guitar render skipped: {str(e)[:60]}")

    # bass line: root-fifth-octave -> slide -> vibrato
    bass_score = [
        {"midi": 40, "dur": 0.45, "decay": 2.5},
        {"midi": 40, "dur": 0.45, "decay": 2.5},
        {"midi": 47, "dur": 0.45, "decay": 2.5},
        {"midi": 52, "dur": 0.5, "decay": 2.5},
        {"midi": 55, "dur": 0.7, "slide": True, "decay": 2.0},
        {"midi": 52, "dur": 1.0, "vibrato": 25, "decay": 1.5},
    ]
    f0b, lob, onb = control_from_score(bass_score)
    try:
        if not _is_new_arch("assets/demos/ddsp_electric_bass.pt"):
            raise RuntimeError("the electric bass model uses the old architecture; wait for the retrain with the new trainer")
        yb = render("assets/demos/ddsp_electric_bass.pt", f0b, lob, dev)
        sf.write(out / "perform_bass_neural.wav", yb, SR)
        print("articulation × neural electric bass -> perform_bass_neural.wav")
    except Exception as e:
        print(f"  bass render skipped: {str(e)[:60]}")
    # ---- per-articulation demo (MIDI-driven, real Fender note as the loudness template) ----
    fender = "assets/demos/ddsp_fender_strat.pt"
    datasets = os.environ.get("QUILL_DATASETS", str(Path.home() / "Desktop/MT/datasets"))
    note_src = (
        datasets + "/IDMT-SMT-GUITAR_V2/"
        "dataset1/Fender Strat Clean Neck SC/audio/G53-45105-1111-00006.wav"
    )
    if Path(fender).exists() and Path(note_src).exists():
        tmpl = note_template(note_src)
        arts = {
            "pluck": [{"midi": 64, "dur": 1.4}],  # single pluck
            "legato": [
                {"midi": 60, "dur": 0.45},
                {"midi": 64, "dur": 0.45, "art": "legato"},
                {"midi": 67, "dur": 1.0, "art": "legato"},
            ],  # hammer-on/pull-off legato (one pluck)
            "slide": [{"midi": 60, "dur": 0.5}, {"midi": 67, "dur": 1.2, "art": "slide"}],
            "mute": [
                {"midi": 52, "dur": 0.4, "art": "mute"},
                {"midi": 52, "dur": 0.4, "art": "mute"},
                {"midi": 52, "dur": 0.4, "art": "mute"},
                {"midi": 52, "dur": 0.4, "art": "mute"},
            ],
            "bend": [{"midi": 67, "dur": 1.6, "art": "bend", "bend": 2.0}],  # string bend + whole tone
            "vibrato": [{"midi": 64, "dur": 1.6, "vibrato": 35}],
        }
        for name, sc in arts.items():
            try:
                y = render_midi(fender, sc, tmpl, dev)
                y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
                sf.write(out / f"art_{name}.wav", y, SR)
                print(f"  articulation[{name}] -> art_{name}.wav")
            except Exception as e:
                print(f"  articulation[{name}] skipped: {str(e)[:60]}")
        try:  # strum an E major chord
            ys = render_strum(fender, [40, 47, 52, 56, 59, 64], tmpl, device=dev)
            sf.write(out / "art_strum.wav", ys, SR)
            print("  articulation[strum] -> art_strum.wav")
        except Exception as e:
            print(f"  articulation[strum] skipped: {str(e)[:60]}")

    # ---- closed-loop demo: real MIDI events -> gesture engine classifies articulation -> neural timbre ----
    if Path(fender).exists() and Path(note_src).exists():
        # an expressive phrase: pluck 60 -> hammer-on 62 (+2 semitones=legato) -> slide 67 (+5=slide) -> vibrato CC
        raw = [
            (0.00, "on", 60, 100),
            (0.45, "on", 62, 96),
            (0.50, "off", 60, 0),
            (0.90, "on", 67, 100),
            (0.95, "off", 62, 0),
            (1.40, "cc", 1, 0, 110),  # vibrato depth 110/127
            (2.40, "off", 67, 0),
        ]
        try:
            y = render_performance(fender, raw, tmpl, device=dev)
            y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
            sf.write(out / "perform_midi_gesture.wav", y, SR)
            print("  closed loop [MIDI->gesture engine->neural timbre] -> perform_midi_gesture.wav")
        except Exception as e:
            print(f"  closed-loop demo skipped: {str(e)[:80]}")

    # ---- electric bass: articulations + closed loop ----
    bass_model = "assets/demos/ddsp_electric_bass.pt"
    bass_note = datasets + "/IDMT-SMT-BASS/ES/NO/BS_1_EQ_1_FS_NO_1_0.wav"
    if Path(bass_model).exists() and _is_new_arch(bass_model) and Path(bass_note).exists():
        btmpl = note_template(bass_note)
        bass_arts = {  # bass range (E1=28..G2=43)
            "pluck": [{"midi": 33, "dur": 1.4}],  # A1 fingerstyle
            "legato": [
                {"midi": 28, "dur": 0.5},
                {"midi": 31, "dur": 0.5, "art": "legato"},
                {"midi": 33, "dur": 1.0, "art": "legato"},
            ],
            "slide": [{"midi": 28, "dur": 0.5}, {"midi": 35, "dur": 1.2, "art": "slide"}],
            "mute": [{"midi": 33, "dur": 0.35, "art": "mute"}] * 4,  # muted staccato
            "vibrato": [{"midi": 33, "dur": 1.6, "vibrato": 25}],
        }
        for name, sc in bass_arts.items():
            try:
                y = render_midi(bass_model, sc, btmpl, dev)
                y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
                sf.write(out / f"bass_art_{name}.wav", y, SR)
                print(f"  bass articulation[{name}] -> bass_art_{name}.wav")
            except Exception as e:
                print(f"  bass articulation[{name}] skipped: {str(e)[:60]}")
        # closed-loop bassline: pluck root -> hammer-on -> octave -> slide back (real MIDI -> gesture engine -> neural bass)
        raw_bass = [
            (0.00, "on", 28, 110),
            (0.50, "off", 28, 0),  # E1 pluck
            (0.50, "on", 28, 100),
            (0.95, "on", 31, 90),  # hammer-on +3 (legato)
            (1.00, "off", 28, 0),
            (1.40, "off", 31, 0),
            (1.40, "on", 40, 105),
            (1.90, "off", 40, 0),  # E2 octave
            (1.90, "on", 40, 100),
            (2.30, "on", 35, 95),  # slide to B1 (-5=slide)
            (2.35, "off", 40, 0),
            (3.10, "off", 35, 0),
        ]
        try:
            yb = render_performance(bass_model, raw_bass, btmpl, device=dev)
            yb = (yb / (np.max(np.abs(yb)) + 1e-9) * 0.9).astype(np.float32)
            sf.write(out / "perform_midi_bass.wav", yb, SR)
            print("  closed loop [MIDI->gesture engine->neural bass] -> perform_midi_bass.wav")
        except Exception as e:
            print(f"  bass closed loop skipped: {str(e)[:80]}")

    print("\nControlFrame -> DDSP integration: gesture score (notes + slides + vibrato) -> neural timbre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
