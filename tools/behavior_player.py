"""Minimal behavior library player — numpy + scipy only, no torch.

Usage:
    python tools/behavior_player.py checkpoints/urmp_vn_trajectories.npz -o demo.wav
    python tools/behavior_player.py checkpoints/glass_style.npz --score 84,2.0,1 79,1.5,0 84,3.0,1

Outputs 250 fps control signals (f0_hz, loudness_z, bright, noise) and renders a sine-synth demo wav.
"""

import argparse
import json

import numpy as np

FPS = 250


SLIDE_RANGE_CAP = 400.0


class BehaviorLibrary:
    """Nearest-neighbor retrieval on a behavior library npz.

    Matches ConditionedNN's normalization and slide filtering.
    """

    def __init__(self, path, slide_cap=SLIDE_RANGE_CAP):
        d = np.load(str(path), allow_pickle=False)
        self.features = d["features"].astype(np.float32)
        self.trajectories = d["trajectories"].astype(np.float32)
        self.meta = json.loads(str(d["meta"]))
        self.use_art = self.features.shape[1] >= 7
        if self.use_art:
            art = self.features[:, 6].astype(int)
            rng = self.trajectories[:, 0].max(axis=1) - self.trajectories[:, 0].min(axis=1)
            keep = (art != 2) | (rng <= slide_cap)
            self.features = self.features[keep]
            self.trajectories = self.trajectories[keep]
            art = art[keep]
            self._pools = {int(a): np.where(art == a)[0] for a in np.unique(art)}
        self.base = self.features[:, :6]
        self._std = self.base.std(axis=0) + 1e-8
        self._norm = self.base / self._std
        self.last_idx = None

    def predict(self, feat):
        """Retrieve the nearest trajectory for a 6-D or 7-D feature vector."""
        q = feat[:6] / self._std
        if self.use_art:
            art = int(feat[6])
            pool = self._pools.get(art)
            if pool is None or len(pool) == 0:
                pool = np.arange(len(self.base))
            dist = np.linalg.norm(self._norm[pool] - q, axis=1)
            idx = pool[int(np.argmin(dist))]
        else:
            dist = np.linalg.norm(self._norm - q, axis=1)
            idx = int(np.argmin(dist))
        self.last_idx = idx
        return self.trajectories[idx].copy()


def _m2f(midi):
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def play(score, library, crossfade_ms=20.0, pitch_glide=True,
         envelope="sustain", traj_gain=1.0, label_processing="full"):
    """Generate f0 (Hz) and loudness (z-score) from a score + behavior library.

    Args:
        score: list of (midi, dur, art) tuples
        library: BehaviorLibrary instance
        crossfade_ms: overlap between notes in ms
        pitch_glide: if True, f0 crossfades at boundaries
        envelope: "sustain" or "pluck"
        traj_gain: trajectory scaling factor
        label_processing: "full", "none", or "uniform"

    Returns:
        f0_hz: numpy array, shape (T,)
        loud_z: numpy array, shape (T,)
        bright: numpy array, shape (T,) — log2 spectral centroid residual
        noise: numpy array, shape (T,) — log10 spectral flatness residual
    """
    cf = max(1, int(crossfade_ms / 1000.0 * FPS))
    segs_f0, segs_loud = [], []
    segs_bright, segs_noise = [], []
    library.last_idx = None

    onsets = []
    cum = 0.0
    for _m, d, _a in score:
        onsets.append(cum)
        cum += d

    for i, (midi, dur, art) in enumerate(score):
        n = max(4, int(dur * FPS))
        prev_gap = 1.0 if i == 0 else max(0.0, onsets[i] - (onsets[i-1] + score[i-1][1]))
        next_gap = 1.0 if i == len(score)-1 else max(0.0, onsets[i+1] - (onsets[i] + dur))

        feat = np.array([
            float(midi), dur, prev_gap, next_gap,
            1.0 if i == 0 else 0.0,
            1.0 if i == len(score) - 1 else 0.0,
            float(art),
        ], np.float32)

        traj = library.predict(feat)

        pitch_d = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[0])
        energy_d = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[1]) * traj_gain
        bright_d = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[2]) * traj_gain
        noise_d = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[3]) * traj_gain

        art_i = int(art)
        anchor = pitch_d[-max(1, n // 4):].mean() if art_i == 2 else pitch_d.mean()
        pitch_d = (pitch_d - anchor) * traj_gain

        if art_i == 2 and i > 0:
            interval = (float(score[i - 1][0]) - float(midi)) * 100.0
            if abs(interval) > 1.0:
                s = pitch_d - float(np.median(pitch_d[:3]))
                hold = float(np.median(s[int(0.5 * n):max(int(0.5 * n) + 2, int(0.85 * n))]))
                u = None
                if abs(hold) > 5.0:
                    u = np.clip(np.maximum.accumulate(s / hold), 0.0, 1.0)
                    t95 = int(np.argmax(u >= 0.95))
                    t05 = int(np.argmax(u >= 0.05))
                    if t95 < max(2, int(0.15 * n)) or t05 > int(0.6 * n):
                        u = None
                if u is None:
                    sl = max(2, min(int(0.12 * FPS), int(n * 0.4)))
                    u = np.ones(n)
                    t01 = np.linspace(0.0, 1.0, sl)
                    u[:sl] = t01 * t01 * (3.0 - 2.0 * t01)
                    resid = np.zeros(n)
                else:
                    resid_raw = s - hold * u
                    sl = int(np.clip(t95, max(2, int(0.05 * FPS)),
                                     min(int(0.2 * FPS), max(2, n // 2))))
                    u = np.ones(n)
                    t01 = np.linspace(0.0, 1.0, sl)
                    u[:sl] = t01 * t01 * (3.0 - 2.0 * t01)
                    t95 = sl
                    land = resid_raw[t95:]
                    if len(land) >= 4:
                        x01 = np.arange(len(land), dtype=np.float64)
                        resid_raw[t95:] = land - np.polyval(np.polyfit(x01, land, 1), x01)
                    resid_raw = resid_raw - float(np.median(resid_raw[t95:]))
                    resid = np.clip(resid_raw, -40.0, 40.0) * u
                pitch_d = interval * (1.0 - u) + resid

        onset_frames = max(1, int(0.030 * FPS))
        if art_i != 2 and onset_frames < n:
            fade = np.linspace(0.0, 1.0, onset_frames)
            pitch_d[:onset_frames] *= fade

        if label_processing == "full" and art_i == 0 and n >= 8:
            from scipy.signal import butter, sosfiltfilt
            cutoff = 3.0
            nyq = FPS / 2.0
            if cutoff < nyq:
                sos = butter(2, cutoff / nyq, btype="low", output="sos")
                pitch_d = sosfiltfilt(sos, pitch_d)
            pitch_d = np.clip(pitch_d, -50.0, 50.0)

        if label_processing == "full" and art_i == 1:
            if n >= 8:
                from scipy.signal import butter, sosfiltfilt
                nyq = FPS / 2.0
                sos_d = butter(2, 2.0 / nyq, btype="low", output="sos")
                pitch_d = pitch_d - sosfiltfilt(sos_d, pitch_d)
            pitch_d = np.clip(pitch_d, -35.0, 35.0)

        if label_processing == "uniform" and art_i != 2:
            if n >= 8:
                from scipy.signal import butter, sosfiltfilt
                nyq = FPS / 2.0
                sos_u = butter(2, 2.0 / nyq, btype="low", output="sos")
                pitch_d = pitch_d - sosfiltfilt(sos_u, pitch_d)
            pitch_d = np.clip(pitch_d, -50.0, 50.0)

        f0_seg = _m2f(midi) * 2.0 ** (pitch_d / 1200.0)

        t = np.arange(n) / FPS
        if envelope == "sustain":
            base = np.ones(n)
            if i == 0:
                a_len = max(1, int(0.03 * FPS))
                base[:a_len] = np.linspace(0.05, 1.0, a_len)
            elif art_i == 2:
                a_len = max(1, int(0.02 * FPS))
                base[:a_len] = np.linspace(0.9, 1.0, a_len)
            else:
                a_len = max(1, int(0.02 * FPS))
                base[:a_len] = np.linspace(0.6, 1.0, a_len)
            if i == len(score) - 1:
                r = max(1, int(min(0.06, dur * 0.2) * FPS))
                base[-r:] *= np.linspace(1.0, 0.3, r)
            else:
                nxt_slide = int(score[i + 1][2]) == 2
                r = max(1, int(min(0.04, dur * 0.15) * FPS))
                base[-r:] *= np.linspace(1.0, 0.95 if nxt_slide else 0.8, r)
        else:
            base = np.exp(-t / max(dur * 0.6, 0.2))
            a_len = max(1, int(0.008 * FPS))
            base[:a_len] *= np.linspace(0.01, 1.0, a_len)

        log_e = np.log(base + 1e-7)
        log_e = (log_e - log_e.mean()) / (log_e.std() + 1e-5)
        if 0 < i < len(score) - 1:
            log_e = np.maximum(log_e, -1.0)
        loud_seg = log_e + energy_d * 0.5

        if label_processing == "full" and art_i == 1 and n >= 8:
            from scipy.signal import detrend
            pitch_osc = detrend(pitch_d)
            peak = np.abs(pitch_osc).max() + 1e-6
            am = 0.15 * (pitch_osc / peak)
            loud_seg = loud_seg + am

        segs_f0.append(f0_seg.astype(np.float64))
        segs_loud.append(loud_seg.astype(np.float64))
        segs_bright.append(bright_d.astype(np.float64))
        segs_noise.append(noise_d.astype(np.float64))

    total = sum(len(s) for s in segs_f0)
    f0_out = np.zeros(total, np.float64)
    loud_out = np.zeros(total, np.float64)
    br_out = np.zeros(total, np.float64)
    nz_out = np.zeros(total, np.float64)
    w_f0 = np.zeros(total, np.float64)
    w_loud = np.zeros(total, np.float64)
    pos = 0
    for i in range(len(segs_f0)):
        seg_n = len(segs_f0[i])
        env = np.ones(seg_n)
        fi = min(cf, seg_n // 2) if i > 0 else 0
        fo = min(cf, seg_n // 2) if i < len(segs_f0) - 1 else 0
        if fi:
            env[:fi] *= np.linspace(0, 1, fi)
        if fo:
            env[-fo:] *= np.linspace(1, 0, fo)

        if pitch_glide:
            env_f0 = env
        else:
            env_f0 = np.ones(seg_n)
            if fi:
                env_f0[:fi // 2] = 0.0
            if fo:
                keep = fo - fo // 2
                env_f0[seg_n - fo + keep:] = 0.0

        if i > 0:
            pos -= min(cf, len(segs_f0[i-1]) // 2, seg_n // 2)
        end = pos + seg_n
        f0_out[pos:end] += segs_f0[i] * env_f0
        loud_out[pos:end] += segs_loud[i] * env
        br_out[pos:end] += segs_bright[i] * env
        nz_out[pos:end] += segs_noise[i] * env
        w_f0[pos:end] += env_f0
        w_loud[pos:end] += env
        pos = end

    T = pos
    sw_f0 = w_f0[:T].copy()
    f0_hz = f0_out[:T] / np.maximum(sw_f0, 1e-8)
    hole = sw_f0 < 1e-8
    if hole.any():
        idx = np.where(~hole)[0]
        f0_hz = np.interp(np.arange(T), idx, f0_hz[idx])
    sw_l = np.maximum(w_loud[:T], 1e-8)
    loud_z = loud_out[:T] / sw_l
    bright = br_out[:T] / sw_l
    noise = nz_out[:T] / sw_l

    return f0_hz, loud_z, bright, noise


def synth_sine(f0_hz, loud_z, sr=16000):
    """Render f0 + loudness to audio with a simple sine oscillator."""
    hop = sr // FPS
    T_samp = len(f0_hz) * hop
    audio = np.zeros(T_samp, np.float64)
    phase = 0.0
    for i in range(len(f0_hz)):
        amp = np.clip(np.exp(loud_z[i] * 0.5) * 0.3, 0.0, 0.9)
        freq = f0_hz[i]
        for j in range(hop):
            audio[i * hop + j] = amp * np.sin(phase)
            phase += 2.0 * np.pi * freq / sr
    return (audio / (np.max(np.abs(audio)) + 1e-9) * 0.8).astype(np.float32)


DEFAULT_SCORE = [
    (62, 0.8, 1), (65, 0.4, 0), (67, 1.2, 1), (65, 0.4, 0),
    (69, 0.4, 0), (72, 1.6, 1), (67, 0.8, 0), (62, 1.6, 1),
]


def main():
    ap = argparse.ArgumentParser(description="Behavior library player")
    ap.add_argument("library", help="Path to behavior library npz")
    ap.add_argument("-o", "--output", default="demo.wav", help="Output wav path")
    ap.add_argument("--score", nargs="+", help="Notes as midi,dur,art (e.g. 62,0.8,1)")
    ap.add_argument("--envelope", default="sustain", choices=["sustain", "pluck"])
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    lib = BehaviorLibrary(args.library)

    if args.score:
        score = []
        for s in args.score:
            parts = s.split(",")
            score.append((float(parts[0]), float(parts[1]), int(parts[2])))
    else:
        score = DEFAULT_SCORE

    print(f"Library: {args.library} ({len(lib.features)} notes)")
    print(f"Score: {len(score)} notes, envelope={args.envelope}")

    f0_hz, loud_z, bright, noise = play(score, lib, envelope=args.envelope)
    print(f"Control signals: {len(f0_hz)} frames ({len(f0_hz)/FPS:.2f} s)")
    print(f"  bright range: [{bright.min():.3f}, {bright.max():.3f}]")
    print(f"  noise  range: [{noise.min():.3f}, {noise.max():.3f}]")

    audio = synth_sine(f0_hz, loud_z, sr=args.sr)
    import soundfile as sf
    sf.write(args.output, audio, args.sr)
    print(f"Wrote {args.output} ({len(audio)/args.sr:.2f} s, {args.sr} Hz)")


if __name__ == "__main__":
    main()
