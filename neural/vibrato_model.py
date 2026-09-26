"""Learned vibrato model: note context → vibrato parameters → pitch trajectory.

Architecture:
  - Input: note context (midi, log_dur, log_prev_gap, log_next_gap, is_first, is_last,
           art_onehot[3], instrument_onehot[3]) = 12 dims
  - Output: rate (Hz), depth (cents RMS), onset_ratio (0-1), envelope (5 pts) = 8 dims
  - Model: MLP 12 → 64 → 32 → 8

Synthesis:
  vib(τ) = depth × envelope(τ) × sin(2π × rate × dur × τ + φ)
  φ is random per render → different takes sound different.

Training data: vibrato-labeled notes from URMP violin/cello + GuitarSet.
Plain notes are included with depth=0 target (model must learn to suppress vibrato).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

FPS = 250
TRAJ_N = 48
ENV_PTS = 5


def _safe_log(x, floor=0.01):
    return float(np.log(max(abs(x), floor)))


def extract_vib_params(pitch_cents, note_dur):
    """Extract vibrato parameters from a 48-point pitch trajectory (cents residual).

    Returns dict with: rate, depth, onset_ratio, envelope (5 pts).
    """
    from scipy.signal import detrend
    n = len(pitch_cents)
    dt = detrend(pitch_cents)
    fs = n / max(note_dur, 0.1)

    freqs = np.fft.rfftfreq(n, 1 / fs)
    fft_mag = np.abs(np.fft.rfft(dt))
    band = (freqs >= 3.0) & (freqs <= 10.0)
    if band.sum() > 0:
        peak_idx = int(np.argmax(fft_mag[band]))
        rate = float(freqs[band][peak_idx])
    else:
        rate = 5.5

    depth = float(np.std(dt))

    cycle_pts = max(2, int(n / max(rate * note_dur, 1)))
    env = np.array([np.std(dt[max(0, j - cycle_pts):j + cycle_pts + 1])
                    for j in range(n)])
    max_env = env.max() + 1e-6
    onset_idx = int(np.argmax(env > 0.3 * max_env))
    onset_ratio = onset_idx / n

    env_norm = env / max_env
    env_5 = np.interp(np.linspace(0, n - 1, ENV_PTS), np.arange(n), env_norm)

    return {"rate": rate, "depth": depth, "onset_ratio": onset_ratio,
            "envelope": env_5.astype(np.float32)}


def note_to_input(midi, dur, prev_gap, next_gap, is_first, is_last, art, instrument_idx):
    """Encode a single note as a 12-dim input vector."""
    art_oh = [0.0, 0.0, 0.0]
    art_oh[min(int(art), 2)] = 1.0
    inst_oh = [0.0, 0.0, 0.0]
    inst_oh[min(int(instrument_idx), 2)] = 1.0
    return np.array([
        (midi - 60.0) / 12.0,
        _safe_log(dur),
        _safe_log(prev_gap),
        _safe_log(next_gap),
        float(is_first),
        float(is_last),
        *art_oh,
        *inst_oh,
    ], dtype=np.float32)


def params_to_target(params):
    """Pack vibrato params dict into 8-dim target vector."""
    return np.array([
        params["rate"] / 10.0,
        params["depth"] / 25.0,
        params["onset_ratio"],
        *params["envelope"],
    ], dtype=np.float32)


def target_to_params(target):
    """Unpack 8-dim target vector into vibrato params dict."""
    t = np.asarray(target, dtype=np.float32)
    return {
        "rate": float(t[0]) * 10.0,
        "depth": float(t[1]) * 25.0,
        "onset_ratio": float(np.clip(t[2], 0, 1)),
        "envelope": np.clip(t[3:8], 0, 1).astype(np.float32),
    }


def synth_vibrato(rate, depth, onset_ratio, envelope, dur, n_frames,
                  phase=None, randomness=0.3):
    """Synthesize a vibrato pitch trajectory (cents) from parameters.

    Args:
        rate: vibrato rate in Hz
        depth: vibrato depth in cents (RMS → peak ≈ depth × √2)
        onset_ratio: where vibrato starts (0-1 in normalized time)
        envelope: 5-point amplitude envelope
        dur: note duration in seconds
        n_frames: number of output frames
        phase: random phase (None → uniform random)
        randomness: 0–1, controls how much natural irregularity to add.
                    0 = pure sinusoidal, 1 = strongly irregular (human-like).

    Returns:
        pitch_cents: (n_frames,) vibrato pitch modulation in cents
    """
    if phase is None:
        phase = np.random.uniform(0, 2 * np.pi)
    tau = np.linspace(0, 1, n_frames)
    t_sec = tau * dur

    # rate jitter: slow random walk on instantaneous frequency
    # ±20% at randomness=1, scaled by randomness parameter
    if randomness > 0 and n_frames > 4:
        rng = np.random.default_rng()
        walk = np.cumsum(rng.normal(0, 0.3, n_frames))
        walk -= np.linspace(walk[0], walk[-1], n_frames)  # zero-drift
        rate_mod = 1.0 + randomness * 0.2 * walk / (np.abs(walk).max() + 1e-6)
        inst_phase = np.cumsum(2 * np.pi * rate * rate_mod / (n_frames / dur))
        osc = np.sin(inst_phase + phase)
    else:
        osc = np.sin(2 * np.pi * rate * t_sec + phase)

    # depth modulation: slow random fluctuation in amplitude
    if randomness > 0 and n_frames > 4:
        rng2 = np.random.default_rng()
        n_ctrl = max(3, int(dur * 2))
        ctrl = 1.0 + randomness * 0.3 * rng2.normal(0, 1, n_ctrl)
        ctrl = np.clip(ctrl, 0.4, 1.6)
        depth_mod = np.interp(np.linspace(0, 1, n_frames),
                              np.linspace(0, 1, n_ctrl), ctrl)
    else:
        depth_mod = 1.0

    # amplitude envelope (interpolate 5 pts)
    env = np.interp(tau, np.linspace(0, 1, ENV_PTS), envelope)

    # onset mask: ramp from 0 to 1 starting at onset_ratio
    onset_width = 0.15
    onset_mask = np.clip((tau - onset_ratio) / onset_width, 0, 1)

    peak = depth * np.sqrt(2)
    return (peak * env * onset_mask * depth_mod * osc).astype(np.float32)


class VibratoMLP(nn.Module):
    """Small MLP: 12 → 64 → 32 → 8."""

    def __init__(self, in_dim=12, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class VibratoPredictor:
    """Inference wrapper: loads checkpoint, predicts vibrato params for a note."""

    def __init__(self, ckpt_path, device="cpu"):
        self.device = device
        self.model = VibratoMLP().to(device)
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, midi, dur, prev_gap, next_gap, is_first, is_last, art,
                instrument_idx=0):
        x = note_to_input(midi, dur, prev_gap, next_gap, is_first, is_last,
                          art, instrument_idx)
        with torch.no_grad():
            out = self.model(torch.tensor(x, device=self.device).unsqueeze(0))
        return target_to_params(out[0].cpu().numpy())

    def synth(self, midi, dur, prev_gap, next_gap, is_first, is_last, art,
              instrument_idx=0, n_frames=None, phase=None, randomness=0.3):
        """Predict params + synthesize vibrato trajectory in one call."""
        params = self.predict(midi, dur, prev_gap, next_gap, is_first, is_last,
                              art, instrument_idx)
        if n_frames is None:
            n_frames = max(1, int(dur * FPS))
        return synth_vibrato(params["rate"], params["depth"], params["onset_ratio"],
                             params["envelope"], dur, n_frames, phase,
                             randomness=randomness), params
