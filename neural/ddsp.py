"""Compact DDSP neural timbre core (Engel et al. ICLR 2020).

A differentiable "harmonic oscillator + filtered noise" synthesiser plus a decoder (f0, loudness) -> control parameters, trained with a multi-scale
spectral loss (MSS) to learn real time-varying harmonic/noise structure from data. The network only generates control parameters; sound is still deterministic DSP,
fed with the articulation layer's ControlFrame (f0, loudness) at the same 250 Hz frame rate.

Depends only on torch/torchaudio; synth/runtime remain pure numpy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000  # training and synthesis sample rate; can be raised for high-pitched timbres
HOP = 64  # 250 Hz frame rate, same as ControlFrame
N_HARM = 60
N_NOISE = 65


def _exp_sigmoid(x, max_v=2.0, exp=10.0):
    """DDSP output activation: smooth and strictly positive."""
    return max_v * torch.sigmoid(x) ** np.log(exp) + 1e-7


def _upsample(x, n):
    """[B,T,C] -> [B,n,C] linear interpolation (control parameters upsampled to the sample rate)."""
    return F.interpolate(x.transpose(1, 2), size=n, mode="linear", align_corners=False).transpose(
        1, 2
    )


def _upsample_slice(x, n, s0, s1):
    """Explicit-index version of `_upsample(x, n)[:, s0:s1]`: materialises only (s1-s0) samples, for chunked synthesis.

    Reproduces the sampling rule of F.interpolate(mode="linear", align_corners=False):
    sample i maps to frame coordinate t = (i + 0.5) * T / n - 0.5, clamped at both ends.
    """
    T = x.shape[1]
    i = torch.arange(s0, s1, device=x.device, dtype=torch.float32)
    t = (i + 0.5) * (T / n) - 0.5
    t = t.clamp(0.0, float(T - 1))
    lo = t.floor().long()
    hi = (lo + 1).clamp(max=T - 1)
    w = (t - lo.float()).unsqueeze(0).unsqueeze(-1)  # [1,m,1]
    return x[:, lo] * (1.0 - w) + x[:, hi] * w  # [B,m,C]


class HarmonicSynth(nn.Module):
    def __init__(self, n_harm=N_HARM, sr=SR, hop=HOP):
        super().__init__()
        self.n_harm, self.sr, self.hop = n_harm, sr, hop
        self.register_buffer("k", torch.arange(1, n_harm + 1).float())

    CHUNK = 32768  # sample-level chunking: peak memory ≈ CHUNK×K×4B×3, independent of phrase length

    def forward(self, f0, total_amp, harm_dist):
        # f0, total_amp: [B,T,1]  harm_dist: [B,T,K]
        n = f0.shape[1] * self.hop
        dist = harm_dist / (harm_dist.sum(-1, keepdim=True) + 1e-7)
        freqs = f0 * self.k  # [B,T,K] harmonic frequencies
        dist = dist * (freqs < self.sr / 2).float()  # anti-aliasing: zero harmonics above Nyquist
        amps_f = total_amp * dist  # [B,T,K] frame level; upsampled per chunk, never materialised in full
        f0_up = _upsample(f0, n)  # [B,n,1]
        phase = torch.cumsum(2 * np.pi * f0_up / self.sr, dim=1)  # [B,n,1] phase continuous over the full length
        # amplitude and sinusoids are accumulated per sample chunk: materialising [B,n,K] in full peaks at about 1 GB for a 60 s phrase, and long phrases OOM.
        # Upsampling uses explicit indexing (_upsample_slice), numerically identical to the full version.
        out = f0.new_empty(f0.shape[0], n)
        for s0 in range(0, n, self.CHUNK):
            s1 = min(n, s0 + self.CHUNK)
            amps = _upsample_slice(amps_f, n, s0, s1)  # [B,s1-s0,K]
            out[:, s0:s1] = (amps * torch.sin(phase[:, s0:s1] * self.k)).sum(-1)
        return out


class FilteredNoise(nn.Module):
    """Time-varying spectrally shaped noise (STFT domain, differentiable): white-noise STFT × upsampled band gains -> ISTFT."""

    def __init__(self, n_bands=N_NOISE, hop=HOP, sr=SR):
        super().__init__()
        self.hop = hop
        self.n_fft = max((n_bands - 1) * 2, hop * 2)
        self.stft_hop = self.n_fft // 2

    def forward(self, noise_mags):
        B, T, N = noise_mags.shape
        n = T * self.hop
        noise = torch.rand(B, n, device=noise_mags.device) * 2 - 1
        win = torch.hann_window(self.n_fft, device=noise_mags.device)
        spec = torch.stft(
            noise, self.n_fft, self.stft_hop, window=win, return_complex=True, center=True
        )  # [B,F,frames]
        Fb, fr = spec.shape[1], spec.shape[2]
        mags = F.interpolate(
            noise_mags.transpose(1, 2), size=fr, mode="linear", align_corners=False
        )  # [B,N,frames]
        mags = F.interpolate(
            mags.transpose(1, 2), size=Fb, mode="linear", align_corners=False
        ).transpose(1, 2)
        out = torch.istft(spec * mags, self.n_fft, self.stft_hop, window=win, length=n, center=True)
        return out


class _PosEnc(nn.Module):
    """Sinusoidal positional encoding (for the transformer temporal module)."""

    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # [B,T,d]
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class DDSPDecoder(nn.Module):
    """(f0, loudness) -> (total_amp, harm_dist[K], noise_mags[N]). MLP + temporal module.

    temporal="gru" (default) | "transformer" (causal self-attention, only past frames, still real-time streamable).
    The two paths are identical apart from the temporal module (input MLP / output heads / activations), for easy A/B comparison.
    """

    def __init__(
        self,
        n_harm=N_HARM,
        n_noise=N_NOISE,
        hidden=256,
        dropout=0.0,
        temporal="gru",
        n_layers=2,
        n_heads=4,
    ):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(),
        )
        self.temporal = temporal
        if temporal == "gru":
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        elif temporal == "transformer":
            self.pos = _PosEnc(hidden)
            layer = nn.TransformerEncoderLayer(
                hidden,
                n_heads,
                dim_feedforward=hidden * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        else:
            raise ValueError(f"unknown temporal={temporal}")
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1 + n_harm + n_noise)
        self.n_harm, self.n_noise = n_harm, n_noise

    def forward(self, f0, loud):
        x = torch.cat([f0 / 500.0, loud], -1)  # coarse normalisation
        h = self.inp(x)
        if self.temporal == "gru":
            h, _ = self.gru(h)
        else:  # causal transformer (only sees ≤t)
            T = h.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
            h = self.tf(self.pos(h), mask=mask)
        h = self.drop(h)
        o = self.out(h)
        amp = _exp_sigmoid(o[..., :1])
        dist = _exp_sigmoid(o[..., 1 : 1 + self.n_harm])
        noise = _exp_sigmoid(o[..., 1 + self.n_harm :])
        return amp, dist, noise


class DDSP(nn.Module):
    def __init__(
        self,
        n_harm=N_HARM,
        n_noise=N_NOISE,
        sr=SR,
        hop=HOP,
        dropout=0.0,
        temporal="gru",
        n_layers=2,
    ):
        super().__init__()
        self.dec = DDSPDecoder(
            n_harm, n_noise, dropout=dropout, temporal=temporal, n_layers=n_layers
        )
        self.harm = HarmonicSynth(n_harm, sr, hop)
        self.noise = FilteredNoise(n_noise, hop, sr)

    def forward(self, f0, loud, noise_gain=1.0, noise_gate_db=-1.5):
        amp, dist, nmag = self.dec(f0, loud)
        noise = self.noise(nmag)
        if noise_gate_db is not None:
            gate = torch.sigmoid((loud.squeeze(-1) - noise_gate_db) * 10.0)  # [B,T]
            gate = F.interpolate(
                gate.unsqueeze(1), size=noise.shape[1], mode="linear", align_corners=False
            ).squeeze(1)
            noise = noise * gate
        y = self.harm(f0, amp, dist) + noise_gain * noise
        return y


# --------------------------------------------------------------------------- #
def multiscale_spectral_loss(a, b, scales=(2048, 1024, 512, 256, 128, 64)):
    """MSS: multi-window |STFT| L1 + log-L1 (DDSP standard reconstruction loss)."""
    loss = 0.0
    for s in scales:
        win = torch.hann_window(s, device=a.device)
        A = torch.stft(a, s, s // 4, window=win, return_complex=True).abs()
        B = torch.stft(b, s, s // 4, window=win, return_complex=True).abs()
        loss = (
            loss + (A - B).abs().mean() + (torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().mean()
        )
    return loss


def extract_features(y, sr_in, sr=SR, hop=HOP):
    """Audio -> (f0[1,T,1], loudness[1,T,1], audio_resampled[1,n]), T=n//hop."""
    import torchaudio

    from analysis.harmonic import track_f0

    y = np.asarray(y, dtype=np.float32)
    f0_raw = track_f0(y, sr=sr_in)[0]  # PESTO, about 10 ms per frame
    yt = torch.tensor(y)[None]
    if sr_in != sr:
        yt = torchaudio.functional.resample(yt, sr_in, sr)
    n = (yt.shape[1] // hop) * hop
    yt = yt[:, :n]
    T = n // hop
    # resample f0 / loudness to T frames
    f0v = np.nan_to_num(np.asarray(f0_raw, np.float32))
    f0 = torch.tensor(np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(f0v)), f0v))[
        None, :, None
    ].float()
    frames = yt[0, : T * hop].reshape(T, hop)
    loud = (frames**2).mean(1).clamp_min(1e-7).log()[None, :, None].float()  # log-RMS
    loud = (loud - loud.mean()) / (loud.std() + 1e-5)
    return f0, loud, yt
