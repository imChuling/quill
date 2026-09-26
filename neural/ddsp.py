"""L2 神经音色核心 · 紧凑 DDSP（Engel et al. ICLR 2020）—— FR-10。

可微「谐波振荡器 + 滤波噪声」合成器 + 解码器(f0,loudness→控制参数)，用多尺度谱损失
(MSS)从数据学真实时变谐波/噪声结构——补加法快照抹平的音色身份(吉他/贝斯 slap 等)。
关键性质(铁律/L2 哲学)：**神经只生成控制参数,发声仍是确定性 DSP**→可实时(BRAVE 因果化后)、
且喂它的就是演奏法层的 ControlFrame(f0,loudness)。本地 MPS 可 smoke-test,Modal 放大训练。

只依赖 torch/torchaudio（neural/ 允许神经依赖；synth/runtime 仍纯 numpy）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000          # DDSP 标准训练采样率（贝斯/多数音色够；高音可升）
HOP = 64            # 帧率 250Hz = ControlFrame 同频
N_HARM = 60
N_NOISE = 65


def _exp_sigmoid(x, max_v=2.0, exp=10.0):
    """DDSP 输出激活：正、平滑、>0。"""
    return max_v * torch.sigmoid(x) ** np.log(exp) + 1e-7


def _upsample(x, n):
    """[B,T,C] -> [B,n,C] 线性插值（控制参数升到样本率）。"""
    return F.interpolate(x.transpose(1, 2), size=n, mode="linear",
                         align_corners=False).transpose(1, 2)


class HarmonicSynth(nn.Module):
    def __init__(self, n_harm=N_HARM, sr=SR, hop=HOP):
        super().__init__()
        self.n_harm, self.sr, self.hop = n_harm, sr, hop
        self.register_buffer("k", torch.arange(1, n_harm + 1).float())

    def forward(self, f0, total_amp, harm_dist):
        # f0,total_amp:[B,T,1]  harm_dist:[B,T,K]
        n = f0.shape[1] * self.hop
        dist = harm_dist / (harm_dist.sum(-1, keepdim=True) + 1e-7)
        freqs = f0 * self.k                                   # [B,T,K] 各谐波频率
        dist = dist * (freqs < self.sr / 2).float()           # 抗混叠：越 Nyquist 置零
        amps = _upsample(total_amp * dist, n)                 # [B,n,K]
        f0_up = _upsample(f0, n)                              # [B,n,1]
        phase = torch.cumsum(2 * np.pi * f0_up / self.sr, dim=1)   # [B,n,1]
        audio = (amps * torch.sin(phase * self.k)).sum(-1)        # [B,n]
        return audio


class FilteredNoise(nn.Module):
    """时变谱整形噪声(STFT 域,可微)：白噪 STFT × 升采样的带增益 → ISTFT。"""
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
        spec = torch.stft(noise, self.n_fft, self.stft_hop, window=win,
                          return_complex=True, center=True)      # [B,F,frames]
        Fb, fr = spec.shape[1], spec.shape[2]
        mags = F.interpolate(noise_mags.transpose(1, 2), size=fr,
                             mode="linear", align_corners=False)  # [B,N,frames]
        mags = F.interpolate(mags.transpose(1, 2), size=Fb,
                             mode="linear", align_corners=False).transpose(1, 2)
        out = torch.istft(spec * mags, self.n_fft, self.stft_hop, window=win,
                          length=n, center=True)
        return out


class _PosEnc(nn.Module):
    """正弦位置编码（transformer 时序模块用）。"""
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                                      # [B,T,d]
        return x + self.pe[:x.shape[1]].unsqueeze(0)


class DDSPDecoder(nn.Module):
    """(f0, loudness) -> (total_amp, harm_dist[K], noise_mags[N])。MLP + 时序模块。

    temporal="gru"（默认，原版）| "transformer"（**因果**自注意力，只看历史帧 → 仍可实时流式）。
    A/B 用：除时序模块外两路完全相同（输入 MLP / 输出头 / 激活），公平比较"注意力是否更好"。
    """
    def __init__(self, n_harm=N_HARM, n_noise=N_NOISE, hidden=256, dropout=0.0,
                 temporal="gru", n_layers=2, n_heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU())
        self.temporal = temporal
        if temporal == "gru":
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        elif temporal == "transformer":
            self.pos = _PosEnc(hidden)
            layer = nn.TransformerEncoderLayer(hidden, n_heads, dim_feedforward=hidden * 2,
                                               dropout=dropout, batch_first=True,
                                               activation="gelu", norm_first=True)
            self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        else:
            raise ValueError(f"未知 temporal={temporal}")
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1 + n_harm + n_noise)
        self.n_harm, self.n_noise = n_harm, n_noise

    def forward(self, f0, loud):
        x = torch.cat([f0 / 500.0, loud], -1)                 # 粗归一
        h = self.inp(x)
        if self.temporal == "gru":
            h, _ = self.gru(h)
        else:                                                 # 因果 transformer（只看 ≤t）
            T = h.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
            h = self.tf(self.pos(h), mask=mask)
        h = self.drop(h)
        o = self.out(h)
        amp = _exp_sigmoid(o[..., :1])
        dist = _exp_sigmoid(o[..., 1:1 + self.n_harm])
        noise = _exp_sigmoid(o[..., 1 + self.n_harm:])
        return amp, dist, noise


class DDSP(nn.Module):
    def __init__(self, n_harm=N_HARM, n_noise=N_NOISE, sr=SR, hop=HOP, dropout=0.0,
                 temporal="gru", n_layers=2):
        super().__init__()
        self.dec = DDSPDecoder(n_harm, n_noise, dropout=dropout, temporal=temporal,
                               n_layers=n_layers)
        self.harm = HarmonicSynth(n_harm, sr, hop)
        self.noise = FilteredNoise(n_noise, hop, sr)

    def forward(self, f0, loud, noise_gain=1.0, noise_gate_db=-1.5):
        amp, dist, nmag = self.dec(f0, loud)
        noise = self.noise(nmag)
        if noise_gate_db is not None:
            gate = torch.sigmoid((loud.squeeze(-1) - noise_gate_db) * 10.0)  # [B,T]
            gate = F.interpolate(gate.unsqueeze(1), size=noise.shape[1],
                                 mode="linear", align_corners=False).squeeze(1)
            noise = noise * gate
        y = self.harm(f0, amp, dist) + noise_gain * noise
        return y


# --------------------------------------------------------------------------- #
def multiscale_spectral_loss(a, b, scales=(2048, 1024, 512, 256, 128, 64)):
    """MSS：多窗 |STFT| L1 + log-L1（DDSP 标准重建损失）。"""
    loss = 0.0
    for s in scales:
        win = torch.hann_window(s, device=a.device)
        A = torch.stft(a, s, s // 4, window=win, return_complex=True).abs()
        B = torch.stft(b, s, s // 4, window=win, return_complex=True).abs()
        loss = loss + (A - B).abs().mean() + (torch.log(A + 1e-5) - torch.log(B + 1e-5)).abs().mean()
    return loss


def extract_features(y, sr_in, sr=SR, hop=HOP):
    """音频 -> (f0[1,T,1], loudness[1,T,1], audio_resampled[1,n])，T=n//hop。"""
    import torchaudio
    from analysis.harmonic import track_f0
    y = np.asarray(y, dtype=np.float32)
    f0_raw = track_f0(y, sr=sr_in)[0]                         # PESTO，~10ms 帧
    yt = torch.tensor(y)[None]
    if sr_in != sr:
        yt = torchaudio.functional.resample(yt, sr_in, sr)
    n = (yt.shape[1] // hop) * hop
    yt = yt[:, :n]
    T = n // hop
    # f0 / loudness 重采样到 T 帧
    f0v = np.nan_to_num(np.asarray(f0_raw, np.float32))
    f0 = torch.tensor(np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(f0v)), f0v))[None, :, None].float()
    frames = yt[0, :T * hop].reshape(T, hop)
    loud = (frames ** 2).mean(1).clamp_min(1e-7).log()[None, :, None].float()  # log-RMS
    loud = (loud - loud.mean()) / (loud.std() + 1e-5)         # 标准化
    return f0, loud, yt


def overfit_one(y, sr_in, steps=400, lr=1e-3, device=None):
    # lr 勿超 1e-3:诊断 D1 实测 3e-3 在合成信号上都卡在 loss~10 不降
    """smoke-test：把 DDSP 过拟合到单个音 → 返回 (model, losses, recon)。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    f0, loud, target = extract_features(y, sr_in)
    f0, loud, target = f0.to(device), loud.to(device), target.to(device)
    model = DDSP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for i in range(steps):
        opt.zero_grad()
        yhat = model(f0, loud)
        m = min(yhat.shape[1], target.shape[1])
        loss = multiscale_spectral_loss(yhat[:, :m], target[:, :m])
        loss.backward(); opt.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        recon = model(f0, loud).cpu().numpy()[0]
    return model, losses, recon
