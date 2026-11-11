"""阶段1 验收脚本 —— FR-2 前半。

对每个输入声音跑 分析 -> 重合成 全链路，产出：
  * A/B wav（原音 vs 重合成）         -> assets/demos/<name>_orig.wav / _resynth.wav
  * H 谐波热图 PNG                     -> assets/demos/<name>_Hheatmap.png
  * 原音 vs 重合成 频谱对比 PNG         -> assets/demos/<name>_spectrum.png
  * 控制台指标：f0、逐谐波 dB 误差、多尺度谱损失、混叠/非谐波能量比

用法::

    python -m eval.stage1_accept                 # 用内置合成长音（人声/钢琴/酒杯）
    python -m eval.stage1_accept a.wav b.wav     # 用真实录音（盲听用）
    python -m eval.stage1_accept --record 4       # 现场录 4s 再分析

“盲听过关”的硬验收需要真实人声 + 钢琴长音：把两个 wav 拖进来即可。
内置合成信号用于无录音时的链路自检与 CI。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 允许 `python eval/stage1_accept.py` 与 `python -m eval.stage1_accept` 两种入口
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.harmonic import f0_scalar, harmonics, track_f0  # noqa: E402
from analysis.io import normalize  # noqa: E402
from synth.additive import resynth  # noqa: E402

SR = 44100
DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "demos"


# --------------------------------------------------------------------------- #
# 内置合成长音（无录音时用）
# --------------------------------------------------------------------------- #
def synth_voice(dur=2.0, sr=SR):
    """类人声长音：双共振峰包络 + 轻微颤音。"""
    t = np.arange(int(sr * dur)) / sr
    f0 = 174.6  # F3
    vib = 1.0 + 0.003 * np.sin(2 * np.pi * 5.0 * t)  # 5Hz 颤音
    phase = np.cumsum(2 * np.pi * f0 * vib / sr)
    y = np.zeros_like(t)
    for k in range(1, 41):
        fk = k * f0
        if fk >= sr / 2:
            break
        # 两个共振峰 ~700Hz / ~1800Hz
        form = np.exp(-(((fk - 700) / 350) ** 2)) + 0.6 * np.exp(-(((fk - 1800) / 600) ** 2)) + 0.03
        y += form * np.cos(k * phase)
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (dur - t) / 0.1)
    return normalize((y * env).astype(np.float32)), "voice"


def synth_piano(dur=2.5, sr=SR):
    """类钢琴长音：1/k 滚降 + 逐谐波指数衰减 + 快起音。"""
    t = np.arange(int(sr * dur)) / sr
    f0 = 261.63  # C4
    y = np.zeros_like(t)
    for k in range(1, 31):
        fk = k * f0
        if fk >= sr / 2:
            break
        decay = np.exp(-t * (1.0 + 0.35 * k))  # 高次谐波衰减更快
        y += (1.0 / k) * decay * np.sin(2 * np.pi * fk * t + 0.1 * k)
    atk = np.minimum(1.0, t / 0.005)
    return normalize((y * atk).astype(np.float32)), "piano"


def synth_wineglass(dur=2.0, sr=SR):
    """准谐波酒杯：部分音相对 k*f0 轻微 stretch（测 partial_freqs 路径）。"""
    t = np.arange(int(sr * dur)) / sr
    partials = [660.0, 1335.0, 2025.0, 2740.0]  # 略高于 k*660
    amps = [1.0, 0.45, 0.2, 0.1]
    y = sum(a * np.sin(2 * np.pi * fp * t) for fp, a in zip(partials, amps))
    env = np.exp(-t * 1.2) * np.minimum(1.0, t / 0.003)
    return normalize((y * env).astype(np.float32)), "wineglass"


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def multiscale_spectral_loss(y, y_re, sr=SR, ffts=(512, 1024, 2048, 4096)):
    """多尺度对数幅度谱 L1（MSS，FR-13 的雏形）。"""
    n = min(len(y), len(y_re))
    y, y_re = y[:n], y_re[:n]
    losses = []
    for nfft in ffts:
        hop = nfft // 4
        S1 = _logmag_stft(y, nfft, hop)
        S2 = _logmag_stft(y_re, nfft, hop)
        m = min(S1.shape[1], S2.shape[1])
        losses.append(float(np.mean(np.abs(S1[:, :m] - S2[:, :m]))))
    return float(np.mean(losses))


def _logmag_stft(y, nfft, hop):
    win = np.hanning(nfft)
    frames = []
    for i in range(0, len(y) - nfft, hop):
        frames.append(np.abs(np.fft.rfft(y[i : i + nfft] * win)))
    if not frames:
        frames = [np.abs(np.fft.rfft(np.zeros(nfft)))]
    S = np.array(frames).T
    return np.log(S + 1e-5)


def inharmonic_energy_ratio(y, f0_ref, sr=SR, tol_hz=25.0):
    """非谐波（含混叠折返）能量占比：远离任意 k*f0 的谱能量 / 总能量。"""
    y = y[len(y) // 8 : -len(y) // 8] if len(y) > 16 else y
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    if not (np.isfinite(f0_ref) and f0_ref > 0):
        return float("nan")
    harm_mask = np.zeros_like(freqs, dtype=bool)
    kmax = int((sr / 2) / f0_ref)
    for k in range(1, kmax + 1):
        harm_mask |= np.abs(freqs - k * f0_ref) <= tol_hz
    harm_mask |= freqs < 40.0  # 忽略极低频 DC 区
    total = spec.sum() + 1e-12
    return float(spec[~harm_mask].sum() / total)


def per_harmonic_db_error(y_re, f0_ref, amps_ref=None, sr=SR, K=20):
    """对重合成再分析得逐谐波幅度（dB），可与参考幅度比对。"""
    H, _, _ = harmonics(y_re, f0_ref, sr=sr, n_harmonics=K)
    T = H.shape[1]
    est = np.median(H[:, int(T * 0.3) : int(T * 0.7)], axis=1)
    return est


# --------------------------------------------------------------------------- #
# 单个声音的处理 + 出图
# --------------------------------------------------------------------------- #
def process(y, sr, name):
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    y = normalize(np.asarray(y, dtype=np.float32))

    f0, conf, ftimes = track_f0(y, sr=sr)
    f0_ref = f0_scalar(f0)
    measure = name == "wineglass"  # 准谐波声源走实测部分音路径
    H, htimes, info = harmonics(y, f0, sr=sr, f0_times=ftimes, measure_partials=measure)
    pf = info["partial_freqs"]
    y_re = resynth(H, info["f0_per_frame"], sr=sr, partial_freqs=pf, f0_ref=f0_ref, length=len(y))
    y_re = normalize(y_re)

    # --- 落盘 A/B wav ---
    orig_p = DEMO_DIR / f"{name}_orig.wav"
    re_p = DEMO_DIR / f"{name}_resynth.wav"
    sf.write(orig_p, y, sr)
    sf.write(re_p, y_re, sr)

    # --- H 热图 ---
    _save_heatmap(H, htimes, name, f0_ref)
    # --- 频谱对比 ---
    _save_spectrum(y, y_re, sr, name, f0_ref)

    # --- 指标 ---
    mss = multiscale_spectral_loss(y, y_re, sr)
    alias_orig = inharmonic_energy_ratio(y, f0_ref, sr)
    alias_re = inharmonic_energy_ratio(y_re, f0_ref, sr)

    print(f"\n=== {name} ===")
    print(f"  f0_ref           : {f0_ref:8.2f} Hz")
    print(f"  有效谐波数(<Nyq) : {int(np.sum(np.any(H > 1e-4, axis=1)))}")
    print(f"  多尺度谱损失 MSS : {mss:8.4f}  (越低越像)")
    print(f"  非谐波能量比 原音: {alias_orig * 100:6.3f}%")
    print(f"  非谐波能量比 重合: {alias_re * 100:6.3f}%  (应接近原音，无混叠折返)")
    if measure and pf is not None:
        print(f"  实测部分音(Hz)   : {np.round(pf[:4], 1).tolist()}")
    print(f"  A/B  -> {orig_p.name} | {re_p.name}")
    return {"name": name, "f0": f0_ref, "mss": mss, "alias_orig": alias_orig, "alias_re": alias_re}


def _save_heatmap(H, times, name, f0_ref):
    fig, ax = plt.subplots(figsize=(8, 4))
    Hdb = 20 * np.log10(H + 1e-6)
    im = ax.imshow(
        Hdb,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[times[0], times[-1], 1, H.shape[0]],
        vmin=-80,
        vmax=0,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Harmonic index k")
    ax.set_title(f"{name}  harmonic amplitude envelope H[k,t]  (f0 ~ {f0_ref:.1f} Hz)")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(DEMO_DIR / f"{name}_Hheatmap.png", dpi=110)
    plt.close(fig)


def _save_spectrum(y, y_re, sr, name, f0_ref):
    def mag_db(sig):
        sig = sig[len(sig) // 8 : -len(sig) // 8]
        S = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
        return np.fft.rfftfreq(len(sig), 1 / sr), 20 * np.log10(S / (S.max() + 1e-12) + 1e-12)

    f1, s1 = mag_db(y)
    f2, s2 = mag_db(y_re)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(f1, s1, lw=0.7, label="Original", alpha=0.8)
    ax.plot(f2, s2, lw=0.7, label="Resynth", alpha=0.8)
    ax.set_xlim(0, min(sr / 2, 8000))
    ax.set_ylim(-90, 3)
    ax.axvline(sr / 2, color="r", ls="--", lw=0.8, label="Nyquist")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title(f"{name}  Original vs Resynth spectrum (no aliasing fold-back)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DEMO_DIR / f"{name}_spectrum.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quill 阶段1 验收")
    ap.add_argument("wavs", nargs="*", help="真实录音 wav（盲听用）")
    ap.add_argument("--record", type=float, default=0.0, help="现场录音秒数（替代文件输入）")
    args = ap.parse_args(argv)

    jobs = []
    if args.record > 0:
        from analysis.io import record

        print(f"录音 {args.record}s …（首次会弹麦克风权限）")
        y = record(args.record, sr=SR)
        jobs.append((y, "recorded"))
    for w in args.wavs:
        from analysis.io import load

        y, _ = load(w, sr=SR)
        jobs.append((y, Path(w).stem))
    if not jobs:
        print("未提供录音，使用内置合成长音（人声/钢琴/酒杯）做链路自检。")
        for fn in (synth_voice, synth_piano, synth_wineglass):
            y, nm = fn()
            jobs.append((y, nm))

    results = [process(y, SR, nm) for y, nm in jobs]

    print("\n---------------- 验收汇总 ----------------")
    ok = True
    for r in results:
        # 链路健康判据：重合成的非谐波能量未显著高于原音（无混叠爆发）
        drift = r["alias_re"] - r["alias_orig"]
        flag = "OK " if drift < 0.05 else "WARN"
        if drift >= 0.05:
            ok = False
        print(
            f"  [{flag}] {r['name']:10s} f0={r['f0']:7.1f}Hz "
            f"MSS={r['mss']:.3f} 非谐波Δ={drift * 100:+.2f}%"
        )
    print(f"产物目录: {DEMO_DIR}")
    print("盲听: 用播放器对比 *_orig.wav 与 *_resynth.wav；目检 *_Hheatmap.png 与 *_spectrum.png。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
