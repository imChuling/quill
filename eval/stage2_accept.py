"""阶段2 验收脚本 —— FR-2 完成 + FR-3。

出口：三例素材（人声 / 吉他单音 / 酒杯）±12 半音可听，共振峰保持。

对每例：
  audio -> analyze_timbre -> TimbreSnapshot(.npz) -> render(-12 / 0 / +12)
  并渲染 +12 的"按序号搬运"错误版（花栗鼠）作对照存档（报告素材）。

产物：
  assets/snapshots/<name>.npz                     音色快照
  assets/demos/<name>_orig.wav                    原音
  assets/demos/<name>_re{0,up12,down12}.wav       共振峰保持重合成（±12）
  assets/demos/<name>_up12_index.wav              按序号搬运错误版（对照）
  assets/demos/<name>_fr3.png                     共振峰保持 vs 搬运 频谱对比

用法：
  python -m eval.stage2_accept                 # 内置合成素材
  python -m eval.stage2_accept guitar.wav      # 真实单音（默认按谐波声源）
  python -m eval.stage2_accept --quasi bowl.wav  # 准谐波声源（酒杯/碗）
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.io import normalize  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from eval.stage1_accept import synth_wineglass  # noqa: E402
from snapshot import save_snapshot  # noqa: E402
from synth.render import render  # noqa: E402

SR = 44100
ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "assets" / "demos"
SNAP_DIR = ROOT / "assets" / "snapshots"


def synth_voice_steady(dur=2.0, sr=SR):
    """类人声**稳态**长音（双共振峰，无颤音）。

    按素材采集协议（A5.5）音色快照只从孤立持续音提取——稳态 f0 是干净残差/
    噪声分解的前提（颤音会在分析窗内令谐波频率漂移、污染残差，属已知限制）。
    """
    t = np.arange(int(sr * dur)) / sr
    f0 = 174.6  # F3
    y = np.zeros_like(t)
    for k in range(1, 41):
        fk = k * f0
        if fk >= sr / 2:
            break
        form = np.exp(-(((fk - 700) / 350) ** 2)) + 0.6 * np.exp(-(((fk - 1800) / 600) ** 2)) + 0.03
        y += form * np.cos(2 * np.pi * fk * t)
    env = np.minimum(1.0, t / 0.05) * np.minimum(1.0, (dur - t) / 0.1)
    return normalize((y * env).astype(np.float32)), "voice"


def synth_guitar(dur=2.0, sr=SR):
    """类吉他单音：谐波指数衰减（高次更快）+ 起音拨弦噪声。"""
    t = np.arange(int(sr * dur)) / sr
    f0 = 146.83  # D3
    y = np.zeros_like(t)
    for k in range(1, 36):
        fk = k * f0
        if fk >= sr / 2:
            break
        tau = 1.2 / (1.0 + 0.5 * k)
        y += (1.0 / k) * np.exp(-t / tau) * np.sin(2 * np.pi * fk * t + 0.2 * k)
    # 拨弦瞬态：~6ms 宽带噪声爆发
    rng = np.random.default_rng(0)
    pluck = rng.standard_normal(len(t)) * np.exp(-t / 0.006)
    y = y + 0.3 * pluck
    return normalize(y.astype(np.float32)), "guitar"


def _spectral_envelope_peak(y, sr, lo=200.0, hi=3000.0, smooth_hz=300.0):
    """粗略谱包络峰频率（Hz）：**线性功率**谱平滑后在 [lo,hi] 取峰。

    对线性功率（而非 log 幅度）平滑——log 域会被谐波间的深谷与密度主导，
    给出错误的"包络峰"；线性功率平滑直接跟随谐波能量分布，定位共振峰可靠。
    """
    seg = y[len(y) // 8 : -len(y) // 8] if len(y) > 16 else y
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2  # 功率谱
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    df = f[1] - f[0]
    win = max(1, int(smooth_hz / df))
    kern = np.ones(win) / win
    sm = np.convolve(S, kern, mode="same")
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return float("nan")
    idx = np.where(band)[0]
    return float(f[idx[np.argmax(sm[idx])]])


def _mag_db(y, sr):
    seg = y[len(y) // 8 : -len(y) // 8] if len(y) > 16 else y
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    return f, 20 * np.log10(S / (S.max() + 1e-12) + 1e-12)


def _smoothed_env_db(y, sr, sigma_hz=250.0):
    """线性功率谱 Gaussian 平滑 -> dB 谱包络曲线（直观展示共振峰位置）。"""
    seg = y[len(y) // 8 : -len(y) // 8] if len(y) > 16 else y
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    df = f[1] - f[0]
    sigma_bins = max(1.0, sigma_hz / df)
    half = int(4 * sigma_bins)
    x = np.arange(-half, half + 1)
    kern = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kern /= kern.sum()
    sm = np.convolve(S, kern, mode="same")
    return f, 10 * np.log10(sm / (sm.max() + 1e-12) + 1e-12)


def _save_fr3_plot(name, orig, re0, up_form, up_idx, sr, formant_hz):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # 细线：原始线谱（淡）；粗线：平滑谱包络（讲故事）
    f, s = _mag_db(up_form, sr)
    ax.plot(f, s, lw=0.5, alpha=0.25, color="C1")
    f, s = _mag_db(up_idx, sr)
    ax.plot(f, s, lw=0.5, alpha=0.25, color="C2")
    for sig, lab, col in (
        (orig, "Original envelope (f0_ref)", "C0"),
        (up_form, "+12 formant-preserving env", "C1"),
        (up_idx, "+12 by-index (chipmunk) env", "C2"),
    ):
        f, e = _smoothed_env_db(sig, sr)
        ax.plot(f, e, lw=2.0, color=col, label=lab)
    if np.isfinite(formant_hz):
        ax.axvline(formant_hz, color="k", ls=":", lw=1.2, label=f"orig formant ~{formant_hz:.0f}Hz")
    ax.set_xlim(0, 5000)
    ax.set_ylim(-60, 3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title(f"{name}  FR-3: formant preserved vs shifted (+12 semitones)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DEMO_DIR / f"{name}_fr3.png", dpi=110)
    plt.close(fig)


def process(y, sr, name, quasi=False):
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    y = normalize(np.asarray(y, dtype=np.float32))

    snap = analyze_timbre(y, sr=sr, name=name, quasi_harmonic=quasi)
    snap_path = save_snapshot(SNAP_DIR / f"{name}.npz", snap)

    f0r = snap["f0_ref"]
    length = len(y)
    # 共振峰保持渲染：-12 / 0 / +12 半音
    re0 = render(snap, f0r, length=length, mode="formant")
    up = render(snap, f0r * 2.0, length=length, mode="formant")  # +12
    down = render(snap, f0r / 2.0, length=length, mode="formant")  # -12
    up_idx = render(snap, f0r * 2.0, length=length, mode="index")  # 错误对照

    outs = {
        "orig": y,
        "re0": re0,
        "up12": up,
        "down12": down,
        "up12_index": up_idx,
    }
    for tag, sig in outs.items():
        sf.write(DEMO_DIR / f"{name}_{tag}.wav", normalize(sig), sr)

    # 共振峰保持度量：谱包络峰（应在 ±12 下基本不动）
    formant_orig = _spectral_envelope_peak(y, sr)
    formant_up_form = _spectral_envelope_peak(up, sr)
    formant_up_idx = _spectral_envelope_peak(up_idx, sr)
    _save_fr3_plot(name, y, re0, up, up_idx, sr, formant_orig)

    # 重合成保真（同基频）：谱包络峰漂移
    drift_re0 = abs(_spectral_envelope_peak(re0, sr) - formant_orig)

    print(f"\n=== {name} ===")
    print(f"  f0_ref               : {f0r:8.2f} Hz")
    print(f"  vibrato              : {snap['vibrato']}")
    print(
        f"  partial_freqs        : "
        f"{'实测前4=' + str(np.round(snap['partial_freqs'][:4], 1).tolist()) if snap['partial_freqs'] is not None else 'None (严格谐波)'}"
    )
    print(f"  谱包络峰 原音         : {formant_orig:8.1f} Hz")
    print(
        f"  谱包络峰 +12 共振峰保持: {formant_up_form:8.1f} Hz  "
        f"(漂移 {abs(formant_up_form - formant_orig):.0f}Hz，应小)"
    )
    print(
        f"  谱包络峰 +12 按序号   : {formant_up_idx:8.1f} Hz  "
        f"(漂移 {abs(formant_up_idx - formant_orig):.0f}Hz，应≈翻倍)"
    )
    print(f"  同基频重合成峰漂移     : {drift_re0:8.1f} Hz")
    print(
        f"  快照 -> {snap_path.name}  ({snap['H_env'].shape[1]} 帧 H, "
        f"{snap['N_env'].shape[1]} 帧 N)"
    )
    return {
        "name": name,
        "f0": f0r,
        "formant_orig": formant_orig,
        "drift_re0": drift_re0,
        "drift_form": abs(formant_up_form - formant_orig),
        "drift_idx": abs(formant_up_idx - formant_orig),
        "clean": bool(np.all(np.isfinite(up)) and np.max(np.abs(up)) < 10.0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Quill 阶段2 验收（FR-2 完成 + FR-3）")
    ap.add_argument("wavs", nargs="*", help="真实单音 wav")
    ap.add_argument(
        "--quasi", action="store_true", help="准谐波声源（酒杯/碗），放宽峰搜索窗并测 partial_freqs"
    )
    args = ap.parse_args(argv)

    jobs = []
    if args.wavs:
        from analysis.io import load

        for w in args.wavs:
            y, _ = load(w, sr=SR)
            jobs.append((y, Path(w).stem, args.quasi))
    else:
        print("未提供录音，使用内置合成单音（人声/吉他/酒杯）。")
        jobs.append((synth_voice_steady()[0], "voice", False))
        jobs.append((synth_guitar()[0], "guitar", False))
        jobs.append((synth_wineglass()[0], "wineglass", True))

    results = [process(y, SR, nm, q) for (y, nm, q) in jobs]

    print("\n---------------- 阶段2 验收汇总 ----------------")
    print(
        "判据：① 同基频重合成保真(峰漂移<150Hz) ② 渲染干净(有限有界) ③ 共振峰保持不劣于按序号搬运"
    )
    print(
        "（注：吉他/酒杯为单调滚降谱、无显式共振峰，两模式自然接近；"
        "人声有 ~700Hz 共振峰，FR-3 优势显著）"
    )
    ok = True
    for r in results:
        faithful = r["drift_re0"] < 150.0
        not_worse = r["drift_form"] <= r["drift_idx"] + 30.0
        ok_row = faithful and not_worse and r["clean"]
        ok = ok and ok_row
        flag = "OK " if ok_row else "WARN"
        note = "  <- 共振峰保持显著" if r["drift_form"] < 0.7 * r["drift_idx"] else ""
        print(
            f"  [{flag}] {r['name']:10s} f0={r['f0']:7.1f}Hz  "
            f"同基频漂移={r['drift_re0']:4.0f}Hz  "
            f"+12: formant={r['drift_form']:4.0f}Hz / index={r['drift_idx']:4.0f}Hz{note}"
        )
    print(f"产物: {DEMO_DIR}  |  快照: {SNAP_DIR}")
    print(
        "盲听 ±12 半音：*_down12 / *_re0 / *_up12 三档音色一致（无花栗鼠）；"
        "对照 *_up12_index 听花栗鼠效应。"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
