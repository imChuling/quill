"""结构性差异测试 —— 回应"和采样器没区别"的诚实检验(2026-08-01)。

只测采样器**结构上做不到**的事(信息不在样本里,脚本补不出来),不测"没做"的事:

T1 · 单素材和弦:同一段录音弹 6 音和弦。
     采样器 = 同一样本移调 6 份相加 → 各声部噪声层**完全相关** → 梳状滤波/相位抵消;
     Quill  = 6 个声部各自色散初相 + 独立噪声读指针 → 像合奏。
     指标:声部间噪声互相关;和弦谱的梳状凹槽深度。

T2 · 短录音长延音:2 秒素材要弹 15 秒长音。
     采样器 = 必须循环(交叉淡化)→ 周期性可听;Quill = 重合成,无循环点。
     指标:包络/谱的自相关周期峰(循环痕迹)。

T3 · 行为迁移:采样器无此概念(对照见 stimuli_2x2)。

输出:客观指标 + 音频对(assets/ear_tests/structural/)。
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from runtime.engine import Engine  # noqa: E402
from snapshot import load_snapshot  # noqa: E402
from synth.instrument import build_instrument  # noqa: E402
from synth.render import render  # noqa: E402

SR = 44100
CHORD = [55, 59, 62, 67, 71, 74]  # 6 音,含八度与五度(最易暴露相位问题)


def rms_norm(y, r=0.08):
    v = np.sqrt((np.asarray(y, np.float64) ** 2).mean())
    return (np.asarray(y, np.float64) * (r / max(v, 1e-9))).astype(np.float32)


def sampler_chord(snap, dur=3.0):
    """采样器和弦:同一样本移调 N 份相加(这就是采样器物理上做的事)。"""
    import librosa

    f0_ref = float(snap["f0_ref"])
    base = render(snap, f0_ref, length=int(dur * SR), mode="formant", add_attack=True)
    ref_midi = 69 + 12 * np.log2(f0_ref / 440.0)
    out = np.zeros(int(dur * SR), np.float64)
    voices = []
    for m in CHORD:
        v = librosa.effects.pitch_shift(base.astype(np.float32), sr=SR, n_steps=float(m - ref_midi))
        v = v[: len(out)]
        voices.append(v)
        out[: len(v)] += v
    return out, voices


def quill_chord(snap, dur=3.0):
    """Quill 和弦:引擎 6 声部(色散初相 + 各自噪声读指针)。"""
    inst = build_instrument(snap, noise_level=0.02)
    eng = Engine(inst, eval_mode=False)  # eval_mode=False → 噪声起点随机化
    eng.reset()
    evs = []
    for m in CHORD:
        evs.append((0.02, "on", m, 100, 0.0, -1))
        evs.append((dur - 0.3, "off", m, 0, 0.0, -1))
    return eng.render_offline(evs, dur).astype(np.float64)


def comb_depth(y):
    """梳状凹槽深度:谱包络去趋势后的谷/峰起伏(dB)。相位抵消越重越大。"""
    S = np.abs(np.fft.rfft(y[: 1 << 16] * np.hanning(min(len(y), 1 << 16))))
    f = np.fft.rfftfreq(min(len(y), 1 << 16), 1 / SR)
    m = (f > 200) & (f < 6000)
    lg = 20 * np.log10(np.maximum(S[m], 1e-9))
    k = 101  # 平滑趋势
    trend = np.convolve(lg, np.ones(k) / k, mode="same")
    d = lg[k:-k] - trend[k:-k]
    return float(np.percentile(d, 95) - np.percentile(d, 5))


def loop_periodicity(y, min_lag_s=0.5):
    """循环痕迹:包络自相关在 >0.5s 滞后处的最大峰(0=无周期,越大越像在循环)。"""
    hop = 441
    n = len(y) // hop * hop
    e = np.sqrt((y[:n].reshape(-1, hop) ** 2).mean(1) + 1e-12)
    e = e[int(0.5 * SR / hop) :]  # 掐掉起音
    e = e - e.mean()
    if len(e) < 40:
        return 0.0
    ac = np.correlate(e, e, mode="full")[len(e) - 1 :]
    ac /= max(ac[0], 1e-12)
    lag0 = int(min_lag_s * SR / hop)
    return float(ac[lag0:].max()) if len(ac) > lag0 + 2 else 0.0


def sampler_sustain(snap, src_len=2.0, out_len=15.0, xfade=0.25):
    """采样器长延音:2 秒素材循环(交叉淡化)到 15 秒——采样器唯一能做的事。"""
    f0_ref = float(snap["f0_ref"])
    src = render(snap, f0_ref, length=int(src_len * SR), mode="formant", add_attack=True).astype(
        np.float64
    )
    xf = int(xfade * SR)
    loop = src[int(0.3 * SR) :]  # 循环区(跳过起音)
    out = list(src[: int(0.3 * SR)])
    body = loop.copy()
    while len(out) < int(out_len * SR):
        if len(out) > xf:  # 等功率交叉淡化(标准采样器循环)
            tail = np.array(out[-xf:])
            head = body[:xf]
            w = np.linspace(0, 1, xf)
            out[-xf:] = list(tail * np.cos(w * np.pi / 2) + head * np.sin(w * np.pi / 2))
            out += list(body[xf:])
        else:
            out += list(body)
    return np.array(out[: int(out_len * SR)])


def quill_sustain(snap, out_len=15.0):
    """Quill 长延音:重合成到任意长度,无循环点。"""
    inst = build_instrument(snap, noise_level=0.02)
    eng = Engine(inst, eval_mode=False)
    eng.reset()
    m = int(round(69 + 12 * np.log2(float(snap["f0_ref"]) / 440.0)))
    evs = [(0.02, "on", m, 100, 0.0, -1), (out_len - 0.5, "off", m, 0, 0.0, -1)]
    return eng.render_offline(evs, out_len).astype(np.float64)


def main():
    import soundfile as sf

    out = ROOT / "assets/ear_tests/structural"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, path in [
        ("glass", "assets/snapshots/wineglass.npz"),
        ("guitar", "assets/snapshots/guitar_rich.npz"),
    ]:
        snap = load_snapshot(str(ROOT / path))

        # T1 和弦
        s_ch, s_voices = sampler_chord(snap)
        q_ch = quill_chord(snap)
        # 声部间噪声相关(采样器:同一样本→高度相关)
        cors = []
        for i in range(len(s_voices)):
            for j in range(i + 1, len(s_voices)):
                a, b = s_voices[i][:SR], s_voices[j][:SR]
                cors.append(abs(np.corrcoef(a, b)[0, 1]))
        sf.write(str(out / f"{name}_chord_sampler.wav"), rms_norm(s_ch), SR)
        sf.write(str(out / f"{name}_chord_quill.wav"), rms_norm(q_ch), SR)

        # T2 长延音
        s_su = sampler_sustain(snap)
        q_su = quill_sustain(snap)
        sf.write(str(out / f"{name}_sustain15s_sampler.wav"), rms_norm(s_su), SR)
        sf.write(str(out / f"{name}_sustain15s_quill.wav"), rms_norm(q_su), SR)

        res[name] = {
            "chord_comb_depth_dB": {
                "sampler": round(comb_depth(s_ch), 1),
                "quill": round(comb_depth(q_ch), 1),
            },
            "chord_voice_corr_mean_sampler": round(float(np.mean(cors)), 3),
            "sustain_loop_periodicity": {
                "sampler": round(loop_periodicity(s_su), 3),
                "quill": round(loop_periodicity(q_su), 3),
            },
        }
        print(f"{name}:")
        print(
            f"  T1 和弦梳状深度(dB,越小越干净): sampler={res[name]['chord_comb_depth_dB']['sampler']}"
            f"  quill={res[name]['chord_comb_depth_dB']['quill']}"
        )
        print(
            f"     采样器声部间相关={res[name]['chord_voice_corr_mean_sampler']}(越高=越像一个声音复制)"
        )
        print(
            f"  T2 15秒长音循环痕迹(0=无循环): sampler={res[name]['sustain_loop_periodicity']['sampler']}"
            f"  quill={res[name]['sustain_loop_periodicity']['quill']}"
        )
    (ROOT / "results/structural_tests.json").write_text(json.dumps(res, indent=1))
    print(f"\n音频 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------- #
# T4 · 参数空间形变(采样器结构上做不到:叠加≠形变)
# --------------------------------------------------------------------- #
def morph_snapshots(sa: dict, sb: dict, w: float = 0.5) -> dict:
    """谱参数插值:得到一件**从未存在过的乐器**(不是两个声音叠加)。
    对齐到 A 的分量频率轴,log 域插值 H_steady/N_env;f0_ref 取几何中点。"""
    out = dict(sa)
    Ha = np.asarray(sa["H_env"], np.float64)
    Hb = np.asarray(sb["H_env"], np.float64)
    Ka = Ha.shape[0]
    fa = float(sa["f0_ref"])
    fb = float(sb["f0_ref"])
    pfa = sa.get("partial_freqs")
    xa = (
        np.asarray(pfa, np.float64)[:Ka]
        if pfa is not None and np.asarray(pfa).size
        else np.arange(1, Ka + 1) * fa
    )
    Kb = Hb.shape[0]
    # B 的包络按**相对谐波序号**重采样到 A 的轴(音高无关的音色形状插值)
    ta = np.arange(1, Ka + 1, dtype=float)
    tb = np.arange(1, Kb + 1, dtype=float)
    T = min(Ha.shape[1], Hb.shape[1])
    Hm = np.empty((Ka, T))
    for t in range(T):
        hb_on_a = np.interp(ta, tb, np.maximum(Hb[:, t], 1e-9))
        Hm[:, t] = np.exp((1 - w) * np.log(np.maximum(Ha[:Ka, t], 1e-9)) + w * np.log(hb_on_a))
    out["H_env"] = Hm.astype(np.float32)
    out["f0_ref"] = float(fa ** (1 - w) * fb**w)
    out["partial_freqs"] = (xa * (out["f0_ref"] / fa)).astype(np.float32)
    Na = np.asarray(sa.get("N_env"), np.float64)
    Nb = np.asarray(sb.get("N_env"), np.float64)
    if Na.size and Nb.size:
        Tn = min(Na.shape[1], Nb.shape[1])
        B = min(Na.shape[0], Nb.shape[0])
        out["N_env"] = np.exp(
            (1 - w) * np.log(np.maximum(Na[:B, :Tn], 1e-9))
            + w * np.log(np.maximum(Nb[:B, :Tn], 1e-9))
        ).astype(np.float32)
    return out


def peak_set(y, n=12):
    """谱峰位置集合(Hz):判断是"两套峰共存"(叠加)还是"一套中间峰"(形变)。"""
    S = np.abs(np.fft.rfft(y[: 1 << 16] * np.hanning(min(len(y), 1 << 16))))
    f = np.fft.rfftfreq(min(len(y), 1 << 16), 1 / SR)
    m = (f > 100) & (f < 5000)
    S, f = S[m], f[m]
    idx = np.argsort(-S)[:400]
    picked = []
    for i in sorted(idx, key=lambda k: -S[k]):
        if all(abs(f[i] - p) > 25 for p in picked):
            picked.append(f[i])
        if len(picked) >= n:
            break
    return np.array(sorted(picked))


def morph_test():
    import soundfile as sf

    out = ROOT / "assets/ear_tests/structural"
    sa = load_snapshot(str(ROOT / "assets/snapshots/wineglass.npz"))
    sb = load_snapshot(str(ROOT / "assets/snapshots/guitar_rich.npz"))
    midi = 62
    f0 = 440 * 2 ** ((midi - 69) / 12)
    n = int(3.0 * SR)
    ya = render(sa, f0, length=n, mode="formant", add_attack=True)
    yb = render(sb, f0, length=n, mode="formant", add_attack=True)
    layer = rms_norm(
        0.5 * (ya / max(np.abs(ya).max(), 1e-9) + yb / max(np.abs(yb).max(), 1e-9))
    )  # 采样器:叠加
    sm = morph_snapshots(sa, sb, 0.5)
    ymorph = rms_norm(render(sm, f0, length=n, mode="formant", add_attack=True))
    sf.write(str(out / "morph_sampler_layered.wav"), layer, SR)
    sf.write(str(out / "morph_quill_50_50.wav"), ymorph, SR)
    for w, tag in ((0.25, "25"), (0.75, "75")):
        sw = morph_snapshots(sa, sb, w)
        sf.write(
            str(out / f"morph_quill_{tag}.wav"),
            rms_norm(render(sw, f0, length=n, mode="formant", add_attack=True)),
            SR,
        )
    pa, pb = peak_set(ya), peak_set(yb)
    pl, pm = peak_set(layer), peak_set(ymorph)

    def frac_from(p, src, tol=30):
        return float(np.mean([any(abs(x - s) < tol for s in src) for x in p])) if len(p) else 0.0

    r = {
        "layered_peaks_from_A": frac_from(pl, pa),
        "layered_peaks_from_B": frac_from(pl, pb),
        "morph_peaks_from_A": frac_from(pm, pa),
        "morph_peaks_from_B": frac_from(pm, pb),
    }
    print("\nT4 参数形变 vs 叠加(谱峰归属):")
    print(
        f"  叠加(采样器能做的): 来自A {r['layered_peaks_from_A']:.0%}  来自B {r['layered_peaks_from_B']:.0%}"
        f"  → 合计 {r['layered_peaks_from_A'] + r['layered_peaks_from_B']:.0%}(>100%=两套峰共存)"
    )
    print(
        f"  参数形变(Quill 独有): 来自A {r['morph_peaks_from_A']:.0%}  来自B {r['morph_peaks_from_B']:.0%}"
        f"  → 合计 {r['morph_peaks_from_A'] + r['morph_peaks_from_B']:.0%}(低=新的中间峰,新乐器)"
    )
    return r


if __name__ != "__main__":
    pass
