"""量化评测:把三条核心主张变成数字。

  1. 可控性:演奏法条件 → 输出音频真实呈现该演奏法的比例(控制保真度矩阵)
     渲染后用 pyin 重新提取 f0,逐音符测 vibrato(3.5-9Hz 微观摆动)和 slide(净漂移)。
  2. 解耦性:2×2 迁移矩阵,f0 微观动态距离应沿"行为源"聚类,
     MFCC 音色距离应沿"音色源"聚类——两轴正交 = 解耦的定量证明。
  3. 数据效率:行为库规模 50→全量,控制保真度曲线。

跑(本机): cd ~/Desktop/MT/project/quill && conda run -n mt python tools/eval_quill.py
产出: checkpoints/eval/ 下渲染音频 + eval_results.json + 终端表格
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
import torch

QUILL = Path(__file__).resolve().parent.parent
CKPT = QUILL / "checkpoints"
OUT = CKPT / "eval"

SR = 16000
FPS = 250

# ---- 测试旋律集:4 段不同风格(midi, dur_s, art) ---- #
MELODIES = {
    "lyrical": [  # 慢板长音,vibrato 为主 + 一处 slide
        (62, 0.8, 1),
        (65, 0.4, 0),
        (67, 1.2, 1),
        (65, 0.4, 0),
        (69, 0.4, 2),
        (72, 1.6, 1),
        (67, 0.8, 0),
        (62, 1.6, 1),
    ],
    "scale_run": [  # 快速级进,全 plain,末音长 vibrato
        (62, 0.2, 0),
        (64, 0.2, 0),
        (65, 0.2, 0),
        (67, 0.2, 0),
        (69, 0.2, 0),
        (71, 0.2, 0),
        (72, 0.2, 0),
        (74, 0.2, 0),
        (76, 0.4, 0),
        (74, 0.4, 0),
        (72, 1.6, 1),
    ],
    "leaps": [  # 跳进 + 两处 slide
        (57, 0.4, 0),
        (64, 0.4, 0),
        (62, 0.4, 2),
        (69, 0.8, 1),
        (65, 0.4, 0),
        (72, 0.4, 2),
        (71, 0.8, 0),
        (64, 1.2, 1),
    ],
    "folk": [  # 民谣句式,混合
        (69, 0.4, 0),
        (71, 0.2, 0),
        (72, 0.4, 0),
        (69, 0.4, 0),
        (67, 0.8, 1),
        (64, 0.4, 0),
        (67, 0.4, 2),
        (69, 1.2, 1),
        (67, 0.4, 0),
        (64, 0.4, 0),
        (62, 1.6, 1),
    ],
}


def strip_art(score):
    return [(m, d, 0) for (m, d, _a) in score]


# ---- 渲染器 ---- #
class Renderer:
    def __init__(self):
        from neural.ddsp import DDSP
        from neural.neural_take import ConditionedNN

        specs = [
            ("violin", "ddsp_urmp_vn_12k.pt", "urmp_vn_trajectories.npz", "sustain"),
            ("trumpet", "ddsp_urmp_tpt_12k.pt", "urmp_tpt_trajectories.npz", "sustain"),
            ("guitar", "ddsp_guitarset.pt", "guitarset_trajectories.npz", "pluck"),
        ]
        self.models, self.nns, self.envelopes = {}, {}, {}
        for name, ck, tr, env in specs:
            if not (CKPT / ck).exists():
                continue
            m = DDSP(dropout=0.0)
            m.load_state_dict(torch.load(str(CKPT / ck), map_location="cpu", weights_only=True))
            m.eval()
            self.models[name] = m
            d = np.load(str(CKPT / tr))
            self.nns[name] = ConditionedNN(d["features"], d["trajectories"])
            self.envelopes[name] = env
        d = np.load(str(CKPT / "urmp_vn_trajectories.npz"))
        self.vn_feats, self.vn_trajs = d["features"], d["trajectories"]

    def render(self, score, timbre="violin", behavior=None, source="v0", nn_override=None):
        """source: v0(检索) / v1(生成) / flat(无行为)。返回 (audio, note_bounds_sec)。"""
        from neural.perform import control_from_behavior

        behavior = behavior or timbre
        notes = [{"midi": m, "dur": d, "art": a} for (m, d, a) in score]

        if source == "flat":
            segs = []
            for m_, d_, _a in score:
                T = int(d_ * FPS)
                f0 = torch.full((1, T, 1), 440.0 * 2 ** ((m_ - 69) / 12.0))
                lo = torch.full((1, T, 1), 0.0)
                with torch.no_grad():
                    segs.append(
                        self.models[timbre](f0, lo, noise_gain=0.6, noise_gate_db=-1.0).numpy()[0]
                    )
            y = np.concatenate(segs)
            y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
            bounds, t = [], 0.0
            for _m, d_, _a in score:
                bounds.append((t, t + d_))
                t += d_
            return y, bounds
        else:
            beh_model = nn_override or self.nns[behavior]
            f0, loud = control_from_behavior(
                notes,
                beh_model,
                pitch_glide=False,
                envelope=self.envelopes[behavior],
                traj_gain=1.0,
            )
            with torch.no_grad():
                y = self.models[timbre](f0, loud, noise_gain=0.6, noise_gate_db=-1.0).numpy()[0]

        y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
        CF_FRAMES = 5
        bounds, t_sec = [], 0.0
        prev_n = None
        for _m, d_, _a in score:
            n_ = max(4, int(d_ * FPS))
            if prev_n is not None:
                t_sec -= min(CF_FRAMES, prev_n // 2, n_ // 2) / FPS
            bounds.append((t_sec, t_sec + d_))
            t_sec += d_
            prev_n = n_
        return y, bounds


# ---- 逐音符 f0 分析 ---- #
def analyze(audio, bounds):
    """pyin 提取 f0 → 每音符 {micro_rms(cents), rate(Hz), drift(cents), rng(cents)}"""
    import librosa

    hop = 160  # 100 fps
    f0, _vfl, _vpr = librosa.pyin(
        audio, fmin=170, fmax=1700, sr=SR, frame_length=1024, hop_length=hop
    )
    t_f = np.arange(len(f0)) * hop / SR
    good = ~np.isnan(f0)
    if good.sum() < 4:
        return [{"micro_rms": 0, "rate": 0, "drift": 0, "rng": 0}] * len(bounds)
    f0i = np.interp(t_f, t_f[good], f0[good])

    out = []
    TRIM = 5  # 掐掉首尾各 50ms(5 frames @100fps),过滤音符过渡区 f0 跳变
    F_RATE = 100  # pyin 帧率
    for t0, t1 in bounds:
        if (t1 - t0) < 0.3:
            out.append({"vib_energy": 0.0, "rate": 0.0, "drift": 0.0, "total_rms": 0.0})
            continue
        i0, i1 = int(t0 * F_RATE) + TRIM, int(t1 * F_RATE) - TRIM
        if i1 <= i0:
            i0, i1 = int(t0 * F_RATE), int(t1 * F_RATE)
        seg = f0i[i0:i1]
        if len(seg) < 8:
            out.append({"vib_energy": 0.0, "rate": 0.0, "drift": 0.0, "total_rms": 0.0})
            continue
        cents = 1200 * np.log2(seg / (np.median(seg) + 1e-9) + 1e-12)

        # 频域 vibrato 检测:Hann-windowed FFT → 3.5-9Hz band 能量
        win = np.hanning(len(cents))
        spec = np.abs(np.fft.rfft((cents - cents.mean()) * win))
        freqs = np.fft.rfftfreq(len(cents), d=1.0 / F_RATE)
        vib_band = (freqs >= 3.5) & (freqs <= 9.0)
        N = max(len(cents), 1)
        vib_energy = float(np.sqrt((spec[vib_band] ** 2).sum() / N**2))
        # 主频率(仅 vib band 内)
        if vib_band.any() and spec[vib_band].max() > 0:
            rate = float(freqs[vib_band][int(np.argmax(spec[vib_band]))])
        else:
            rate = 0.0
        # 净漂移:首 1/4 中位 → 尾 1/4 中位
        q = max(2, len(cents) // 4)
        drift = float(np.median(cents[-q:]) - np.median(cents[:q]))
        total_rms = float(np.sqrt((cents**2).mean()))
        out.append({"vib_energy": vib_energy, "rate": rate, "drift": drift, "total_rms": total_rms})
    return out


def is_vibrato(a):
    return a["vib_energy"] > 3.0 and 3.5 <= a["rate"] <= 9.5


def is_slide(a):
    return abs(a["drift"]) > 60.0


# ---- 实验 1:控制保真度(控制信号级分析) ---- #
def analyze_control(f0_np, score):
    """直接分析进入 DDSP 之前的 f0 控制信号(250Hz),不受 DDSP 音质干扰。
    使用每个音符的中间 60% 区间,避开 crossfade 重叠区和首尾过渡。"""
    FPS_CTRL = 250
    CF = 5  # control_from_behavior 的 crossfade 帧数(20ms@250Hz),重叠会累积,必须对齐
    out = []
    t_frame = 0
    prev_T = None
    for _m, dur, art in score:
        T = max(4, int(dur * FPS_CTRL))
        if prev_T is not None:
            t_frame -= min(CF, prev_T // 2, T // 2)  # 与 control_from_behavior 同步的重叠
        # vibrato/plain 取中间 60%(避开 crossfade + attack/release);
        # slide 的现象在音符头部(前 40% 滑入),头部只跳过 crossfade(~24ms)。
        margin = max(12, T // 5)
        i0 = t_frame + (6 if art == 2 else margin)
        i1 = t_frame + T - margin
        t_frame += T
        prev_T = T
        if i1 <= i0 or i1 > len(f0_np) or dur < 0.3:
            out.append(
                {"vib_energy": np.nan, "total_rms": np.nan, "drift": 0.0, "dur": dur, "art": art}
            )
            continue
        seg = f0_np[i0:i1]
        cents = 1200 * np.log2(seg / (np.median(seg) + 1e-9) + 1e-12)
        win = np.hanning(len(cents))
        spec = np.abs(np.fft.rfft((cents - cents.mean()) * win))
        freqs = np.fft.rfftfreq(len(cents), d=1.0 / FPS_CTRL)
        vib_band = (freqs >= 3.5) & (freqs <= 9.0)
        Nc = max(len(cents), 1)
        vib_e = float(np.sqrt((spec[vib_band] ** 2).sum() / Nc**2))
        total_rms = float(np.sqrt((cents**2).mean()))
        q = max(2, len(cents) // 4)
        # 起点取窗口头 4 帧(不是前 1/4 中位):快滑 80ms 内完成,长窗中位会把滑音吞掉
        qh = max(2, min(4, len(cents) // 8))
        drift = float(np.median(cents[-q:]) - np.median(cents[:qh]))
        out.append(
            {"vib_energy": vib_e, "total_rms": total_rms, "drift": drift, "dur": dur, "art": art}
        )
    return out


class RandSameLabel:
    """消融基线:标签过滤 ✓ context 检索 ✗ —— 同标签池内随机取轨迹。"""

    def __init__(self, nn, seed=42):
        self.nn = nn
        self.rng = np.random.default_rng(seed)

    def predict(self, feat):
        feat = np.asarray(feat, dtype=np.float32)
        a = int(round(float(feat[6]))) if len(feat) >= 7 else 0
        pool = self.nn._pools.get(a)
        if pool is None or len(pool) == 0:
            pool = np.arange(len(self.nn.base))
        return self.nn.trajectories[int(self.rng.choice(pool))]


class NNNoLabel:
    """消融基线:标签过滤 ✗ context 检索 ✓ —— 全库 6-D 最近邻,忽略标签。"""

    def __init__(self, nn):
        self.nn = nn

    def predict(self, feat):
        q = np.asarray(feat, dtype=np.float32)[:6] / self.nn._std
        dist = np.linalg.norm(self.nn._norm - q, axis=1)
        return self.nn.trajectories[int(np.argmin(dist))]


def exp1_controllability(R):
    print("\n=== 实验 1:演奏法控制保真度(控制信号级,4 段旋律) ===")
    from neural.perform import control_from_behavior

    results = {}
    ablations = {"nn_nolabel": NNNoLabel(R.nns["violin"])}
    # v1(VAE)/v2(参数化)基线模型已随代码清理移除;其历史数字保留在
    # checkpoints/eval/eval_results.json(论文 Sec.2 的 unpaired 对比引用自该存档)
    for source in ["v0", "nn_nolabel"]:
        vib_es, plain_es, slide_drifts = [], [], []
        slide_ok, slide_n = 0, 0
        for mname, score in MELODIES.items():
            notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
            beh_model = ablations.get(source, R.nns["violin"])
            f0, loud = control_from_behavior(
                notes, beh_model, pitch_glide=False, envelope="sustain", traj_gain=1.0
            )
            f0_np = f0.numpy()[0, :, 0]
            # 同时渲染保存音频(给耳测用;消融基线只做控制层分析)
            if source == "v0":
                audio, bounds = R.render(score, source=source)
                sf.write(str(OUT / f"{mname}_{source}.wav"), audio, SR)
            ana = analyze_control(f0_np, score)
            for a in ana:
                if np.isnan(a["vib_energy"]):
                    continue
                if a["art"] == 1:
                    vib_es.append(a["vib_energy"])
                elif a["art"] == 2:
                    slide_drifts.append(abs(a["drift"]))
                    slide_n += 1
                    slide_ok += abs(a["drift"]) > 50
                else:
                    plain_es.append(a["vib_energy"])

        vib_m = float(np.median(vib_es)) if vib_es else 0
        plain_m = float(np.median(plain_es)) if plain_es else 0
        contrast = vib_m / max(plain_m, 0.1)
        r = {
            "vib_median": vib_m,
            "plain_median": plain_m,
            "contrast": contrast,
            "slide_ok": slide_ok,
            "slide_n": slide_n,
            "n_vib": len(vib_es),
            "n_plain": len(plain_es),
        }
        results[source] = r
        print(
            f"  {source:>10s}  vib中位数 {vib_m:.0f}c  plain中位数 {plain_m:.0f}c  "
            f"对比度 {contrast:.1f}×   "
            f"slide命中 {slide_ok}/{slide_n}"
        )

    # rand_label 消融:100 个随机种子,报分布而非单点(单种子结论不可靠)
    from neural.perform import control_from_behavior

    rand_contrasts = []
    for seed in range(100):
        rl = RandSameLabel(R.nns["violin"], seed=seed)
        vib_es, plain_es = [], []
        for _mname, score in MELODIES.items():
            notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
            f0, _ = control_from_behavior(
                notes, rl, pitch_glide=False, envelope="sustain", traj_gain=1.0
            )
            for a in analyze_control(f0.numpy()[0, :, 0], score):
                if np.isnan(a["vib_energy"]):
                    continue
                if a["art"] == 1:
                    vib_es.append(a["vib_energy"])
                elif a["art"] == 0:
                    plain_es.append(a["vib_energy"])
        vm = float(np.median(vib_es)) if vib_es else 0
        pm = float(np.median(plain_es)) if plain_es else 0
        rand_contrasts.append(vm / max(pm, 0.1))
    rc = np.array(rand_contrasts)
    results["rand_label_multiseed"] = {
        "n_seeds": len(rc),
        "median": round(float(np.median(rc)), 2),
        "p2.5": round(float(np.percentile(rc, 2.5)), 2),
        "p97.5": round(float(np.percentile(rc, 97.5)), 2),
        "frac_gt1": round(float((rc > 1).mean()), 2),
    }
    print(
        f"  rand_label(100 种子)  中位 {np.median(rc):.2f}×  "
        f"95% 范围 [{np.percentile(rc, 2.5):.2f}, {np.percentile(rc, 97.5):.2f}]  "
        f"{int((rc > 1).sum())}/100 种子 >1"
    )

    # flat baseline
    for mname, score in MELODIES.items():
        audio, bounds = R.render(score, source="flat")
        sf.write(str(OUT / f"{mname}_flat.wav"), audio, SR)
    print("  flat  (无行为轨迹,所有指标=0,已保存音频)")
    return results


# ---- 实验 2:解耦性(2×2 迁移矩阵) ---- #
def behavior_sig(ana):
    """行为签名 = 逐音符 [vib_energy, drift, total_rms]"""
    return np.array([[a["vib_energy"], a["drift"], a["total_rms"]] for a in ana]).ravel()


def timbre_sig(audio):
    import librosa

    mf = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=13)[1:]  # 去 c0(能量)
    return mf.mean(axis=1)


def exp2_decoupling(R):
    insts = list(R.models.keys())  # violin / guitar / cello(有 checkpoint 才进)
    n = len(insts)
    print(f"\n=== 实验 2:行为×音色解耦({n}×{n},lyrical+folk 汇总) ===")
    bsig, tsig = {}, {}
    for mname in ["lyrical", "folk"]:
        score = MELODIES[mname]
        for beh in insts:
            for tim in insts:
                audio, bounds = R.render(score, timbre=tim, behavior=beh)
                sf.write(str(OUT / f"x_{mname}_{beh[:2]}B_{tim[:2]}T.wav"), audio, SR)
                key = (beh, tim)
                bsig[key] = bsig.get(key, []) + [behavior_sig(analyze(audio, bounds))]
                tsig[key] = tsig.get(key, []) + [timbre_sig(audio)]

    def dist(sig, k1, k2):
        return float(np.mean([np.linalg.norm(a - b) for a, b in zip(sig[k1], sig[k2])]))

    # 对每个跨乐器组合 (behB, timT) 做双检验:
    #   行为距离(到同行为对角 bB×bT) < 行为距离(到同音色对角 tB×tT) → 行为跟随行为源
    #   音色距离(到同音色对角 tB×tT) < 音色距离(到同行为对角 bB×bT) → 音色跟随音色源
    results, ok = {}, 0
    total = 0
    for beh in insts:
        for tim in insts:
            if beh == tim:
                continue
            x = (beh, tim)
            b_beh = dist(bsig, x, (beh, beh))
            b_tim = dist(bsig, x, (tim, tim))
            t_beh = dist(tsig, x, (beh, beh))
            t_tim = dist(tsig, x, (tim, tim))
            c1, c2 = b_beh < b_tim, t_tim < t_beh
            ok += c1 + c2
            total += 2
            results[f"{beh[:2]}B_{tim[:2]}T"] = {
                "b_beh": b_beh,
                "b_tim": b_tim,
                "t_beh": t_beh,
                "t_tim": t_tim,
                "behavior_follows": bool(c1),
                "timbre_follows": bool(c2),
            }
            print(
                f"  {beh:>6s}行为 × {tim:>6s}音色:  "
                f"行为跟随行为源 {'✓' if c1 else '✗'} ({b_beh:.0f} vs {b_tim:.0f})   "
                f"音色跟随音色源 {'✓' if c2 else '✗'} ({t_tim:.0f} vs {t_beh:.0f})"
            )
    print(f"  通过 {ok}/{total} 项检验")
    results["passed"] = f"{ok}/{total}"
    return results


# ---- 实验 3:行为库规模曲线 ---- #
def exp3_datacurve(R):
    """控制信号层分析(与实验 1 同口径):不受 DDSP 音质/pyin 重提取噪声干扰。"""
    from neural.neural_take import ConditionedNN
    from neural.perform import control_from_behavior

    print("\n=== 实验 3:行为库规模 → 控制保真度(控制信号级) ===")
    sizes = [50, 150, 500, 1500, 7582]
    n_seeds = 8  # 子采样噪声大(库小时检索命中随机),多种子平均
    curve = []
    for n in sizes:
        c_list, s_list, v_list, p_list = [], [], [], []
        for seed in range(n_seeds):
            order = np.random.default_rng(42 + seed).permutation(len(R.vn_feats))
            idx = order[: min(n, len(order))]
            nn = ConditionedNN(R.vn_feats[idx], R.vn_trajs[idx])
            vib_es, plain_es, slide_ok, slide_n = [], [], 0, 0
            for _mname, score in MELODIES.items():
                notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
                f0, _loud = control_from_behavior(
                    notes, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0
                )
                for a in analyze_control(f0.numpy()[0, :, 0], score):
                    if np.isnan(a["vib_energy"]):
                        continue
                    if a["art"] == 1:
                        vib_es.append(a["vib_energy"])
                    elif a["art"] == 2:
                        slide_n += 1
                        slide_ok += abs(a["drift"]) > 50
                    else:
                        plain_es.append(a["vib_energy"])
            vib_m = float(np.median(vib_es)) if vib_es else 0
            plain_m = float(np.median(plain_es)) if plain_es else 0
            c_list.append(vib_m / max(plain_m, 0.1))
            s_list.append(slide_ok / max(slide_n, 1))
            v_list.append(vib_m)
            p_list.append(plain_m)
        row = {
            "n": int(min(n, len(R.vn_feats))),
            "vib_median": float(np.mean(v_list)),
            "plain_median": float(np.mean(p_list)),
            "contrast": float(np.mean(c_list)),
            "contrast_std": float(np.std(c_list) / np.sqrt(len(c_list))),
            "slide_hit": float(np.mean(s_list)),
        }
        curve.append(row)
        print(
            f"  {row['n']:>5d} notes  对比度 {row['contrast']:.1f}×±{row['contrast_std']:.1f}  "
            f"slide命中 {row['slide_hit']:.0%}"
        )
    return curve


# ---- 实验 4:音频级演奏法验证(pYIN 重提取) ---- #
def exp4_audio_controllability(R):
    """对渲染后的 WAV 跑 pYIN,逐音符测 vibrato band energy,
    验证控制信号层的对比度在音频端依然成立。"""
    print("\n=== 实验 4:音频级演奏法验证(pYIN 重提取) ===")
    results = {}
    for source in ["v0", "flat"]:
        vib_es, plain_es = [], []
        for _mname, score in MELODIES.items():
            audio, bounds = R.render(score, source=source)
            ana = analyze(audio, bounds)
            for a, (_m, _d, art) in zip(ana, score):
                if a["vib_energy"] == 0.0 and a["total_rms"] == 0.0:
                    continue
                if art == 1:
                    vib_es.append(a["vib_energy"])
                elif art == 0:
                    plain_es.append(a["vib_energy"])

        vib_m = float(np.median(vib_es)) if vib_es else 0
        plain_m = float(np.median(plain_es)) if plain_es else 0
        contrast = vib_m / max(plain_m, 0.1)
        results[source] = {
            "vib_median": round(vib_m, 1),
            "plain_median": round(plain_m, 1),
            "contrast": round(contrast, 1),
            "n_vib": len(vib_es),
            "n_plain": len(plain_es),
        }
        print(
            f"  {source:>4s}  vib中位数 {vib_m:.1f}c  plain中位数 {plain_m:.1f}c  "
            f"对比度 {contrast:.1f}×  (n_vib={len(vib_es)}, n_plain={len(plain_es)})"
        )
    return results


# ---- 实验 5:配对反事实(同音符换标签) ---- #
def _bootstrap_median_ci(arr, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(42)
    medians = np.array(
        [float(np.median(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    )
    lo = float(np.percentile(medians, 100 * alpha / 2))
    hi = float(np.percentile(medians, 100 * (1 - alpha / 2)))
    return lo, hi


def exp5_paired_counterfactual(R):
    """对每段旋律的每个音符,分别赋 vibrato 和 plain 标签渲染,
    配对比较 vib_energy → 消除时长/音高/位置混杂。
    使用 log-ratio + epsilon 保留全部预定义合格音符(无 outcome-dependent exclusion);
    音频端每对旋律前重置 torch seed 确保相同 DDSP noise realization。"""
    from neural.perform import control_from_behavior

    print("\n=== 实验 5:配对反事实(同音符 vibrato vs plain,控制信号级+音频级) ===")

    EPS = 1e-4
    ctrl_logr, audio_logr = [], []
    ctrl_gt1, audio_gt1 = 0, 0
    ctrl_by_mel, audio_by_mel = {}, {}
    for mi, (mname, score) in enumerate(MELODIES.items()):
        score_vib = [(m, d, 1) for (m, d, _a) in score]
        score_pln = [(m, d, 0) for (m, d, _a) in score]
        notes_vib = [{"midi": m, "dur": d, "art": a} for m, d, a in score_vib]
        notes_pln = [{"midi": m, "dur": d, "art": a} for m, d, a in score_pln]
        nn = R.nns["violin"]

        f0_v, _ = control_from_behavior(
            notes_vib, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0
        )
        f0_p, _ = control_from_behavior(
            notes_pln, nn, pitch_glide=False, envelope="sustain", traj_gain=1.0
        )
        ana_v = analyze_control(f0_v.numpy()[0, :, 0], score_vib)
        ana_p = analyze_control(f0_p.numpy()[0, :, 0], score_pln)
        _mel = []
        for av, ap in zip(ana_v, ana_p):
            if np.isnan(av["vib_energy"]) or np.isnan(ap["vib_energy"]):
                continue
            lr = np.log2((av["vib_energy"] + EPS) / (ap["vib_energy"] + EPS))
            ctrl_logr.append(lr)
            _mel.append(lr)
            if lr > 0:
                ctrl_gt1 += 1
        ctrl_by_mel[mname] = _mel

        pair_seed = 1000 + mi
        torch.manual_seed(pair_seed)
        audio_v, bounds_v = R.render(score_vib, source="v0")
        torch.manual_seed(pair_seed)
        audio_p, bounds_p = R.render(score_pln, source="v0")
        aana_v = analyze(audio_v, bounds_v)
        aana_p = analyze(audio_p, bounds_p)
        _mel = []
        for av, ap, (_m, d_, _a) in zip(aana_v, aana_p, score):
            if d_ < 0.3:
                continue
            lr = np.log2((av["vib_energy"] + EPS) / (ap["vib_energy"] + EPS))
            audio_logr.append(lr)
            _mel.append(lr)
            if lr > 0:
                audio_gt1 += 1
        audio_by_mel[mname] = _mel

    ctrl_arr = np.array(ctrl_logr)
    audio_arr = np.array(audio_logr)
    ctrl_med = float(2 ** np.median(ctrl_arr))
    audio_med = float(2 ** np.median(audio_arr))
    ctrl_ci = tuple(float(2**x) for x in _bootstrap_median_ci(ctrl_arr))
    audio_ci = tuple(float(2**x) for x in _bootstrap_median_ci(audio_arr))

    print(
        f"  控制信号: 配对中位比 {ctrl_med:.1f}× bootstrap 95% CI [{ctrl_ci[0]:.1f}, {ctrl_ci[1]:.1f}]  "
        f"{ctrl_gt1}/{len(ctrl_logr)} ratio>1"
    )
    print(
        f"  音频端:   配对中位比 {audio_med:.1f}× bootstrap 95% CI [{audio_ci[0]:.1f}, {audio_ci[1]:.1f}]  "
        f"{audio_gt1}/{len(audio_logr)} ratio>1"
    )

    results = {
        "control_median_ratio": round(ctrl_med, 2),
        "control_bootstrap_ci95": [round(ctrl_ci[0], 2), round(ctrl_ci[1], 2)],
        "control_n": len(ctrl_logr),
        "control_gt1": ctrl_gt1,
        "audio_median_ratio": round(audio_med, 2),
        "audio_bootstrap_ci95": [round(audio_ci[0], 2), round(audio_ci[1], 2)],
        "audio_n": len(audio_logr),
        "audio_gt1": audio_gt1,
        # 逐旋律中位比:音符共享旋律上下文(item CI 有伪重复),
        # cluster 只有 4 个,直接报逐旋律值供读者判断稳健性
        "control_per_melody": {
            m: round(float(2 ** np.median(v)), 2) for m, v in ctrl_by_mel.items() if v
        },
        "audio_per_melody": {
            m: round(float(2 ** np.median(v)), 2) for m, v in audio_by_mel.items() if v
        },
    }
    print("  逐旋律(控制):", results["control_per_melody"])
    print("  逐旋律(音频):", results["audio_per_melody"])
    return results


# ---- 实验 6:后处理归因消融(配对反事实 × 三种后处理模式) ---- #
def exp6_processing_ablation(R):
    """回答"对比度来自检索还是手写滤波":同一配对反事实协议,
    对比 label_processing = none(检索原样)/ uniform(统一处理)/ full(完整)。"""
    from neural.perform import control_from_behavior

    print("\n=== 实验 6:后处理归因(配对反事实,控制信号级) ===")
    results = {}
    for mode in ["none", "uniform", "full"]:
        logr, gt1 = [], 0
        by_melody = {}
        for mname, score in MELODIES.items():
            score_vib = [(m, d, 1) for (m, d, _a) in score]
            score_pln = [(m, d, 0) for (m, d, _a) in score]
            notes_vib = [{"midi": m, "dur": d, "art": a} for m, d, a in score_vib]
            notes_pln = [{"midi": m, "dur": d, "art": a} for m, d, a in score_pln]
            nn = R.nns["violin"]
            f0_v, _ = control_from_behavior(
                notes_vib,
                nn,
                pitch_glide=False,
                envelope="sustain",
                traj_gain=1.0,
                label_processing=mode,
            )
            f0_p, _ = control_from_behavior(
                notes_pln,
                nn,
                pitch_glide=False,
                envelope="sustain",
                traj_gain=1.0,
                label_processing=mode,
            )
            ana_v = analyze_control(f0_v.numpy()[0, :, 0], score_vib)
            ana_p = analyze_control(f0_p.numpy()[0, :, 0], score_pln)
            EPS = 1e-4
            mel_lrs = []
            for av, ap in zip(ana_v, ana_p):
                if np.isnan(av["vib_energy"]) or np.isnan(ap["vib_energy"]):
                    continue
                lr = np.log2((av["vib_energy"] + EPS) / (ap["vib_energy"] + EPS))
                logr.append(lr)
                mel_lrs.append(lr)
                if lr > 0:
                    gt1 += 1
            by_melody[mname] = mel_lrs
        arr = np.array(logr)
        med = float(2 ** np.median(arr))
        ci = tuple(float(2**x) for x in _bootstrap_median_ci(arr))
        # 旋律级:每旋律的配对中位比(音符共享旋律上下文,item CI 有伪重复,
        # 4 个 cluster 不足以做稳定 cluster bootstrap → 直接报逐旋律值)
        per_mel = {m: round(float(2 ** np.median(v)), 2) for m, v in by_melody.items() if v}
        results[mode] = {
            "paired_median_ratio": round(med, 2),
            "item_ci95": [round(ci[0], 2), round(ci[1], 2)],
            "n": len(logr),
            "gt1": gt1,
            "per_melody_median": per_mel,
        }
        pm_str = " ".join(f"{m}:{v:.1f}×" for m, v in per_mel.items())
        print(
            f"  {mode:>8s}: 配对中位比 {med:.1f}× item-CI [{ci[0]:.1f}, {ci[1]:.1f}]  "
            f"{gt1}/{len(logr)} ratio>1  | 逐旋律 {pm_str}"
        )
    return results


# ---- 实验 7:glass×violin 2×2 正式归因 ---- #
def exp7_glass_2x2(R):
    """glass/violin 双资产 2×2:行为签名跟随行为源、音色签名跟随音色源。
    全部 cell 移高八度(glass 天然音域),包络统一 sustain 保证可比性。"""
    from neural.ddsp import DDSP
    from neural.neural_take import ConditionedNN
    from neural.perform import control_from_behavior

    ck = CKPT / "ddsp_glass.pt"
    st = CKPT / "glass_style.npz"
    if not ck.exists() or not st.exists():
        print("\n=== 实验 7:跳过(glass checkpoint 缺失) ===")
        return {"skipped": True}
    print("\n=== 实验 7:glass×violin 2×2 归因 ===")
    glass = DDSP(dropout=0.0)
    glass.load_state_dict(torch.load(str(ck), map_location="cpu", weights_only=True))
    glass.eval()
    d_g = np.load(str(st))
    nn_g = ConditionedNN(d_g["features"], d_g["trajectories"])
    models = {"glass": glass, "violin": R.models["violin"]}
    nns = {"glass": nn_g, "violin": R.nns["violin"]}

    TR = 12
    bsig, tsig = {}, {}
    for mname in ["lyrical", "folk"]:
        # slide 音符改 plain:glass 库无 slide 池,ConditionedNN 会静默回退全库
        # 且 slide 主要由手写规则重建 → 保留会污染"行为交换"的归因
        score = [(m + TR, d, 0 if a == 2 else a) for (m, d, a) in MELODIES[mname]]
        for beh in ["glass", "violin"]:
            for tim in ["glass", "violin"]:
                notes = [{"midi": m, "dur": d, "art": a} for m, d, a in score]
                torch.manual_seed(42)
                f0, loud = control_from_behavior(
                    notes, nns[beh], pitch_glide=False, envelope="sustain", traj_gain=1.0
                )
                with torch.no_grad():
                    y = models[tim](f0, loud, noise_gain=0.6, noise_gate_db=-1.0).numpy()[0]
                y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
                CF_FRAMES = 5
                bounds, t_sec, prev_n = [], 0.0, None
                for _m, d_, _a in score:
                    n_ = max(4, int(d_ * FPS))
                    if prev_n is not None:
                        t_sec -= min(CF_FRAMES, prev_n // 2, n_ // 2) / FPS
                    bounds.append((t_sec, t_sec + d_))
                    t_sec += d_
                    prev_n = n_
                sf.write(str(OUT / f"g22_{mname}_{beh[:2]}B_{tim[:2]}T.wav"), y, SR)
                key = (beh, tim)
                bsig[key] = bsig.get(key, []) + [behavior_sig(analyze(y, bounds))]
                tsig[key] = tsig.get(key, []) + [timbre_sig(y)]

    def dist(sig, k1, k2):
        return float(np.mean([np.linalg.norm(a - b) for a, b in zip(sig[k1], sig[k2])]))

    results, ok, total = {}, 0, 0
    for beh, tim in [("glass", "violin"), ("violin", "glass")]:
        x = (beh, tim)
        b_beh = dist(bsig, x, (beh, beh))
        b_tim = dist(bsig, x, (tim, tim))
        t_beh = dist(tsig, x, (beh, beh))
        t_tim = dist(tsig, x, (tim, tim))
        c1, c2 = b_beh < b_tim, t_tim < t_beh
        ok += c1 + c2
        total += 2
        results[f"{beh[:2]}B_{tim[:2]}T"] = {
            "b_beh": b_beh,
            "b_tim": b_tim,
            "t_beh": t_beh,
            "t_tim": t_tim,
            "behavior_follows": bool(c1),
            "timbre_follows": bool(c2),
        }
        print(
            f"  {beh:>6s}行为 × {tim:>6s}音色:  "
            f"行为跟随 {'✓' if c1 else '✗'} ({b_beh:.0f} vs {b_tim:.0f})   "
            f"音色跟随 {'✓' if c2 else '✗'} ({t_tim:.0f} vs {t_beh:.0f})"
        )
    print(f"  通过 {ok}/{total}")
    results["passed"] = f"{ok}/{total}"
    return results


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    R = Renderer()
    res = {
        "controllability": exp1_controllability(R),
        "decoupling": exp2_decoupling(R),
        "data_curve": exp3_datacurve(R),
        "audio_controllability": exp4_audio_controllability(R),
        "paired_counterfactual": exp5_paired_counterfactual(R),
        "processing_ablation": exp6_processing_ablation(R),
        "glass_2x2": exp7_glass_2x2(R),
    }
    with open(str(OUT / "eval_results.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n结果已存 {OUT / 'eval_results.json'};渲染音频在 {OUT}/")


if __name__ == "__main__":
    main()
