"""可演奏乐器：音色快照 -> 实时引擎用的预计算结构。

把时变的分析端快照压成实时声部需要的稳态数据：
  H_steady[K]   稳态谐波幅度（持续段中位数）
  ratio_k[K]    各分量频率比例（严格谐波=k；准谐波=partial_freqs/f0_ref）
  oct_k[K]      log2(ratio_k)，亮度倾斜按八度算 gain
  attack_grain  起音残差颗粒（note-on 回放）
  atk_traj      起音谱轨迹，spec_traj/amp_traj 持续段谱形与响度轨迹，amp_env 自然幅度包络
  noise_table   固定 seed 预渲染的噪声纹理，实时只做查表，结果可复现

只依赖 numpy。热切换：Engine 持有 instrument 引用，worker 线程算好新 Instrument 后直接
重绑引用（CPython 引用赋值是原子的），旧的由 GC 回收。
"""

from __future__ import annotations

import numpy as np

from synth.noise import synth_noise


class Instrument:
    def __init__(
        self,
        H_steady,
        ratio_k,
        attack_grain,
        noise_table,
        f0_ref,
        sr,
        name="untitled",
        noise_level=0.5,
        vib_rate=None,
        amp_env=None,
        amp_hop=512,
        spec_traj=None,
        traj_fps=30.0,
        amp_traj=None,
        amp_tail_ratio=1.0,
        atk_traj=None,
        atk_fps=86.13,
    ):
        self.H_steady = np.ascontiguousarray(H_steady, dtype=np.float64)
        self.ratio_k = np.ascontiguousarray(ratio_k, dtype=np.float64)
        self.oct_k = np.log2(np.maximum(self.ratio_k, 1e-9))
        self.attack_grain = np.ascontiguousarray(attack_grain, dtype=np.float64)
        self.noise_table = np.ascontiguousarray(noise_table, dtype=np.float64)
        self.f0_ref = float(f0_ref)
        self.sr = int(sr)
        self.K = int(len(self.H_steady))
        # 起音谱轨迹 [K,Fa]：相对 H_steady 的逐分音因子，原生帧率不降采样。每个分音自己的
        # 起音/早期衰减是拨弦感的物理来源：高次分音衰减快、基频留得久。
        at_ = np.asarray(atk_traj, dtype=np.float64) if atk_traj is not None else None
        self.atk_traj = (
            np.ascontiguousarray(at_)
            if at_ is not None and at_.ndim == 2 and at_.shape[0] == self.K and at_.shape[1] >= 2
            else None
        )
        self.atk_fps = float(atk_fps)
        self.name = name
        self.noise_level = float(noise_level)
        # CC1 颤音速率取源片段实测值（snapshot.vibrato.rate_hz），None 用引擎默认
        self.vib_rate = float(vib_rate) if vib_rate else None
        # 自然幅度包络（峰后衰减∈[0,1]，自然速率）：拨弦类衰减、持续类≈1。只衰减响度，
        # 谱形仍由 gain_k 定。
        self.amp_env = (
            np.ascontiguousarray(amp_env, dtype=np.float64)
            if amp_env is not None and len(amp_env) > 1
            else None
        )
        self.amp_hop = int(amp_hop)
        # 持续段谱形轨迹 [K,F]：相对 H_steady 的逐帧因子，能量已归一，只含形状运动。
        # 渲染端沿它乒乓循环，持续音带上源录音自己的起伏。
        st = np.asarray(spec_traj, dtype=np.float64) if spec_traj is not None else None
        self.spec_traj = (
            np.ascontiguousarray(st)
            if st is not None and st.ndim == 2 and st.shape[1] >= 2
            else None
        )
        self.traj_fps = float(traj_fps)
        # 响度轨迹 [F]，与 spec_traj 同帧：源持续段的音量起伏，呼吸的可听主体
        at = np.asarray(amp_traj, dtype=np.float64) if amp_traj is not None else None
        self.amp_traj = (
            np.ascontiguousarray(at)
            if at is not None and self.spec_traj is not None and len(at) == self.spec_traj.shape[1]
            else None
        )
        self.amp_tail_ratio = float(amp_tail_ratio)  # 超出 amp_env 表长后每帧继续衰减的比率


def build_instrument(
    snap: dict,
    noise_level: float = 0.5,
    noise_table_s: float = 2.0,
    noise_seed: int = 1234,
    pick: float = 0.5,
) -> Instrument:
    """从快照构造 Instrument（一次性预计算，非实时路径）。

    pick∈[0,1]：拨片厚度，整形起音颗粒（见 synth.attack_shape）。自然幅度包络从 H_env
    能量轮廓提取（峰后衰减），让拨弦类在实时引擎里也衰减。
    """
    sr = int(snap.get("meta", {}).get("sr", 44100))
    harm_hop = int(snap.get("meta", {}).get("harm_hop", 512))
    H = np.asarray(snap["H_env"], dtype=np.float64)  # [K, T]
    K, T = H.shape
    # 稳态谐波幅度：取中段 50% 帧的中位数，避开起音与收尾
    lo, hi = int(T * 0.25), max(int(T * 0.75), int(T * 0.25) + 1)
    H_steady = np.median(H[:, lo:hi], axis=1)

    # 自然幅度包络 = 峰后能量的单调衰减趋势（对数域线性拟合）。拟合窗止于 75% 处，避开
    # 片尾收音：收音不是乐器的延音行为。拨弦/敲击给出干净的指数衰减；持续系斜率≈0，
    # 包络≈平，片内起伏归 amp_traj。
    energy = np.sqrt(np.maximum((H**2).sum(axis=0), 0.0))
    pk = int(np.argmax(energy)) if energy.size else 0
    epk = energy[pk] if energy.size and energy[pk] > 0 else 1.0
    amp_env = np.ones(T, dtype=np.float64)
    amp_tail_ratio = 1.0
    fit_start = max(pk, int(T * 0.25))  # 从持续区起拟合：起音到延音的安定归 ADSR，不算衰减
    fit_end = max(int(T * 0.75), fit_start + 5)
    if T > 1 and fit_end - fit_start >= 5 and fit_end <= T:
        tt = (np.arange(fit_start, fit_end) - fit_start) * (harm_hop / sr)
        ln_e = np.log(np.maximum(energy[fit_start:fit_end] / epk, 1e-6))
        k = min(float(np.polyfit(tt, ln_e, 1)[0]), 0.0)  # 只允许衰减；ln 域斜率，单位 1/s
        t_post = np.maximum((np.arange(T) - pk), 0) * (harm_hop / sr)
        amp_env = np.exp(k * t_post)
        amp_tail_ratio = float(np.exp(k * (harm_hop / sr)))  # 超出表长后继续按同一速率衰减

    # 起音谱轨迹：取 [0, 峰值+ATK_MS] 内 H(t)/H_steady 的逐分音因子，原生帧率，起音需要
    # 时间分辨率。谐波层自己完成拨弦，而不是靠外贴噪声颗粒。
    ATK_MS = 320.0
    atk_traj = None
    frame_ms = harm_hop / sr * 1000.0
    n_atk = int(min(T, max(4, round((ATK_MS + pk * frame_ms) / frame_ms))))
    if T >= 6 and n_atk >= 4:
        a_seg = H[:, :n_atk]
        # 逐帧能量归一（与 spec_traj 同规范）：起音轨迹只携带谱形变化，不携带响度变化，
        # 响度由 ADSR/amp_env 负责。带电平会双重计账，分音瞬时放大数倍，母线削顶成爆破音。
        ref_e = float(np.sqrt((H_steady**2).sum()))
        seg_e = np.sqrt(np.maximum((a_seg**2).sum(axis=0), 1e-24))
        a_seg = a_seg * (ref_e / seg_e)[None, :]
        at_ = a_seg / np.maximum(H_steady, 1e-9)[:, None]
        at_ = np.clip(np.nan_to_num(at_, nan=1.0, posinf=3.0), 0.0, 3.0)
        # 末端 6 帧平滑归 1，与稳态无缝衔接
        m = min(6, at_.shape[1])
        ramp = np.linspace(0.0, 1.0, m)[None, :]
        at_[:, -m:] = at_[:, -m:] * (1 - ramp) + ramp
        atk_traj = at_

    # 谱形与响度轨迹：持续段 H(t) 降采样到约 30 fps。谱形轨迹是每帧能量归一后的形状因子；
    # 响度轨迹是每帧能量起伏，持续系全量、拨弦类按持续度折减，避免与 amp_env 双重计账。
    frame_fps = sr / max(harm_hop, 1)
    step = max(1, int(round(frame_fps / 30.0)))
    seg = H[:, lo:hi:step]
    traj_fps = frame_fps / step
    spec_traj = None
    amp_traj = None
    if seg.shape[1] >= 4:
        # 呼吸带低通：不稳的录音里轨迹帧间跳动可达 0.15（正常呼吸 < 0.01），按帧回放是
        # 粗糙杂音。呼吸是 < 5 Hz 的现象：两遍滑动平均（截止约 2.5 Hz）去掉快于呼吸的分析
        # 噪声，再设深度上限，自然持续音不会有 ±12 dB 的呼吸。
        w = max(3, int(round(traj_fps / 5.0)) | 1)

        def _ma2(x2d, win):
            pad = win // 2
            kk = np.ones(win) / win
            out = x2d
            for _ in range(2):
                xp = np.pad(out, ((0, 0), (pad, pad)), mode="edge")
                out = np.stack(
                    [
                        np.convolve(xp[i], kk, mode="valid")[: x2d.shape[1]]
                        for i in range(x2d.shape[0])
                    ]
                )
            return out

        ref_norm = float(np.sqrt((H_steady**2).sum()))
        en = np.sqrt(np.maximum((seg**2).sum(axis=0), 1e-24))
        seg_n = seg * (ref_norm / en)[None, :]
        traj = np.clip(seg_n / np.maximum(H_steady, 1e-9)[:, None], 0.5, 2.0)
        traj = _ma2(traj, w)
        wgt = H_steady / (H_steady.sum() + 1e-12)
        dev = float((np.abs(traj - 1.0) * wgt[:, None]).sum(axis=0).mean())
        if dev > 0.20:  # 谱形呼吸深度上限
            traj = 1.0 + (traj - 1.0) * (0.20 / dev)
        spec_traj = traj
        # 持续度，amp_traj 的门：amp_env 是衰减趋势拟合，持续系≈平则 1，拨弦中段已衰则 0
        sus = float(np.clip((float(np.median(amp_env[lo:hi])) - 0.55) / 0.3, 0.0, 1.0))
        raw = np.clip(en / max(ref_norm, 1e-12), 0.7, 1.4)
        at = 1.0 + (raw - 1.0) * sus
        at = _ma2(at[None, :], w)[0]
        adev = float(np.abs(at - 1.0).mean())
        if adev > 0.12:  # 响度呼吸深度上限，均值约 ±1 dB
            at = 1.0 + (at - 1.0) * (0.12 / adev)
        amp_traj = at

    pf = snap.get("partial_freqs")
    f0_ref = float(snap["f0_ref"])
    if pf is not None and np.asarray(pf).size:
        ratio_k = np.asarray(pf, dtype=np.float64)[:K] / f0_ref
    else:
        ratio_k = np.arange(1, K + 1, dtype=np.float64)

    attack = np.asarray(snap.get("attack_res", np.zeros(0)), dtype=np.float64)
    if attack.size:  # 按拨片厚度整形起音颗粒
        from synth.attack_shape import shape_attack

        attack = np.asarray(shape_attack(attack, pick, sr), dtype=np.float64)
        np.nan_to_num(attack, copy=False)  # 被 NaN 污染的颗粒渲染时会被冲成 0，产生突变咔哒

    # 噪声纹理表：固定 seed 预渲染，实时端只查表
    N_env = snap.get("N_env")
    centers = snap.get("noise_centers_hz")
    tlen = int(noise_table_s * sr)
    if N_env is not None and np.asarray(N_env).size and noise_level > 0:
        # 多渲一段 crossfade，把表尾等功率混入表头，循环无缝；否则持续音每越过一次表长
        # 就在回绕处听到一声咔哒
        xf = int(0.05 * sr)
        ext = synth_noise(np.asarray(N_env), sr, tlen + xf, centers_hz=centers, seed=noise_seed)
        noise_table = ext[:tlen].copy()
        ramp = np.sin(np.linspace(0, np.pi / 2, xf)) ** 2  # 等功率 0→1
        noise_table[:xf] = noise_table[:xf] * ramp + ext[tlen : tlen + xf] * ramp[::-1]
    else:
        noise_table = np.zeros(tlen, dtype=np.float32)

    vib = snap.get("vibrato") or {}
    return Instrument(
        H_steady,
        ratio_k,
        attack,
        noise_table,
        f0_ref,
        sr,
        name=snap.get("meta", {}).get("name", "untitled"),
        noise_level=noise_level,
        vib_rate=vib.get("rate_hz"),
        amp_env=amp_env,
        amp_hop=harm_hop,
        spec_traj=spec_traj,
        traj_fps=traj_fps,
        amp_traj=amp_traj,
        amp_tail_ratio=amp_tail_ratio,
        atk_traj=atk_traj,
        atk_fps=sr / max(harm_hop, 1),
    )
