"""音色分析：音频 -> TimbreSnapshot —— FR-2 完成（阶段2 Day3）。

把 track_f0 / harmonics / residual 串成一条线，按素材采集协议从**孤立单音**
提取 Part B 的 TimbreSnapshot（只在 analysis 侧用 librosa；产物纯数据）。
"""
from __future__ import annotations

import numpy as np

from analysis.harmonic import track_f0, harmonics, f0_scalar
from analysis.residual import extract_residual, noise_envelope, attack_residual
from snapshot import make_snapshot

try:
    from quill_config import CFG
    _SR = int(CFG["audio"]["sr"])
    _NH = int(CFG["harmonic"]["n_harmonics"])
    _HOP = int(CFG["harmonic"]["hop"])
except Exception:  # pragma: no cover
    _SR, _NH, _HOP = 44100, 60, 512


def detect_vibrato(f0, times, fmin=4.0, fmax=8.0,
                   min_depth_cents=8.0, prominence=4.0):
    """从 f0 轨迹检测颤音 -> {"rate_hz","depth_cents"} | None。

    思路：f0 转 cents 偏差、去均值、加窗 FFT，在 4-8Hz 找显著峰
    （坑位提示：先抽周期性颤音，再判 slide 单调性——本函数即"抽周期性"那步）。
    """
    f0 = np.asarray(f0, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    good = np.isfinite(f0) & (f0 > 0)
    if good.sum() < 16 or times.size < 2:
        return None
    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return None
    med = np.median(f0[good])
    idx = np.arange(len(f0))
    cents = np.interp(idx, idx[good], 1200.0 * np.log2(f0[good] / med))
    # **线性去趋势**（坑位：先抽周期性再判 slide）——去掉单调滑音的线性分量，
    # 否则 slide 的线性 cents 斜坡会在低频留能量被误判成颤音。颤音是去趋势后仍存的振荡。
    coef = np.polyfit(idx, cents, 1)
    cents = cents - (coef[0] * idx + coef[1])

    n = len(cents)
    w = np.hanning(n)
    C = np.abs(np.fft.rfft(cents * w))
    freqs = np.fft.rfftfreq(n, dt)
    band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if band.size == 0:
        return None
    pk = band[int(np.argmax(C[band]))]
    amp_cents = 2.0 * C[pk] / np.sum(w)         # 单边幅度（峰值偏移）
    bg = np.median(C[(freqs > 1.0) & (freqs < 20.0)]) + 1e-9
    if amp_cents < min_depth_cents or C[pk] < prominence * bg:
        return None
    return {"rate_hz": float(freqs[pk]), "depth_cents": float(amp_cents)}


def _snap_to_hint(f0, hint, max_dev_cents: float = 50.0):
    """用已知基频 hint 修正 f0 轨迹的八度/谐波倍频错（factor∈{1/3,1/2,1,2,3}）。

    数据集单音自带真值音高时用——低音(贝斯/大号)PESTO 易锁到 2/3 次谐波，整条快照
    据此建错谐波；按 median(f0)/hint 选最近简单比值整体校正，保留相对轮廓（颤音）。
    极低音(<40Hz)PESTO 直接失效、误差非简单比值——校正后仍偏离 >max_dev_cents 则退化为
    hint 常量（保证谐波定位正确，代价是丢颤音；深贝斯本就近乎无颤音）。
    """
    f0 = np.asarray(f0, dtype=np.float64).copy()
    if hint <= 0:
        return f0
    good = np.isfinite(f0) & (f0 > 0)
    if good.sum() == 0:
        return np.full_like(f0, hint)                # 全无有效 f0 -> 常量 hint
    med = float(np.median(f0[good]))
    if med > 0:
        cands = np.array([1 / 3, 1 / 2, 1.0, 2.0, 3.0])
        best = cands[int(np.argmin(np.abs(np.log((med / hint) / cands))))]
        if abs(np.log(best)) > 1e-6:
            f0[good] = f0[good] / best
    medc = float(np.median(f0[good])) if good.any() else 0.0
    if medc <= 0 or abs(1200.0 * np.log2(max(medc, 1e-6) / hint)) > max_dev_cents:
        f0[:] = hint                                 # 极低音兜底：常量真值
    return f0


def select_stable_window(y, sr, f0, f0_times, min_s=0.30, max_s=2.0,
                         stable_cents=150.0):
    # stable_cents 150(7.4):分析端逐帧跟 f0,表达性演唱(vibrato/起伏)完全可分析;
    # 70 音分会把"有性格"的段在资格赛就刷掉,只剩死板哼鸣(lead pad 问题的帮凶)。
    """从录音里自动挑最适合取色的**稳定音高窗口**(7.4)。

    用户自然录音(唱一句/哼一段)里,音高会随旋律/字词移动——但取色只需其中一个
    "延长的音"(某个持续元音)。这里找最长的连续「有音高 + 局部音高稳(含 vibrato)
    + 有能量」区间返回其采样范围;找不到(纯噪声/快速说话/无持续音)返回 None
    (交由守门拒收)。这让"record a few seconds of any sound"成立,而非强制单音。
    """
    T = len(f0)
    if T < 4 or len(f0_times) != T:
        return None
    voiced = np.isfinite(f0) & (f0 > 0)
    if voiced.sum() < 4:
        return None
    med = float(np.median(f0[voiced]))
    cents = np.where(voiced, 1200.0 * np.log2(np.where(voiced, f0, med) / med), np.nan)
    dt = float(np.median(np.diff(f0_times))) if T > 1 else 0.01
    half = max(1, int(0.12 / max(dt, 1e-4)))          # 局部中位窗 ±120ms(容 vibrato)
    win = max(1, int(dt * sr))
    dev = np.full(T, np.inf)
    energy = np.zeros(T)
    for i in range(T):
        a, b = max(0, i - half), min(T, i + half + 1)
        seg = cents[a:b]; seg = seg[np.isfinite(seg)]
        if seg.size and np.isfinite(cents[i]):
            dev[i] = abs(cents[i] - np.median(seg))
        c = int(f0_times[i] * sr)
        e0, e1 = max(0, c - win), min(len(y), c + win)
        if e1 > e0:
            energy[i] = float(np.sqrt(np.mean(np.asarray(y[e0:e1], np.float64) ** 2)))
    emax = float(energy.max()) + 1e-12
    good = voiced & (dev < stable_cents) & (energy > 0.15 * emax)
    # 候选段打分 = 时长^0.3 × 能量 × 音色丰富度(高频带占比)——只按"音高最稳"选会
    # 系统性偏爱最暗最没个性的哼鸣段(7.4 UAT 实锤:12s 录音里有丰富人声,却每次
    # 都选中闭口哼,取出的全是"lead pad"),用户唱的精华在"稳定且有性格"的段里。
    def _richness(a_i, b_i):
        c = int((f0_times[a_i] + f0_times[b_i]) / 2 * sr)
        seg = np.asarray(y[max(0, c - sr // 2): c + sr // 2], np.float64)
        if len(seg) < 2048:
            return 0.3
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        fr = np.fft.rfftfreq(len(seg), 1.0 / sr)
        tot = float((S[(fr > 60) & (fr < 5000)] ** 2).sum()) + 1e-12
        hi = float((S[(fr > 700) & (fr < 5000)] ** 2).sum())
        return hi / tot
    runs = []
    i = 0
    while i < T:
        if good[i]:
            j = i
            while j + 1 < T and good[j + 1]:
                j += 1
            if (f0_times[j] - f0_times[i]) >= min_s:
                dur = float(f0_times[j] - f0_times[i])
                erg = float(energy[i:j + 1].mean()) / emax
                score = (dur ** 0.3) * erg * (0.25 + _richness(i, j))
                runs.append((score, i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return None
    _, best_a, best_b = max(runs)
    a_s = int(f0_times[best_a] * sr)
    b_s = int(min(len(y), f0_times[best_b] * sr + win))
    if (b_s - a_s) / sr > max_s:                       # 过长 → 取能量峰附近 max_s
        pk = best_a + int(np.argmax(energy[best_a:best_b + 1]))
        c = int(f0_times[pk] * sr)
        a_s = max(a_s, c - int(max_s * sr / 2))
        b_s = min(b_s, c + int(max_s * sr / 2))
    return a_s, b_s


def analyze_timbre(y, sr: int = _SR, name: str = "untitled",
                   source_file: str = "", n_harmonics: int = _NH,
                   n_mels: int = 40, quasi_harmonic: bool = False,
                   f0_hint: float | None = None, denoise: bool = False,
                   window: str | None = None, cross_check: bool = True) -> dict:
    """孤立单音 -> TimbreSnapshot（Part B）。

    quasi_harmonic=True 用于酒杯/碗：放宽峰值搜索窗并记录实测 partial_freqs。
    f0_hint：已知真值基频(Hz)，修正低音区 PESTO 的八度/谐波倍频错（数据集语料构建用）。
    denoise=True(取色地板 P3,app 捕捉路径用)：噪声包络逐带减去稳态底噪
    ——房间嗡声/嘶声是恒定的(≈每带时间下分位),减掉;呼吸/弓噪随演奏起伏,保留。
    默认 False 保持库行为(合成测试信号的稳态噪声是有意成分)。
    """
    y = np.asarray(y, dtype=np.float32)
    f0, conf, ftimes = track_f0(y, sr=sr, cross_check=cross_check)
    if f0_hint is not None and f0_hint > 0:
        f0 = _snap_to_hint(f0, float(f0_hint))
    # 自动选稳定窗(window="auto",app 取色路径):自然录音里挑出可取色的持续音段。
    window_found = None
    window_mid_cut = False
    if window == "auto":
        w = select_stable_window(y, sr, f0, ftimes)
        window_found = w is not None
        if w is not None:
            a, b = w
            # 切口在录音中段(非自然起音处)→ 该段**没有真实起音**,残差开头只会是
            # 谐波模型在人工切口的失配瞬态(边界垃圾)——若存成起音颗粒,每键重放一声"咔"
            # (7.4 UAT 用户实锤:颗粒/谐波比 0.25 vs 健康 0.02)。此时跳过起音提取。
            # 但拨弦的稳定窗天然始于起音之后(起音期音高不稳)——回看 0.45s:若存在
            # 静默→爆发的真实起音,把窗回退含住它,颗粒照常提取(拨弦味的来源)。
            window_mid_cut = a > int(0.12 * sr)
            if window_mid_cut:
                lb = np.asarray(y[max(0, a - int(0.45 * sr)):a], np.float64)
                head = np.asarray(y[a:a + int(0.25 * sr)], np.float64)
                if lb.size > int(0.1 * sr) and head.size:
                    hopq = 256
                    nq = len(lb) // hopq
                    rq = np.sqrt((lb[:nq * hopq].reshape(nq, hopq) ** 2).mean(1))
                    head_rms = float(np.sqrt((head ** 2).mean())) + 1e-12
                    if rq.size and float(rq.min()) < 0.15 * head_rms:
                        onset_i = (a - nq * hopq) + int(np.argmin(rq)) * hopq
                        a = max(0, onset_i)              # 回退到静默点,含住真实起音
                        window_mid_cut = False
            fm = (ftimes >= a / sr) & (ftimes < b / sr)
            if int(fm.sum()) >= 4:
                y = np.ascontiguousarray(y[a:b])
                f0, conf, ftimes = f0[fm], conf[fm], ftimes[fm] - a / sr
    f0_ref = f0_scalar(f0)
    # 乐音守门指标(7.4;选窗后计算 → 唱旋律/句子里的稳定音段 cv 会降到乐音水平)。
    # 实测分离度:乐音 cv≤0.008;白噪 0.44/伪语音 0.39/复音和弦 0.21(→引导走 Mode 3)。
    _good = np.isfinite(f0) & (f0 > 0)
    voiced_ratio = float(_good.mean()) if f0.size else 0.0
    f0_cv = (float(np.std(f0[_good]) / np.mean(f0[_good]))
             if _good.sum() > 3 else 9.9)

    H, htimes, info = harmonics(y, f0, sr=sr, f0_times=ftimes,
                               n_harmonics=n_harmonics,
                               measure_partials=quasi_harmonic)
    f0_frame = info["f0_per_frame"]
    partial_freqs = info["partial_freqs"] if quasi_harmonic else None

    res = extract_residual(y, H, f0_frame, sr=sr, n_fft=info["n_fft"],
                           hop=_HOP, partial_freqs=partial_freqs, f0_ref=f0_ref)
    N_env, centers = noise_envelope(res, sr=sr, n_mels=n_mels)
    if denoise and N_env.ndim == 2 and N_env.shape[1] >= 8:
        floor = np.percentile(N_env, 20, axis=1, keepdims=True)   # 每带稳态底噪(时间下分位)
        N_env = np.sqrt(np.maximum(N_env ** 2 - (1.1 * floor) ** 2, 0.0))  # 功率域谱减
    attack = (np.zeros(0, dtype=np.float32) if window_mid_cut
              else attack_residual(res, sr=sr))
    vibrato = detect_vibrato(f0, ftimes)

    return make_snapshot(
        f0_ref, H, N_env, attack, partial_freqs=partial_freqs,
        vibrato=vibrato, noise_centers_hz=centers, name=name,
        source_file=source_file, sr=sr,
        harm_hop=_HOP, harm_nfft=int(info["n_fft"]),
        voiced_ratio=voiced_ratio, f0_cv=f0_cv,
        window_found=bool(window_found) if window_found is not None else None)
