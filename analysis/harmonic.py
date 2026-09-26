"""f0 估计 + 谐波幅度提取 —— FR-2 前半（阶段1 Day1 晚 + Day2）。

* track_f0(y)         : PESTO 主力 + librosa.pyin 交叉校验，输出 (f0, conf, times)
* harmonics(y, f0)    : STFT(4096/512) 上对 k<=60 个谐波做峰值跟踪 -> H[60, T]

谐波峰值估计采用抛物线插值（McAulay-Quatieri 1986）：已知 f0，无需完整
partial tracking，只在 k*f0±search_cents 窗内取局部峰并对 log 幅度做二次插值，
既校正 Hann 窗扇贝损失（<1dB 单测的前提），又顺带测得准谐波声源的实测部分音频率。
"""
from __future__ import annotations

import warnings

import numpy as np

try:
    from quill_config import CFG
except Exception:  # pragma: no cover
    CFG = {
        "audio": {"sr": 44100},
        "f0": {"step_size_ms": 10.0, "pesto_model": "mir-1k_g7",
               "conf_threshold": 0.6, "octave_tol_cents": 50.0,
               "pyin_fmin": 55.0, "pyin_fmax": 2000.0},
        "harmonic": {"n_harmonics": 60, "n_fft": 4096, "hop": 512,
                     "search_cents": 30.0, "quasi_search_cents": 80.0,
                     "low_f0_nfft_threshold": 100.0, "low_f0_nfft": 8192,
                     "nyquist_margin_hz": 50.0},
    }

_SR = int(CFG["audio"]["sr"])
_FCFG = CFG["f0"]
_HCFG = CFG["harmonic"]


# --------------------------------------------------------------------------- #
# f0 估计
# --------------------------------------------------------------------------- #
def track_f0(y: np.ndarray, sr: int = _SR,
             step_size_ms: float | None = None,
             conf_threshold: float | None = None,
             cross_check: bool = True):
    """估计逐帧基频。

    返回 (f0, conf, times)：
      f0    float32[T]  Hz，低置信度帧置 NaN（坑位：f0 倍频常伴 conf 跌落）
      conf  float32[T]  置信度 0-1
      times float32[T]  帧中心秒

    主力 PESTO（ISMIR'23，<30k 参数实时）；`cross_check=True` 时用
    librosa.pyin 比对中位数，倍频错误（≈2x / 0.5x）发 warning。
    PESTO 不可用时自动回落到 pyin（实施计划既定退路）。
    """
    step = float(step_size_ms if step_size_ms is not None
                 else _FCFG["step_size_ms"])
    thr = float(conf_threshold if conf_threshold is not None
                else _FCFG["conf_threshold"])
    y = np.asarray(y, dtype=np.float32)

    f0, conf, times, backend = _pesto_f0(y, sr, step)
    if f0 is None:  # PESTO 不可用 -> pyin 兜底
        f0, conf, times = _pyin_f0(y, sr, step)
        backend = "pyin"
        warnings.warn("PESTO 不可用，已回落到 librosa.pyin", RuntimeWarning)
    elif cross_check:
        _octave_cross_check(f0, conf, y, sr, thr)

    # 置信度过滤：低于阈值的帧 f0 置 NaN，交给下游忽略
    f0 = f0.astype(np.float32).copy()
    f0[conf < thr] = np.nan
    return f0, conf.astype(np.float32), times.astype(np.float32)


def _pesto_f0(y, sr, step):
    try:
        import torch
        import pesto
    except Exception:
        return None, None, None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts_ms, pitch_hz, conf, _act = pesto.predict(
                torch.from_numpy(y), sr, step_size=step,
                model_name=_FCFG["pesto_model"])
        f0 = pitch_hz.detach().cpu().numpy().astype(np.float32)
        conf = conf.detach().cpu().numpy().astype(np.float32)
        times = ts_ms.detach().cpu().numpy().astype(np.float32) / 1000.0
        return f0, conf, times, "pesto"
    except Exception as exc:  # 权重下载失败/形状异常 -> 让上层兜底
        warnings.warn(f"PESTO 推断异常：{exc}", RuntimeWarning)
        return None, None, None, None


def _pyin_f0(y, sr, step):
    import librosa
    hop = max(1, int(round(step / 1000.0 * sr)))
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, sr=sr, fmin=float(_FCFG["pyin_fmin"]), fmax=float(_FCFG["pyin_fmax"]),
        hop_length=hop)
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    conf = np.asarray(voiced_prob, dtype=np.float32)
    times = (np.arange(len(f0)) * hop / sr).astype(np.float32)
    return f0, conf, times


def _octave_cross_check(f0, conf, y, sr, thr):
    """用 pyin 中位数比对 PESTO，疑似倍频错误时 warning（不擅自改值）。"""
    try:
        import librosa
        f0_py, _, vp = librosa.pyin(
            y, sr=sr, fmin=float(_FCFG["pyin_fmin"]),
            fmax=float(_FCFG["pyin_fmax"]))
    except Exception:
        return
    m_pesto = _voiced_median(f0, conf >= thr)
    m_pyin = _voiced_median(f0_py, np.asarray(vp) >= 0.5)
    if not (np.isfinite(m_pesto) and np.isfinite(m_pyin) and m_pyin > 0):
        return
    cents = 1200.0 * np.log2(m_pesto / m_pyin)
    if abs(cents) > float(_FCFG["octave_tol_cents"]):
        warnings.warn(
            f"f0 交叉校验失配：PESTO≈{m_pesto:.1f}Hz vs pyin≈{m_pyin:.1f}Hz "
            f"({cents:+.0f}cents)，疑似倍频错误", RuntimeWarning)


def _voiced_median(f0, mask):
    vals = np.asarray(f0)[np.asarray(mask) & np.isfinite(np.asarray(f0))]
    vals = vals[vals > 0]
    return float(np.median(vals)) if vals.size else np.nan


def f0_scalar(f0: np.ndarray) -> float:
    """有声帧 f0 中位数，作为 TimbreSnapshot 的 f0_ref。"""
    return _voiced_median(f0, np.ones_like(f0, dtype=bool))


# --------------------------------------------------------------------------- #
# 谐波幅度提取
# --------------------------------------------------------------------------- #
def harmonics(y: np.ndarray, f0, sr: int = _SR, f0_times=None,
              n_harmonics: int | None = None, n_fft: int | None = None,
              hop: int | None = None, search_cents: float | None = None,
              partial_freqs=None, measure_partials: bool = False,
              min_prominence: float = 2.0):
    """逐帧提取前 `n_harmonics` 个谐波的幅度包络 H[K, T]。

    参数
    ----
    f0 : 标量 或 数组
        标量 -> 全程恒定基频；数组 -> 逐帧基频。配 `f0_times` 时按时间插值到
        STFT 帧中心；否则按帧序号重采样。NaN（无声/低置信）帧用最近有声值填充。
    partial_freqs : float[K] 或 None
        准谐波声源（酒杯/碗）的实测部分音频率（@f0_ref）。给定则目标频率=
        partial_freqs * (f0/f0_ref)，否则严格 k*f0。
    measure_partials : bool
        True 则放宽搜索窗到 quasi_search_cents 并返回各谐波实测频率均值
        （写入 TimbreSnapshot.partial_freqs，解决高位泛音偏离 k*f0 的丢失）。

    返回
    ----
    H : float32[K, T]   谐波幅度（真实正弦幅度，已校正 Hann 窗相干增益）
    times : float32[T]  帧中心秒
    info : dict         {"partial_freqs": float32[K] 或 None,
                         "f0_per_frame": float32[T], "n_fft": int}
    """
    import librosa

    y = np.asarray(y, dtype=np.float32)
    K = int(n_harmonics if n_harmonics is not None else _HCFG["n_harmonics"])
    hop = int(hop if hop is not None else _HCFG["hop"])
    cents = float(search_cents if search_cents is not None else (
        _HCFG["quasi_search_cents"] if measure_partials else _HCFG["search_cents"]))

    # 逐帧 f0（用于决定低音是否升 n_fft）
    f0_arr = np.atleast_1d(np.asarray(f0, dtype=np.float64))
    f0_med = _voiced_median(f0_arr, np.ones_like(f0_arr, dtype=bool))

    if n_fft is None:
        n_fft = int(_HCFG["n_fft"])
        if np.isfinite(f0_med) and f0_med < float(_HCFG["low_f0_nfft_threshold"]):
            n_fft = int(_HCFG["low_f0_nfft"])  # 坑位：低音谐波间距不足 -> 加大窗
    n_fft = int(n_fft)

    win = np.hanning(n_fft).astype(np.float64)  # 与 librosa 默认 'hann' 一致
    coherent_gain = win.sum()  # 正弦幅度复原常数：A = 2|X_peak|/cg

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    mag = np.abs(D).astype(np.float64)            # [n_fft/2+1, T]
    n_bins, T = mag.shape
    times = (np.arange(T) * hop / sr).astype(np.float32)
    bin_hz = sr / n_fft
    nyquist = sr / 2.0
    nyq_margin = float(_HCFG["nyquist_margin_hz"])

    f0_frame = _resample_f0(f0_arr, f0_times, times, sr, hop)

    # 准谐波基准频率（相对 f0_ref 的比例），默认严格 k*f0
    if partial_freqs is not None:
        pf = np.asarray(partial_freqs, dtype=np.float64)[:K]
        f0_ref = f0_med if np.isfinite(f0_med) and f0_med > 0 else f0_frame[0]
        ratio_k = pf / f0_ref  # 目标 = ratio_k * f0_frame
    else:
        ratio_k = np.arange(1, K + 1, dtype=np.float64)  # 严格谐波 k

    H = np.zeros((K, T), dtype=np.float32)
    measured = np.zeros((K, T), dtype=np.float64)
    measured_cnt = np.zeros(K, dtype=np.float64)
    fac = 2.0 ** (cents / 1200.0)  # 搜索窗半宽比例

    for ti in range(T):
        f0_t = f0_frame[ti]
        if not (np.isfinite(f0_t) and f0_t > 0):
            continue
        col = mag[:, ti]
        for k in range(K):
            target = ratio_k[k] * f0_t
            if target >= nyquist - nyq_margin:  # 坑位：k*f0>=Nyquist 截断
                break
            lo_bin = int(np.floor((target / fac) / bin_hz))
            hi_bin = int(np.ceil((target * fac) / bin_hz))
            lo_bin = max(1, lo_bin)
            hi_bin = min(n_bins - 2, hi_bin)
            if hi_bin < lo_bin:
                continue
            seg = col[lo_bin:hi_bin + 1]
            j = lo_bin + int(np.argmax(seg))
            # 突出度门控(7.4):真谐波=从邻域噪声地板凸起的尖峰(≥6dB);噪声采收的"峰"
            # 与邻域齐平。不检验时,嘈杂录音的噪声地板会被当高次谐波采收,合成端用**稳定
            # 正弦**重放=取色杂音(Air/Noise 旋钮管不到谐波路径)。邻域=主瓣外、相邻谐波
            # 瓣内的两侧带(按谐波间距自适应)。
            spacing = f0_t / bin_hz
            o_lo = max(3, int(0.15 * spacing))
            o_hi = max(o_lo + 2, int(0.45 * spacing))
            a = max(1, j - o_hi)
            b = max(a + 1, j - o_lo + 1)
            c = min(n_bins - 1, j + o_lo)
            d = min(n_bins, j + o_hi + 1)
            nb = np.concatenate([col[a:b], col[c:d]])
            floor = float(np.median(nb)) if nb.size else 0.0
            if min_prominence > 0 and col[j] < min_prominence * floor:
                continue                      # 突出度不足 → 视为噪声地板,该帧该谐波不采
            amp, true_bin = _parabolic_peak(col, j, coherent_gain)
            H[k, ti] = amp
            measured[k, ti] += true_bin * bin_hz
            measured_cnt[k] += 1.0

    info = {"f0_per_frame": f0_frame.astype(np.float32), "n_fft": n_fft,
            "partial_freqs": None}
    if measure_partials:
        with np.errstate(invalid="ignore", divide="ignore"):
            pf_out = np.where(measured_cnt > 0,
                              measured.sum(axis=1) / np.maximum(measured_cnt, 1),
                              np.arange(1, K + 1) * (f0_med if np.isfinite(f0_med) else 0.0))
        info["partial_freqs"] = pf_out.astype(np.float32)
    return H, times, info


def _parabolic_peak(col: np.ndarray, j: int, coherent_gain: float):
    """对 log 幅度在峰 bin j 处做二次插值，返回 (正弦幅度, 真实 bin 位置)。

    标准 McAulay-Quatieri / Smith-Serra 峰值插值，校正 Hann 窗扇贝损失。
    A = 2 * |X_peak| / coherent_gain（librosa.stft 不归一化窗）。
    """
    eps = 1e-12
    if j <= 0 or j >= len(col) - 1:
        peak_lin = col[j]
        return float(2.0 * peak_lin / coherent_gain), float(j)
    a = 20.0 * np.log10(col[j - 1] + eps)
    b = 20.0 * np.log10(col[j] + eps)
    c = 20.0 * np.log10(col[j + 1] + eps)
    denom = (a - 2.0 * b + c)
    p = 0.5 * (a - c) / denom if abs(denom) > eps else 0.0
    p = float(np.clip(p, -0.5, 0.5))
    peak_db = b - 0.25 * (a - c) * p
    peak_lin = 10.0 ** (peak_db / 20.0)
    return float(2.0 * peak_lin / coherent_gain), float(j + p)


def _resample_f0(f0_arr, f0_times, frame_times, sr, hop):
    """把 f0 重采样到 STFT 帧中心；NaN 用最近有声值填充。"""
    f0_arr = np.asarray(f0_arr, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if f0_arr.size == 1:
        return np.full(frame_times.shape, f0_arr[0], dtype=np.float64)

    filled = _fill_nan_nearest(f0_arr)
    if f0_times is not None:
        src_t = np.asarray(f0_times, dtype=np.float64)
    else:
        # 假定 f0 帧率与自身长度均匀分布在信号时长上
        total = frame_times[-1] if frame_times.size else (len(filled) - 1)
        src_t = np.linspace(0.0, total, num=len(filled))
    return np.interp(frame_times, src_t, filled)


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).copy()
    good = np.isfinite(a) & (a > 0)
    if not good.any():
        return np.full_like(a, np.nan)
    idx = np.arange(len(a))
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a
