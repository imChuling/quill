"""演奏法分类 —— FR-6 基础（阶段5 Day2-3）。

规则优先级树（阈值全部读 config.yaml，可调）：
    同窗 onset≥3 且间隔<80ms                 -> strum
    |f0 净斜率|>阈 且 持续>60ms（且非颤音）   -> slide   (Basic Pitch bend 为直接证据)
    与前音衔接、无新强起音、幅度连续、音程小   -> legato
    log_attack<25ms 且 质心>2kHz             -> pluck
    其余                                      -> sustain；4-8Hz f0 峰显著 -> +vibrato

坑位：slide 与 vibrato 混淆 -> 先抽 vibrato（周期性，看 vibrato 字段/净斜率≈0），
再判 slide（单调性，净斜率大）。本树先判 strum/slide-with-vibrato-guard。

M7 随机森林（sklearn）作兜底 + 规则 vs 森林 ablation（实施计划 Day3）。
分类升级路线：规则树 -> 随机森林 -> MERT 探针（阶段8）。
"""
from __future__ import annotations

import numpy as np

try:
    from quill_config import CFG
    _A = CFG.get("articulation", {})
except Exception:  # pragma: no cover
    _A = {}

_LABELS = ("pluck", "slide", "legato", "strum", "sustain")

_DEF = {
    "strum_window_ms": 80.0, "strum_min_onsets": 3,
    "slide_slope_cents_s": 500.0, "slide_min_dur_ms": 60.0,
    "slide_monotonic_frac": 0.70,
    "legato_max_gap_ms": 60.0, "legato_pitch_jump_max_st": 7.0,
    "pluck_attack_ms": 25.0, "pluck_centroid_hz": 2000.0,
    "vibrato_rate_lo": 4.0, "vibrato_rate_hi": 8.0,
}


def _f0_monotonic_frac(track):
    """f0 轨迹同向步占比 = max(上行比, 下行比)。真滑音≈1；八度跳变/噪声≈0.5。"""
    if len(track) < 3:
        return 0.0
    hz = np.array([p[1] for p in track], dtype=np.float64)
    d = np.diff(hz)
    d = d[np.abs(d) > 1e-6]
    if d.size == 0:
        return 0.0
    up = float(np.mean(d > 0))
    return max(up, 1.0 - up)


def _cfg(key):
    return float(_A.get(key, _DEF[key]))


def _is_vibrato(ev):
    """事件是否带 4-8Hz 颤音（先抽周期性，坑位）。"""
    v = ev.get("vibrato")
    if not v:
        return False
    r = v.get("rate_hz", 0.0)
    return _cfg("vibrato_rate_lo") <= r <= _cfg("vibrato_rate_hi")


def _strum_windows(events):
    """返回属于 strum 的事件 id 集合：滑窗内 onset 数≥阈、间隔<window。"""
    win = _cfg("strum_window_ms") / 1000.0
    need = int(_A.get("strum_min_onsets", _DEF["strum_min_onsets"]))
    ev = sorted(events, key=lambda e: e["t_on"])
    strum_ids = set()
    n = len(ev)
    for i in range(n):
        j = i
        while j < n and ev[j]["t_on"] - ev[i]["t_on"] < win:
            j += 1
        if j - i >= need:                       # [i, j) 同窗 onset 数达标
            for k in range(i, j):
                strum_ids.add(ev[k]["id"])
    return strum_ids


def classify_events(events, prev_aware=True, valid_labels=None):
    """规则优先级树逐事件打标，原地写 ev["label"]（并返回 events）。

    valid_labels: 限定输出标签空间（如 IDMT 等孤立单音数据集只有 pluck/slide/sustain，
    无 legato/strum——此时关掉 strum/legato 检测、pluck 作为默认，避免对数据集不存在的
    类做无意义预测。None=完整 5 类（演奏phrasing 域）。
    """
    if not events:
        return events
    if valid_labels is None:
        valid_labels = set(_LABELS)
    else:
        valid_labels = set(valid_labels)
    use_strum = "strum" in valid_labels
    use_legato = "legato" in valid_labels and prev_aware
    ev_sorted = sorted(events, key=lambda e: e["t_on"])
    strum_ids = _strum_windows(events) if use_strum else set()

    slope_thr = _cfg("slide_slope_cents_s")
    slide_min_dur = _cfg("slide_min_dur_ms") / 1000.0
    gap_max = _cfg("legato_max_gap_ms") / 1000.0
    jump_max = _cfg("legato_pitch_jump_max_st")
    atk_thr = np.log10(_cfg("pluck_attack_ms") / 1000.0)   # log_attack 阈
    cen_thr = _cfg("pluck_centroid_hz")

    restricted = not (use_strum or use_legato)   # 孤立单音域：pluck 为默认
    prev = None
    for ev in ev_sorted:
        f = ev.get("features", {})
        vib = _is_vibrato(ev)
        slope = abs(float(f.get("f0_slope_cents_s", 0.0) or 0.0))
        dur = ev["t_off"] - ev["t_on"]
        log_atk = f.get("log_attack")
        cen = f.get("centroid_mean")
        mono = _f0_monotonic_frac(ev.get("f0_track", []))
        # slide：净斜率大 + 持续够 + 单调（坑位）+ 不是颤音（先抽周期性，再判单调性）
        is_slide = ((not vib) and slope > slope_thr and dur > slide_min_dur
                    and mono >= _cfg("slide_monotonic_frac"))

        if use_strum and ev["id"] in strum_ids:
            label = "strum"
        elif is_slide:
            label = "slide"
        elif vib:                                  # 颤音 -> sustain（先抽周期性）
            label = "sustain"
        elif use_legato and prev is not None and \
                _is_legato(ev, prev, gap_max, jump_max, atk_thr):
            label = "legato"
        elif restricted:                           # 孤立单音：非滑非颤即 pluck
            label = "pluck"
        elif (log_atk is not None and np.isfinite(log_atk) and log_atk < atk_thr
              and cen is not None and np.isfinite(cen) and cen > cen_thr):
            label = "pluck"
        else:
            label = "sustain"

        ev["label"] = label
        prev = ev
    return events


def _is_legato(ev, prev, gap_max, jump_max, atk_thr):
    """legato：与前音衔接（gap 小）、音程在范围内、本音无强起音（attack 慢）、幅度连续。"""
    gap = ev["t_on"] - prev["t_off"]
    if gap > gap_max:
        return False
    p0, p1 = prev.get("pitch_midi"), ev.get("pitch_midi")
    if p0 is None or p1 is None:
        return False
    if abs(p1 - p0) > jump_max or abs(p1 - p0) < 0.5:
        return False          # 同音不算 legato；跨度过大更像独立起音
    log_atk = ev.get("features", {}).get("log_attack")
    soft = (log_atk is None) or (not np.isfinite(log_atk)) or (log_atk >= atk_thr)
    # 幅度连续：本音起始力度与前音收尾相近（无重击）
    cont = ev.get("velocity_est", 0.0) <= prev.get("velocity_est", 1.0) * 1.3 + 0.1
    return bool(soft and cont)


# --------------------------------------------------------------------------- #
# M7 随机森林兜底（规则 vs 森林 ablation）
# --------------------------------------------------------------------------- #
_FEATURE_KEYS = ("log_attack", "centroid_mean", "centroid_slope",
                 "f0_slope_cents_s", "polyphony", "hnr")


def feature_vector(ev):
    """事件 -> 数值特征向量（森林输入）。"""
    f = ev.get("features", {})
    vec = []
    for k in _FEATURE_KEYS:
        v = f.get(k)
        vec.append(float(v) if v is not None and np.isfinite(v) else 0.0)
    vec.append(1.0 if _is_vibrato(ev) else 0.0)
    vec.append(float(ev["t_off"] - ev["t_on"]))     # 时长
    return np.asarray(vec, dtype=np.float64)


def train_forest(events, labels, n_estimators=100, seed=0):
    """IDMT/自录标注秒级训练随机森林（M7 兜底）。返回 sklearn 分类器。"""
    from sklearn.ensemble import RandomForestClassifier
    X = np.array([feature_vector(e) for e in events])
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                 class_weight="balanced")
    clf.fit(X, labels)
    return clf


def forest_predict(clf, events):
    X = np.array([feature_vector(e) for e in events])
    return list(clf.predict(X))
