"""MIDI 方言适配层:任何方言进来,统一 IR 出去。

SMF 容器只有一种,但演奏法信息按乐器社区的约定藏在不同位置(方言)。
本模块自动检测方言并收敛到 Quill 的统一中间表示:

    [{"midi": int, "dur": float, "art": int, "onset": float}, ...]
    art: 0=plain 1=vibrato 2=slide (与 neural/perform.py 一致)

设计约束(docs/roadmap_midi_dialects.md):
- 引擎零改动:本模块只产出 score,不 import neural.*
- 自动检测优先,置信度低时报告里给出候选让上层 UI 决定
- 鼓是明确非目标:检测到纯鼓文件直接拒绝

方言与适配动作:
  piano         直通(velocity/CC64 不携带 Quill 需要的信息)
  keyswitch     剔除音域外触发音符,latching 映射到 art 标签
  multichannel  吉他/MPE:按乐器轨合并,bend 曲线转 art 标签
  mono_lead     单声部 + bend:bend 曲线转 art 标签
  breath        CC2 连续曲线:提取逐音符能量提示(cc_energy,引擎忽略额外键)
  drums         拒绝(pitched monophonic 管线不适用)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pretty_midi

# ---- 常量 ---- #
BEND_RANGE_ST = 2.0        # 标准 GM 弯音范围 ±2 半音(文件可覆写,SMF 里常缺失)
KS_GAP_ST = 12             # keyswitch 簇与演奏音域之间的最小间隔(半音)
KS_MAX_DUR = 0.30          # keyswitch 音符时值上限(秒)
SLIDE_MIN_ST = 1.0         # bend 判 slide 的最小净位移(半音)
SLIDE_MONO_FRAC = 0.8      # bend 判 slide 的单调步比例
VIB_RATE_LO, VIB_RATE_HI = 3.0, 12.0   # bend 判 vibrato 的频率窗(Hz)
VIB_MIN_AMP_ST = 0.08      # bend 判 vibrato 的最小峰值幅度(半音,≈8 cents)
CHORD_EPS = 0.010          # 同一 onset 判和弦的时间容差(秒)

# keyswitch 簇内偏移 → art 标签(可由调用方覆写)
DEFAULT_KS_MAP = {0: 0, 1: 1, 2: 2}    # 簇底=plain, +1=vibrato, +2=slide

ART_NAMES = {0: "plain", 1: "vibrato", 2: "slide"}


@dataclass
class ImportReport:
    dialect: str = "piano"
    confidence: float = 1.0
    candidates: list = field(default_factory=list)   # 低置信度时的备选方言
    n_notes_in: int = 0
    n_notes_out: int = 0
    n_channels_merged: int = 0
    n_keyswitches_stripped: int = 0
    ks_map_used: dict = field(default_factory=dict)  # 触发音高 → art 名
    n_bend_slide: int = 0
    n_bend_vibrato: int = 0
    n_overlaps_trimmed: int = 0
    n_chord_notes_dropped: int = 0
    n_drum_tracks_dropped: int = 0
    cc_energy_source: str = ""                       # "cc1"/"cc2"/"cc11"/""
    warnings: list = field(default_factory=list)



class DrumFileError(ValueError):
    """纯鼓文件:note number 是乐器身份而非音高,管线明确不支持。"""


# ---- 检测 ---- #

def _melodic_instruments(pm):
    return [ins for ins in pm.instruments if not ins.is_drum and ins.notes]


def _find_keyswitch_cluster(notes):
    """在单乐器音符集里找 keyswitch 簇。

    判据:按音高排序后存在 ≥KS_GAP_ST 半音的断档,断档以下的音符
    全部短促(≤KS_MAX_DUR)。返回 (簇内音高集合, 簇底音高) 或 (set(), None)。
    """
    if len(notes) < 4:
        return set(), None
    pitches = sorted({n.pitch for n in notes})
    for i in range(len(pitches) - 1):
        if pitches[i + 1] - pitches[i] >= KS_GAP_ST:
            low = set(pitches[: i + 1])
            low_notes = [n for n in notes if n.pitch in low]
            main_notes = [n for n in notes if n.pitch not in low]
            if not low_notes or not main_notes:
                continue
            if all(n.end - n.start <= KS_MAX_DUR for n in low_notes) and \
                    len(main_notes) >= len(low_notes):
                return low, min(low)
    return set(), None


def _cc_density(ins, number):
    ccs = [c for c in ins.control_changes if c.number == number]
    if not ins.notes or len(ccs) < 4:
        return 0.0
    span = max(n.end for n in ins.notes) - min(n.start for n in ins.notes)
    return len(ccs) / max(span, 1e-6)


def detect_dialect(pm):
    """返回 (dialect, confidence, candidates, signals_dict)。"""
    melodic = _melodic_instruments(pm)
    drums = [ins for ins in pm.instruments if ins.is_drum and ins.notes]
    signals = {"n_melodic_tracks": len(melodic), "n_drum_tracks": len(drums)}

    if not melodic:
        if drums:
            return "drums", 1.0, [], signals
        return "piano", 0.3, ["empty"], signals

    n_bends = sum(len(ins.pitch_bends) for ins in melodic)
    signals["n_pitch_bends"] = n_bends
    cc2 = max((_cc_density(ins, 2) for ins in melodic), default=0.0)
    signals["cc2_density"] = round(cc2, 2)

    if len(melodic) >= 3:
        # 多轨且带 bend → 吉他/MPE 家族;不带 bend 也按多轨合并处理
        conf = 0.9 if n_bends > 0 else 0.6
        return "multichannel", conf, (["piano"] if conf < 0.7 else []), signals

    ks, _base = _find_keyswitch_cluster(melodic[0].notes) if len(melodic) == 1 \
        else (set(), None)
    if ks:
        signals["keyswitch_pitches"] = sorted(ks)
        return "keyswitch", 0.85, [], signals

    if cc2 >= 5.0:                       # 每秒 ≥5 个 CC2 事件 = 连续气息曲线
        return "breath", 0.8, ["mono_lead"], signals

    if n_bends >= 8:
        return "mono_lead", 0.8, ["piano"], signals

    return "piano", 0.9, [], signals


# ---- bend 曲线 → art 标签 ---- #

def _classify_bend(ins, note):
    """音符区间内的 bend 曲线 → art 标签(0/1/2)。规则见 roadmap 冲突消解节。"""
    pts = [(b.time, b.pitch / 8192.0 * BEND_RANGE_ST)
           for b in ins.pitch_bends if note.start <= b.time < note.end]
    if len(pts) < 3:
        return 0
    vals = np.array([v for _, v in pts])
    dur = max(note.end - note.start, 1e-6)

    steps = np.diff(vals)
    steps = steps[np.abs(steps) > 1e-9]
    net = vals[-1] - vals[0]
    if len(steps) and abs(net) >= SLIDE_MIN_ST:
        mono = np.mean(np.sign(steps) == np.sign(net))
        if mono >= SLIDE_MONO_FRAC:
            return 2                                   # slide

    centered = vals - vals.mean()
    if np.max(np.abs(centered)) >= VIB_MIN_AMP_ST:
        crossings = np.sum(np.diff(np.sign(centered)) != 0)
        rate = crossings / 2.0 / dur
        if VIB_RATE_LO <= rate <= VIB_RATE_HI:
            return 1                                   # vibrato
    return 0


# ---- 单声部化 ---- #

def _monophonize(events, report):
    """排序、去和弦(保最高音)、截断重叠。events: [(onset, dur, midi, art, extra)]"""
    events = sorted(events, key=lambda e: (e[0], -e[2]))
    out = []
    for ev in events:
        if out and abs(ev[0] - out[-1][0]) <= CHORD_EPS:
            report.n_chord_notes_dropped += 1          # 同 onset:已保最高音
            continue
        out.append(list(ev))
    for i in range(len(out) - 1):
        end_i = out[i][0] + out[i][1]
        if end_i > out[i + 1][0]:
            out[i][1] = max(0.02, out[i + 1][0] - out[i][0])
            report.n_overlaps_trimmed += 1
    return out


# ---- 逐音符 CC 能量提示 ---- #

def _cc_energy_per_note(ins, notes_events, cc_number):
    ccs = sorted([(c.time, c.value) for c in ins.control_changes
                  if c.number == cc_number])
    if not ccs:
        return None
    t = np.array([x for x, _ in ccs])
    v = np.array([y for _, y in ccs], dtype=float) / 127.0
    vals = []
    for onset, dur, *_ in notes_events:
        m = (t >= onset) & (t < onset + dur)
        vals.append(float(v[m].mean()) if m.any() else float(
            v[np.searchsorted(t, onset).clip(0, len(v) - 1)]))
    return vals


# ---- 主入口 ---- #

def import_midi(path, ks_map=None, bend_range_st=None):
    """解析任意方言的 MIDI 文件 → (score, ImportReport)。

    score 直接可喂 neural.perform.control_from_behavior(额外键被引擎忽略)。
    纯鼓文件抛 DrumFileError。
    """
    global BEND_RANGE_ST
    if bend_range_st is not None:
        BEND_RANGE_ST = float(bend_range_st)
    ks_map = dict(DEFAULT_KS_MAP if ks_map is None else ks_map)

    pm = pretty_midi.PrettyMIDI(str(path))
    report = ImportReport()
    dialect, conf, cands, signals = detect_dialect(pm)
    report.dialect, report.confidence, report.candidates = dialect, conf, cands

    if dialect == "drums":
        raise DrumFileError(
            "纯鼓文件:note number 是乐器身份(36=底鼓/38=军鼓),不是音高;"
            "Quill 的 pitched monophonic 管线不适用。")

    drums = [ins for ins in pm.instruments if ins.is_drum and ins.notes]
    if drums:
        report.n_drum_tracks_dropped = len(drums)
        report.warnings.append(f"丢弃 {len(drums)} 条鼓轨")

    melodic = _melodic_instruments(pm)
    report.n_notes_in = sum(len(ins.notes) for ins in melodic)

    events = []          # (onset, dur, midi, art, cc_energy_or_None)

    if dialect == "keyswitch":
        ins = melodic[0]
        ks_pitches, base = _find_keyswitch_cluster(ins.notes)
        switches = sorted([(n.start, n.pitch) for n in ins.notes
                           if n.pitch in ks_pitches])
        report.n_keyswitches_stripped = len(switches)
        report.ks_map_used = {
            int(p): ART_NAMES.get(ks_map.get(p - base, 0), "plain")
            for p in ks_pitches}
        unknown = {p - base for _, p in switches} - set(ks_map)
        if unknown:
            report.warnings.append(
                f"未映射的 keyswitch 偏移 {sorted(unknown)} → 按 plain 处理")
        sw_t = [t for t, _ in switches]
        sw_art = [ks_map.get(p - base, 0) for _, p in switches]
        for n in ins.notes:
            if n.pitch in ks_pitches:
                continue
            k = int(np.searchsorted(sw_t, n.start, side="right")) - 1
            art = sw_art[k] if k >= 0 else 0           # latching:首个 KS 前默认 plain
            events.append((n.start, n.end - n.start, n.pitch, art, None))

    elif dialect in ("multichannel", "mono_lead"):
        report.n_channels_merged = len(melodic) if dialect == "multichannel" else 0
        for ins in melodic:
            for n in ins.notes:
                art = _classify_bend(ins, n)
                if art == 2:
                    report.n_bend_slide += 1
                elif art == 1:
                    report.n_bend_vibrato += 1
                events.append((n.start, n.end - n.start, n.pitch, art, None))

    else:                                              # piano / breath
        ins = melodic[0]
        for n in ins.notes:
            events.append((n.start, n.end - n.start, n.pitch, 0, None))

    # 逐音符 CC 能量提示(roadmap 优先级 3):单乐器方言里取最密的连续 CC 曲线,
    # 吹奏式优先 CC2(breath),音源式优先 CC11(expression)/CC1(mod)。
    if len(melodic) == 1 and dialect != "multichannel":
        ins = melodic[0]
        for cc_num, name in ((2, "cc2"), (11, "cc11"), (1, "cc1")):
            if _cc_density(ins, cc_num) >= 5.0:
                raw = [(e[0], e[1]) for e in events]
                cc_vals = _cc_energy_per_note(ins, raw, cc_num)
                if cc_vals:
                    report.cc_energy_source = name
                    events = [(o, d, m, a, cc_vals[i])
                              for i, (o, d, m, a, _) in enumerate(events)]
                break

    events = _monophonize(events, report)
    score = []
    for onset, dur, midi, art, cc in events:
        note = {"midi": int(midi), "dur": round(float(dur), 4),
                "art": int(art), "onset": round(float(onset), 4)}
        if cc is not None:
            note["cc_energy"] = round(float(cc), 3)
        score.append(note)
    report.n_notes_out = len(score)
    return score, report
