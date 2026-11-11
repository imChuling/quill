"""手势解释器：MIDI 演奏手势 -> 演奏法事件。

    note_on 无 held              -> pluck（普通起音）
    note_on 有 held: |音程|≤2半音 -> legato（不重触发，快滑过渡）
                     |音程|>2半音 -> slide（f0=f0_start·2^(g(t/T)·interval/12)，力度→滑速）
    <80ms 内 ≥3 音               -> strum（弦序时差错峰触发，力度→扫速）
    pitch_bend                   -> f0 直驱
    CC1                          -> vibrato 深度（速率取库实测值）

产物是 Engine 能消费的 cooked 事件（on/off/slide/legato/bend/vibrato/pedal）。阈值来自
config.yaml 的 articulation / gesture 节；slide 曲线与 strum 模板可从 .tplib 注入。
只依赖 numpy。
"""

from __future__ import annotations

import numpy as np

try:
    from quill_config import CFG

    _A = CFG.get("articulation", {})
    _G = CFG.get("gesture", {})
except Exception:  # pragma: no cover
    _A, _G = {}, {}


def _ms(d, k, default):
    return float(d.get(k, default)) / 1000.0


def canonical_slide_curve(lib, max_interval_st=7.0):
    """从库里挑干净的 slide（|音程|小、单调）取平均形状 g(τ)∈[0,1]；无则返回 None（线性）。"""
    if not lib or not lib.get("slide"):
        return None
    curves = [
        np.asarray(s["curve_ctrl"], dtype=np.float64)
        for s in lib["slide"]
        if abs(s.get("interval_st", 99)) <= max_interval_st and len(s.get("curve_ctrl", [])) >= 2
    ]
    if not curves:
        return None
    n = min(len(c) for c in curves)
    g = np.mean(
        [np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(c)), c) for c in curves], axis=0
    )
    # 单调化并归一到 [0,1]，抑制个别噪声
    g = np.maximum.accumulate(g)
    g = (g - g[0]) / (g[-1] - g[0]) if g[-1] > g[0] else np.linspace(0, 1, n)
    return g.astype(np.float64)


def strum_offsets(lib):
    """库里的 strum 时差向量（ms）与方向；无则 (None, None)。"""
    if lib and lib.get("strum"):
        st = lib["strum"][0]
        return np.asarray(st["offsets_ms"], dtype=np.float64), st.get("direction", "down")
    return None, None


class GestureInterpreter:
    """MIDI 手势 -> 演奏法事件。offline: process(raw)；realtime: feed(ev,t)。"""

    def __init__(self, library=None, cfg=None):
        a = (cfg or {}).get("articulation", _A) if cfg else _A
        g = (cfg or {}).get("gesture", _G) if cfg else _G
        self.strum_window = float(a.get("strum_window_ms", 80.0)) / 1000.0
        self.strum_min = int(a.get("strum_min_onsets", 3))
        self.legato_max_st = float(g.get("legato_max_st", 2))
        self.slide_dur = _ms(g, "slide_dur_ms", 130)
        self.slide_dur_min = _ms(g, "slide_dur_min_ms", 50)
        self.legato_dur = _ms(g, "legato_dur_ms", 40)
        self.strum_spread = _ms(g, "strum_spread_ms", 14)
        self.vib_max_cents = float(g.get("vibrato_max_cents", 50.0))

        self.slide_curve = canonical_slide_curve(library)
        self._strum_off, self._strum_dir = strum_offsets(library)

    def apply_to_engine(self, engine):
        """把 slide 曲线 / 颤音速率注入引擎。"""
        engine.set_gesture_templates(slide_curve=self.slide_curve)
        engine.vib_max_cents = self.vib_max_cents

    def _slide_dur(self, vel):
        """力度→滑速：力度越大滑得越快（时长越短）。"""
        f = max(0, min(127, vel)) / 127.0
        return self.slide_dur_min + (self.slide_dur - self.slide_dur_min) * (1 - f)

    def _strum_event_offsets(self, n, vel):
        """n 弦的错峰时差（秒），力度越大越紧。"""
        if self._strum_off is not None and len(self._strum_off) >= 2:
            base = (
                np.interp(
                    np.linspace(0, 1, n),
                    np.linspace(0, 1, len(self._strum_off)),
                    np.sort(np.abs(self._strum_off)),
                )
                / 1000.0
            )
        else:
            base = np.arange(n) * self.strum_spread
        f = max(0, min(127, vel)) / 127.0
        return base * (1.3 - 0.6 * f)

    def process(self, raw_events):
        """raw: [(t_sec, kind, note, vel)]，kind∈{on,off,bend,cc} -> cooked 事件列表。

        cooked: [(t, kind, note, vel, value, note2)]，可直接喂 Engine.render_offline。
        """
        evs = sorted(raw_events, key=lambda e: (e[0], 0 if e[1] != "on" else 1))
        # 预扫 strum 簇：连续 note_on 落在 window 内且数量 ≥ strum_min
        on_idx = [i for i, e in enumerate(evs) if e[1] == "on"]
        strum_member = set()
        i = 0
        clusters = []
        while i < len(on_idx):
            j = i
            while (
                j + 1 < len(on_idx)
                and evs[on_idx[j + 1]][0] - evs[on_idx[i]][0] < self.strum_window
            ):
                j += 1
            if j - i + 1 >= self.strum_min:
                grp = on_idx[i : j + 1]
                clusters.append(grp)
                strum_member.update(grp)
                i = j + 1
            else:
                i += 1

        # ---- 走一遍，发 cooked 事件 ----
        cooked = []
        held = []  # 当前按下的物理键（有序）
        done_clusters = set()
        for idx, e in enumerate(evs):
            t, kind = e[0], e[1]
            note = e[2] if len(e) > 2 else -1
            vel = e[3] if len(e) > 3 else 90
            if kind == "on":
                if idx in strum_member:
                    cl = next(c for c in clusters if idx in c)
                    if id(cl) in done_clusters:
                        continue  # 簇内后续音已在首音处理时入 held，勿重复
                    done_clusters.add(id(cl))
                    notes = [(evs[k][2], evs[k][3] if len(evs[k]) > 3 else 90) for k in cl]
                    order = sorted(range(len(notes)), key=lambda x: notes[x][0])  # 低→高
                    if self._strum_dir == "down":
                        order = order[::-1]
                    offs = self._strum_event_offsets(len(notes), vel)
                    t0 = evs[cl[0]][0]
                    for rank, oi in enumerate(order):
                        nn, vv = notes[oi]
                        cooked.append((t0 + offs[rank], "on", nn, vv, 0.0, -1))
                        held.append(nn)
                    continue
                # 非 strum：有 held 则 legato/slide，否则 pluck
                frm = held[-1] if held else None
                if frm is not None and frm != note:
                    interval = abs(note - frm)
                    if interval <= self.legato_max_st:
                        cooked.append((t, "legato", note, vel, self.legato_dur, frm))
                    else:
                        cooked.append((t, "slide", note, vel, self._slide_dur(vel), frm))
                else:
                    cooked.append((t, "on", note, vel, 0.0, -1))
                held.append(note)
            elif kind == "off":
                cooked.append((t, "off", note, 0, 0.0, -1))
                if note in held:
                    held.remove(note)
            elif kind == "bend":
                cooked.append((t, "bend", -1, 0, float(vel if len(e) <= 4 else e[4]), -1))
            elif kind == "cc":
                depth = (e[4] if len(e) > 4 else vel) / 127.0
                cooked.append((t, "vibrato", -1, 0, float(depth), -1))
            elif kind == "pedal":  # CC64 延音踏板，直通引擎
                cooked.append((t, "pedal", -1, 0, float(e[4] if len(e) > 4 else vel), -1))
        cooked.sort(key=lambda x: x[0])
        return cooked

    # ---- 卷帘/DAW 音符（带显式演奏法标签）→ cooked 事件 ----
    def strum_offsets(self, seq):
        """按 start 排好序的音符里，同一簇内标 strum 的音按出现顺序错峰 k×spread（秒），其余 0。"""
        offs, cl_t0, k = [], None, -1
        for nt in seq:
            t0 = float(nt["start"])
            if nt.get("art") == "strum":
                if cl_t0 is not None and t0 - cl_t0 < self.strum_window:
                    k += 1
                else:
                    cl_t0, k = t0, 0
                offs.append(k * self.strum_spread)
            else:
                cl_t0, k = None, -1
                offs.append(0.0)
        return offs

    def notes_to_events(self, notes):
        """[{midi,start,len,vel,art}] → [(t_sec, kind, note, vel, value, note2)]。

        slide/legato 续接上一音（时长按力度与旋钮），strum 错峰，其余普通起音。卷帘回放、
        导出与网页渲染共用这一个函数。
        """
        seq = sorted(notes, key=lambda x: float(x["start"]))
        evs, prev = [], -1
        for nt, off in zip(seq, self.strum_offsets(seq)):
            m, v = int(nt["midi"]), int(nt.get("vel", 100))
            art = nt.get("art", "pluck")
            t0 = float(nt["start"]) + off
            t1 = t0 + float(nt["len"])
            if art == "slide" and prev >= 0:
                evs.append((t0, "slide", m, v, self._slide_dur(v), prev))
            elif art == "legato" and prev >= 0:
                evs.append((t0, "legato", m, v, self.legato_dur, prev))
            else:
                evs.append((t0, "on", m, v, 0.0, -1))
            evs.append((t1, "off", m, 0, 0.0, -1))
            prev = m
        return evs
