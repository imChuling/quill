"""单声部 Voice：相位数组、ADSR 状态机、起音颗粒。

每个 Voice 持有自己的持久状态（相位、ADSR、控制斜坡、颗粒/噪声读指针），逐块把贡献累加进
主混音块。note-on 重置相位；被偷的声部先 3–5 ms 快速淡出再复用；loudness 与 f0 在块内线性
斜坡；包络尾低于 -120 dB 时清零。本模块只依赖 numpy 和 numba 内核。
"""

from __future__ import annotations

import numpy as np

from synth.oscillator import voice_block_add

_SILENCE = 10 ** (-120 / 20.0)  # -120 dB：包络尾清零阈值，防 denormal

# ADSR 段
_ATK, _DEC, _SUS, _REL, _DONE = 0, 1, 2, 3, 4

# 不可变常量(不通过旋钮/config 改变)
_GRAIN_WITH_ATK = 0.30
_BEND_RATE_ST_S = 250.0


class LiveParams:
    """Voice 层的可变实时参数，由引擎/config/旋钮通过本模块的 setter 更新。"""
    __slots__ = (
        "ramp_hz", "hum_cents", "hum_amp",
        "vib_onset_ms", "vib_ramp_ms",
        "live_spec", "live_dark", "live_breath",
        "slide_noise", "regrain_legato", "regrain_slide",
        "art_dip_legato", "art_dip_slide",
        "tone_tilt",
    )

    def __init__(self):
        self.ramp_hz = 600.0
        self.hum_cents = 6.0
        self.hum_amp = 0.05
        self.vib_onset_ms = 110.0
        self.vib_ramp_ms = 220.0
        self.live_spec = 0.85
        self.live_dark = 0.0
        self.live_breath = 0.8
        self.slide_noise = 0.30
        self.regrain_legato = 0.16
        self.regrain_slide = 0.10
        self.art_dip_legato = 0.30  # 换音凹陷深度(线性幅度,0.30 ≈ -3 dB)
        self.art_dip_slide = 0.22
        self.tone_tilt = 0.0


_P = LiveParams()

_SLIDE_FEEL_BASE = (0.30, 0.16, 0.10, 0.30, 0.22)


class Voice:
    def __init__(self, K: int, sr: int):
        self.K = int(K)
        self.sr = int(sr)
        self.phase = np.zeros(K, dtype=np.float64)
        # Schroeder 相位（最小峰值因子）：note-on 用它色散初相，避免所有谐波在 0 相对齐成
        # 脉冲串的机械嗡声；初相确定，渲染可复现。
        ki = np.arange(K, dtype=np.float64)
        self._init_phase = (np.pi * ki * (ki + 1.0) / max(K, 1)) % (2.0 * np.pi)
        self.gain_k = np.zeros(K, dtype=np.float64)
        self._gain_buf = np.zeros(K, dtype=np.float64)  # 谱形轨迹/倾斜的写缓冲，热路径免分配
        self._gain_prev = np.zeros(K, dtype=np.float64)  # 上一块的 gain，内核块内渐变的起点
        self.ratio = np.arange(1, K + 1, dtype=np.float64)
        self.active = False
        self.note = -1
        self.age = 0  # note-on 计数，用于偷音选最旧
        self._reset_runtime()

    def _reset_runtime(self):
        self.stage = _DONE
        self.env = 0.0
        self.vel_loud = 0.0
        self._tilt = 0.0
        self.f0 = 220.0
        self.f0_target = 220.0
        self.f0_base = 220.0  # 当前发声基频，glide 在此上移动
        self.bend = 1.0  # pitch-bend 比例，平滑后
        self.bend_prev = 1.0
        self._bend_target = 1.0  # pitch-bend 目标，set_bend 设，逐块平滑追
        # glide（slide/legato 续奏：log 域沿曲线滑 f0，不重置相位/ADSR）
        self._glide_from = 220.0
        self._glide_to = 220.0
        self._glide_total = 1
        self._glide_pos = 0
        self._glide_curve = None
        # vibrato LFO（CC1 深度，速率取库实测值或默认）
        self._vib_depth = 0.0  # cents
        self._vib_rate = 5.5
        self._vib_phase = 0.0
        self._vib_age = 0  # 揉弦起振后流逝的采样数，用于延迟和渐入
        self._hum_off = 0.0  # 人性化 LFO 相位偏移，按音高错开，避免齐奏锁相
        self._gain_from = None  # gain_k 沿 glide 渐变，去滑音/连音起点的咔哒
        self._gain_to = None
        self._traj_pos = 0.0  # 谱形轨迹读位置（帧，浮点），沿源录音的持续段走
        self._traj_dir = 1.0  # 乒乓方向（±1），避免回绕接缝
        self._traj_rate = 1.0  # 每声部速率微差，和弦不同步呼吸
        self._tilt_s = 0.0  # 动态倾斜的平滑状态（150 ms 一阶），消逐块跳变毛刺
        self._amp_prev = 1.0  # 响度呼吸上一块因子：a 用它、b 用本块，内核块内渐变无台阶
        self._grain_scale = 1.0  # 起音颗粒倍率：note_on 按力度定，连/滑重触发用低电平
        self._atk_traj_pos = 1 << 30  # 起音谱轨迹位置（采样；大值=不活动）
        self._glide_stren = 0.0  # 滑音强度（按音程），驱动摩擦噪声隆起
        self._art_pos = 1 << 30  # 换音凹陷包络位置（采样；大值=不活动）
        self._art_len = 1
        self._art_depth = 0.0
        self.samples_since_on = 0
        self.attack_pos = 0
        self.noise_pos = 0
        self._inst = None
        # 线性 ADSR 每段每采样增量，note_on 时按 sr 算
        self._atk_inc = self._dec_inc = self._rel_inc = 0.0
        self._sustain = 0.7
        self._fadein_len = 1
        # 控制驱动模式：外部 (f0 Hz, 线性幅度) 帧流直接定每块的 f0/响度，绕过 glide、
        # 包络与人性化。None 表示普通事件驱动。
        self._ctl_fps = 250.0
        self._ctl_last_amp = 1.0
        self._ctl_last_f0 = 220.0
        self._clear_ctl()

    def _clear_ctl(self):
        """清除控制驱动状态；note_on（复用的声部不得带上一个控制流的残留）与运行时复位共用。"""
        self._ctl_f0 = self._ctl_amp = self._ctl_tilt = self._ctl_nz = None
        self._ctl_done = False
        self._ctl_s0 = 0
        self._ctl_xf_from = None
        self._ctl_xf_f0 = None
        self._ctl_xf_len = 0

    # ---- 控制驱动：与 DDSP 相同的 (f0, loudness) 帧流接口 ----
    def drive(self, f0_hz, amp, fps: float = 250.0, tilt_db=None, noise_mul=None):
        """进入控制驱动：此后 render_into 的基频/响度取自这两条帧流（线性插值，越界钳末帧）。

        读位置以 samples_since_on 计，所以必须紧跟 note_on 调用。tilt_db（dB/oct）与
        noise_mul（噪声层倍率）可选，承载行为库的亮度/气声运动。
        """
        switching = self._ctl_f0 is not None and self.active  # 连音换轨迹，声部已在响
        self._ctl_f0 = np.ascontiguousarray(f0_hz, dtype=np.float64)
        self._ctl_amp = np.ascontiguousarray(amp, dtype=np.float64)
        self._ctl_fps = float(fps)
        self._ctl_tilt = None if tilt_db is None else np.ascontiguousarray(tilt_db, dtype=np.float64)
        self._ctl_nz = None if noise_mul is None else np.ascontiguousarray(noise_mul, dtype=np.float64)
        self._ctl_done = False  # 轨迹走完前库里的揉弦说了算，引擎 LFO 不叠加
        # 读位置从本次 drive 起算；连音换轨迹时 samples_since_on 不归零
        self._ctl_s0 = int(self.samples_since_on) if switching else 0
        if switching:
            # 换音：响度与基频在 20 ms 内从当前值过渡到新流（响度线性、基频几何）。一块内
            # 硬切几个半音会发出咔哒，20 ms 的几何过渡听不成滑音但斜率连续。共振峰跟随的
            # 慢变基频不重置，由 120 ms 平滑自己走过去；重置意味着谐波增益瞬跳。
            self._ctl_xf_from = float(self._ctl_last_amp)
            self._ctl_xf_f0 = float(self._ctl_last_f0)
            self._ctl_xf_len = max(1, int(0.02 * self.sr))
        else:
            self._ctl_f0_slow = float(self._ctl_f0[0])  # 共振峰映射跟随的慢变基频，不跟颤音
            self._ctl_xf_from = None
            self._ctl_xf_f0 = None
            self._ctl_xf_len = 0

    def _ctl_at(self, s: int):
        s_rel = s - self._ctl_s0
        p = s_rel * self._ctl_fps / self.sr
        last = len(self._ctl_f0) - 1
        if p >= last:
            if not self._ctl_done:  # 轨迹走完：末帧保持，引擎 LFO 揉弦从此接力
                self._ctl_done = True
                self._vib_age = 0
            f0, amp = self._ctl_f0[last], self._ctl_amp[last]
        else:
            i = max(0, int(p))
            fr = max(0.0, p - i)
            f0 = self._ctl_f0[i] * (1.0 - fr) + self._ctl_f0[i + 1] * fr
            amp = self._ctl_amp[i] * (1.0 - fr) + self._ctl_amp[i + 1] * fr
        if self._ctl_xf_from is not None and s_rel < self._ctl_xf_len:
            w = max(0.0, s_rel / self._ctl_xf_len)
            amp = self._ctl_xf_from * (1.0 - w) + amp * w
            if self._ctl_xf_f0:
                f0 = self._ctl_xf_f0 ** (1.0 - w) * f0**w
        self._ctl_last_amp = amp
        self._ctl_last_f0 = f0
        return f0, amp

    def _ctl_aux(self, arr, s: int):
        p = (s - self._ctl_s0) * self._ctl_fps / self.sr
        last = len(arr) - 1
        if p >= last:
            return float(arr[last])
        i = int(p)
        fr = p - i
        return float(arr[i] * (1.0 - fr) + arr[i + 1] * fr)

    def retrigger_attack(self, scale: float = 0.6):
        """控制驱动下的音符边界：只重放起音颗粒/起音谱轨迹，不碰相位、淡入与包络
        （响度弧由行为库负责）。"""
        self.attack_pos = 0
        self._grain_scale = float(scale)
        if getattr(self._inst, "atk_traj", None) is not None:
            self._grain_scale *= _GRAIN_WITH_ATK
            self._atk_traj_pos = 0

    def note_on(
        self,
        note: int,
        vel: int,
        inst,
        adsr: dict,
        vel_cfg: dict,
        runtime_cfg: dict,
        age: int,
        noise_pos0: int = 0,
    ):
        """触发：设目标 f0、力度映射、亮度倾斜与 gain_k；重置相位、颗粒与 ADSR。"""
        self._inst = inst
        self.note = int(note)
        self.age = int(age)
        self.active = True

        self.f0_target = 440.0 * 2.0 ** ((note - 69) / 12.0)
        self.f0 = self.f0_base = self.f0_target
        self.bend = self.bend_prev = self._bend_target = 1.0
        # 复位 glide / vibrato：note_on 是拨弦式起音，无滑无颤
        self._glide_from = self._glide_to = self.f0_target
        self._glide_total, self._glide_pos, self._glide_curve = 1, 0, None
        self._vib_depth, self._vib_phase, self._vib_age = 0.0, 0.0, 0
        self._hum_off = (int(note) % 12) * 0.523  # 按音高错开人性化相位
        st = inst.spec_traj  # 谱形轨迹起点/速率按音符号确定性错开，可复现
        if st is not None:
            F = st.shape[1]
            self._traj_pos = float((int(note) * 7.13) % max(F - 1, 1))
            self._traj_rate = 1.0 + 0.12 * np.sin(int(note) * 1.7)
            self._traj_dir = 1.0 if (int(note) % 2 == 0) else -1.0
        self._amp_prev = 1.0
        self._tilt_s = 0.0
        self._atk_traj_pos = 0  # 起音谱轨迹读位置（采样），每分音自己的起音
        # 新音从事件驱动开始，控制驱动要在 note_on 之后显式 drive()；否则复用的声部会带着
        # 上一个控制流的残留把普通起音当成控制驱动，或把新的控制音当成连音换轨迹。
        self._clear_ctl()

        v = max(0, min(127, int(vel))) / 127.0
        self.vel_loud = v ** float(vel_cfg["loud_exp"])
        self._tilt = -(1.0 - v) * float(vel_cfg["tilt_db_oct"])  # dB/oct，轻弹更暗
        self._grain_scale = 0.6 + 0.6 * v  # 力度→起音颗粒占比，重弹瞬态更突出
        if getattr(inst, "atk_traj", None) is not None:
            self._grain_scale *= _GRAIN_WITH_ATK  # 谐波层自己完成拨弦时，外贴颗粒退为点缀
        self._glide_stren = 0.0
        self.ratio[:] = inst.ratio_k[: self.K]
        # gain_k = 共振峰保持重定向 × 亮度倾斜，按当前演奏音高定
        self.gain_k[:] = self._compute_gain_k(self.f0_target)
        self._gain_prev[:] = self.gain_k  # note-on 首块不渐变，起点即终点
        self._gain_to = None

        self.phase[:] = self._init_phase  # 色散初相，去机械感
        self.attack_pos = 0
        self.noise_pos = int(noise_pos0)
        self.samples_since_on = 0

        sr = self.sr
        self._sustain = float(adsr["sustain"])
        self._atk_inc = 1.0 / max(1, adsr["attack_ms"] / 1000.0 * sr)
        self._dec_inc = (1.0 - self._sustain) / max(1, adsr["decay_ms"] / 1000.0 * sr)
        self._rel_inc = 1.0 / max(1, adsr["release_ms"] / 1000.0 * sr)
        self._fadein_len = max(1, int(runtime_cfg["attack_fade_ms"] / 1000.0 * sr))
        self.stage = _ATK
        self.env = 0.0

    def _compute_gain_k(self, f0_play):
        """共振峰保持：第 k 分量幅度 = 谱包络在绝对频率 ratio_k·f0_play 处的值，再乘亮度倾斜。

        包络由快照 (ratio_k·f0_ref, H_steady) 定义。不按谐波序号搬运（那会让共振峰随音高
        平移，人声丢元音），而按绝对频率重采样：共振峰钉在原频率，换音不变色。返回新数组。
        """
        inst = self._inst
        K = self.K
        Hs = inst.H_steady[:K]
        ratio = inst.ratio_k[:K]
        xp = ratio * inst.f0_ref  # 快照各分量的绝对频率，即包络采样点
        targets = ratio * f0_play  # 当前演奏音各分量的绝对频率
        g = np.interp(targets, xp, Hs, left=Hs[0], right=Hs[-1])
        hi = xp[-1]  # 支撑外的高频按 12 dB/oct 衰减外推
        above = targets > hi
        if np.any(above) and Hs[-1] > 0 and hi > 0:
            octv = np.log2(np.maximum(targets[above], hi) / hi)
            g[above] = Hs[-1] * 10.0 ** (-12.0 * octv / 20.0)
        # 共振峰保持只对有共振峰结构的源成立。近正弦源（口哨、擦玻璃：能量几乎全在基频）的
        # 包络就是一个尖峰，按绝对频率重采样会把播放音的第 2/3 泛音采到源基频峰上，
        # 故按谱质心在共振峰保持与谐波保持之间混合。
        fw = getattr(inst, "_formant_w", None)
        if fw is None:
            hs = np.abs(np.asarray(inst.H_steady, np.float64))
            cen = float((np.arange(1, len(hs) + 1) * hs).sum() / (hs.sum() + 1e-12))
            fw = inst._formant_w = float(min(1.0, max(0.0, (cen - 1.2) / 1.0)))
        if fw < 1.0:
            g = fw * g + (1.0 - fw) * Hs
        g = g * (10.0 ** (self._tilt * inst.oct_k[:K] / 20.0))
        gmax = float(np.max(np.abs(g))) if K else 0.0
        if gmax > 0:  # 剪掉 -60 dB 以下的分量，内核跳过它们
            g[np.abs(g) < 1e-3 * gmax] = 0.0
        return g

    def note_off(self):
        if self.active and self.stage != _DONE:
            if self._ctl_f0 is not None:
                # 控制驱动时包络未参与，释放必须从满幅起步，否则先掉到 sustain 值发出 tick
                self.env = 1.0
            self.stage = _REL

    def steal(self, steal_ms: float):
        """偷音：进入快速释放（3–5 ms），淡完即由引擎回收复用。"""
        if self.active:
            self._rel_inc = 1.0 / max(1, steal_ms / 1000.0 * self.sr)
            self.stage = _REL

    def set_bend(self, semitones: float):
        self._bend_target = 2.0 ** (semitones / 12.0)  # render_into 逐块平滑追，消 MIDI 稀疏台阶

    def init_bend(self, semitones: float):
        """note_on 继承通道弯音：瞬时到位，新音直接在弯音音高起振，不从中心滑过来。"""
        self.bend = self.bend_prev = self._bend_target = 2.0 ** (semitones / 12.0)

    def glide_to(
        self, f0_hz: float, dur_samp: int, curve=None, kind: str | None = None, vel: int = 90
    ):
        """从当前 f0_base 沿 curve（g(τ)∈[0,1]）在 dur_samp 内滑到 f0_hz。

        续奏不重置相位/ADSR（legato 快滑无重击，slide 按库曲线）。kind 为 'legato' 或
        'slide' 时低电平重触发起音颗粒使换音可闻，并按音程设摩擦噪声隆起强度（渲染期作用于
        噪声层，随机纹理无步进毛刺）。
        """
        self._glide_from = self.f0_base
        self._glide_to = float(f0_hz)
        self._glide_total = max(1, int(dur_samp))
        self._glide_pos = 0
        self._glide_curve = np.asarray(curve, dtype=np.float64) if curve is not None else None
        self.f0_target = float(f0_hz)
        if kind in ("legato", "slide"):
            vv = max(0, min(127, int(vel))) / 127.0
            base = _P.regrain_legato if kind == "legato" else _P.regrain_slide
            self.attack_pos = 0  # 低电平重触发，颗粒强的源有效
            self._grain_scale = base * (0.6 + 0.6 * vv)
            self._art_pos = 0  # 换音凹陷：50 ms 的短促弧，换弓/换气感
            self._art_len = max(1, int(0.05 * self.sr))
            self._art_depth = _P.art_dip_legato if kind == "legato" else _P.art_dip_slide
            if self._glide_from > 0:
                oct_span = abs(np.log2(max(self._glide_to, 1e-6) / self._glide_from))
                self._glide_stren = min(1.0, oct_span * 1.6)  # 音程越大摩擦越明显
            else:
                self._glide_stren = 0.0
        # 共振峰随目标音高重定向，但 gain_k 沿 glide 渐变：突变即每谐波幅度跳变，
        # 滑音/连音起点会发出咔哒
        if self._inst is not None:
            self._gain_from = self.gain_k.copy()
            self._gain_to = self._compute_gain_k(f0_hz)

    def set_vibrato(self, depth_cents: float, rate_hz: float = None):
        """CC1 颤音深度（cents）；速率取库实测值或默认。depth=0 关。"""
        if depth_cents > 0.0 and self._vib_depth <= 0.0:
            self._vib_age = 0  # 新揉弦起振，重新走延迟和渐入
        self._vib_depth = float(depth_cents)
        if rate_hz:
            self._vib_rate = float(rate_hz)

    def _glide_f0(self, pos):
        if self._glide_total <= 1 or pos >= self._glide_total:
            return self._glide_to
        tau = pos / self._glide_total
        if self._glide_curve is not None and len(self._glide_curve) >= 2:
            g = float(np.interp(tau, np.linspace(0, 1, len(self._glide_curve)), self._glide_curve))
        else:
            g = tau
        ratio = self._glide_to / self._glide_from if self._glide_from > 0 else 1.0
        return self._glide_from * ratio**g

    def _advance_adsr(self, n: int):
        """推进 ADSR n 个采样，返回 (env_start, env_end)。线性段。"""
        e0 = self.env
        e = self.env
        st = self.stage
        if st == _ATK:
            e += self._atk_inc * n
            if e >= 1.0:
                e = 1.0
                self.stage = _DEC
        elif st == _DEC:
            e -= self._dec_inc * n
            if e <= self._sustain:
                e = self._sustain
                self.stage = _SUS
        elif st == _SUS:
            e = self._sustain
        elif st == _REL:
            e -= self._rel_inc * n
            if e <= _SILENCE:
                e = 0.0
                self.stage = _DONE
                self.active = False
        self.env = e
        return e0, e

    def render_into(self, block: np.ndarray, n: int):
        """把本声部 n 个采样累加进 block[:n]（block 为主混音缓冲的视图）。"""
        if not self.active:
            return
        inst = self._inst
        env_a, env_b = self._advance_adsr(n)

        # 谐波淡入（与起音颗粒交叠约 20 ms）：0→1
        s0 = self.samples_since_on
        fi_a = min(1.0, s0 / self._fadein_len)
        fi_b = min(1.0, (s0 + n) / self._fadein_len)

        loud_a = self.vel_loud * env_a * fi_a
        loud_b = self.vel_loud * env_b * fi_b

        # 自然幅度包络（拨弦类衰减，持续类≈1）：按 note 流逝时间采样；超出表长后按尾部速率
        # 继续衰减，钳末帧会冻在残响电平上永远响
        ae = inst.amp_env
        ib = 0
        if ae is not None:
            hop = inst.amp_hop
            L = len(ae) - 1
            ja, jb = s0 // hop, (s0 + n) // hop
            tr = inst.amp_tail_ratio
            fa = ae[min(ja, L)] * (tr ** max(0, ja - L))
            fb = ae[min(jb, L)] * (tr ** max(0, jb - L))
            ib = min(jb, L)
            loud_a *= fa
            loud_b *= fb

        ctl = self._ctl_f0 is not None
        if ctl:  # 控制驱动：响度 = 力度 × 外部幅度帧；ADSR 只在释放段起作用
            cf0_a, camp_a = self._ctl_at(s0)
            cf0_b, camp_b = self._ctl_at(s0 + n)
            # 释放段乘释放包络，含本块内刚进入 _DONE 的那一块：否则最后一块按满幅渲染再静音，
            # 松键后一声咔哒。其余段的包络交给控制流。
            rel_a, rel_b = (env_a, env_b) if self.stage in (_REL, _DONE) else (1.0, 1.0)
            loud_a = self.vel_loud * camp_a * fi_a * rel_a
            loud_b = self.vel_loud * camp_b * fi_b * rel_b

        # f0 块内斜坡 = glide（基频沿曲线滑）× pitch-bend × vibrato
        pos0 = self._glide_pos
        pos1 = min(self._glide_total, pos0 + n)
        base_a = self._glide_f0(pos0)
        base_b = self._glide_f0(pos1)
        self._glide_pos = pos1
        self.f0_base = base_b
        # 滑音进行度（0→1→0 的弧），驱动摩擦噪声隆起
        slide_u = (
            np.sin(np.pi * pos1 / self._glide_total)
            if (self._glide_total > 1 and pos1 < self._glide_total and self._glide_stren > 0)
            else 0.0
        )
        # gain_k 沿 glide 渐变到目标共振峰映射，去 slide/legato 起点的咔哒
        if self._gain_to is not None:
            frac = pos1 / self._glide_total if self._glide_total > 0 else 1.0
            if frac >= 1.0:
                self.gain_k[:] = self._gain_to
                self._gain_to = None
            else:
                self.gain_k[:] = self._gain_from + (self._gain_to - self._gain_from) * frac
        if self._vib_depth > 0.0 and (not ctl or self._ctl_done):
            # 揉弦延迟 vib_onset_ms 起振，深度在 vib_ramp_ms 内 smoothstep 渐入
            # （控制驱动时库轨迹走完才接力，避免双重揉弦）
            on = _P.vib_onset_ms * self.sr / 1000.0
            rp = max(1.0, _P.vib_ramp_ms * self.sr / 1000.0)
            u = min(1.0, max(0.0, (self._vib_age - on) / rp))
            venv = u * u * (3.0 - 2.0 * u)  # smoothstep，揉弦由浅入深
            self._vib_age += n
            depth = self._vib_depth * venv
            va = 2.0 ** (depth * np.sin(self._vib_phase) / 1200.0)
            self._vib_phase += 2.0 * np.pi * self._vib_rate * n / self.sr
            vb = 2.0 ** (depth * np.sin(self._vib_phase) / 1200.0)
        else:
            va = vb = 1.0
        # pitch-bend 速率限幅（半音域）：触摸条式弯音会瞬移（±4 st / 12 ms，1000+ st/s），
        # 如实渲染是咕噜声。乐句极限约 163 st/s（13 Hz 手动颤音 × ±2 st），限 250 st/s 不影响
        # 真实演奏，瞬移变成约 16 ms 的滑音。无弯音时 bend == target == 1.0，不进分支。
        tgt = self._bend_target
        if self.bend != tgt:
            cur_st = 12.0 * np.log2(self.bend)
            step = 12.0 * np.log2(tgt) - cur_st
            lim = _BEND_RATE_ST_S * (n / self.sr)
            if step > lim:
                step = lim
            elif step < -lim:
                step = -lim
            self.bend = tgt if abs(step) < 1e-9 else 2.0 ** ((cur_st + step) / 12.0)
        if ctl:  # 控制驱动：基频来自外部帧流
            base_a, base_b = cf0_a, cf0_b
            self.f0_base = base_b
            # 共振峰映射跟 120 ms 一阶平滑后的基频：滑音/换音照样换色，颤音（5–7 Hz）
            # 不在谱峰上来回采样，否则泛音幅度随颤音抖
            self._ctl_f0_slow += (base_b - self._ctl_f0_slow) * (1.0 - np.exp(-n / (0.12 * self.sr)))
            self.gain_k[:] = self._compute_gain_k(self._ctl_f0_slow)
        f0_a = base_a * self.bend_prev * va
        f0_b = base_b * self.bend * vb
        self.bend_prev = self.bend

        # 人性化：微音高抖动 + 微幅度颤动，两组非公约频率的 LFO，确定性。
        # 控制驱动时关闭：微观动态由行为库提供，再叠一层抖动会糊掉检索到的颤音
        if (_P.hum_cents > 0.0 or _P.hum_amp > 0.0) and not ctl:
            t0 = s0 / self.sr
            t1 = (s0 + n) / self.sr
            ph = self._hum_off
            jit0 = _P.hum_cents * (
                0.6 * np.sin(2 * np.pi * 4.3 * t0 + ph)
                + 0.4 * np.sin(2 * np.pi * 6.7 * t0 + 1.7 * ph)
            )
            jit1 = _P.hum_cents * (
                0.6 * np.sin(2 * np.pi * 4.3 * t1 + ph)
                + 0.4 * np.sin(2 * np.pi * 6.7 * t1 + 1.7 * ph)
            )
            f0_a *= 2.0 ** (jit0 / 1200.0)
            f0_b *= 2.0 ** (jit1 / 1200.0)
            loud_a *= 1.0 + _P.hum_amp * np.sin(2 * np.pi * 3.1 * t0 + 0.5 * ph)
            loud_b *= 1.0 + _P.hum_amp * np.sin(2 * np.pi * 3.1 * t1 + 0.5 * ph)

        # ---- 表现层 ----
        # 谱形轨迹：沿源录音持续段的形状运动乒乓循环，每声部错开起点/速率/方向。
        # 亮度随电平：响则亮、衰减则暗（dark_db·oct_k 倾斜）。
        gk = self.gain_k
        # 起音谱轨迹：每个分音自己的起音/早期衰减，拨弦感的物理来源，由谐波层自己完成
        atk = inst.atk_traj
        if atk is not None and self._atk_traj_pos < (1 << 29):
            Fa = atk.shape[1]
            p_a = self._atk_traj_pos * inst.atk_fps / self.sr
            if p_a >= Fa - 1:
                self._atk_traj_pos = 1 << 30  # 起音结束，交回稳态路径
            else:
                i0 = int(p_a)
                i1 = min(i0 + 1, Fa - 1)
                fr = p_a - i0
                Kv = min(self.K, atk.shape[0])
                avec = atk[:Kv, i0] * (1.0 - fr) + atk[:Kv, i1] * fr
                buf = self._gain_buf
                buf[: self.K] = gk[: self.K]
                buf[:Kv] *= avec
                gk = buf
                self._atk_traj_pos += n
        live_amp = 1.0  # 响度呼吸因子，源持续段的音量起伏
        st = inst.spec_traj
        if st is not None and _P.live_spec > 0.0:
            F = st.shape[1]
            Kv = min(self.K, st.shape[0])
            p = self._traj_pos
            i0 = int(p)
            i1 = min(i0 + 1, F - 1)
            fr = p - i0
            fvec = st[:Kv, i0] * (1.0 - fr) + st[:Kv, i1] * fr
            buf = self._gain_buf
            if gk is not buf:  # 可能已被起音轨迹改写
                buf[: self.K] = gk[: self.K]
            buf[:Kv] *= 1.0 + _P.live_spec * (fvec - 1.0)  # 深度即因子向 1 收缩
            gk = buf
            at = inst.amp_traj
            if at is not None:  # 响度呼吸：a 用上一块因子、b 用本块，无边界台阶
                af = at[i0] * (1.0 - fr) + at[i1] * fr
                now = 1.0 + _P.live_spec * (af - 1.0)
                loud_a *= self._amp_prev
                loud_b *= now
                live_amp = 0.5 * (self._amp_prev + now)  # 噪声层用中点，随机纹理掩蔽台阶
                self._amp_prev = now
            adv = self._traj_dir * self._traj_rate * inst.traj_fps * (n / self.sr)
            p2 = p + adv
            if p2 >= F - 1.0:  # 乒乓反弹，无回绕接缝
                p2 = 2.0 * (F - 1.0) - p2
                self._traj_dir = -self._traj_dir
            if p2 <= 0.0:
                p2 = -p2
                self._traj_dir = abs(self._traj_dir)
            self._traj_pos = min(max(p2, 0.0), F - 1.0)
        if self._art_pos < self._art_len:  # 换音凹陷：短促的正弦弧，a/b 采样使块内渐变连续
            pa = self._art_pos / self._art_len
            pb = min(1.0, (self._art_pos + n) / self._art_len)
            loud_a *= 1.0 - self._art_depth * np.sin(np.pi * pa)
            loud_b *= 1.0 - self._art_depth * np.sin(np.pi * pb)
            self._art_pos += n
        tilt_dyn = _P.tone_tilt  # 动态倾斜（dB/oct）：Brightness 旋钮、暗化、亮度呼吸合并一次幂
        if ctl and self._ctl_tilt is not None:  # 行为库的亮度运动，质心残差 → dB/oct
            tilt_dyn += self._ctl_aux(self._ctl_tilt, s0 + n // 2)
        if _P.live_dark > 0.0:
            lvl = env_b * fi_b
            if ae is not None:
                lvl *= ae[ib]
            tilt_dyn -= _P.live_dark * (1.0 - min(1.0, lvl))  # 电平越低越暗
        if _P.live_breath > 0.0:
            tb = s0 / self.sr  # 亮度慢呼吸：双非公约频率，每声部错相
            ph = self._hum_off * 2.1
            tilt_dyn += _P.live_breath * (
                0.6 * np.sin(2 * np.pi * 0.31 * tb + ph)
                + 0.4 * np.sin(2 * np.pi * 0.47 * tb + 1.9 * ph)
            )
        # 150 ms 一阶平滑消倾斜跳变，无门连续应用：幅度门会在 LFO 过零处开关，产生残差脉冲。
        # 旋钮归零后 _tilt_s 仍需滑回 0，否则最后一块的倾斜会冻结在声部里。
        if _P.live_breath > 0.0 or _P.live_dark > 0.0 or tilt_dyn != 0.0 or self._tilt_s != 0.0:
            alpha = 1.0 - np.exp(-n / (0.15 * self.sr))
            self._tilt_s += (tilt_dyn - self._tilt_s) * alpha
            if gk is self.gain_k:  # 未经轨迹分支，先拷进缓冲
                buf = self._gain_buf
                buf[: self.K] = gk[: self.K]
                gk = buf
            gk[: self.K] *= 10.0 ** (self._tilt_s * inst.oct_k[: self.K] / 20.0)

        nyq = self.sr * 0.5
        voice_block_add(
            block,
            n,
            self.phase,
            self.ratio,
            self._gain_prev,
            gk,
            self.K,
            f0_a,
            f0_b,
            loud_a,
            loud_b,
            float(self.sr),
            nyq,
            _P.ramp_hz,
        )
        self._gain_prev[: self.K] = gk[: self.K]  # 本块终值即下块起点，跨块连续

        # 起音颗粒回放（力度缩放），与谐波淡入交叠
        grain = inst.attack_grain
        gp = self.attack_pos
        if gp < len(grain):
            m = min(n, len(grain) - gp)
            block[:m] += grain[gp : gp + m] * self.vel_loud * self._grain_scale
        self.attack_pos = gp + n

        # 噪声纹理（查表，ADSR 缩放）；固定表加复位指针，渲染可复现
        tbl = inst.noise_table
        nl = inst.noise_level
        if ctl and self._ctl_nz is not None:  # 行为库的气声/摩擦运动，平坦度残差 → 倍率
            nl = nl * self._ctl_aux(self._ctl_nz, s0 + n // 2)
        # 滑音摩擦按绝对量注入，不依赖 Air/Noise 底值，但仍用源自己的噪声纹理
        if slide_u > 0.0:
            nl = (
                nl + _P.slide_noise * slide_u * self._glide_stren
            )
        if nl > 0 and len(tbl) > 0:
            np_ = self.noise_pos
            env_mid = 0.5 * (env_a + env_b) * self.vel_loud * nl * live_amp
            if np_ + n <= len(tbl):
                block[:n] += tbl[np_ : np_ + n] * env_mid
                self.noise_pos = (np_ + n) % len(tbl)
            else:  # 环绕表尾
                m = len(tbl) - np_
                block[:m] += tbl[np_:] * env_mid
                block[m:n] += tbl[: n - m] * env_mid
                self.noise_pos = n - m

        self.samples_since_on = s0 + n


# ---- 公共 setter：引擎/config/旋钮 → _P ----


def set_nyquist_ramp_hz(hz: float):
    _P.ramp_hz = float(hz)


def set_humanize(cents: float, amp: float):
    _P.hum_cents = float(cents)
    _P.hum_amp = float(amp)


def set_vibrato_shape(onset_ms: float, ramp_ms: float):
    _P.vib_onset_ms = float(onset_ms)
    _P.vib_ramp_ms = float(ramp_ms)


def set_slide_feel(
    noise_bump: float,
    regrain_legato: float,
    regrain_slide: float,
    dip_legato: float = 0.30,
    dip_slide: float = 0.22,
):
    _P.slide_noise = float(noise_bump)
    _P.regrain_legato = float(regrain_legato)
    _P.regrain_slide = float(regrain_slide)
    _P.art_dip_legato = float(dip_legato)
    _P.art_dip_slide = float(dip_slide)


def set_tone_tilt(db_oct: float):
    _P.tone_tilt = float(db_oct)


def set_articulation_intensity(scale: float):
    s = max(0.0, float(scale))
    n, rl, rs, dl, ds = _SLIDE_FEEL_BASE
    set_slide_feel(n * s, rl * s, rs * s, min(0.6, dl * s), min(0.6, ds * s))


def regrain_legato() -> float:
    return _P.regrain_legato


def set_liveness(spec: float, dark_db_oct: float, breath_db_oct: float = 0.6):
    _P.live_spec = float(spec)
    _P.live_dark = float(dark_db_oct)
    _P.live_breath = float(breath_db_oct)
