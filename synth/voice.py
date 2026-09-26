"""单声部 Voice：相位数组 + ADSR 状态机 + 起音颗粒 —— FR-4（阶段3 Day1）。

每个 Voice 持有持久状态（相位、ADSR、控制斜坡、颗粒/噪声读指针），逐块把自己的
贡献累加进主混音块。铁律：note-on 重置相位（#6）；偷音 3-5ms 快淡出再复用（#4）；
loudness/f0 块内线性斜坡（#3）；包络尾 <-120dB 清零（#7）。

引擎铁律 #12：只依赖 numpy + numba 内核。
"""
from __future__ import annotations

import numpy as np

from synth.oscillator import voice_block_add

_SILENCE = 10 ** (-120 / 20.0)   # -120dB：包络尾清零阈值（防 denormal）

# ADSR 段
_ATK, _DEC, _SUS, _REL, _DONE = 0, 1, 2, 3, 4


class Voice:
    def __init__(self, K: int, sr: int):
        self.K = int(K)
        self.sr = int(sr)
        self.phase = np.zeros(K, dtype=np.float64)
        # Schroeder 相位（最小峰值因子）：note-on 用它**色散**初相，避免所有谐波
        # 在 0 相对齐成脉冲串那种机械"嗡嗡"声；确定性=保留可复现（铁律 #6 的本意）。
        ki = np.arange(K, dtype=np.float64)
        self._init_phase = (np.pi * ki * (ki + 1.0) / max(K, 1)) % (2.0 * np.pi)
        self.gain_k = np.zeros(K, dtype=np.float64)
        self._gain_buf = np.zeros(K, dtype=np.float64)   # 谱形轨迹/暗化的写缓冲(热路径免分配)
        self._gain_prev = np.zeros(K, dtype=np.float64)  # 上一块的 gain(内核块内渐变的起点)
        self.ratio = np.arange(1, K + 1, dtype=np.float64)
        self.active = False
        self.note = -1
        self.age = 0                  # note-on 计数，用于偷音选最旧
        self._reset_runtime()

    def _reset_runtime(self):
        self.stage = _DONE
        self.env = 0.0
        self.vel_loud = 0.0
        self._tilt = 0.0
        self.f0 = 220.0
        self.f0_target = 220.0
        self.f0_base = 220.0          # 当前发声基频（glide 在此上移动）
        self.bend = 1.0               # pitch-bend 比例(平滑后)
        self.bend_prev = 1.0
        self._bend_target = 1.0       # pitch-bend 目标(set_bend 设,逐块平滑追)
        # glide（slide/legato 续奏：log 域沿曲线滑 f0，不重置相位/ADSR）
        self._glide_from = 220.0
        self._glide_to = 220.0
        self._glide_total = 1
        self._glide_pos = 0
        self._glide_curve = None
        # vibrato LFO（CC1 深度，速率取库实测值或默认）
        self._vib_depth = 0.0         # cents
        self._vib_rate = 5.5
        self._vib_phase = 0.0
        self._vib_age = 0             # 揉弦起振后流逝采样（用于延迟+渐入"绽放"）
        self._hum_off = 0.0           # 人性化 LFO 相位偏移（按音高错开，避免齐奏锁相）
        self._gain_from = None        # gain_k 沿 glide 渐变（去滑音/连音起点咔哒）
        self._gain_to = None
        self._traj_pos = 0.0          # 谱形轨迹读位置(帧,浮点)——沿源录音的"呼吸"走
        self._traj_dir = 1.0          # 乒乓方向(±1),避免回绕接缝
        self._traj_rate = 1.0         # 每声部速率微差(和弦不同步呼吸)
        self._tilt_s = 0.0            # 动态倾斜的平滑状态(150ms 一阶,消起音/释音期的逐块跳变毛刺)
        self._amp_prev = 1.0          # 响度呼吸上一块因子(a 用它/b 用本块 -> 内核块内渐变,无边界台阶)
        self._grain_scale = 1.0       # 起音颗粒倍率:note_on 按力度定;连/滑重触发用低电平(P2)
        self._atk_traj_pos = 1 << 30  # 起音谱轨迹位置(大值=不活动)
        self._glide_stren = 0.0       # 滑音强度(按音程),驱动摩擦噪声隆起
        self._art_pos = 1 << 30       # 换音"凹陷"包络位置(采样;大值=不活动)
        self._art_len = 1
        self._art_depth = 0.0
        self.samples_since_on = 0
        self.attack_pos = 0
        self.noise_pos = 0
        self._inst = None
        # 线性 ADSR 每段每样本增量（note_on 时按 sr 算）
        self._atk_inc = self._dec_inc = self._rel_inc = 0.0
        self._sustain = 0.7
        self._fadein_len = 1

    # ----------------------------------------------------------------- #
    def note_on(self, note: int, vel: int, inst, adsr: dict,
                vel_cfg: dict, runtime_cfg: dict, age: int,
                noise_pos0: int = 0):
        """触发：设目标 f0、力度映射、亮度倾斜 gain_k；重置相位/颗粒/ADSR。"""
        self._inst = inst
        self.note = int(note)
        self.age = int(age)
        self.active = True

        self.f0_target = 440.0 * 2.0 ** ((note - 69) / 12.0)
        self.f0 = self.f0_base = self.f0_target
        self.bend = self.bend_prev = self._bend_target = 1.0
        # 复位 glide / vibrato（note_on = pluck 起音，无滑无颤）
        self._glide_from = self._glide_to = self.f0_target
        self._glide_total, self._glide_pos, self._glide_curve = 1, 0, None
        self._vib_depth, self._vib_phase, self._vib_age = 0.0, 0.0, 0
        self._hum_off = (int(note) % 12) * 0.523       # 按音高错开人性化相位
        st = inst.spec_traj                            # 谱形轨迹起点/速率:按音符号确定性错开(可复现)
        if st is not None:
            F = st.shape[1]
            self._traj_pos = float((int(note) * 7.13) % max(F - 1, 1))
            self._traj_rate = 1.0 + 0.12 * np.sin(int(note) * 1.7)
            self._traj_dir = 1.0 if (int(note) % 2 == 0) else -1.0
        self._amp_prev = 1.0
        self._tilt_s = 0.0
        self._atk_traj_pos = 0        # 起音谱轨迹读位置(采样;每分音自己的起音)

        v = max(0, min(127, int(vel))) / 127.0
        self.vel_loud = v ** float(vel_cfg["loud_exp"])
        self._tilt = -(1.0 - v) * float(vel_cfg["tilt_db_oct"])   # dB/oct（暗->亮）
        self._grain_scale = 0.6 + 0.6 * v         # 力度→起音噪声占比(重弹瞬态更突出,P2/MIDI-DDSP 维)
        if getattr(inst, "atk_traj", None) is not None:
            self._grain_scale *= _GRAIN_WITH_ATK   # 谐波层已自己完成拨弦 → 外贴颗粒退为点缀
                                                    # (否则就是用户听到的"明显杂音")
        self._glide_stren = 0.0
        self.ratio[:] = inst.ratio_k[:self.K]
        # gain_k = 共振峰保持重定向（FR-3）× 亮度倾斜，按当前演奏音高定
        self.gain_k[:] = self._compute_gain_k(self.f0_target)
        self._gain_prev[:] = self.gain_k        # note-on 首块不渐变(起点=终点)
        self._gain_to = None          # 无 gain 交叉淡化进行中

        self.phase[:] = self._init_phase                      # 色散初相（铁律 #6，去机械感）
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
        """FR-3 共振峰保持：第 k 分量幅度 = 谱包络在**绝对频率** ratio_k·f0_play 处的值
        （包络由快照 (ratio_k·f0_ref, H_steady) 定义），再乘亮度倾斜。**返回**数组。

        关键：不按谐波序号搬运（那会让共振峰随音高平移=花栗鼠、人声丢元音），
        而按绝对频率重采样——元音共振峰钉在原频率，换音不变色。
        """
        inst = self._inst
        K = self.K
        Hs = inst.H_steady[:K]
        ratio = inst.ratio_k[:K]
        xp = ratio * inst.f0_ref                 # 快照各分量绝对频率（包络采样点）
        targets = ratio * f0_play                # 当前演奏音各分量绝对频率
        g = np.interp(targets, xp, Hs, left=Hs[0], right=Hs[-1])
        hi = xp[-1]                              # 支撑外高频按 12dB/oct 衰减外推
        above = targets > hi
        if np.any(above) and Hs[-1] > 0 and hi > 0:
            octv = np.log2(np.maximum(targets[above], hi) / hi)
            g[above] = Hs[-1] * 10.0 ** (-12.0 * octv / 20.0)
        g = g * (10.0 ** (self._tilt * inst.oct_k[:K] / 20.0))
        gmax = float(np.max(np.abs(g))) if K else 0.0
        if gmax > 0:                             # 剪枝 <-60dB（内核跳过省热路径）
            g[np.abs(g) < 1e-3 * gmax] = 0.0
        return g

    def note_off(self):
        if self.active and self.stage != _DONE:
            self.stage = _REL

    def steal(self, steal_ms: float):
        """偷音：进入快速释放（3-5ms），淡完即由引擎回收复用。"""
        if self.active:
            self._rel_inc = 1.0 / max(1, steal_ms / 1000.0 * self.sr)
            self.stage = _REL

    def set_bend(self, semitones: float):
        self._bend_target = 2.0 ** (semitones / 12.0)   # 设目标;render_into 平滑追(消 MIDI 稀疏台阶)

    def init_bend(self, semitones: float):
        """note_on 继承通道弯音:瞬时到位(新音直接在弯音音高起振,不从中心滑过来)。"""
        self.bend = self.bend_prev = self._bend_target = 2.0 ** (semitones / 12.0)

    def glide_to(self, f0_hz: float, dur_samp: int, curve=None,
                 kind: str | None = None, vel: int = 90):
        """从当前 f0_base 沿 curve(g(τ)∈[0,1]) 在 dur_samp 内滑到 f0_hz。

        续奏：**不**重置相位/ADSR（legato=快滑无重击，slide=按库曲线）。
        kind='legato'/'slide' 时(P2 演奏法可听度):①低电平重触发起音颗粒=听得见换音;
        ②按音程设摩擦噪声隆起强度(渲染期作用于噪声层,随机纹理无步进毛刺)。
        """
        self._glide_from = self.f0_base
        self._glide_to = float(f0_hz)
        self._glide_total = max(1, int(dur_samp))
        self._glide_pos = 0
        self._glide_curve = (np.asarray(curve, dtype=np.float64)
                             if curve is not None else None)
        self.f0_target = float(f0_hz)
        if kind in ("legato", "slide"):
            vv = max(0, min(127, int(vel))) / 127.0
            base = _REGRAIN_LEGATO if kind == "legato" else _REGRAIN_SLIDE
            self.attack_pos = 0                                   # 低电平重触发(颗粒强的源有效)
            self._grain_scale = base * (0.6 + 0.6 * vv)
            self._art_pos = 0                                     # 换音"凹陷":短促 -3dB 弧(换弓/换气感,普适)
            self._art_len = max(1, int(0.05 * self.sr))
            self._art_depth = _ART_DIP_LEGATO if kind == "legato" else _ART_DIP_SLIDE
            if self._glide_from > 0:
                oct_span = abs(np.log2(max(self._glide_to, 1e-6) / self._glide_from))
                self._glide_stren = min(1.0, oct_span * 1.6)      # 音程越大摩擦越明显
            else:
                self._glide_stren = 0.0
        # 共振峰随目标音高重定向（FR-3）——但 gain_k **沿 glide 渐变**，不可突变
        # （突变=每谐波幅度跳变=滑音/连音起点一声咔哒，曾是"节拍器"元凶）
        if self._inst is not None:
            self._gain_from = self.gain_k.copy()
            self._gain_to = self._compute_gain_k(f0_hz)

    def set_vibrato(self, depth_cents: float, rate_hz: float = None):
        """CC1 颤音深度（cents）；速率取库实测值或默认。depth=0 关。"""
        if depth_cents > 0.0 and self._vib_depth <= 0.0:
            self._vib_age = 0          # 新揉弦起振 → 从延迟+渐入重新"绽放"
        self._vib_depth = float(depth_cents)
        if rate_hz:
            self._vib_rate = float(rate_hz)

    def _glide_f0(self, pos):
        if self._glide_total <= 1 or pos >= self._glide_total:
            return self._glide_to
        tau = pos / self._glide_total
        if self._glide_curve is not None and len(self._glide_curve) >= 2:
            g = float(np.interp(tau, np.linspace(0, 1, len(self._glide_curve)),
                                self._glide_curve))
        else:
            g = tau
        ratio = self._glide_to / self._glide_from if self._glide_from > 0 else 1.0
        return self._glide_from * ratio ** g

    # ----------------------------------------------------------------- #
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

        # 谐波淡入（与起音颗粒交叠 20ms）：0->1
        s0 = self.samples_since_on
        fi_a = min(1.0, s0 / self._fadein_len)
        fi_b = min(1.0, (s0 + n) / self._fadein_len)

        loud_a = self.vel_loud * env_a * fi_a
        loud_b = self.vel_loud * env_b * fi_b

        # 自然幅度包络（拨弦类衰减；持续类≈1 无回归）——按 note 流逝时间自然速率采样;
        # 超出表长后按尾部速率**继续衰减**(钳末帧=冻在残响电平上永远响,不自然)
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

        # f0 块内斜坡 = glide（基频沿曲线滑）× pitch-bend × vibrato
        pos0 = self._glide_pos
        pos1 = min(self._glide_total, pos0 + n)
        base_a = self._glide_f0(pos0)
        base_b = self._glide_f0(pos1)
        self._glide_pos = pos1
        self.f0_base = base_b
        # 滑音进行度(0→1→0 的弧):驱动摩擦噪声隆起(P2"手在动")
        slide_u = (np.sin(np.pi * pos1 / self._glide_total)
                   if (self._glide_total > 1 and pos1 < self._glide_total and self._glide_stren > 0)
                   else 0.0)
        # gain_k 沿 glide 渐变到目标共振峰映射（去 slide/legato 起点咔哒）
        if self._gain_to is not None:
            frac = pos1 / self._glide_total if self._glide_total > 0 else 1.0
            if frac >= 1.0:
                self.gain_k[:] = self._gain_to
                self._gain_to = None
            else:
                self.gain_k[:] = self._gain_from + (self._gain_to - self._gain_from) * frac
        if self._vib_depth > 0.0:
            # 揉弦"绽放":延迟 _VIB_ONSET_MS 起振,深度在 _VIB_RAMP_MS 内 smoothstep 渐入
            on = _VIB_ONSET_MS * self.sr / 1000.0
            rp = max(1.0, _VIB_RAMP_MS * self.sr / 1000.0)
            u = min(1.0, max(0.0, (self._vib_age - on) / rp))
            venv = u * u * (3.0 - 2.0 * u)               # smoothstep（真人揉弦由浅入深）
            self._vib_age += n
            depth = self._vib_depth * venv
            va = 2.0 ** (depth * np.sin(self._vib_phase) / 1200.0)
            self._vib_phase += 2.0 * np.pi * self._vib_rate * n / self.sr
            vb = 2.0 ** (depth * np.sin(self._vib_phase) / 1200.0)
        else:
            va = vb = 1.0
        # pitch-bend 速率限幅(半音域):实机踪迹(7.10)证实 S88 MK3 触摸条会"瞬移"
        # (±4st/12ms,1000+st/s;物理轮不可能),引擎如实渲染=咕噜声。乐句极限≈163st/s
        # (13Hz 手动颤音×±2st),限 250 st/s:真实演奏不受影响,瞬移变 ~16ms 利落滑音。
        # 无弯音时 bend==target==1.0 恒不进分支 -> 非弯音路径逐位不变。
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
        f0_a = base_a * self.bend_prev * va
        f0_b = base_b * self.bend * vb
        self.bend_prev = self.bend

        # 人性化：微音高抖动 + 微幅度颤动（去 L0 静态机械感；两组非公约 LFO，确定性）
        if _HUM_CENTS > 0.0 or _HUM_AMP > 0.0:
            t0 = s0 / self.sr
            t1 = (s0 + n) / self.sr
            ph = self._hum_off
            jit0 = _HUM_CENTS * (0.6 * np.sin(2 * np.pi * 4.3 * t0 + ph)
                                 + 0.4 * np.sin(2 * np.pi * 6.7 * t0 + 1.7 * ph))
            jit1 = _HUM_CENTS * (0.6 * np.sin(2 * np.pi * 4.3 * t1 + ph)
                                 + 0.4 * np.sin(2 * np.pi * 6.7 * t1 + 1.7 * ph))
            f0_a *= 2.0 ** (jit0 / 1200.0)
            f0_b *= 2.0 ** (jit1 / 1200.0)
            loud_a *= 1.0 + _HUM_AMP * np.sin(2 * np.pi * 3.1 * t0 + 0.5 * ph)
            loud_b *= 1.0 + _HUM_AMP * np.sin(2 * np.pi * 3.1 * t1 + 0.5 * ph)

        # —— P1 表现层(MIDI-DDSP 维度的规则版;见 docs/research_borrowings.md A1)——
        # ①谱形轨迹:沿源录音持续段的形状运动乒乓循环(每声部错开起点/速率/方向)
        # ②亮度随电平:响=亮、衰减=暗(dark_db·oct_k 倾斜)
        gk = self.gain_k
        # —— 起音谱轨迹:每个分音自己的起音/早期衰减(拨弦感的物理来源)——
        # 旧架构所有分音共用一条 amp_env + 统一淡入,拨弦只能靠外贴噪声颗粒
        # (用户耳测:"lead 合成器加了个蹩脚的拨弦动作")。这里让谐波层自己弹那一下。
        atk = inst.atk_traj
        if atk is not None and self._atk_traj_pos < (1 << 29):
            Fa = atk.shape[1]
            p_a = self._atk_traj_pos * inst.atk_fps / self.sr
            if p_a >= Fa - 1:
                self._atk_traj_pos = 1 << 30                 # 起音结束,交回稳态路径
            else:
                i0 = int(p_a); i1 = min(i0 + 1, Fa - 1); fr = p_a - i0
                Kv = min(self.K, atk.shape[0])
                avec = atk[:Kv, i0] * (1.0 - fr) + atk[:Kv, i1] * fr
                buf = self._gain_buf
                buf[:self.K] = gk[:self.K]
                buf[:Kv] *= avec
                gk = buf
                self._atk_traj_pos += n
        live_amp = 1.0                                       # 响度呼吸因子(源持续段的音量起伏)
        st = inst.spec_traj
        if st is not None and _LIVE_SPEC > 0.0:
            F = st.shape[1]
            Kv = min(self.K, st.shape[0])
            p = self._traj_pos
            i0 = int(p)
            i1 = min(i0 + 1, F - 1)
            fr = p - i0
            fvec = st[:Kv, i0] * (1.0 - fr) + st[:Kv, i1] * fr
            buf = self._gain_buf
            if gk is not buf:                                # 可能已被起音轨迹改写
                buf[:self.K] = gk[:self.K]
            buf[:Kv] *= 1.0 + _LIVE_SPEC * (fvec - 1.0)     # 深度=因子向 1 收缩
            gk = buf
            at = inst.amp_traj
            if at is not None:                               # 响度呼吸(可听主体):a=上一块因子,b=本块 -> 无边界台阶
                af = at[i0] * (1.0 - fr) + at[i1] * fr
                now = 1.0 + _LIVE_SPEC * (af - 1.0)
                loud_a *= self._amp_prev
                loud_b *= now
                live_amp = 0.5 * (self._amp_prev + now)      # 噪声层用中点(随机纹理,台阶被掩蔽)
                self._amp_prev = now
            adv = self._traj_dir * self._traj_rate * inst.traj_fps * (n / self.sr)
            p2 = p + adv
            if p2 >= F - 1.0:                                # 乒乓反弹(无回绕接缝)
                p2 = 2.0 * (F - 1.0) - p2
                self._traj_dir = -self._traj_dir
            if p2 <= 0.0:
                p2 = -p2
                self._traj_dir = abs(self._traj_dir)
            self._traj_pos = min(max(p2, 0.0), F - 1.0)
        if self._art_pos < self._art_len:                    # P2 换音凹陷:短促 -3dB 弧(a/b 采样=块内渐变,连续)
            pa = self._art_pos / self._art_len
            pb = min(1.0, (self._art_pos + n) / self._art_len)
            loud_a *= 1.0 - self._art_depth * np.sin(np.pi * pa)
            loud_b *= 1.0 - self._art_depth * np.sin(np.pi * pb)
            self._art_pos += n
        tilt_dyn = 0.0                                       # 动态倾斜(dB/oct):暗化 + 亮度呼吸合并一次幂
        if _LIVE_DARK > 0.0:
            lvl = env_b * fi_b
            if ae is not None:
                lvl *= ae[ib]
            tilt_dyn -= _LIVE_DARK * (1.0 - min(1.0, lvl))   # 电平低=更暗
        if _LIVE_BREATH > 0.0:
            tb = s0 / self.sr                                # 亮度慢呼吸(静态源也"活";双非公约频率,每声部错相)
            ph = self._hum_off * 2.1
            tilt_dyn += _LIVE_BREATH * (0.6 * np.sin(2 * np.pi * 0.31 * tb + ph)
                                        + 0.4 * np.sin(2 * np.pi * 0.47 * tb + 1.9 * ph))
        # 150ms 一阶平滑消倾斜跳变;**无门**连续应用(|tilt|>0.02 的门会在 LFO 过零处开关=残差脉冲)
        if _LIVE_BREATH > 0.0 or _LIVE_DARK > 0.0:
            alpha = 1.0 - np.exp(-n / (0.15 * self.sr))
            self._tilt_s += (tilt_dyn - self._tilt_s) * alpha
            if gk is self.gain_k:                            # 未经轨迹分支:先拷进缓冲
                buf = self._gain_buf
                buf[:self.K] = gk[:self.K]
                gk = buf
            gk[:self.K] *= 10.0 ** (self._tilt_s * inst.oct_k[:self.K] / 20.0)

        nyq = self.sr * 0.5
        voice_block_add(block, n, self.phase, self.ratio,
                        self._gain_prev, gk, self.K,
                        f0_a, f0_b, loud_a, loud_b, float(self.sr), nyq,
                        _RAMP_HZ)
        self._gain_prev[:self.K] = gk[:self.K]      # 本块终值 = 下块起点(跨块连续)

        # 起音颗粒回放（力度缩放），与谐波淡入交叠
        grain = inst.attack_grain
        gp = self.attack_pos
        if gp < len(grain):
            m = min(n, len(grain) - gp)
            block[:m] += grain[gp:gp + m] * self.vel_loud * self._grain_scale
        self.attack_pos = gp + n

        # 噪声纹理（查表，ADSR 缩放）；固定表 + 复位指针 => 可复现（铁律 #6）
        tbl = inst.noise_table
        nl = inst.noise_level
        if slide_u > 0.0:                                        # P2 滑音摩擦:**绝对量**注入(不依赖 Air/Noise
            nl = nl + _SLIDE_NOISE * slide_u * self._glide_stren  # 底值;仍用源自己的噪声纹理=有源色彩)
        if nl > 0 and len(tbl) > 0:
            np_ = self.noise_pos
            env_mid = 0.5 * (env_a + env_b) * self.vel_loud * nl * live_amp
            if np_ + n <= len(tbl):
                block[:n] += tbl[np_:np_ + n] * env_mid
                self.noise_pos = (np_ + n) % len(tbl)
            else:                          # 环绕表尾
                m = len(tbl) - np_
                block[:m] += tbl[np_:] * env_mid
                block[m:n] += tbl[:n - m] * env_mid
                self.noise_pos = n - m

        self.samples_since_on = s0 + n


# Nyquist 软淡出宽度（从 config 注入；模块级默认，引擎初始化时覆盖）
_RAMP_HZ = 600.0
# 有起音谱轨迹时的颗粒折减:谐波层自己弹拨弦,录音颗粒只留一点"手指触弦"的质感
_GRAIN_WITH_ATK = 0.30
# pitch-bend 速率上限(半音/秒)。乐句极限≈163(13Hz 颤音×±2st),250 留足余量;
# 触摸条"瞬移"(1000+st/s)被限成 ~16ms 滑音。过小会拖慢真实快甩。
_BEND_RATE_ST_S = 250.0
# 人性化抖动深度（引擎初始化时从 config 覆盖）
_HUM_CENTS = 6.0
_HUM_AMP = 0.05
# 揉弦"绽放"：起振延迟 + 深度渐入时长（ms）——真人揉弦先清音再由浅入深
_VIB_ONSET_MS = 110.0
_VIB_RAMP_MS = 220.0
# P1 表现层深度(引擎初始化时从 config 覆盖):
_LIVE_SPEC = 0.85     # 谱形轨迹深度∈[0,1](0=关=旧行为,A/B 用)
_LIVE_DARK = 0.0      # 亮度随电平暗化(dB/oct)。默认关:残差分析证实即便 150ms 平滑仍产生
                      # 孤立脉冲(块粒度增益步进的固有毛刺);亮度运动由 breath+轨迹承担。
                      # 待 7.8 后给内核加逐样本增益渐变再启用。
_LIVE_BREATH = 0.8    # 亮度慢呼吸幅度(dB/oct,±):静态源的"活气"来源
# P2 演奏法可听度:
_SLIDE_NOISE = 0.30   # 滑音摩擦噪声峰值(绝对 noise_level 当量,随音程强度/滑程弧缩放)
_REGRAIN_LEGATO = 0.16  # 连音换音的低电平起音颗粒(颗粒强的源有效)
_REGRAIN_SLIDE = 0.10   # 滑音到达的更轻颗粒
_ART_DIP_LEGATO = 0.30  # 连音换音凹陷深度(≈-3dB 短弧,换弓/换气感,普适所有音色)
_ART_DIP_SLIDE = 0.22


def set_nyquist_ramp_hz(hz: float):
    global _RAMP_HZ
    _RAMP_HZ = float(hz)


def set_humanize(cents: float, amp: float):
    global _HUM_CENTS, _HUM_AMP
    _HUM_CENTS = float(cents)
    _HUM_AMP = float(amp)


def set_vibrato_shape(onset_ms: float, ramp_ms: float):
    """揉弦绽放:起振延迟 + 渐入时长（ms）。从 config gesture 注入。"""
    global _VIB_ONSET_MS, _VIB_RAMP_MS
    _VIB_ONSET_MS = float(onset_ms)
    _VIB_RAMP_MS = float(ramp_ms)


def set_slide_feel(noise_bump: float, regrain_legato: float, regrain_slide: float,
                   dip_legato: float = 0.30, dip_slide: float = 0.22):
    """P2 演奏法可听度参数。(0,0,0,0,0)=旧行为(A/B 与回滚)。"""
    global _SLIDE_NOISE, _REGRAIN_LEGATO, _REGRAIN_SLIDE, _ART_DIP_LEGATO, _ART_DIP_SLIDE
    _SLIDE_NOISE = float(noise_bump)
    _REGRAIN_LEGATO = float(regrain_legato)
    _REGRAIN_SLIDE = float(regrain_slide)
    _ART_DIP_LEGATO = float(dip_legato)
    _ART_DIP_SLIDE = float(dip_slide)


def set_liveness(spec: float, dark_db_oct: float, breath_db_oct: float = 0.6):
    """P1 表现层深度:谱形轨迹 + 亮度随电平 + 亮度呼吸。(0,0,0)=完全旧行为(A/B 与回滚)。"""
    global _LIVE_SPEC, _LIVE_DARK, _LIVE_BREATH
    _LIVE_SPEC = float(spec)
    _LIVE_DARK = float(dark_db_oct)
    _LIVE_BREATH = float(breath_db_oct)
