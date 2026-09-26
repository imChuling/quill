"""ControlFrame → DDSP 集成 —— 演奏法 × 神经音色（Quill 合体，FR-8×FR-10）。

把一段手势谱（音符 + 滑音 + 揉弦）转成 250Hz 的 (f0, loudness) 控制轨迹（= Part B 的
ControlFrame），喂给训好的 DDSP 神经音色 → 渲染出"用某把琴的神经音色、按你的演奏法弹"的乐句。
这正是 Quill 独一份的合体：**per-琴神经音色(few-shot 克隆) × 共享演奏法库**。

跑：python -W ignore -m neural.perform
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural.ddsp import DDSP, SR, HOP  # noqa: E402

FPS = SR / HOP                          # 250Hz 帧率


def _m2f(midi):
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def _is_new_arch(path):
    """新解码器(含 dropout 层,inp.4=第二 Linear)的 checkpoint 才兼容当前 DDSP。"""
    if not Path(path).exists():
        return False
    try:
        w = torch.load(path, map_location="cpu").get("dec.inp.4.weight")
        return w is not None and w.dim() == 2     # 新arch inp.4=Linear(2D)；旧=LayerNorm(1D)
    except Exception:
        return False


def note_template(real_note_path, sr_in=44100, pre=3):
    """从一个干净真实乐器音抽 loudness 包络模板（裁到主起音，去掉预触发的"双音头"）。

    MIDI 每个音套这个真实包络 → 动态真实(音头/衰减对)，且无预触发双起音。
    """
    from analysis.io import load
    from neural.ddsp import extract_features
    from scipy.signal import find_peaks
    y, _ = load(real_note_path, sr=sr_in)
    _, loud, _ = extract_features(y, sr_in)
    t = loud[0, :, 0].numpy()
    pks, _ = find_peaks(t, prominence=0.8, distance=20)
    if len(pks):
        t = t[max(0, int(pks[0]) - pre):]                 # 裁到主起音 = 一次拨弦
    return t


def control_from_score(score, vib_rate=5.5, template=None):
    """score: [{midi, dur, slide?, vibrato?(cents), decay?}] -> (f0[1,T,1], loud[1,T,1])。

    slide=True：从上一音滑到本音(前 40%)；vibrato=cents：本音正弦揉弦；
    loudness：每音拨弦包络(瞬起+指数衰减)，全句 log 后标准化(对齐训练口径)。
    """
    f0_segs, loud_segs, onsets, prev, cum = [], [], [], None, 0
    for seg in score:
        n = max(2, int(seg["dur"] * FPS))
        onsets.append((cum * HOP, not seg.get("slide")))     # (样本位, 是否拨弦起音)
        cum += n
        f0_t = _m2f(seg["midi"])
        f0 = np.full(n, f0_t, dtype=np.float64)
        if seg.get("slide") and prev is not None:
            sl = max(2, int(n * 0.4))
            f0[:sl] = _m2f(prev) * (f0_t / _m2f(prev)) ** np.linspace(0, 1, sl)
        if seg.get("vibrato"):
            t = np.arange(n) / FPS
            u = np.clip((t - 0.11) / 0.22, 0.0, 1.0)          # 揉弦绽放:延迟 110ms + 渐入 220ms
            venv = u * u * (3.0 - 2.0 * u)                    # smoothstep(由浅入深,像真人)
            f0 = f0 * 2.0 ** (venv * seg["vibrato"] * np.sin(2 * np.pi * vib_rate * t) / 1200.0)
        if template is not None:                              # 真实包络模板(推荐)：动态真实、无双音头
            loge = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(template)), template)
        else:                                                 # 兜底：合成拨弦包络(动态偏假)
            t = np.arange(n) / FPS
            env = np.exp(-t / seg.get("tau", 0.45))
            a = max(1, int(0.010 * FPS))
            env[:a] *= np.linspace(0.0, 1.0, a)
            loge = np.log(env + 1e-6)
            loge = (loge - loge.mean()) / (loge.std() + 1e-5)
        f0_segs.append(f0); loud_segs.append(loge)
        prev = seg["midi"]
    return (torch.tensor(np.concatenate(f0_segs))[None, :, None].float(),
            torch.tensor(np.concatenate(loud_segs))[None, :, None].float(), onsets)


def control_from_behavior(score, behavior_model, crossfade_ms=20.0,
                          pitch_glide=True, envelope="pluck", traj_gain=1.0,
                          vibrato_source="retrieval", vibrato_predictor=None,
                          instrument_idx=0, randomness=0.3,
                          label_processing="full",
                          concat_lambda=0.0):
    """score + 行为模型 → (f0[1,T,1], loud[1,T,1]) 控制流。

    与 control_from_score 的区别:这里的微观动态(pitch vibrato/bend、energy envelope)
    来自行为模型预测的 4×48 轨迹,而不是手编规则。

    score: [{midi, dur, art?}]  art: 0=plain 1=vibrato 2=slide,缺省 0
    behavior_model: 有 .predict(feat) -> (4,48) 的对象;若其 use_art=True,
                    feat 会带上第 7 维演奏法标签(条件化检索)。
    pitch_glide: True 时 f0 在音符边界 crossfade(会产生滑音感);
                 False 时 f0 在重叠区中点硬切换,只有 loudness 平滑过渡。
    envelope: "pluck"(指数衰减,拨弦) / "sustain"(起音-保持-释放,弓弦/管乐)
    traj_gain: 轨迹作用强度(1.0=原样;>1 夸张微观动态,跨乐器迁移听感更明显)
    vibrato_source: "retrieval"(默认,pitch 全部来自检索轨迹) /
                    "model"(vibrato 音符的 pitch 由学习模型合成,plain/slide 不变)
    vibrato_predictor: VibratoPredictor 实例(vibrato_source="model" 时必须提供)
    instrument_idx: 乐器索引(0=violin 1=cello 2=guitar),传给 vibrato model
    randomness: 颤音随机性 0–1(仅 vibrato_source="model" 时有效)
    label_processing: "full"(默认,标签条件化后处理:plain 低通/钳制,
                      vibrato 去漂移/钳制/AM) / "none"(检索轨迹原样,
                      无任何标签后处理——归因消融用) / "uniform"(所有音符
                      统一去漂移+钳制 ±50c,不看标签——归因消融用)
    """
    cf = max(1, int(crossfade_ms / 1000.0 * FPS))
    use_art = bool(getattr(behavior_model, "use_art", False))
    segs_f0, segs_loud = [], []

    # precompute onsets so gap calculation is O(1) per note
    _onsets = []
    _cum = 0.0
    for s in score:
        _onsets.append(s.get("onset", _cum))
        _cum = _onsets[-1] + s["dur"]

    prev_traj = None
    for i, note in enumerate(score):
        midi, dur = note["midi"], note["dur"]
        n = max(4, int(dur * FPS))

        if i > 0:
            prev_gap = max(0.0, float(_onsets[i] - (_onsets[i-1] + score[i-1]["dur"])))
        else:
            prev_gap = 1.0
        if i < len(score) - 1:
            next_gap = max(0.0, float(_onsets[i+1] - (_onsets[i] + dur)))
        else:
            next_gap = 1.0
        base_feat = [float(midi), dur, prev_gap, next_gap,
                     1.0 if i == 0 else 0.0,
                     1.0 if i == len(score) - 1 else 0.0]
        if use_art:
            base_feat.append(float(note.get("art", 0)))
        feat = np.array(base_feat, np.float32)

        if concat_lambda > 0 and prev_traj is not None:
            traj = behavior_model.predict(feat, prev_traj=prev_traj,
                                          concat_lambda=concat_lambda)
        else:
            traj = behavior_model.predict(feat)
        pitch_d = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[0])
        energy_d = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, 48), traj[1]) * traj_gain
        # 音准锚定:真实轨迹带演奏者音准误差(均值≠0),gain 会放大跑调。
        # plain/vibrato 减全轨迹均值(保形状去偏);slide 锚尾段(滑完落在目标音上)。
        art_i = int(note.get("art", 0))
        anchor = pitch_d[-max(1, n // 4):].mean() if art_i == 2 else pitch_d.mean()
        pitch_d = (pitch_d - anchor) * traj_gain

        # slide 音程对齐:检索特征里没有音程方向,检索到的滑音轨迹方向/幅度
        # 可能与实际旋律音程不符(听感=滑音音准跑偏)。分解:滑音的"时机"
        # (什么时候滑、滑多快)取自检索轨迹的累积位移进度曲线;"幅度与方向"
        # 由旋律真实音程决定 → 从上一个音出发、精确落到本音,形状仍是演奏者的。
        if art_i == 2 and i > 0:
            interval = (float(score[i-1]["midi"]) - float(midi)) * 100.0
            if abs(interval) > 1.0:
                # 滑音进度 = 向终点的净位移比例(单调化+钳制)。不用累积 |位移|:
                # 轨迹尾部的 release 小抖动会稀释进度,导致滑到 85% 就平台,
                # 音准悬空 interval×15%(实测 -62c)。净位移法对尾部抖动免疫。
                s = pitch_d - float(np.median(pitch_d[:3]))
                # 分母用"保持段"(50%–85% 区间)中位值而非末帧:有的库轨迹整个
                # 保持段都偏 50c、到 release 才修正(演奏者滑走音了),按末帧
                # 归一会把走音平台也复现出来;按保持段归一 = 滑完即落准。
                hold = float(np.median(s[int(0.5 * n):max(int(0.5 * n) + 2, int(0.85 * n))]))
                u = None
                if abs(hold) > 5.0:
                    u = np.clip(np.maximum.accumulate(s / hold), 0.0, 1.0)
                    # 守卫:滑得太早(前 15% 就完成=硬跳)或起步太晚(60% 处还没
                    # 开始=held-then-jump)都是库里的假滑音轨迹,弃用
                    t95 = int(np.argmax(u >= 0.95))
                    t05 = int(np.argmax(u >= 0.05))
                    if t95 < max(2, int(0.15 * n)) or t05 > int(0.6 * n):
                        u = None
                if u is None:
                    # 兜底:快速换把式滑入(小提琴 shift ~50-120ms,不是匀速 portamento)
                    sl = max(2, min(int(0.12 * FPS), int(n * 0.4)))
                    u = np.ones(n)
                    t01 = np.linspace(0.0, 1.0, sl)
                    u[:sl] = t01 * t01 * (3.0 - 2.0 * t01)
                    resid = np.zeros(n)   # 假轨迹的纹理含跳变本身,不可回叠
                else:
                    # 残差 = 检索轨迹刨掉它自己的宏观滑线后剩下的微观纹理
                    # (过冲回稳、滑后接揉弦、收尾抖动)——按轨迹自己的节奏定义。
                    resid_raw = s - hold * u
                    # 宏观滑线不直接用检索进度曲线:抖动轨迹经单调化会变阶梯
                    # (每次回抖出一个平台),整音连滑的轨迹又慢成抹音。
                    # 只取"滑音时长"(t95,钳到换把范围 50–200ms),形状用
                    # smoothstep——一次干净加速滑到位,这才是弓弦换把的听感。
                    sl = int(np.clip(t95, max(2, int(0.05 * FPS)),
                                     min(int(0.2 * FPS), max(2, n // 2))))
                    u = np.ones(n)
                    t01 = np.linspace(0.0, 1.0, sl)
                    u[:sl] = t01 * t01 * (3.0 - 2.0 * t01)
                    t95 = sl
                    # 落地后重新锚零 + 线性去趋势(人声轨迹落地后仍持续上漂,
                    # 尾部悬 +30c;去趋势保留抖动、去掉漂移),
                    # 钳制 ±40c 防宏观泄漏,随进度渐入(滑行中不干扰音准)。
                    land = resid_raw[t95:]
                    if len(land) >= 4:
                        x01 = np.arange(len(land), dtype=np.float64)
                        resid_raw[t95:] = land - np.polyval(np.polyfit(x01, land, 1), x01)
                    resid_raw = resid_raw - float(np.median(resid_raw[t95:]))
                    resid = np.clip(resid_raw, -40.0, 40.0) * u
                pitch_d = interval * (1.0 - u) + resid

        # 起音稳定:轨迹开头几帧可能带演奏者的 pitch 不稳(尤其弓弦乐器咬弦瞬间),
        # 前 ~30ms 把 pitch_d 从 0 淡入,让音头音准干净,之后轨迹正常接管。
        # slide 除外:滑音的头部偏移正是"从上一个音出发",不能抹掉。
        onset_frames = max(1, int(0.030 * FPS))
        if art_i != 2 and onset_frames < n:
            fade = np.linspace(0.0, 1.0, onset_frames)
            pitch_d[:onset_frames] *= fade

        # plain 音符抑振:URMP "plain" 池的轨迹常含自然揉弦,不压制则 plain/vibrato 无对比度。
        # 低通滤波去掉 >3Hz pitch 振荡(保留慢速漂移,去掉 vibrato 频段)。
        if label_processing == "full" and art_i == 0 and n >= 8:
            from scipy.signal import butter, sosfiltfilt
            cutoff = 3.0  # Hz
            nyq = FPS / 2.0
            if cutoff < nyq:
                sos = butter(2, cutoff / nyq, btype='low', output='sos')
                pitch_d = sosfiltfilt(sos, pitch_d)
            # plain = 音高稳定;人声库的起音上挑(scoop)可达 ±400c,快音里
            # 前 1/3 都走音。钳制 ±50c:保留慢速漂移,掐掉大幅起音挑滑。
            pitch_d = np.clip(pitch_d, -50.0, 50.0)

        # vibrato 去漂移 + 幅度钳制:揉弦的定义是绕稳定中心的振荡;
        # 库轨迹(尤其人声)常带 <2Hz 慢漂移,减掉它保住音准中心,
        # 5-7Hz 振荡原样保留;再钳制 ±35cents 防过宽揉弦。
        if label_processing == "full" and art_i == 1:
            if n >= 8:
                from scipy.signal import butter, sosfiltfilt
                nyq = FPS / 2.0
                sos_d = butter(2, 2.0 / nyq, btype='low', output='sos')
                pitch_d = pitch_d - sosfiltfilt(sos_d, pitch_d)
            pitch_d = np.clip(pitch_d, -35.0, 35.0)

        # uniform 模式:所有音符统一去漂移+钳制 ±50c,不看标签(归因消融)
        if label_processing == "uniform" and art_i != 2:
            if n >= 8:
                from scipy.signal import butter, sosfiltfilt
                nyq = FPS / 2.0
                sos_u = butter(2, 2.0 / nyq, btype='low', output='sos')
                pitch_d = pitch_d - sosfiltfilt(sos_u, pitch_d)
            pitch_d = np.clip(pitch_d, -50.0, 50.0)

        # vibrato model 分支:art=1(vibrato) 时用学习模型替代检索的 pitch 轨迹
        if vibrato_source == "model" and art_i == 1 and vibrato_predictor is not None:
            prev_gap_v = prev_gap
            next_gap_v = next_gap
            vib_cents, _ = vibrato_predictor.synth(
                midi, dur, prev_gap_v, next_gap_v,
                1.0 if i == 0 else 0.0,
                1.0 if i == len(score) - 1 else 0.0,
                art_i, instrument_idx=instrument_idx,
                n_frames=n, randomness=randomness)
            pitch_d = vib_cents

        f0_seg = _m2f(midi) * 2.0 ** (pitch_d / 1200.0)

        t = np.arange(n) / FPS
        if envelope == "sustain":
            # 弓弦包络按乐句位置分角色:句内音符是连弓,不能每音都重起音
            # (否则每个边界 -13dB 深坑,听感"断/卡";滑音内的坑最刺耳)。
            base = np.ones(n)
            if i == 0:
                a = max(1, int(0.03 * FPS))
                base[:a] = np.linspace(0.05, 1.0, a)      # 句首:真起音
            elif art_i == 2:
                a = max(1, int(0.02 * FPS))
                base[:a] = np.linspace(0.9, 1.0, a)       # 滑音:连弓滑入,近无缝
            else:
                a = max(1, int(0.02 * FPS))
                base[:a] = np.linspace(0.6, 1.0, a)       # 句内换音:轻起音
            if i == len(score) - 1:
                r = max(1, int(min(0.06, dur * 0.2) * FPS))
                base[-r:] *= np.linspace(1.0, 0.3, r)     # 句尾:收弓
            else:
                nxt_slide = int(score[i + 1].get("art", 0)) == 2
                r = max(1, int(min(0.04, dur * 0.15) * FPS))
                base[-r:] *= np.linspace(1.0, 0.95 if nxt_slide else 0.8, r)
        else:
            base = np.exp(-t / max(dur * 0.6, 0.2))
            a = max(1, int(0.008 * FPS))
            base[:a] *= np.linspace(0.01, 1.0, a)
        log_e = np.log(base + 1e-7)
        log_e = (log_e - log_e.mean()) / (log_e.std() + 1e-5)
        # 句内音符 z-score 后加谷底下限:逐音符标准化保住每音的动态弧度
        # (这是听感有生命力的来源),但句内边界不许跌成深坑(断/卡感);
        # 句首起音、句尾收弓保留全深度。
        if 0 < i < len(score) - 1:
            log_e = np.maximum(log_e, -1.0)
        loud_seg = log_e + energy_d * 0.5

        # vibrato amplitude modulation: real bowed vibrato causes loudness to
        # fluctuate at the vibrato rate (harmonics drift in/out of body resonances).
        # Extract the oscillation from pitch_d and couple ~15% to loudness.
        if label_processing == "full" and art_i == 1 and n >= 8:
            from scipy.signal import detrend as _detrend
            pitch_osc = _detrend(pitch_d)
            peak = np.abs(pitch_osc).max() + 1e-6
            am = 0.15 * (pitch_osc / peak)  # ±15% loudness modulation
            loud_seg = loud_seg + am

        segs_f0.append(f0_seg.astype(np.float64))
        segs_loud.append(loud_seg.astype(np.float64))
        prev_traj = traj

    total = sum(len(s) for s in segs_f0)
    f0_out = np.zeros(total, np.float64)
    loud_out = np.zeros(total, np.float64)
    w_f0 = np.zeros(total, np.float64)
    w_loud = np.zeros(total, np.float64)
    pos = 0
    for i in range(len(segs_f0)):
        n = len(segs_f0[i])
        env = np.ones(n)
        fi = min(cf, n // 2) if i > 0 else 0
        fo = min(cf, n // 2) if i < len(segs_f0) - 1 else 0
        if fi:
            env[:fi] *= np.linspace(0, 1, fi)
        if fo:
            env[-fo:] *= np.linspace(1, 0, fo)

        if pitch_glide:
            env_f0 = env
        else:
            # f0 不参与 crossfade:重叠区前半归上一个音,后半归当前音 → 无插值滑音
            env_f0 = np.ones(n)
            if fi:
                env_f0[:fi // 2] = 0.0
            if fo:
                keep = fo - fo // 2          # 尾部重叠的前半仍归本音
                env_f0[n - fo + keep:] = 0.0

        if i > 0:
            pos -= min(cf, len(segs_f0[i-1]) // 2, n // 2)
        end = pos + n
        f0_out[pos:end] += segs_f0[i] * env_f0
        loud_out[pos:end] += segs_loud[i] * env
        w_f0[pos:end] += env_f0
        w_loud[pos:end] += env
        pos = end
    T = pos
    # f0 权重为 0 的帧(硬切换边界的空隙)用前值填充
    sw_f0 = w_f0[:T].copy()
    f0_n = f0_out[:T] / np.maximum(sw_f0, 1e-8)
    hole = sw_f0 < 1e-8
    if hole.any():
        idx = np.where(~hole)[0]
        f0_n = np.interp(np.arange(T), idx, f0_n[idx])
    sw_l = np.maximum(w_loud[:T], 1e-8)
    return (torch.tensor(f0_n)[None, :, None].float(),
            torch.tensor(loud_out[:T] / sw_l)[None, :, None].float())


def _pluck_grain(tplib="assets/libraries/guitar.tplib", sr_out=SR):
    """从 .tplib 取一个拨弦颗粒，重采样到 DDSP 采样率（起音瞬态混合用）。"""
    import torchaudio
    from analysis.tplib import load_tplib
    lib = load_tplib(tplib)
    if not lib.get("pluck"):
        return None
    g = np.asarray(lib["pluck"][0]["grain"], dtype=np.float32)     # 60ms @44100
    g = torchaudio.functional.resample(torch.tensor(g)[None], 44100, sr_out).numpy()[0]
    return g / (np.max(np.abs(g)) + 1e-9)



@torch.no_grad()
def render(model_path, f0, loud, device=None, noise_gain=1.0):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = DDSP(dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    y = model(f0.to(device), loud.to(device), noise_gain=noise_gain).cpu().numpy()[0]
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)



def _mute_template(template, tau=0.15):
    """闷音(palm mute)：在真实模板上叠快速衰减 → 短促闷响。"""
    t = np.arange(len(template)) / FPS
    lin = np.exp(template) * np.exp(-t / tau)
    lg = np.log(lin + 1e-6)
    return (lg - lg.mean()) / (lg.std() + 1e-5)


def render_midi(model_path, score, template, device=None, vib_rate=5.5):
    """MIDI 单声部乐句 -> 神经音色。控制曲线类演奏法(f0/响度轨迹):

    f0 类:pluck/legato/slide/bend/vibrato/**trill(颤指,两音方波交替)**/**gliss(长滑奏)**;
    响度类:mute(闷)/**tremolo(震音,快速反复起音)**/**swell(音量渐入,无拨弦头)**。
    起音类各自落 loudness 模板;legato/slide **不重起音**(续衰减只滑 f0)。
    """
    frames = [max(2, int(s["dur"] * FPS)) for s in score]
    total = sum(frames)
    f0 = np.zeros(total, np.float64)
    loud = np.full(total, float(template[-1]), np.float64)     # 默认=模板尾(续衰减)
    cum, prev = 0, None
    for seg, n in zip(score, frames):
        midi = seg["midi"]; art = seg.get("art", "pluck")
        tt = np.arange(n) / FPS
        sf0 = np.full(n, _m2f(midi))
        if art in ("legato", "slide") and prev is not None:    # 续奏:滑 f0,不重起音
            sl = max(2, int(n * (0.22 if art == "legato" else 0.4)))
            sf0[:sl] = _m2f(prev) * (_m2f(midi) / _m2f(prev)) ** np.linspace(0, 1, sl)
        elif art == "trill":                                   # 颤指:本音↔本音+iv 方波交替
            iv = seg.get("trill", 2); rate = seg.get("trill_rate", 8.0)
            sf0 = _m2f(midi) * 2 ** ((np.sin(2 * np.pi * rate * tt) > 0) * iv / 12.0)
        elif art == "gliss":                                   # 长滑奏:整音滑到目标
            tgt = seg.get("gliss_to", midi + 7)
            sf0 = _m2f(midi) * (_m2f(tgt) / _m2f(midi)) ** np.linspace(0, 1, n)
        if art == "bend":                                      # 推弦:~bend_ms 内快速升到 +bend 半音再保持
            b = seg.get("bend", 2.0)
            h = max(2, min(n - 1, int(seg.get("bend_ms", 170) / 1000.0 * FPS)))
            sf0[:h] *= 2 ** (np.linspace(0, b, h) / 12.0); sf0[h:] *= 2 ** (b / 12.0)
        if seg.get("vibrato"):
            vu = np.clip((tt - 0.07) / 0.16, 0.0, 1.0)         # 揉弦绽放(solo:更早起振、更快渐入)
            ve = vu * vu * (3.0 - 2.0 * vu)
            sf0 *= 2 ** (ve * seg["vibrato"] * np.sin(2 * np.pi * vib_rate * tt) / 1200.0)
        f0[cum:cum + n] = sf0
        # ---- 响度 ----
        if art in ("legato", "slide"):                         # 续衰减,不动 loud
            pass
        elif art == "tremolo":                                 # 震音:快速反复落模板
            period = max(2, int(FPS / seg.get("trem_rate", 10.0)))
            for st in range(0, n, period):
                m = min(len(template), n - st); loud[cum + st:cum + st + m] = template[:m]
        elif art == "swell":                                   # 渐入:升余弦淡入,无拨弦头
            r = min(n, max(2, int(n * seg.get("swell_frac", 0.5))))
            base, peak = float(template[-1]), float(np.percentile(template, 75))
            seg_loud = np.full(n, peak)
            seg_loud[:r] = base + (peak - base) * 0.5 * (1 - np.cos(np.linspace(0, np.pi, r)))
            loud[cum:cum + n] = seg_loud
        else:                                                  # pluck/bend/trill/gliss/mute:单次起音
            tpl = _mute_template(template) if art == "mute" else template
            m = min(len(tpl), total - cum); loud[cum:cum + m] = tpl[:m]
        cum += n; prev = midi
    f0t = torch.tensor(f0)[None, :, None].float()
    lot = torch.tensor(loud)[None, :, None].float()
    return render(model_path, f0t, lot, device)


def render_strum(model_path, chord, template, dur=1.6, spread_ms=110, device=None):
    """扫弦:和弦各音错开 spread_ms 触发(神经音色单声部,音频级错峰叠加)。

    spread_ms = **扫弦速度旋钮**(产品里给用户调:小=快扫紧、大=慢扫逐根进)。
    """
    offs = np.linspace(0, spread_ms, len(chord))
    parts = []
    for m, off in zip(chord, offs):
        y = render_midi(model_path, [{"midi": int(m), "dur": dur}], template, device)
        parts.append((int(SR * off / 1000.0), np.asarray(y)))
    total = max(s + len(y) for s, y in parts)
    buf = np.zeros(total, np.float64)
    for s, y in parts:
        buf[s:s + len(y)] += y
    return (buf / (np.max(np.abs(buf)) + 1e-9) * 0.9).astype(np.float32)


def control_from_cooked(cooked, template, tail_s=1.0, vib_rate=5.5,
                        vib_max_cents=50.0, bend_is_semitones=True):
    """手势引擎的 cooked 事件 -> 250Hz (f0, loudness) 控制轨(= Part B 的 ControlFrame)。

    单声部:on=拨弦(落模板)、legato/slide=续衰减只滑 f0、bend/vibrato=按时间起作用的 f0 调制。
    (和弦扫弦是复音,走 render_strum;这里按时间线建单声部控制。)
    """
    events = sorted(cooked, key=lambda e: e[0])
    onsets = [(int(round(e[0] * FPS)), e[1], e[2], e[5])
              for e in events if e[1] in ("on", "legato", "slide")]
    if not onsets:
        z = np.zeros(int(FPS))
        return z, np.full_like(z, float(template[-1]))
    n = int(round((events[-1][0] + tail_s) * FPS)) + 1
    f0 = np.zeros(n, np.float64)
    loud = np.full(n, float(template[-1]), np.float64)
    first = onsets[0][0]
    if first > 0:
        loud[:first] = float(template[-1]) - 6.0                # 首音前压低(近静音)
    for k, (fr, art, note, frm) in enumerate(onsets):
        nxt = onsets[k + 1][0] if k + 1 < len(onsets) else n
        nxt = max(nxt, fr + 2)
        seg = np.full(nxt - fr, _m2f(note))
        if art in ("legato", "slide") and frm is not None and frm >= 0:
            r = min(nxt - fr, max(2, int((nxt - fr) * (0.22 if art == "legato" else 0.4))))
            seg[:r] = _m2f(frm) * (_m2f(note) / _m2f(frm)) ** np.linspace(0, 1, r)
        f0[fr:nxt] = seg
        if art == "on":                                         # 拨弦:落模板(续衰减留给后续)
            m = min(len(template), n - fr); loud[fr:fr + m] = template[:m]
    # bend / vibrato:按事件时间起,持续到下个同类事件(stream 用 0 关闭)
    bend = np.zeros(n); vib = np.zeros(n)
    for e in events:
        fr = min(n - 1, int(round(e[0] * FPS)))
        if e[1] == "bend":
            bend[fr:] = float(e[4]) if bend_is_semitones else float(e[4]) / 8192.0 * 2.0
        elif e[1] == "vibrato":
            vib[fr:] = float(e[4]) * vib_max_cents
    tt = np.arange(n) / FPS
    f0 = f0 * 2 ** (bend / 12.0) * 2 ** (vib * np.sin(2 * np.pi * vib_rate * tt) / 1200.0)
    return f0, loud


def render_performance(model_path, raw_events, template, library=None, device=None):
    """真实 MIDI 演奏 -> 神经音色。raw:[(t,kind,note,vel[,val])] 经手势引擎判演奏法再渲染。

    闭环:弹 MIDI(音符+手势) -> GestureInterpreter 判 pluck/legato/slide/bend/vibrato
    -> control_from_cooked 建 (f0,loudness) -> DDSP 神经音色发声。
    """
    from runtime.gesture import GestureInterpreter
    cooked = GestureInterpreter(library=library).process(raw_events)
    f0, loud = control_from_cooked(cooked, template)
    f0t = torch.tensor(f0)[None, :, None].float()
    lot = torch.tensor(loud)[None, :, None].float()
    return render(model_path, f0t, lot, device)


def main(argv=None):
    import soundfile as sf
    out = Path("assets/demos"); out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    # 吉他乐句：拨几个音 → 滑音 64->67 → 末音揉弦
    guitar_score = [
        {"midi": 60, "dur": 0.5}, {"midi": 64, "dur": 0.5},
        {"midi": 67, "dur": 0.5}, {"midi": 69, "dur": 0.6, "slide": True},
        {"midi": 67, "dur": 0.5}, {"midi": 64, "dur": 1.2, "vibrato": 30},
    ]
    f0, lo, onsets = control_from_score(guitar_score)
    g_grain = _pluck_grain("assets/libraries/guitar.tplib")
    # 木吉他模型好了优先用，否则退回 Fender；起音叠真实拨弦颗粒
    g_model = ("assets/demos/ddsp_acoustic_guitar.pt"
               if Path("assets/demos/ddsp_acoustic_guitar.pt").exists()
               and _is_new_arch("assets/demos/ddsp_acoustic_guitar.pt")
               else "assets/demos/ddsp_fender_strat.pt")
    try:
        y = render(g_model, f0, lo, dev)                      # 不叠颗粒(那版更糟,已撤)
        sf.write(out / "perform_guitar_neural.wav", y, SR)
        print(f"演奏法×神经吉他({Path(g_model).stem}) -> perform_guitar_neural.wav")
    except Exception as e:
        print(f"  吉他渲染跳过: {str(e)[:60]}")

    # 贝斯线：根-五-八 → 滑音 → 揉
    bass_score = [
        {"midi": 40, "dur": 0.45, "decay": 2.5}, {"midi": 40, "dur": 0.45, "decay": 2.5},
        {"midi": 47, "dur": 0.45, "decay": 2.5}, {"midi": 52, "dur": 0.5, "decay": 2.5},
        {"midi": 55, "dur": 0.7, "slide": True, "decay": 2.0},
        {"midi": 52, "dur": 1.0, "vibrato": 25, "decay": 1.5},
    ]
    f0b, lob, onb = control_from_score(bass_score)
    b_grain = _pluck_grain("assets/libraries/bass.tplib")
    try:
        if not _is_new_arch("assets/demos/ddsp_electric_bass.pt"):
            raise RuntimeError("电贝斯模型是旧架构,等新训练器重训完")
        yb = render("assets/demos/ddsp_electric_bass.pt", f0b, lob, dev)
        sf.write(out / "perform_bass_neural.wav", yb, SR)
        print("演奏法×神经电贝斯 -> perform_bass_neural.wav")
    except Exception as e:
        print(f"  贝斯渲染跳过: {str(e)[:60]}")
    # ---- 逐个演奏法 demo（MIDI 驱动，真实 Fender 音做 loudness 模板）----
    fender = "assets/demos/ddsp_fender_strat.pt"
    note_src = ("/Users/lichuling/Desktop/MT/datasets/IDMT-SMT-GUITAR_V2/"
                "dataset1/Fender Strat Clean Neck SC/audio/G53-45105-1111-00006.wav")
    if Path(fender).exists() and Path(note_src).exists():
        tmpl = note_template(note_src)
        arts = {
            "pluck":   [{"midi": 64, "dur": 1.4}],                                   # 单拨弦
            "legato":  [{"midi": 60, "dur": 0.45}, {"midi": 64, "dur": 0.45, "art": "legato"},
                        {"midi": 67, "dur": 1.0, "art": "legato"}],                  # 锤勾连奏(一拨)
            "slide":   [{"midi": 60, "dur": 0.5}, {"midi": 67, "dur": 1.2, "art": "slide"}],
            "mute":    [{"midi": 52, "dur": 0.4, "art": "mute"}, {"midi": 52, "dur": 0.4, "art": "mute"},
                        {"midi": 52, "dur": 0.4, "art": "mute"}, {"midi": 52, "dur": 0.4, "art": "mute"}],
            "bend":    [{"midi": 67, "dur": 1.6, "art": "bend", "bend": 2.0}],       # 推弦 +全音
            "vibrato": [{"midi": 64, "dur": 1.6, "vibrato": 35}],
        }
        for name, sc in arts.items():
            try:
                y = render_midi(fender, sc, tmpl, dev)
                y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
                sf.write(out / f"art_{name}.wav", y, SR)
                print(f"  演奏法[{name}] -> art_{name}.wav")
            except Exception as e:
                print(f"  演奏法[{name}] 跳过: {str(e)[:60]}")
        try:                                                                        # 扫弦 E 大和弦
            ys = render_strum(fender, [40, 47, 52, 56, 59, 64], tmpl, device=dev)
            sf.write(out / "art_strum.wav", ys, SR)
            print("  演奏法[strum] -> art_strum.wav")
        except Exception as e:
            print(f"  演奏法[strum] 跳过: {str(e)[:60]}")

    # ---- 闭环 demo：真实 MIDI 事件 -> 手势引擎判演奏法 -> 神经音色 ----
    if Path(fender).exists() and Path(note_src).exists():
        # 一段表情乐句:拨弦 60 -> 锤勾 62(+2半音=legato) -> 滑音 67(+5=slide) -> 揉弦 CC
        raw = [
            (0.00, "on", 60, 100), (0.45, "on", 62, 96), (0.50, "off", 60, 0),
            (0.90, "on", 67, 100), (0.95, "off", 62, 0),
            (1.40, "cc", 1, 0, 110),                       # 揉弦深度 110/127
            (2.40, "off", 67, 0),
        ]
        try:
            y = render_performance(fender, raw, tmpl, device=dev)
            y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
            sf.write(out / "perform_midi_gesture.wav", y, SR)
            print("  闭环[MIDI→手势引擎→神经音色] -> perform_midi_gesture.wav")
        except Exception as e:
            print(f"  闭环 demo 跳过: {str(e)[:80]}")

    # ---- C：电贝斯也跑一遍演奏法 + 闭环（广度验证）----
    bass_model = "assets/demos/ddsp_electric_bass.pt"
    bass_note = ("/Users/lichuling/Desktop/MT/datasets/IDMT-SMT-BASS/"
                 "ES/NO/BS_1_EQ_1_FS_NO_1_0.wav")
    if Path(bass_model).exists() and _is_new_arch(bass_model) and Path(bass_note).exists():
        btmpl = note_template(bass_note)
        bass_arts = {                                          # 贝斯音域(E1=28..G2=43)
            "pluck":   [{"midi": 33, "dur": 1.4}],            # A1 指弹
            "legato":  [{"midi": 28, "dur": 0.5}, {"midi": 31, "dur": 0.5, "art": "legato"},
                        {"midi": 33, "dur": 1.0, "art": "legato"}],
            "slide":   [{"midi": 28, "dur": 0.5}, {"midi": 35, "dur": 1.2, "art": "slide"}],
            "mute":    [{"midi": 33, "dur": 0.35, "art": "mute"}] * 4,   # 闷音断奏
            "vibrato": [{"midi": 33, "dur": 1.6, "vibrato": 25}],
        }
        for name, sc in bass_arts.items():
            try:
                y = render_midi(bass_model, sc, btmpl, dev)
                y = (y / (np.max(np.abs(y)) + 1e-9) * 0.9).astype(np.float32)
                sf.write(out / f"bass_art_{name}.wav", y, SR)
                print(f"  贝斯演奏法[{name}] -> bass_art_{name}.wav")
            except Exception as e:
                print(f"  贝斯演奏法[{name}] 跳过: {str(e)[:60]}")
        # 闭环 bassline groove：拨根→锤勾五度→八度→滑回（真 MIDI→手势引擎→神经贝斯）
        raw_bass = [
            (0.00, "on", 28, 110), (0.50, "off", 28, 0),          # E1 拨
            (0.50, "on", 28, 100), (0.95, "on", 31, 90),          # 锤勾 +3(legato)
            (1.00, "off", 28, 0), (1.40, "off", 31, 0),
            (1.40, "on", 40, 105), (1.90, "off", 40, 0),          # E2 八度
            (1.90, "on", 40, 100), (2.30, "on", 35, 95),          # 滑到 B1(-5=slide)
            (2.35, "off", 40, 0), (3.10, "off", 35, 0),
        ]
        try:
            yb = render_performance(bass_model, raw_bass, btmpl, device=dev)
            yb = (yb / (np.max(np.abs(yb)) + 1e-9) * 0.9).astype(np.float32)
            sf.write(out / "perform_midi_bass.wav", yb, SR)
            print("  闭环[MIDI→手势引擎→神经贝斯] -> perform_midi_bass.wav")
        except Exception as e:
            print(f"  贝斯闭环 跳过: {str(e)[:80]}")

    print("\nControlFrame → DDSP 集成：手势谱(音符+滑音+揉弦) -> 神经音色发声。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
