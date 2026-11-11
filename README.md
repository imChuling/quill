# Quill · 声音取色笔

> 从一段录音中同时吸取**音色**与**演奏法**，秒级生成一件可用 MIDI 键盘实时演奏的新乐器。
> License: CC BY-NC 4.0 (`LICENSE`) | Demo video: [YouTube](https://www.youtube.com/watch?v=EhepbJxW9SM)
> | ISMIR 2026 LBD: [paper page](https://ismir2026program.ismir.net/lbd_613.html#details) | Model cards: `checkpoints/MODEL_CARDS.md`
> | Behavior library format: `docs/behavior_library_format.md`

## Why Quill (the innovation)

Turning a recording into something you can play already exists — but every existing tool is a player, not an instrument. A sampler stores the recorded audio itself: press a key and it replays that fixed clip, pitch-shifted — no technique it didn't record, no genuine note length. Commercial MIDI instruments (MODO Bass, Ample Guitar) solve articulation and sustain beautifully, but their timbre is authored by the vendor.

**Quill stores a recipe, not a recording.** It decomposes a few seconds of any pitched sound into harmonic and noise components and resynthesizes it sample-by-sample — any pitch, any duration, any articulation. Playing techniques (slides, legato, strums) are mined few-shot from a single clip as retargetable parameter trajectories, decoupled from timbre and recombinable across sources — a voice that slides like a guitar. To our knowledge, few-shot articulation capture combined with real-time MIDI playability on an arbitrary captured timbre is occupied by neither commercial products nor academic systems.

## Quickstart — run the app

```bash
conda create -n quill python=3.10 && conda activate quill
pip install -r requirements.txt          # 精确复现用 requirements-lock.txt（含传递依赖）
python ui/app_web.py          # native desktop app: capture any sound → play it, with its techniques
```

In the app: **PRESS TO CAPTURE** (record ~5s of any pitched sound, or **FROM A SONG** to
extract an instrument from a mix) → play the keyboard (overlap keys = legato/slide;
chord bursts = strum) → record into the piano roll → swap the tone, replay the same
performance. External class-compliant MIDI keyboards work out of the box.

One recording, two assets. **SAVE** in the capture window stores the tone and, in the
background, extracts a same-named *behavior library* from the same take (needs a few
notes, not one held pitch). **⚗ BAKE DDSP** fine-tunes a neural timbre from the take on
the Mac GPU (8 s+ of audio, ~4 min) and adds a 🧪 voice to the piano roll.

Where the two assets meet — one pair, two lists:
- **TIMBRE** (left): captured tones (marked *live*: they play the keys and render), built-in
  DDSP timbres (violin, trumpet, wine glass) and baked 🧪 timbres (*render*: piano roll only).
- **BEHAVIOR** (right): *rules · gesture* (the interpreter, live only) and every library:
  built-in, distilled, and the ones extracted from your recordings. ⟷ marks a timbre and
  a library that came from the same take.
- The current **timbre × behavior** pair is shown at the top (NOW) and on the piano roll.
  It drives both the live keys and the roll's render; the roll has no separate selectors.
  When a DDSP timbre is selected, the keys keep the current captured tone (shown next to
  the pair) because the live engine is the spectral one.
- After **SAVE** finishes extracting, a notice offers the cross-use in one click: your new
  tone played with the violin library.
- Every knob on the panel acts on the sound (brightness, air/noise, intensity, strum,
  glide, pick, vibrato, ADSR); captured tones get an automatic noise floor from their
  spectrum, adjustable with AIR / NOISE.

**Hear it without installing:** the [demo video](https://www.youtube.com/watch?v=EhepbJxW9SM),
or the `web_demo/` page (a prerendered timbre × style-library matrix, no server needed).

## Evaluation highlights

| What | Number |
|---|---|
| Articulation classification (IDMT-SMT-Guitar, 5164 events, 3-class macro-F1) | rule-tree 0.545 → RF 0.774 → MERT probe **0.828** |
| Formant preservation (+12 st, real voice) | formant drift **22 Hz** (vs 249 Hz naive) |
| Real-time: 16-voice full load, per-block render | median **2.65 ms**, p95 **2.7–3.1 ms** on a quiet M2 (5.8 ms budget @256/44.1k) |
| Note segmentation (synthetic set) | recall 100% |
| Full-chain articulation validation (raw keys → FR-8 → engine, 3 timbres) | slide landing ≤ 0.2% pitch error |

Honest negative result: RAVE-style neural codecs capture timbre near studio grade but
are **not playable** via external control (latents leave the training manifold) — see
§6 of the project report. Playable claims cover the spectral core (L0) only.

## Download checkpoints

Model weights and behavior libraries are not in the git tree. The published set is a
GitHub Release asset; unpack it into `checkpoints/`:

```bash
curl -L https://github.com/imChuling/quill/releases/download/v1.0/quill-checkpoints-v1.0.tar.gz -o quill-checkpoints-v1.0.tar.gz
tar xzf quill-checkpoints-v1.0.tar.gz --strip-components=1 -C checkpoints/

# Verify (first 16 hex chars of SHA-256):
python -c "
import hashlib, pathlib
expected = {
    'ddsp_glass.pt':              'b718be519aee9e87',
    'ddsp_guitarset.pt':          'bdc153946fec299a',
    'glass_style.npz':            '198675c22031590b',
    'guitarset_trajectories.npz': 'ac41f33d253d86bb',
}
ckpt = pathlib.Path('checkpoints')
ok = 0
for name, prefix in expected.items():
    p = ckpt / name
    if not p.exists():
        print(f'MISSING  {name}')
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    status = 'OK' if h == prefix else f'MISMATCH (got {h})'
    print(f'{status:10s}  {name}')
    ok += h == prefix
print(f'\n{ok}/{len(expected)} files verified.')
"
```

| File | Size | SHA-256 prefix | In the release |
|---|---|---|---|
| `ddsp_glass.pt` | 1.9 MB | `b718be51` | yes |
| `ddsp_guitarset.pt` | 1.9 MB | `bdc15394` | yes (GuitarSet audio, CC BY-NC 4.0) |
| `glass_style.npz` | 36 KB | `198675c2` | yes |
| `guitarset_trajectories.npz` | 10 MB | `ac41f33d` | yes (derived from GuitarSet, CC BY-NC 4.0) |
| `ddsp_urmp_vn_12k.pt`, `ddsp_urmp_tpt_12k.pt` | 1.9 MB each | | no: trained on URMP audio |
| `urmp_vn_trajectories.npz`, `urmp_tpt_trajectories.npz` | 4.5 / 2.4 MB | | no: derived from URMP |
| `behavior_distilled.pt`, `vibrato_model.pt` | 6.4 MB / 16 KB | | no: trained on URMP-derived trajectories |

URMP ships no standard license, requires registration and citation, and does not state
terms for derived data, so nothing derived from it is redistributed. With the release
set you get the wine-glass and guitar timbres and libraries, which is what the desktop
app and the web page need. The violin and trumpet results in the paper need the URMP
files: register for URMP, then rebuild the libraries with
`tools/extract_trajectories_urmp.py` and train the two DDSP models with
`neural/train_ddsp.py` (see [Behavior libraries](#behavior-libraries)).

## Reproduce the paper

All numbers from the ISMIR LBD 2026 paper are stored in `checkpoints/eval/eval_results.json`.
To verify them:

Both commands need the checkpoint files listed in `checkpoints/MODEL_CARDS.md` in place under
`checkpoints/` (they are not part of the git tree; the check verifies their hashes first):

```bash
# Structural check (hashes the checkpoints, compares stored results, no GPU needed, <1 s):
python tools/reproduce_lbd.py --check-only

# Full reproduction (re-runs all evaluations, needs checkpoints + GPU, ~10 min):
python tools/reproduce_lbd.py
```

Expected output with the checkpoints present:

```
PASS: all paper numbers reproduced within tolerance.
```

Key numbers: paired counterfactual median ratio 7.01x (CI [4.79, 18.65]), decoupling 12/12,
glass 2x2 4/4, processing ablation none 1.55x / full 7.01x.

## Web demo (no capture, just play)

`web_demo/` is a single static page served by the render service: a 3 × 4 timbre × behavior
matrix of prerendered clips (two melodies; the guitar DDSP timbre is left out for quality, guitar behavior stays), plus "your melody" — the desktop piano roll ported to the browser:
record from the on-screen keys, the computer keyboard or a MIDI keyboard (Web MIDI) with a
light preview tone, drag/resize notes, right-click to mark vibrato or slide, load a MIDI file,
then Play renders the phrase on the server with any timbre × behavior pair. Capture, baking
and real-time Quill sound stay in the desktop app.

```bash
python tools/build_web_demo_matrix.py   # regenerate web_demo/audio + manifest.json
QUILL_WEB_DIR=$PWD/web_demo python -m uvicorn ui.render_service:app --port 8726
```


## Behavior libraries

A behavior library is an `.npz` of note-level performance trajectories (pitch, energy,
brightness, noise; 4 x 48 frames per note) plus retrieval features. Format spec:
`docs/behavior_library_format.md`. A torch-free player that turns a score + library into
250 fps control signals: `tools/behavior_player.py`.

| Library | Source | License | Distributed |
|---|---|---|---|
| `glass_style.npz` | self-recorded wine glass | CC BY-NC 4.0 | yes (GitHub Release v1.0) |
| `guitarset_trajectories.npz` | GuitarSet | derived, source CC BY-NC 4.0 | yes (GitHub Release v1.0) |
| `urmp_vn_trajectories.npz` | URMP violin | derived; URMP has no standard license | **no** |
| `urmp_tpt_trajectories.npz` | URMP trumpet | derived; URMP has no standard license | **no** |

The two URMP-derived libraries are kept in the repository only to reproduce the paper's
numbers and are not included in any release package. URMP requires registration and
citation and does not state terms for derived data, so we do not redistribute them.
To rebuild them locally, register for URMP, then:

```bash
python tools/extract_trajectories_urmp.py --inst vn  --urmp /path/to/URMP -o checkpoints/urmp_vn_trajectories.npz
python tools/extract_trajectories_urmp.py --inst tpt --urmp /path/to/URMP -o checkpoints/urmp_tpt_trajectories.npz
```

This is the script that produced the shipped libraries. `tests/test_urmp_extraction_rules.py`
verifies that its labeling and context rules reproduce every stored label and feature.

## Citing

If you use Quill, please cite:

```bibtex
@inproceedings{li2026quill,
  title={Quill: Learning Timbre and Performance Behavior as Separate, Swappable Assets},
  author={Li, Chuling},
  booktitle={Extended Abstracts for the Late-Breaking Demo Session of the 27th Int. Society for Music Information Retrieval Conf.},
  year={2026}
}
```

Or see `CITATION.cff` for machine-readable format.

## 三层架构

```
L2  神经表现力层（9-11月）   DDSP 微调音色 · 神经过渡 · 表现力控制生成
L1  演奏法引擎（6.20→7.7）   事件切分 · 演奏法分类 · .tplib 库 · MIDI 手势解释器  ★核心创新
L0  谱建模合成核心（6.11→6.19）SMS 谐波+噪声分解 · 任意音高重合成 · 实时 MIDI 演奏
```

> **RAVE Tier-1 探索（6.18–6.23,已收口)**:per-乐器神经 codec 做"近录音室取色"。**音色捕捉成立**
> (贝斯盲听"像贝斯",DDSP/L0 都败的硬乐器拿下);**但可弹性撞墙**——潜码被外部控制操纵即出分布、解码
> glitch(回归桥/真潜码搬运三方案均不可接受)。结论:RAVE 当"studio 音色捕捉"天花板展示,可弹神经
> codec 列研究方向;**7.8 主角 = 取色 × 演奏法**(对标商业音源:别人音色写死、Quill 弹任意捕捉音色)。
> 完整记录见项目工作日志「RAVE Tier-1 探索」(未随仓库公开)。

## 仓库结构（附录 B）

```
quill/
├─ analysis/   io.py harmonic.py residual.py timbre.py tplib.py segment.py features.py
├─ synth/      dsp.py additive.py noise.py voice.py oscillator.py instrument.py render.py
├─ runtime/    engine.py gesture.py midi_in.py
├─ neural/     ddsp.py behavior_distill.py bake_ddsp.py perform.py
├─ ui/         app_web.py bridge.py render_service.py take_store.py
├─ web_demo/   the web version (static page + prerendered clips), served by render_service
├─ eval/       metrics.py oracle_gate.py hero_unified.py protocol_ab.py
├─ tools/      extract_style.py reproduce_lbd.py eval_quill.py
├─ tests/      266 tests (pytest)
├─ checkpoints/ DDSP models + behavior libraries + MODEL_CARDS.md
└─ docs/       ismir_lbd_2026/ changelog.md tplib_format.md
```

> **引擎铁律 #12**：`librosa`/`pesto`/`torch` 等重依赖只许活在 `analysis/`；
> `synth/`、`runtime/` 只依赖 `numpy`(+`numba`)——离线渲染、单测、未来插件移植的共同前提。

## 环境

本项目复用既有 conda 环境 **`mt`**（Python 3.10，已含完整音频栈）。

```bash
conda activate mt
# 阶段1 额外补装（已完成）：
pip install pytest pesto-pitch
```

完整依赖见 `requirements.txt`。若需从零重建环境：

```bash
conda create -n quill python=3.10
conda activate quill
pip install -r requirements.txt
```

## 阶段 1（6.11-6.13）· FR-2 前半 · 已完成 ✅

谐波分析与重合成。入口：无 → 出口：重建盲听过关。

| 交付 | 文件 | 状态 |
|---|---|---|
| FR-1 雏形：录音/导入/增益归一 | `analysis/io.py` | ✅ |
| f0 估计（PESTO 主力 + pyin 交叉校验） | `analysis/harmonic.py::track_f0` | ✅ |
| 谐波提取 `H[60,T]`（STFT 4096/512 + 抛物线峰值插值） | `analysis/harmonic.py::harmonics` | ✅ |
| 加法重合成（相位累加 + 帧间插值，numpy-only） | `synth/additive.py::resynth` | ✅ |
| 单测：已知谐波信号提取误差 < 1dB + 往返 + 无混叠 | `tests/` | ✅ 11 passed |
| 验收：A/B wav + H 热图 + 频谱混叠检查 | `eval/stage1_accept.py` | ✅ |

### 跑验收

```bash
conda activate mt
cd quill
python -m pytest -q                       # 单元测试（11 passed）
python -m eval.stage1_accept              # 内置合成长音自检 -> assets/demos/
python -m eval.stage1_accept voice.wav piano.wav   # 真实录音盲听
python -m eval.stage1_accept --record 4   # 现场录 4s
```

产物落在 `assets/demos/`：`*_orig.wav` / `*_resynth.wav`（盲听 A/B）、
`*_Hheatmap.png`（谐波包络）、`*_spectrum.png`（原音 vs 重合成，验证 Nyquist 无折返）。

### 阶段1 验收口径（实测）

- 合成已知谐波信号提取误差 **< 1dB**（on-bin / off-bin / 低音 65Hz 三档，单测覆盖）
- Nyquist 截断生效，重合成无混叠折返峰（非谐波能量 Δ < 0.05%）
- 低音 f0 < 100Hz 自动把 `n_fft` 升到 8192（谐波间距不足坑位）
- 准谐波声源（酒杯）实测部分音入 `partial_freqs`，重合成按实测比例发振

## 阶段 2（6.14-6.16）· FR-2 完成 + FR-3 · 已完成 ✅

残差/噪声建模 + 共振峰保持的音高重定向 + 音色快照序列化。出口：三例素材 ±12 半音可听。

| 交付 | 文件 | 状态 |
|---|---|---|
| 相位对齐残差提取（STFT 主瓣幅度减法，保相位） | `analysis/residual.py::extract_residual` | ✅ |
| 40 mel 带噪声包络（SMS stochastic, Serra & Smith 1990） | `analysis/residual.py::noise_envelope` | ✅ |
| 噪声整形合成（numpy STFT/ISTFT + mel，分析-修改-合成） | `synth/noise.py::synth_noise` | ✅ |
| FR-3 共振峰保持音高重定向（True Envelope, Röbel-Rodet 2005） | `synth/additive.py::redirect_harmonics` | ✅ |
| 按序号搬运错误版（花栗鼠对照，报告素材） | `redirect_harmonics(mode="index")` | ✅ |
| attack 残差 60ms 入库（`float32[2646]`，首尾淡变） | `analysis/residual.py::attack_residual` | ✅ |
| `TimbreSnapshot` 序列化（Part B 契约，宽容加载器） | `snapshot.py` | ✅ |
| 音色分析管线（audio→snapshot，含颤音检测） | `analysis/timbre.py::analyze_timbre` | ✅ |
| 离线渲染（快照→任意基频，谐波+噪声+attack） | `synth/render.py::render` | ✅ |
| 单测：残差对消/噪声往返/共振峰保持/快照往返/颤音 | `tests/test_stage2.py` | ✅ 12 passed |
| 验收：三例 ±12 半音 + 共振峰 vs 搬运对照 | `eval/stage2_accept.py` | ✅ |

### 跑验收

```bash
conda activate mt && cd quill
python -m pytest -q                          # 23 passed（阶段1+2）
python -m eval.stage2_accept                 # 内置人声/吉他/酒杯 -> 快照 + ±12 渲染
python -m eval.stage2_accept guitar.wav      # 真实单音（谐波声源）
python -m eval.stage2_accept --quasi bowl.wav  # 准谐波声源（酒杯/碗）
```

产物：`assets/snapshots/<name>.npz`（音色快照）；`assets/demos/<name>_{down12,re0,up12}.wav`
（共振峰保持 ±12）、`<name>_up12_index.wav`（花栗鼠对照）、`<name>_fr3.png`（谱包络对比）。

### 阶段2 验收口径（实测，合成素材）

- 同基频重合成保真：谱包络峰漂移 voice 3Hz / guitar 0Hz / wineglass 25Hz
- **FR-3 共振峰保持**（人声，有 ~700Hz 共振峰）：+12 半音谱包络峰漂移
  **3Hz（保持）vs 751Hz（按序号搬运）**——无花栗鼠，见 `voice_fr3.png`
- 残差对消纯谐波 **−24dB**；高带噪声保留比 ~1（单测覆盖）
- 准谐波（酒杯）实测部分音 `[660.2, 1335, 2025.1, 2740]` ≈ 真值
- 已知限制：颤音/滑音等**非稳态**单音会在分析窗内令谐波漂移、污染残差；音色快照
  按采集协议只取**孤立稳态单音**（颤音的表现力建模属 L2/后续阶段）

## 阶段 3（6.17-6.19）· FR-4 · 已完成 ✅

实时块基引擎 + MIDI。出口：实测延迟数字 + 满复音无 xrun。

| 交付 | 文件 | 状态 |
|---|---|---|
| numba 加法振荡器内核（相量递推快路径 + 变频慢路径 + Nyquist 软淡出） | `synth/oscillator.py` | ✅ |
| Voice（持久相位 + 线性 ADSR + 起音颗粒 + 20ms 谐波淡入 + 偷音） | `synth/voice.py` | ✅ |
| Instrument（快照→稳态 H/ratio/颗粒/固定 seed 噪声表，热切换） | `synth/instrument.py` | ✅ |
| 块基引擎（VoicePool16 + 采样级事件分段 + 母线 + gc 控制 + 离线路径） | `runtime/engine.py` | ✅ |
| MIDI 输入（mido callback→SimpleQueue，拔插安全） | `runtime/midi_in.py` | ✅ |
| 单测：ADSR ±10% / 力度映射 / 偷音 / 采样级触发 / 离线确定性 | `tests/test_stage3.py` | ✅ 9 passed |
| 验收：16 复音满载耗时 + 黄金回归 + 延迟报告 | `eval/stage3_accept.py` | ✅ |

> **架构核心**：`Engine.process_block` 是实时 `sd.OutputStream` 回调与离线
> `render_offline` 的**唯一共用代码路径**（铁律 #12；离线渲染=单测/回归/ABX 刺激音基座）。
> 附录 C 16 条音频引擎铁律已逐条落位（gc.freeze/disable、采样级时间戳、块内斜坡、
> 偷音淡出、Nyquist 软淡出、相位重置、denormal 清零、母线 tanh、try/except 静音、JIT 预热…）。

### 跑验收

```bash
conda activate mt && cd quill
python -m pytest -q                          # 32 passed（阶段1+2+3）
python -m eval.stage3_accept                 # 16 复音压力 + 黄金回归 + 延迟报告
python -m eval.stage3_accept voice           # 用指定快照
python -m eval.stage3_accept --update-golden # 改引擎后刷新回归基线
```

实时演奏（需 MIDI 键盘 + 声卡）：

```python
from runtime.engine import engine_from_snapshot_file
from runtime.midi_in import MidiInput

eng = engine_from_snapshot_file("assets/snapshots/guitar.npz", eval_mode=False)
midi = MidiInput().open()  # 默认取第一个 MIDI 输入口
eng.start(queue=midi.queue)  # 进入演奏模式（gc.freeze/disable）
# ... 弹琴 ...
eng.stop()
midi.close()
```

### 阶段3 验收口径（实测，本机离线）

- **满复音无 xrun**：16 复音满载 `process_block` 耗时 mean ~0.33ms / **p95 ~3-10% 块预算**
  （预算 5.8ms@blocksize256），判据 <50% **PASS**（相量递推内核 + 演奏模式 gc.disable）
- **离线确定性**：固定事件序列 + 固定 seed → 逐样本可复现（黄金文件回归 PASS）
- **延迟**：块缓冲预算 5.8ms；端到端 <30ms 需低延迟设备（环回法在有声卡机器实测填表，
  脚本打印操作步骤）。注：内置扬声器 CoreAudio 默认缓冲较大（~130ms），演示需 ASIO/小缓冲

## 阶段 4（6.20-6.23）· FR-1 完整 + FR-5 · 已完成 ✅

事件切分与特征化（双前端）。出口：events.json 召回 ≥90%。

| 交付 | 文件 | 状态 |
|---|---|---|
| FR-1 完整：导入/录音 ≤60s 截断 + mp3 + 自动增益 | `analysis/io.py` | ✅ |
| 前端1 Basic Pitch（音符 + pitch-bend，CoreML 后端） | `analysis/segment.py::basic_pitch_events` | ✅ |
| 前端2 自建 onset(backtrack)+PESTO f0 分段 | `analysis/segment.py::onset_f0_events` | ✅ |
| 幽灵八度过滤（置信度阈值 + PESTO f0 一致性，高/低八度） | `analysis/segment.py::ghost_octave_filter` | ✅ |
| 双前端求并集 + 差异样本（人工裁决清单） | `analysis/segment.py::union_events` | ✅ |
| 逐事件特征（log-attack/质心/f0斜率/复音度/HNR，Peeters 子集） | `analysis/features.py` | ✅ |
| events.json 序列化（Part B 契约） | `analysis/features.py::save_events` | ✅ |
| 手标工具（点击播放+按键打标）+ 召回计算 | `eval/label_tool.py` | ✅ |
| 单测：FR-1 截断 / events.json / 幽灵过滤 / 召回 / 特征 | `tests/test_stage4.py` | ✅ 8 passed |
| 验收：闭环渲染已知曲 → 分析 → 召回 ≥90% | `eval/stage4_accept.py` | ✅ |

### 跑验收

```bash
conda activate mt && cd quill
python -m pytest -q                          # 40 passed（阶段1-4）
python -m eval.stage4_accept                 # 闭环：渲染已知曲 -> FR-5 -> 召回
python -m eval.stage4_accept guitar.wav      # 真实素材 -> events.json + 手标提示
```

### 阶段4 验收口径（实测）

- **events.json 召回 100%**（闭环：引擎渲染 11 事件的已知曲 = 8 单音 + 3 音和弦 →
  双前端分析 → 与真值匹配 11/11，判据 ≥90% **PASS**），精确率 91.7%
- **幽灵八度坑位已治**：Basic Pitch 在每个起音吐低置信度 D3 幽灵（amp 0.26-0.38 vs
  真音 0.74）→ 置信度阈值 0.4 + PESTO f0 一致性的高/低八度过滤，本例滤掉 7 个
- 逐事件特征齐全（log_attack/centroid/f0_slope/polyphony/hnr）；f0_track 含 bend
- 注：闭环用合成已知曲验证 FR-5 管线；真实吉他 50 事件召回用 `label_tool` 手标后
  对照 `compute_recall`（方法论：双前端差异样本人工裁决，写报告）

## 阶段 5（6.24-6.28）· FR-6 基础 + FR-7 · 已完成 ✅

演奏法分类（规则优先级树）+ `.tplib` 演奏法库。出口：.tplib 往返无损 + 目检 ≥80%。

| 交付 | 文件 | 状态 |
|---|---|---|
| 规则优先级树（strum/slide/legato/pluck/sustain，阈值全进 config.yaml） | `analysis/articulation.py::classify_events` | ✅ |
| slide vs vibrato 解纠缠（先线性去趋势抽周期，再判单调净斜率） | `analysis/timbre.py::detect_vibrato` | ✅ |
| M7 随机森林兜底 + 规则 vs 森林 ablation | `analysis/articulation.py::train_forest` | ✅ |
| IDMT-SMT-Guitar loader（统一为 events 格式，缺失优雅跳过） | `analysis/datasets.py` | ✅ |
| FR-7 ArticulationLibrary（pluck 颗粒 RMS 三桶/slide 样条/legato 过渡/strum 时差） | `analysis/tplib.py::build_library` | ✅ |
| `.tplib`（zip=manifest+npz）读写往返无损 | `analysis/tplib.py::save/load_tplib` | ✅ |
| 6.25 中途集成检查点（sustain-only：analysis→assets→runtime 全通路） | `eval/integration_check.py` | ✅ |
| 单测：规则各类 / vibrato-vs-slide / .tplib 往返 / 森林 / strum 窗 / IDMT | `tests/test_stage5.py` | ✅ 13 passed |
| 验收：已知演奏法集分类准确率 + .tplib 往返 | `eval/stage5_accept.py` | ✅ |

### 跑验收

```bash
conda activate mt && cd quill
python -m pytest -q                          # 53 passed（阶段1-5）
python -m eval.stage5_accept                 # 已知演奏法 -> 分类准确率 + .tplib 往返
python -m eval.integration_check             # 6.25 sustain-only 端到端通路
```

### 阶段5 验收口径（实测）

- **规则树分类准确率 100%**（构造的 30 个已知演奏法事件：pluck/slide/legato/strum/sustain
  各类全对角线，替代"自录目检≥80%"的可复现口径），颤音检出 3/3
- **.tplib 往返无损 PASS**（pluck 颗粒 RMS 三桶 / slide 归一化样条控制点 /
  legato ±50ms 过渡曲线 / strum 时差向量 / sustain=TimbreSnapshot 引用）
- 森林 ablation：测试集 72.7%（< 规则树）——印证"规则可解释、少样本即工作"的设计取舍
- 坑位已治：slide 被误判 vibrato（detect_vibrato 改**线性去趋势**后抽周期）→ slide 6/6
- 6.25 集成检查点 PASS：合成单音→快照→引擎演奏 note64=330Hz(+2cents)、全通路无接口问题
- 注：自录/IDMT 宏 F1 双口径（规则≥0.7 / MERT 探针≥0.8）在阶段8 正式评估；
  IDMT loader 已就位，数据集需手动下载

## 阶段 6（6.29-7.2）· FR-8 · 已完成 ✅

手势解释器 + 吉他链路集成。出口：吉他链路集成 demo。

| 交付 | 文件 | 状态 |
|---|---|---|
| Voice 扩展：glide（slide/legato 续奏，log 域沿曲线滑 f0）+ vibrato LFO | `synth/voice.py` | ✅ |
| 引擎 slide/legato/vibrato 事件 + 手势模板注入 | `runtime/engine.py` | ✅ |
| 手势解释器状态机（重叠键→legato/slide、和弦→strum、PB、CC1→颤音） | `runtime/gesture.py` | ✅ |
| 真吉他 .tplib 构建（从演奏段自动分类） | `eval/build_library.py` | ✅ |
| 吉他链路集成 demo（真音色×真演奏法×手势） | `eval/stage6_accept.py` | ✅ |
| 单测：pluck/legato/slide/strum/vibrato 决策 + glide 渲染 | `tests/test_stage6.py` | ✅ 10 passed |

### 跑验收

```bash
conda activate mt && cd quill
python -m pytest -q                          # 64 passed（阶段1-6）
python -m eval.build_library assets/recordings/performance/guitar_perf_01.m4a  # 建真吉他库
python -m eval.stage6_accept                 # 吉他链路集成 demo（手势→演奏法）
```

### 阶段6 验收口径（实测）

- 手势→演奏法决策正确：重叠键 |音程|≤2半音→legato、>2→slide(力度→滑速)、
  <80ms 内≥3音→strum 错峰、CC1→颤音、PB→f0 直驱
- **吉他链路集成 demo**：真吉他音色快照 + 真吉他演奏法库（slide 曲线库实测 8 点）+
  手势解释器 → 一段拨/连/滑/扫/颤的"那把吉他"（`stage6_guitar_chain.wav`）
- f0 轨迹图 `stage6_f0_trajectory.png`：legato/slide/bend 渲染为可听的 f0 滑动（pyin 实测）

## 下一步

阶段 7（7.3-7.7）｜FR-12 + 参赛包：最小 UI（录音键/波形/演奏法标签/延迟数）+ demo 视频
（双层叙事，滑音放第一分钟）+ **FR-9 杀手锏**（会吉他滑音的人声）→ 7.8 提交。
