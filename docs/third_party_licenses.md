# 第三方依赖与许可证清单(Sprint 0 · 2026-07-29)

> 冲刺本子 §13 合规项。范围 = requirements.txt 全部 + 数据集 + 预训练模型。
> 结论先行:**运行时核心链路全部为宽松许可(BSD/MIT/Apache/ISC),无 GPL 传染;
> 合规注意点集中在"数据集限研究用途"与"Demucs 预训练权重"两处。**

## 1. 运行时核心(随产品分发)

| 依赖 | 许可证 | 风险 |
|---|---|---|
| numpy / scipy | BSD-3 | 无 |
| numba | BSD-2 | 无 |
| librosa | ISC | 无 |
| soundfile | BSD-3(自带 libsndfile LGPL-2.1 动态链接) | 动态链接合规,勿静态打包 |
| sounddevice | MIT(PortAudio MIT) | 无 |
| mido / python-rtmidi | MIT(RtMidi MIT-like) | 无 |
| pywebview | BSD-3 | 无 |
| PyYAML | MIT | 无 |
| React / ReactDOM 18.3.1(`ui/web/vendor/`,桌面 UI 随库打包,许可证文本同目录) | MIT | 无 |
| Cormorant Garamond / Cormorant SC / Italianno / Spline Sans Mono(`ui/web/vendor/fonts/`,latin 子集 woff2,随桌面 UI 打包,OFL.txt 同目录) | SIL OFL 1.1 | 无 |

## 2. 预训练模型(重点)

| 组件 | 代码许可证 | 权重/数据注意 |
|---|---|---|
| **PESTO**(f0 前端,主链路) | LGPL-3.0(代码) | 以 pip 包动态依赖形式使用=合规;**不得**把其代码拷入本仓库;对外表述:"uses the pretrained PESTO pitch estimator" |
| **Demucs** htdemucs_6s(Mode 3 可选) | 代码 MIT | **htdemucs 权重非商业条款风险**:训练数据含 MUSDB 等限研究数据;Let's Build 为非商业赛事=OK,但 README/ModelCard 须标 "research/non-commercial use" |
| **basic-pitch**(FR-5 转录) | Apache-2.0 | 无(Spotify 官方开源) |
| **MERT-v1-95M**(研究探针,不进产品) | 权重 CC-BY-NC-4.0 | **非商业**;只用于离线评测,不随产品分发,论文引用即可 |
| torch / transformers / nnAudio | BSD/Apache/MIT | 无 |

## 3. 数据集(只用于评测,不随产品分发)

| 数据集 | 许可证 | 允许用途 |
|---|---|---|
| TinySOL | CC-BY-4.0 | 评测+论文 ✓,注明出处 |
| URMP(violin/trumpet DDSP 训练 + 行为库来源) | **无标准许可**,需注册并引用,未规定派生物条款 | 评测+论文 ✓;派生行为库 **不公开发布**(2026-10-27 拍板,仅仓库内复现用) |
| GuitarSet | MIT(标注)/CC-BY-4.0(音频) | 评测+论文 ✓ |
| IDMT-SMT-Guitar / Bass | **仅限非商业研究** | 评测+论文 ✓;**不得**把音频塞进 demo 视频当素材 |
| NSynth | CC-BY-4.0 | 评测+训练 prior ✓ |
| 用户自录(voice/guitar/wineglass 等) | 自有 | 需演奏者 publicity 授权(吉他手,冲刺本子 §13) |

## 4. 对外表述红线(进 claim ledger SCOPE-01)

- 允许:"the capture-to-performance path requires **no training**; **no per-user/per-timbre training**"
- 禁止:"no pretrained models"(PESTO/Demucs/basic-pitch/MERT 都是预训练组件)
- Let's Build 提交若含 Demucs 功能演示,页面须带一句 non-commercial 声明;
- ISMIR LBD 摘要 acknowledgements 引 TinySOL/GuitarSet/NSynth 出处。

## 5. 待办

- [ ] Audiotool Nexus SDK(@audiotool/nexus)许可证与 T&C —— 等注册后确认(Sprint 1 前);
- [ ] 若 Atlas 过门:NSynth/TinySOL 训练用途再核对一次(CC-BY 允许,标注出处即可)。
