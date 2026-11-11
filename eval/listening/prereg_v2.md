# 听测 v2 预注册(数据收集前定稿)

定稿日期:2026-08-09(在任何 v2 回答收集之前)。
本文档一旦开始收集数据即不再修改;分析按此执行,偏离须在论文中披露。

## 与 v1(2026-08-05 / N=13, N=18)的关系

v1 逐人数据未留存且无参与者间随机化;v2 结果**整体替换**论文中的
v1 数字。v1 降为未报告的 pilot。参与者可能与 v1 重叠,论文披露
"participants overlap with an earlier pilot using similar stimuli"。

## 设计

- 单一自包含 HTML(`checkpoints/study_v2.html`),音频内嵌。
- 参与者输入自选两位数字编号 → 前端以编号种子化 PRNG:
  **逐人独立**的题目顺序与 A/B 位置(counterbalancing)。
- 页面末尾生成答案码(记录编号、乱序映射、逐题选择的刺激索引),
  参与者复制发回;`tools/decode_study_v2.py` 解码为匿名 CSV。

## 题目

### Part 1 主听测(6 题,forced choice / 评分)

| # | 对比 | 假设方向 |
|---|---|---|
| M1 | v0 检索行为 vs flat 无行为(lyrical) | 选 v0 |
| M2 | 带演奏法标注 vs 全 plain(folk) | 选带标注 |
| M3 | v2 参数化 vs v0 检索(lyrical) | 无方向(探索) |
| M4 | violin 行为 vs guitar 行为,同 violin 音色(folk) | 听出不同 |
| M5 | 自然度评分:Quill vs MIDI-DDSP(各 1-5 分) | 无方向(基准对照) |
| M6 | glass 音色 × violin 行为 vs glass 音色 × glass 行为(lyrical, +12) | 听出不同 |

### Part 2 跨乐器 ABX(8 题)

问题措辞(v2 修订):**"哪个 Candidate 的音高与力度起伏(演奏方式)
更接近 Reference?"** 选项 A / B / 不确定。

| # | Reference | Candidates(同音色) | match |
|---|---|---|---|
| X1 | viB×viT lyrical | trT:viB vs trB | viB |
| X2 | viB×viT folk | trT:viB vs trB | viB |
| X3 | trB×trT lyrical | viT:trB vs viB | trB |
| X4 | trB×trT folk | viT:trB vs viB | trB |
| X5 | guB×guT lyrical | viT:guB vs viB | guB |
| X6 | guB×guT folk | viT:guB vs viB | guB |
| X7 | viB×viT lyrical(+12) | glassT:viB vs glB | viB |
| X8 | glB×glT folk(+12) | viT:glB vs viB | glB |

X7/X8 为 glass 新增。X8 是行为带宽假设的感知检验:预期
**接近随机**(glass 行为迁移到 violin 感知上弱)——无论结果如何都报告。

## 统计计划

- 主听测二元题(M1/M2/M4/M6):单侧二项检验(方向见上表),M3 双侧。
- M5:配对 Wilcoxon(评分)。
- ABX:unsure **剔除出正确率但报告数量**(预注册规则);
  pooled 单侧二项;逐题二项 + Bonferroni(0.05/8);
  per-participant 准确率 + Wilcoxon signed-rank vs 0.5。
  X7/X8 单独报告,不与 X1–X6 合并入"跨乐器可辨认"主张。
- 样本目标:尽力 ≥18,接受实际回收数;不做序贯查看后加收。

## 剔除规则

- 同一编号提交多份:取第一份。
- 答案码不完整/无法解码:整人剔除并报告数量。
- 无其他事后剔除。
