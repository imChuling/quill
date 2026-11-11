# 听测数据归档

论文中全部人类实验数字的可审计来源。`analyze_listening.py` 从匿名化 CSV
重算每个数字,输出 `listening_stats.json` 与论文对账。

## 文件

| 文件 | 内容 |
|---|---|
| `responses_main.csv` | 主听测(N=13)逐人回答,匿名 ID P01–P13 |
| `responses_abx.csv` | ABX(N=18)逐人逐题回答,匿名 ID A01–A18 |
| `analyze_listening.py` | 统计脚本(binomial / Bonferroni / Wilcoxon) |
| `listening_stats.json` | 脚本输出,与论文数字一一对应 |

## 数据格式

`responses_main.csv`:
```
participant_id,musical_training,q_expressive,q_style_distinct
P01,1,labeled,yes
```
- `musical_training`: 1=受过正式音乐训练
- `q_expressive`: labeled / plain(哪个版本更有表现力)
- `q_style_distinct`: yes / no(换行为源后是否听出不同演奏风格)

`responses_abx.csv`:
```
participant_id,trial_id,response
A01,1,match
```
- `trial_id`: 1–6
- `response`: match(选中同行为源候选)/ nonmatch / unsure

## 收集流程与剔除规则(预先规定)

- 刺激由 `tools/build_listening_study.py` / `tools/build_abx_study.py` 生成的
  HTML 页面呈现;**刺激顺序对所有参与者固定,无 per-participant
  counterbalancing**(局限,已在论文中披露)。
- ABX 中 `unsure` 回答剔除出正确率统计(论文 N=107 有效 = 108 − 1 unsure)。
- 匿名化:仅保留顺序 ID 与是否受训;不含姓名、联系方式与作答时间。

## 复算

```
conda run -n mt python eval/listening/analyze_listening.py
```
