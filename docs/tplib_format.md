# `.tplib` —— 演奏法库格式规范 v1

> Quill 的开放交换格式。**演奏法库是一组可重定向的控制曲线，不是音频切片**——这是与
> 采样器的根本区别：一条滑音存为可重定向到任意音程/时长的归一化曲线，一次扫弦存为可作用
> 于任意和弦的弦序时差。社区可分享、可交换、可叠加（GTM：社区种子）。

## 容器

`.tplib` = **ZIP**，含两个成员：
```
manifest.json   # 结构 + 标量 + 字符串（人可读）
arrays.npz      # 所有 float32 数组（numpy 压缩）
```
manifest 用 `"key"` 字符串引用 `arrays.npz` 里的数组键，实现"结构与数据分离、读写往返无损"。

## 数据契约（Part B）

```python
ArticulationLibrary = {
  "pluck":  [ {"rms_bucket": 0|1|2, "grain": float32[2646]} ],     # 60ms@44.1k 瞬态残差，按力度三桶
  "slide":  [ {"interval_st": float, "dur_s": float,
               "curve_ctrl": float32[8]} ],                        # 归一化样条 g(τ)∈[0,1]（音程×时长归一）
  "legato": [ {"amp_curve": float32[25], "tilt_curve": float32[25]} ],  # ±50ms@250Hz 过渡曲线
  "strum":  [ {"offsets_ms": float32[N], "direction": "up"|"down"} ],   # 弦序时差向量 + 方向
  "sustain": TimbreSnapshot,                                       # 音色快照引用（内嵌，自包含）
  "meta":   {"instrument_guess": str, "source": str,
             "stats": {pluck,slide,legato,strum 计数}, "format_version": 1}
}
```

### 各字段语义
- **pluck.grain**：拨弦瞬态的残差颗粒，granular 回放（Roads 2001）；`rms_bucket` 按力度选桶。
- **slide.curve_ctrl**：滑音的**归一化形状** g(τ)∈[0,1]，τ=时间分数。合成端按
  `f0 = f0_start · 2^(g(t/T)·interval/12)` 重定向到**任意**起止音高与时长（力度→滑速）。
- **legato.amp_curve / tilt_curve**：连音衔接处 ±50ms 的幅度与亮度过渡（250Hz，25 点）。
- **strum.offsets_ms**：各弦相对首弦的发声时差（毫秒）；`direction` 上行/下行扫。作用于任意和弦。
- **sustain**：音色端（`TimbreSnapshot`：f0_ref + H_env[60,T] + N_env[40,T] + attack_res[2646]
  + partial_freqs|None + vibrato|None）。**与演奏法解耦** → 可与任意其它库的演奏法交叉重组（FR-9）。

## 读写
```python
from analysis.tplib import build_library, save_tplib, load_tplib

lib = build_library(labeled_events, y, sr, sustain_snapshot=snap)
save_tplib("guitar.tplib", lib)  # 往返无损
lib2 = load_tplib("guitar.tplib")
```
加载器**宽容**：新增字段给默认、未知键忽略（前/后向兼容）。`format_version` 标版本。

## 兼容性约定
- 数组一律 `float32`；标量/字符串进 manifest.json（UTF-8）。
- 时间单位：秒（dur_s）/毫秒（offsets_ms）；音程：半音（interval_st）。
- 采样率默认 44100；sustain 内 meta 自带 sr。
- 交叉重组只要求 sustain 与演奏法各自合法，音色与演奏法**可来自不同 `.tplib`**。
