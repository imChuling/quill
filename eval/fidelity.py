"""跨音高音色保真评估 —— FR-13（音色保真硬指标）。

核心主张的量化证据：**Quill 的共振峰保持重合成，在远离参考音高处比朴素采样器移调
更忠于真实乐器。** 做法：

  从音高 A 的真录音建 TimbreSnapshot → 重合成到音高 B
    Quill   : 共振峰保持加法重合成（render_offline / synth.render）
    Sampler : librosa.effects.pitch_shift（相位声码器移调，无共振峰校正）= 典型采样器基线
  两者各与**音高 B 的真录音**比谱包络距（稳态段平均 log 幅度谱 L1，对时序鲁棒）。
  Quill 距 < Sampler 距 = 共振峰保持更保真（共振峰乐器尤其显著）。

输入是一组 (pitch_midi, audio) 同乐器多音高样本；现支持用户人声三音高 / IDMT 吉他 /
NSynth（下好后）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.io import load  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from synth.render import render  # noqa: E402

SR = 44100


# --------------------------------------------------------------------------- #
# 谱包络距（稳态段平均 log 幅度谱 L1，对相位/时序鲁棒）
# --------------------------------------------------------------------------- #
def _steady_logspec(y, sr=SR, n_fft=2048, fmax=8000.0):
    seg = y[len(y) // 5 : -len(y) // 5] if len(y) > 10 else y
    if len(seg) < n_fft:
        seg = np.pad(seg, (0, n_fft - len(seg)))
    S = []
    for i in range(0, len(seg) - n_fft, n_fft // 2):
        S.append(np.abs(np.fft.rfft(seg[i : i + n_fft] * np.hanning(n_fft))))
    mag = np.mean(S, axis=0) if S else np.abs(np.fft.rfft(seg[:n_fft] * np.hanning(n_fft)))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    mag = mag[f <= fmax]
    mag = mag / (mag.sum() + 1e-12)  # 归一化（去整体响度，比形状）
    return np.log(mag + 1e-6)


def spectral_distance(y1, y2, sr=SR):
    a, b = _steady_logspec(y1, sr), _steady_logspec(y2, sr)
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def _sampler_pitch_shift(y, semitones, sr=SR):
    """采样器基线：相位声码器移调（librosa），不保共振峰。"""
    import librosa

    if abs(semitones) < 1e-3:
        return np.asarray(y, dtype=np.float32)
    return librosa.effects.pitch_shift(
        np.asarray(y, dtype=np.float32), sr=sr, n_steps=float(semitones)
    ).astype(np.float32)


# --------------------------------------------------------------------------- #
# 一件乐器的跨音高保真
# --------------------------------------------------------------------------- #
def evaluate_instrument(samples, name="instrument", sr=SR, quasi=False, max_semis=None):
    """samples: [(pitch_midi, audio), ...] 同乐器多音高真录音。

    取中位音高为参考建快照；**每样本的真实 f0 由 PESTO 实测**（不依赖标注/估计音高），
    Quill 渲染到目标真实 f0、采样器按真实 f0 比例移调；与目标真录音比谱距。
    max_semis 限制目标移调幅度（音乐上合理区间，避免 3 个八度的非现实极端拖累两方）。
    """
    from analysis.harmonic import f0_scalar, track_f0

    enriched = []
    for pitch, audio in samples:
        y = np.asarray(audio, np.float32)
        f0v = f0_scalar(track_f0(y, sr=sr)[0])
        if np.isfinite(f0v) and f0v > 0:
            enriched.append((f0v, pitch, y))
    enriched.sort(key=lambda s: s[0])
    if len(enriched) < 2:
        return [], None
    ref_i = len(enriched) // 2
    f0_ref_real, ref_pitch, ref_audio = enriched[ref_i]
    snap = analyze_timbre(ref_audio, sr=sr, name=name, quasi_harmonic=quasi)
    f0_ref = snap["f0_ref"]

    rows = []
    for i, (f0_tgt_real, _pitch, real) in enumerate(enriched):
        if i == ref_i:
            continue
        semis = 12.0 * np.log2(f0_tgt_real / f0_ref_real)
        if max_semis is not None and abs(semis) > max_semis:
            continue
        f0_tgt = f0_ref * (f0_tgt_real / f0_ref_real)  # 快照参考基频 × 真实比例
        n = len(real)
        quill = render(snap, f0_tgt, length=n, mode="formant", add_attack=False)
        sampler = _sampler_pitch_shift(ref_audio, semis, sr)
        dq = spectral_distance(quill, real, sr)
        ds = spectral_distance(sampler, real, sr)
        rows.append(
            {"semis": int(round(semis)), "quill": dq, "sampler": ds, "quill_better": dq < ds}
        )
    return rows, ref_pitch


def _report(rows, name):
    if not rows:
        print(f"  [{name}] 样本不足（需≥2 音高）")
        return None
    q = np.mean([r["quill"] for r in rows])
    s = np.mean([r["sampler"] for r in rows])
    win = np.mean([r["quill_better"] for r in rows])
    print(f"\n=== {name} ({len(rows)} 个目标音高) ===")
    print(f"  {'移调(半音)':>10} {'Quill谱距':>10} {'采样器谱距':>10}  优胜")
    for r in sorted(rows, key=lambda x: x["semis"]):
        print(
            f"  {r['semis']:>+10d} {r['quill']:>10.4f} {r['sampler']:>10.4f}  "
            f"{'Quill' if r['quill_better'] else '采样器'}"
        )
    print(
        f"  均值: Quill {q:.4f} vs 采样器 {s:.4f}  "
        f"-> Quill {'更保真' if q < s else '未优于'}（越低越像真录音）；"
        f"Quill 胜率 {win * 100:.0f}%"
    )
    return {"name": name, "quill": q, "sampler": s, "win_rate": win}


# --------------------------------------------------------------------------- #
# 数据源
# --------------------------------------------------------------------------- #
def voice_samples():
    """用户人声三音高（强共振峰 -> Quill 优势应显著）。"""
    base = Path(__file__).resolve().parents[1] / "assets" / "recordings" / "tones"
    out = []
    midi = {"low": 50, "mid": 60, "high": 69}  # 估计音区（仅作参考序，真 f0 用 analyze）
    for reg in ("low", "mid", "high"):
        for ext in (".m4a", ".wav", ".mp3"):
            p = base / f"voice_tone_{reg}{ext}"
            if p.exists():
                y, _ = load(str(p), sr=SR)
                out.append((midi[reg], y))
                break
    return out


def idmt_samples(root, guitar_subdir=None, max_notes=12):
    """IDMT 同一把吉他的多音高单音（弱共振峰 -> Quill 优势温和）。"""
    import glob
    import os

    from analysis.datasets import load_idmt_smt_guitar

    evs = load_idmt_smt_guitar(root)
    # 选一个 subdir 的单音文件（每文件常含数音；取 pluck/normal 的）
    from collections import defaultdict

    byf = defaultdict(list)
    for e in evs:
        byf[e["_source_file"]].append(e)
    out = []
    seen_pitch = set()
    for f, es in byf.items():
        if len(es) != 1:  # 取单音文件，纯净
            continue
        e = es[0]
        if e["pitch_midi"] in seen_pitch:
            continue
        stem = os.path.splitext(f)[0]
        hits = glob.glob(os.path.join(root, "**", stem + ".wav"), recursive=True)
        if not hits:
            continue
        y, _ = load(hits[0], sr=SR)
        out.append((int(e["pitch_midi"]), y))
        seen_pitch.add(e["pitch_midi"])
        if len(out) >= max_notes:
            break
    return out


# --------------------------------------------------------------------------- #
# 多乐器广度数据源（TinySOL 管弦真录音 / NSynth 基准）
# --------------------------------------------------------------------------- #
_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _note_to_midi(s):
    """音名 -> MIDI：'A4'->69, 'C#5', 'Bb3', 支持 #/♯ b/♭。"""
    s = s.strip()
    name = s[0].upper()
    i, acc = 1, 0
    while i < len(s) and s[i] in "#b♯♭":
        acc += 1 if s[i] in "#♯" else -1
        i += 1
    return 12 * (int(s[i:]) + 1) + _NOTE[name] + acc


def _spread(samples, max_n):
    """从已按音高排序的样本里均匀抽 max_n 个（含两端），控制运行时长。"""
    if len(samples) <= max_n:
        return samples
    idx = np.linspace(0, len(samples) - 1, max_n).round().astype(int)
    return [samples[i] for i in sorted(set(idx))]


def tinysol_groups(root, technique="ord", dynamics="mf", max_pitch=9, min_pitch=3):
    """TinySOL -> [(TinySOL/Family/Instrument, [(midi,audio),...]), ...]。

    真实布局 <Family>/<Instrument>/<technique>/<Instr-tech-Pitch-Dyn-...>.wav
    （如 Brass/Bass_Tuba/ordinario/BTb-ord-F#1-pp-N-N.wav）。Family/Instrument 取
    路径倒数第 4/3 段（对根目录多层嵌套鲁棒）；文件名解析 tech/pitch/dyn，固定
    ordinario + 单一 dynamics 保音色一致。
    """
    import glob
    import os
    from collections import defaultdict

    root = str(root)
    groups = defaultdict(list)
    for wav in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        comp = wav.split(os.sep)
        if len(comp) < 4:
            continue
        fam, instr = comp[-4], comp[-3]  # …/Family/Instrument/technique/file.wav
        base = os.path.basename(wav)[:-4]
        parts = base.split("-")
        if len(parts) < 4:
            continue
        tech, pitch, dyn = parts[1], parts[2], parts[3]
        if tech != technique or dyn != dynamics:
            continue
        try:
            midi = _note_to_midi(pitch)
        except Exception:
            continue
        groups[(fam, instr)].append((midi, wav))
    out = []
    for (fam, instr), items in sorted(groups.items()):
        items = sorted(set(items))
        if len(items) < min_pitch:
            continue
        items = _spread(items, max_pitch)
        loaded = [(m, load(p, sr=SR)[0]) for m, p in items]
        out.append((f"TinySOL/{fam}/{instr}", loaded))
    return out


def nsynth_groups(root, per_family=1, max_pitch=9, min_pitch=3, families=None, velocity=None):
    """NSynth -> [(NSynth/family, [(midi,audio),...]), ...]。

    文件名 <family>_<source>_<instrID>-<pitch>-<velocity>.wav，pitch 即 MIDI；
    按乐器（family_source_instrID）分组，每个 family 取 per_family 件乐器。
    """
    import glob
    import os
    from collections import defaultdict

    root = str(root)
    audio_dir = os.path.join(root, "audio")
    files = glob.glob(os.path.join(audio_dir, "*.wav"))
    if not files:
        files = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
    by_instr = defaultdict(list)  # instr_key -> [(midi, vel, path)]
    fam_of = {}
    for wav in files:
        base = os.path.basename(wav)[:-4]
        if "-" not in base:
            continue
        head, pitch, vel = base.rsplit("-", 2)
        fam = head.split("_")[0]
        if families and fam not in families:
            continue
        try:
            by_instr[head].append((int(pitch), int(vel), wav))
            fam_of[head] = fam
        except ValueError:
            continue
    # 每 family 选 per_family 件音高跨度最大的乐器，velocity 固定
    fam_instrs = defaultdict(list)
    for head, items in by_instr.items():
        fam_instrs[fam_of[head]].append((head, items))
    out = []
    for fam, instrs in sorted(fam_instrs.items()):
        instrs.sort(key=lambda hi: -len({m for m, v, p in hi[1]}))
        for _head, items in instrs[:per_family]:
            vels = sorted({v for m, v, p in items})
            vel = velocity if velocity in vels else vels[len(vels) // 2]
            ps = sorted({(m, p) for m, v, p in items if v == vel})
            if len(ps) < min_pitch:
                continue
            ps = _spread(ps, max_pitch)
            loaded = [(m, load(p, sr=SR)[0]) for m, p in ps]
            out.append((f"NSynth/{fam}", loaded))
    return out


def _run_breadth(groups, quasi=False, max_semis=18.0):
    """跑一批 (name, samples) 分组，按 family 前缀聚合返回 per-group + per-family。"""
    from collections import defaultdict

    fam_rows = defaultdict(list)
    per_group = []
    for name, samples in groups:
        rows, _ = evaluate_instrument(samples, name=name, quasi=quasi, max_semis=max_semis)
        if not rows:
            continue
        q = float(np.mean([r["quill"] for r in rows]))
        s = float(np.mean([r["sampler"] for r in rows]))
        win = float(np.mean([r["quill_better"] for r in rows]))
        per_group.append({"name": name, "quill": q, "sampler": s, "win_rate": win, "n": len(rows)})
        fam = "/".join(name.split("/")[:2])
        fam_rows[fam].extend(rows)
        print(
            f"  {name:32s} Quill {q:.3f} vs 采样器 {s:.3f}  "
            f"胜率 {win * 100:3.0f}%  ({'✓' if q < s else '×'})"
        )
    fam_summary = []
    for fam, rows in sorted(fam_rows.items()):
        q = float(np.mean([r["quill"] for r in rows]))
        s = float(np.mean([r["sampler"] for r in rows]))
        win = float(np.mean([r["quill_better"] for r in rows]))
        fam_summary.append({"name": fam, "quill": q, "sampler": s, "win_rate": win, "n": len(rows)})
    return per_group, fam_summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="跨音高音色保真 vs 采样器基线")
    ap.add_argument("--idmt", default=None, help="IDMT 根目录（加测吉他）")
    ap.add_argument("--tinysol", default=None, help="TinySOL 根目录（管弦广度）")
    ap.add_argument("--nsynth", default=None, help="NSynth 根目录（基准广度）")
    ap.add_argument("--per-family", type=int, default=1, help="NSynth 每 family 取几件乐器")
    args = ap.parse_args(argv)

    summary = []
    vs = voice_samples()
    if len(vs) >= 2:
        rows, _ = evaluate_instrument(vs, name="voice(人声·强共振峰)")
        r = _report(rows, "voice(人声·强共振峰)")
        if r:
            summary.append(r)
    else:
        print("（人声样本 <2，跳过；需 tones/voice_tone_low/mid/high）")

    if args.idmt:
        gs = idmt_samples(args.idmt)
        if len(gs) >= 2:
            rows, _ = evaluate_instrument(gs, name="guitar(吉他·弱共振峰)")
            r = _report(rows, "guitar(吉他·弱共振峰)")
            if r:
                summary.append(r)

    # ---- 多乐器广度 ----
    fam_all = []
    if args.tinysol:
        print("\n========== TinySOL 管弦广度（真录音·强共振峰） ==========")
        groups = tinysol_groups(args.tinysol)
        print(f"  （{len(groups)} 件乐器）")
        _, fam = _run_breadth(groups)
        fam_all += fam
    if args.nsynth:
        print("\n========== NSynth 基准广度 ==========")
        groups = nsynth_groups(args.nsynth, per_family=args.per_family)
        print(f"  （{len(groups)} 件乐器）")
        _, fam = _run_breadth(groups)
        fam_all += fam

    if summary:
        print("\n---------------- 跨音高保真汇总（用户素材） ----------------")
        for r in summary:
            print(
                f"  {r['name']:24s} Quill {r['quill']:.4f} vs 采样器 {r['sampler']:.4f}  "
                f"胜率 {r['win_rate'] * 100:.0f}%  ({'✓ Quill 更保真' if r['quill'] < r['sampler'] else '×'})"
            )
        print("  口径：谱距越低越忠于真录音；共振峰乐器上 Quill 优势应更明显。")

    if fam_all:
        print("\n---------------- 多乐器广度汇总（按族） ----------------")
        n_win = sum(f["quill"] < f["sampler"] for f in fam_all)
        for f in fam_all:
            print(
                f"  {f['name']:22s} Quill {f['quill']:.3f} vs 采样器 {f['sampler']:.3f}  "
                f"胜率 {f['win_rate'] * 100:3.0f}%  n={f['n']:<3d} "
                f"({'✓ Quill 更保真' if f['quill'] < f['sampler'] else '×'})"
            )
        gq = np.mean([f["quill"] for f in fam_all])
        gs = np.mean([f["sampler"] for f in fam_all])
        print(
            f"  ── 全体 {n_win}/{len(fam_all)} 族 Quill 更保真；"
            f"宏平均 Quill {gq:.3f} vs 采样器 {gs:.3f} ──"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
