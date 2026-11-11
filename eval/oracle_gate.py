"""Gate 0 · Oracle —— Neural Timbre Atlas 的 go/no-go 前置实验。

冲刺本子 §5.8 Gate 0:**若真实 target-note snapshot 的 DSP render 未超过预注册的
static-source 改善阈值,先检查/增强 renderer,停止 Atlas。**

问题:Atlas 想做的事是"给定未见乐器的 1-3 个 reference note,预测该乐器在新 pitch/
velocity 下相对静态快照的参数修正"。一个**完美**的这种模型,最好也只能预测出
「目标音自己的快照参数」。所以先量这个上界:

    static : snapshot(参考音 A) → 渲染到 f0_B        ← 当前 L0 出货路径
    oracle : snapshot(**真实目标音 B**) → 渲染到 f0_B  ← 任何参数预测模型的上界
    sampler: pitch_shift(A → B)                      ← 朴素采样器基线(已知可听差异)

三者都与**目标音 B 的真录音**比稳态 log 谱距(analysis 与产品同链路)。

若 oracle ≈ static:参数空间里没有可赚的东西,瓶颈在 renderer(合成器表达力)而不在
参数预测 → 训模型是浪费,应先增强 renderer。
若 oracle ≫ static:存在真实 headroom,Atlas 值得投 Sprint 2。

--------------------------------------------------------------------------- #
预注册判据(2026-07-29 写死,**先于任何结果**;改动须在 claim ledger 留痕)
--------------------------------------------------------------------------- #
H1 绝对 headroom : median (static-oracle)/static  ≥ 0.10   (至少 10% 谱距可赚)
H2 校准 headroom : median (static-oracle)/(sampler-static) ≥ 0.50
                   ——可赚的量至少是「已被证实可听」的 static-vs-sampler 差距的一半。
                   这条把抽象谱距锚到一个已知有听感意义的尺度上。
H3 一致性        : oracle 优于 static 的 (乐器,目标音) 对占比 ≥ 0.70

判定:
  GO        = H1 ∧ H3 通过,且 H2 通过        → Atlas 进 Sprint 2
  WEAK      = H1 ∧ H3 通过,H2 未过           → headroom 存在但可能不可听;
                                               Sprint 2 降为 2 天 spike,先做 ridge(Gate 1)
  NO-GO     = H1 或 H3 未过                   → 停 Atlas,转而增强 renderer;L0 继续出货

用法:
    python eval/oracle_gate.py --dataset tinysol --limit 6      # 快速 pilot
    python eval/oracle_gate.py --dataset tinysol                # 全量
    python eval/oracle_gate.py --dataset guitarset --out results/oracle_gate_gset.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.harmonic import f0_scalar, track_f0  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402
from eval.fidelity import SR, _sampler_pitch_shift, spectral_distance  # noqa: E402
from synth.render import render  # noqa: E402

# 预注册阈值(勿在看到结果后修改)
H1_MIN_REL = 0.10
H2_MIN_CALIB = 0.50
H3_MIN_WINRATE = 0.70


def oracle_rows(samples, name, sr=SR, quasi=False, max_semis=18.0):
    """一件乐器:对每个非参考音算 static / oracle / sampler 三个谱距。"""
    enriched = []
    for pitch, audio in samples:
        y = np.asarray(audio, np.float32)
        f0v = f0_scalar(track_f0(y, sr=sr)[0])
        if np.isfinite(f0v) and f0v > 0:
            enriched.append((f0v, pitch, y))
    enriched.sort(key=lambda s: s[0])
    if len(enriched) < 2:
        return []

    ref_i = len(enriched) // 2
    f0_ref_real, ref_pitch, ref_audio = enriched[ref_i]
    snap_s = analyze_timbre(ref_audio, sr=sr, name=name, quasi_harmonic=quasi)
    f0_ref = snap_s["f0_ref"]

    rows = []
    for i, (f0_tgt_real, pitch, real) in enumerate(enriched):
        if i == ref_i:
            continue
        semis = 12.0 * np.log2(f0_tgt_real / f0_ref_real)
        if abs(semis) > max_semis:
            continue
        n = len(real)

        # static: 参考音快照 → 渲染到目标 f0(当前出货路径)
        f0_tgt = f0_ref * (f0_tgt_real / f0_ref_real)
        y_static = render(snap_s, f0_tgt, length=n, mode="formant", add_attack=False)

        # oracle: 目标音**自己的**快照 → 在自己的参考基频上渲染(零移调=参数上界)
        try:
            snap_o = analyze_timbre(real, sr=sr, name=f"{name}_oracle", quasi_harmonic=quasi)
            y_oracle = render(snap_o, snap_o["f0_ref"], length=n, mode="formant", add_attack=False)
        except Exception as exc:  # 取色失败的音跳过,不静默计成好结果
            rows.append(
                {
                    "pitch": int(pitch),
                    "semis": round(semis, 2),
                    "skipped": f"oracle_capture_failed:{type(exc).__name__}",
                }
            )
            continue

        y_samp = _sampler_pitch_shift(ref_audio, semis, sr)

        d_static = spectral_distance(y_static, real, sr)
        d_oracle = spectral_distance(y_oracle, real, sr)
        d_samp = spectral_distance(y_samp, real, sr)
        rows.append(
            {
                "pitch": int(pitch),
                "semis": round(semis, 2),
                "d_static": d_static,
                "d_oracle": d_oracle,
                "d_sampler": d_samp,
                "rel_headroom": (d_static - d_oracle) / d_static if d_static > 0 else 0.0,
                "calib_headroom": (
                    (d_static - d_oracle) / (d_samp - d_static)
                    if (d_samp - d_static) > 1e-9
                    else float("nan")
                ),
                "oracle_better": bool(d_oracle < d_static),
            }
        )
    return rows


def summarize(all_rows):
    ok = [r for r in all_rows if "d_static" in r]
    if not ok:
        return {"error": "no valid rows"}
    rel = np.array([r["rel_headroom"] for r in ok])
    win = np.array([r["oracle_better"] for r in ok], dtype=float)
    calib = np.array([r["calib_headroom"] for r in ok], dtype=float)
    calib_ok = calib[np.isfinite(calib)]

    h1 = float(np.median(rel))
    h2 = float(np.median(calib_ok)) if calib_ok.size else float("nan")
    h3 = float(win.mean())
    p1, p3 = h1 >= H1_MIN_REL, h3 >= H3_MIN_WINRATE
    p2 = bool(np.isfinite(h2) and h2 >= H2_MIN_CALIB)

    verdict = "GO" if (p1 and p3 and p2) else ("WEAK" if (p1 and p3) else "NO-GO")
    return {
        "n_pairs": len(ok),
        "n_skipped": len(all_rows) - len(ok),
        "H1_median_rel_headroom": h1,
        "H1_pass": bool(p1),
        "H1_min": H1_MIN_REL,
        "H2_median_calib_headroom": h2,
        "H2_pass": p2,
        "H2_min": H2_MIN_CALIB,
        "H3_oracle_winrate": h3,
        "H3_pass": bool(p3),
        "H3_min": H3_MIN_WINRATE,
        "median_d_static": float(np.median([r["d_static"] for r in ok])),
        "median_d_oracle": float(np.median([r["d_oracle"] for r in ok])),
        "median_d_sampler": float(np.median([r["d_sampler"] for r in ok])),
        "verdict": verdict,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tinysol", choices=["tinysol", "guitarset", "nsynth"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 件乐器(pilot)")
    ap.add_argument("--max-pitch", type=int, default=7)
    ap.add_argument("--out", default="results/oracle_gate.json")
    a = ap.parse_args(argv)

    dsets = Path.home() / "Desktop" / "MT" / "datasets"
    if a.dataset == "tinysol":
        from eval.fidelity import tinysol_groups

        groups = tinysol_groups(a.root or dsets / "TinySOL", max_pitch=a.max_pitch)
    elif a.dataset == "nsynth":
        from eval.fidelity import nsynth_groups

        groups = nsynth_groups(a.root or dsets / "nsynth-test", max_pitch=a.max_pitch)
    else:
        from eval.fidelity import idmt_samples

        s = idmt_samples(a.root or dsets / "GuitarSet_mic")
        groups = [("GuitarSet/mic", s)] if s else []

    if not groups:
        print(f"[oracle-gate] 数据集为空: {a.dataset}")
        return 1
    if a.limit:
        groups = groups[: a.limit]

    print(
        f"[oracle-gate] {a.dataset}: {len(groups)} 件乐器 "
        f"(预注册 H1≥{H1_MIN_REL} H2≥{H2_MIN_CALIB} H3≥{H3_MIN_WINRATE})"
    )
    t0 = time.time()
    all_rows, per_inst = [], []
    for name, samples in groups:
        rows = oracle_rows(samples, name)
        ok = [r for r in rows if "d_static" in r]
        if ok:
            m_rel = float(np.median([r["rel_headroom"] for r in ok]))
            wr = float(np.mean([r["oracle_better"] for r in ok]))
            print(f"  {name:42s} n={len(ok):2d} rel_headroom={m_rel:+.3f} winrate={wr:.2f}")
            per_inst.append(
                {
                    "instrument": name,
                    "n": len(ok),
                    "median_rel_headroom": m_rel,
                    "oracle_winrate": wr,
                }
            )
        else:
            print(f"  {name:42s} (无有效样本)")
        for r in rows:
            r["instrument"] = name
        all_rows += rows

    summ = summarize(all_rows)
    summ["dataset"] = a.dataset
    summ["elapsed_s"] = round(time.time() - t0, 1)
    summ["n_instruments"] = len(per_inst)

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summ, "per_instrument": per_inst, "rows": all_rows}, indent=2)
    )

    print("\n" + "=" * 66)
    print(
        f"H1 绝对 headroom  median (static-oracle)/static = {summ['H1_median_rel_headroom']:+.3f} "
        f"(≥{H1_MIN_REL}) {'PASS' if summ['H1_pass'] else 'FAIL'}"
    )
    print(
        f"H2 校准 headroom  median /(sampler-static)      = {summ['H2_median_calib_headroom']:+.3f} "
        f"(≥{H2_MIN_CALIB}) {'PASS' if summ['H2_pass'] else 'FAIL'}"
    )
    print(
        f"H3 oracle 胜率                                  = {summ['H3_oracle_winrate']:.3f} "
        f"(≥{H3_MIN_WINRATE}) {'PASS' if summ['H3_pass'] else 'FAIL'}"
    )
    print(
        f"谱距中位: static={summ['median_d_static']:.4f} "
        f"oracle={summ['median_d_oracle']:.4f} sampler={summ['median_d_sampler']:.4f}"
    )
    print(
        f"\n>>> GATE 0 判定: {summ['verdict']}   "
        f"(n={summ['n_pairs']} 对, {summ['n_instruments']} 件乐器, {summ['elapsed_s']}s)"
    )
    print("=" * 66)
    print(f"artifact: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
