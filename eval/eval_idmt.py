"""IDMT-SMT-Guitar 正式评估 —— FR-6 双口径（阶段8 提前）。

三方 ablation：规则树 vs 随机森林 vs MERT 探针，在 IDMT 公开集上算宏 F1 + 混淆矩阵。
出口口径（产品计划成功指标）：规则树宏F1≥0.7 / MERT探针≥0.8。

数据集需手动下载（Fraunhofer IDMT-SMT-Guitar，多 GB）；缺失时优雅提示。
用法：
  python -m eval.eval_idmt --root /path/to/IDMT-SMT-GUITAR_V2 [--mert] [--max-files 40]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.articulation import classify_events, forest_predict, train_forest  # noqa: E402
from analysis.datasets import available, load_idmt_smt_guitar  # noqa: E402
from analysis.features import compute_event_features  # noqa: E402
from eval.metrics import classification_report, plot_confusion  # noqa: E402

SR = 44100
DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "demos"


def _find_wav(root, stem):
    """在数据集树里找与标注同名的音频（wav 优先）。"""
    for ext in (".wav", ".WAV", ".flac", ".mp3"):
        hits = glob.glob(os.path.join(root, "**", stem + ext), recursive=True)
        if hits:
            return hits[0]
    return None


def _load_events_with_features(root, max_files=None, want_audio_for_mert=False):
    from analysis.io import load

    events_all = load_idmt_smt_guitar(root)
    if not events_all:
        return [], {}
    by_file = defaultdict(list)
    for e in events_all:
        by_file[e.get("_source_file", "?")].append(e)

    files = list(by_file)
    if max_files and max_files < len(files):
        import random

        random.Random(0).shuffle(files)  # 打乱保证子集覆盖 slide/sustain
        files = files[:max_files]
    out, audio_by_file = [], {}
    n = len(files)
    for k, f in enumerate(files):
        if k % 25 == 0:
            print(f"  特征提取 {k}/{n} 文件，事件累计 {len(out)} …", flush=True)
        stem = os.path.splitext(f)[0]
        wav = _find_wav(root, stem)
        if wav is None:
            continue
        try:
            y, _ = load(wav, sr=SR, do_normalize=True, max_seconds=60)
        except Exception:
            continue
        evs = by_file[f]
        compute_event_features(y, SR, evs)  # 补特征 + f0_track
        out.extend(evs)
        if want_audio_for_mert:
            audio_by_file[f] = y
    return out, audio_by_file


def main(argv=None):
    ap = argparse.ArgumentParser(description="IDMT 正式评估（规则 vs 森林 vs MERT）")
    ap.add_argument("--root", required=True, help="IDMT-SMT-Guitar 根目录")
    ap.add_argument("--mert", action="store_true", help="加 MERT 探针（慢，逐事件抽嵌入）")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--refresh", action="store_true", help="忽略缓存重新提取特征")
    args = ap.parse_args(argv)

    if not available(args.root):
        print(f"[缺失] 未找到 IDMT 数据集: {args.root}")
        print("  下载 Fraunhofer IDMT-SMT-Guitar 后传 --root 指向解压目录。")
        return 2

    # 特征缓存（提取耗时 ~20min，缓存后重跑秒级；--refresh 强刷）
    import hashlib
    import pickle
    from collections import Counter, defaultdict

    key = hashlib.md5(f"{args.root}|{args.max_files}".encode()).hexdigest()[:12]
    cache = Path(f"/tmp/quill_idmt_cache_{key}.pkl")
    audio_by_file = {}
    if cache.exists() and not args.refresh:
        print(f"载入特征缓存 {cache.name} …")
        with open(cache, "rb") as fh:
            events = pickle.load(fh)
    else:
        print("载入 IDMT 标注 + 逐事件特征 …")
        events, audio_by_file = _load_events_with_features(
            args.root, args.max_files, want_audio_for_mert=args.mert
        )
        if events:
            with open(cache, "wb") as fh:
                pickle.dump(events, fh)
    if not events:
        print("[空] 解析到 0 个带音频的事件——检查目录结构 / 音频文件名是否与标注同名。")
        return 2
    true = [e["label"] for e in events]
    print(f"  {len(events)} 事件，分布: {dict(Counter(true))}")

    results = {}

    # 规则树：**按文件分组分类**（避免跨文件 t_on 碰撞误判 strum——多文件事件起点都
    # 集中在 0.2-2.5s，全量混在一起会被同窗 strum 规则错判）。
    present = set(true)  # 数据集实际标签空间
    by_file = defaultdict(list)
    for e in events:
        by_file[e.get("_source_file", "?")].append(e)
    for evs in by_file.values():
        classify_events(evs, valid_labels=present)
    rule_pred = [e["label"] for e in events]
    results["rule"] = classification_report(true, rule_pred)
    plot_confusion(true, rule_pred, DEMO_DIR / "idmt_confusion_rule.png", title="IDMT rule-tree")

    # 随机森林 + MERT：分层 train/test
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(events))
    tr, te = train_test_split(idx, test_size=0.35, random_state=0, stratify=true)
    ytr = [true[i] for i in tr]
    yte = [true[i] for i in te]

    etr = [events[i] for i in tr]
    ete = [events[i] for i in te]
    clf = train_forest(etr, ytr)
    pf = forest_predict(clf, ete)
    results["forest"] = classification_report(yte, pf)
    plot_confusion(yte, pf, DEMO_DIR / "idmt_confusion_forest.png", title="IDMT forest")

    if args.mert:
        print("抽 MERT 嵌入（逐事件，较慢）…")
        from analysis.io import load as _load
        from analysis.mert_probe import mert_embed, probe_predict, train_probe

        # 逐事件按其所属文件音频抽嵌入；缓存命中时 audio_by_file 为空 -> 按需重读 wav
        audio_cache = dict(audio_by_file)
        emb = []
        for k, e in enumerate(events):
            if k % 200 == 0:
                print(f"  MERT 嵌入 {k}/{len(events)} …", flush=True)
            sf_ = e.get("_source_file")
            y = audio_cache.get(sf_)
            if y is None:
                wav = _find_wav(args.root, os.path.splitext(sf_)[0]) if sf_ else None
                if wav:
                    try:
                        y, _ = _load(wav, sr=SR, max_seconds=60)
                    except Exception:
                        y = None
                audio_cache[sf_] = y
            if y is None:
                emb.append(np.zeros(768, np.float32))
                continue
            a, b = int(e["t_on"] * SR), int(e["t_off"] * SR)
            seg = y[a:b] if b > a else y[a : a + 1]
            emb.append(mert_embed(seg, SR))
        emb = np.array(emb)
        pipe = train_probe(emb[tr], ytr)
        pm = probe_predict(pipe, emb[te])
        results["mert"] = classification_report(yte, pm)
        plot_confusion(yte, pm, DEMO_DIR / "idmt_confusion_mert.png", title="IDMT MERT probe")

    # 汇总
    print("\n---------------- IDMT 三方 ablation（宏 F1）----------------")
    targets = {"rule": 0.7, "mert": 0.8}
    for k in ("rule", "forest", "mert"):
        if k not in results:
            continue
        r = results[k]
        tgt = targets.get(k)
        flag = "" if tgt is None else (" PASS" if r["macro_f1"] >= tgt else " <目标")
        print(
            f"  {k:8s}: 宏F1={r['macro_f1']:.3f}  acc={r['accuracy']:.3f}"
            + (f"  (目标≥{tgt}{flag})" if tgt else "")
        )
    print(f"混淆矩阵图 -> {DEMO_DIR}/idmt_confusion_*.png")
    print("口径（产品计划）：规则树≥0.7 / MERT探针≥0.8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
