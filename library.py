"""SnapshotLibrary：标签化的 TimbreSnapshot 语料与预设库。

两种用法：按 tag（bass / guitar / violin…）或语义描述符检索一张代表快照直接演奏；
或把 (tag, instrument, pitch) 元数据与全部快照展平成参数矩阵，供条件生成先验训练，
合成端仍是确定性的实时 DSP。

语料是一个目录的 .npz 快照加 index.json（tag/family/instrument/midi/source 元数据）。
数据源来自 eval.fidelity 的 tinysol_groups / nsynth_groups。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.timbre import analyze_timbre
from snapshot import load_snapshot, save_snapshot

SR = 44100


def _slug(s):
    return s.replace("/", "_").replace(" ", "")


def build_corpus(groups, out_dir, source="dataset", quasi=False, verbose=True):
    """groups: [(name, [(midi, audio), ...]), ...]，name 形如 'TinySOL/Strings/Violin'。

    逐样本 analyze_timbre -> 存 .npz -> 累积 index.json。返回 index(list of dict)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / "index.json"
    index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    seen = {(e["instrument"], e["midi"]) for e in index}

    for name, samples in groups:
        parts = name.split("/")
        tag = parts[-1]
        family = parts[1] if len(parts) > 2 else (parts[0] if parts else name)
        for midi, audio in samples:
            if (name, int(midi)) in seen:
                continue
            f0_hint = 440.0 * 2 ** ((int(midi) - 69) / 12.0)  # 真值音高纠八度错
            try:
                snap = analyze_timbre(
                    np.asarray(audio, np.float32),
                    sr=SR,
                    name=f"{name}@{midi}",
                    quasi_harmonic=quasi,
                    f0_hint=f0_hint,
                )
            except Exception as e:
                if verbose:
                    print(f"  [skip] {name}@{midi}: {e}")
                continue
            if not (np.isfinite(snap["f0_ref"]) and snap["f0_ref"] > 0):
                continue
            fn = f"{_slug(name)}__{int(midi)}.npz"
            save_snapshot(out_dir / fn, snap)
            entry = {
                "file": fn,
                "tag": tag,
                "family": family,
                "instrument": name,
                "midi": int(midi),
                "f0_ref": float(snap["f0_ref"]),
                "source": source,
            }
            index.append(entry)
            seen.add((name, int(midi)))
            if verbose:
                print(f"  + {fn}  f0={snap['f0_ref']:.1f}Hz")
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=0))
    if verbose:
        print(f"语料 {len(index)} 张快照 -> {idx_path}")
    return index


# 质量白名单：谱建模重合成效果好的族，共鸣强、持续音、共振峰保持占优。排除会塌的：
# mallet 非谐打击、accordion 自由簧、弓弦颤音加弓噪、气声长笛/双簧。拨弦吉他的身份在
# 拨弦瞬态，静态快照抓不住，不入白名单，吉他走 DDSP 神经音色；低频拨弦的 bass 尚可，保留。
GOOD_TAGS = {
    "bass",
    "brass",
    "organ",
    "keyboard",
    "vocal",
    "reed",
    "Bass_Tuba",
    "Contrabass",
    "Bassoon",
    "Horn",
    "Trombone",
    "Trumpet_C",
    "Clarinet_Bb",
    "Sax_Alto",
}

# 语义描述符 → 候选 tag：按听感挑音色而非乐器学名，均落在 GOOD_TAGS 内
DESCRIPTORS = {
    "low 低沉": ["bass", "Contrabass", "Bass_Tuba"],
    "brass 铜管": ["brass", "Trumpet_C", "Trombone", "Horn", "Bass_Tuba"],
    "warm 温暖": ["Bassoon", "Horn", "Contrabass", "organ"],
    "bright 明亮": ["Trumpet_C", "reed", "Sax_Alto"],
    "reedy 簧管": ["reed", "Clarinet_Bb", "Sax_Alto"],
    "vocal 人声": ["vocal"],
    "organ 风琴": ["organ", "keyboard"],
}


class SnapshotLibrary:
    """加载语料；按 tag 或语义描述符检索可演奏的代表快照。"""

    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())

    def tags(self):
        return sorted({e["tag"] for e in self.index})

    def entries(self, tag=None, family=None):
        out = self.index
        if tag is not None:
            out = [e for e in out if e["tag"] == tag]
        if family is not None:
            out = [e for e in out if e["family"] == family]
        return out

    def retrieve(self, tag=None, family=None, pitch_midi=60):
        """取最接近目标音高的代表快照；共振峰保持让它可以弹到全音域。"""
        cand = self.entries(tag=tag, family=family)
        if not cand:
            raise KeyError(f"no snapshot for tag={tag} family={family}")
        f0_tgt = 440.0 * 2 ** ((pitch_midi - 69) / 12)
        best = min(cand, key=lambda e: abs(np.log2(max(e["f0_ref"], 1.0) / f0_tgt)))
        return load_snapshot(self.root / best["file"]), best

    def good_entries(self):
        """质量白名单内的条目；demo 与检索默认只露这些。"""
        return [e for e in self.index if e["tag"] in GOOD_TAGS]

    def descriptors(self):
        """可用的语义描述符（语料里有 ≥1 张对应的好快照）→ {描述符: 张数}。"""
        out = {}
        for desc, tags in DESCRIPTORS.items():
            n = sum(1 for e in self.good_entries() if e["tag"] in tags)
            if n:
                out[desc] = n
        return out

    def browse(self):
        """浏览库：返回 [(描述符, 张数, 代表族)]。"""
        rows = []
        for desc, n in self.descriptors().items():
            tags = DESCRIPTORS[desc]
            fams = sorted({e["tag"] for e in self.good_entries() if e["tag"] in tags})
            rows.append((desc, n, fams))
        return rows

    def retrieve_descriptor(self, descriptor, pitch_midi=60):
        """按语义描述符（"warm 温暖" 等）检索最接近音高的好快照；也接受直接给 tag。"""
        tags = DESCRIPTORS.get(descriptor)
        if tags is None:
            tags = [descriptor]
        cand = [e for e in self.good_entries() if e["tag"] in tags]
        if not cand:
            raise KeyError(f"no good snapshot for descriptor={descriptor!r}")
        f0_tgt = 440.0 * 2 ** ((pitch_midi - 69) / 12)
        best = min(cand, key=lambda e: abs(np.log2(max(e["f0_ref"], 1.0) / f0_tgt)))
        return load_snapshot(self.root / best["file"]), best

    def matrix(self):
        """返回 (X[N,D] 展平快照参数, meta[N])，给生成先验训练用（参数空间，非波形）。"""
        X, meta = [], []
        for e in self.index:
            snap = load_snapshot(self.root / e["file"])
            X.append(_flatten_snapshot(snap))
            meta.append(e)
        return np.asarray(X, dtype=np.float32), meta
def main(argv=None):
    """构建标签化快照语料：python -m library --tinysol <root> --nsynth <root> --out <dir>"""
    import argparse
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent))
    from eval.fidelity import nsynth_groups, tinysol_groups

    ap = argparse.ArgumentParser(description="构建 SnapshotLibrary 语料")
    ap.add_argument("--out", default="../../datasets/quill_corpus")
    ap.add_argument("--tinysol", default=None)
    ap.add_argument("--nsynth", default=None)
    ap.add_argument("--max-pitch", type=int, default=15)
    ap.add_argument("--per-family", type=int, default=4)
    args = ap.parse_args(argv)

    groups = []
    if args.tinysol:
        groups += tinysol_groups(args.tinysol, max_pitch=args.max_pitch, min_pitch=3)
    if args.nsynth:
        groups += nsynth_groups(
            args.nsynth, per_family=args.per_family, max_pitch=args.max_pitch, min_pitch=3
        )
    print(f"待分析 {len(groups)} 件乐器 -> {args.out}")
    idx = build_corpus(groups, args.out, source="dataset")
    from collections import Counter

    print("按 tag 分布:", dict(Counter(e["tag"] for e in idx)))
    return 0


def _flatten_snapshot(snap, n_harm=60, n_mel=40, t_steady=1):
    """把快照压成定长向量（稳态谐波 log 幅度 + 噪声带 log 均值 + log f0_ref），供先验建模。"""
    H = np.asarray(snap["H_env"], np.float32)  # [K, T]
    N = np.asarray(snap["N_env"], np.float32)  # [M, T]
    h = np.log(np.maximum(H.mean(axis=1), 1e-6))[:n_harm]
    h = np.pad(h, (0, max(0, n_harm - len(h))))[:n_harm]
    nz = np.log(np.maximum(N.mean(axis=1), 1e-6))[:n_mel]
    nz = np.pad(nz, (0, max(0, n_mel - len(nz))))[:n_mel]
    f0 = np.array([np.log(max(snap["f0_ref"], 1.0))], np.float32)
    return np.concatenate([h, nz, f0]).astype(np.float32)


def reconstruct_snapshot(flat, skeleton, n_harm=60, n_mel=40):
    """_flatten_snapshot 的逆：把生成的稳态谱形嫁接到检索 skeleton 的时间动态与起音残差上。

    生成先验只产出稳态截面（101 维），没有时间演化和 attack；本函数按谐波/噪声带把
    skeleton 的时变包络重新着色到目标稳态谱，保留 skeleton 的起音、衰减与残差颗粒。
    返回可渲染快照。
    """
    flat = np.asarray(flat, dtype=np.float64)
    h_t = np.exp(flat[:n_harm])  # 目标稳态谐波幅度
    n_t = np.exp(flat[n_harm : n_harm + n_mel])  # 目标稳态噪声带
    H = np.asarray(skeleton["H_env"], np.float64).copy()
    N = np.asarray(skeleton["N_env"], np.float64).copy()
    K = min(n_harm, H.shape[0])
    M = min(n_mel, N.shape[0])
    sk_h = H[:K].mean(axis=1)  # skeleton 稳态截面
    sk_n = N[:M].mean(axis=1)
    H[:K] *= (h_t[:K] / (sk_h + 1e-8))[:, None]  # 逐谐波重着色，保留时间形状
    N[:M] *= (n_t[:M] / (sk_n + 1e-8))[:, None]
    snap = dict(skeleton)  # 沿用 attack_res/f0_ref/centers/partials
    snap["H_env"] = H.astype(np.float32)
    snap["N_env"] = N.astype(np.float32)
    snap["meta"] = dict(skeleton.get("meta", {}))
    snap["meta"]["name"] = "generated"
    snap["meta"]["skeleton"] = skeleton.get("meta", {}).get("name", "?")
    return snap


if __name__ == "__main__":
    raise SystemExit(main())
