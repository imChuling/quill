"""Atlas 数据管道 —— Sprint 2 第一块(episodic support/query 数据集)。

表示(v0,与 L0 精确兼容):
  每个音 → TimbreSnapshot → 谱包络采到**固定 log 频率格点**(80Hz–12kHz, 64 点, log 幅度)。
  共振峰保持渲染(mode="formant")在包络空间的静态预测 = support 的包络本身,
  因此 **residual = env(target) − env(support)**(log 域)。
  回渲染:H_steady'_k = H_steady_k · exp(residual(freq_k))(freq_k=support 各分量绝对频率),
  residual≡0 ⇒ 快照逐位不变 ⇒ α=0 精确回退 L0(引擎零改动)。

episode(v0 = 单 support;1-vs-3 消融属后续):
  (instrument, support_note, target_note),|Δsemitone| ≤ 18
  x = [Δst, support_f0(log2), env_support(64)]
  y = env_target − env_support(64)

split 铁律:按 **instrument instance** 切,测试乐器全程不见(冲刺本子 §5.6)。
缓存:results/atlas/snapshots.npz(逐音包络+元数据;analyze_timbre 每音 ~0.7s,只算一次)。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.harmonic import f0_scalar, track_f0  # noqa: E402
from analysis.timbre import analyze_timbre  # noqa: E402

GRID = np.geomspace(80.0, 12000.0, 64)  # 固定包络格点(Hz)
MAX_SEMIS = 18.0
CACHE = ROOT / "results" / "atlas" / "notes_v2.npz"
SNAP_DIR = ROOT / "results" / "atlas" / "snaps"  # 全快照缓存(换表示免重分析)

# 预注册 held-out(按族覆盖:弦/木管/铜管各留 1-2 件;Gate 2 要求改善不只在训练乐器)
HELD_OUT = [
    "TinySOL/Strings/Violoncello",
    "TinySOL/Winds/Oboe",
    "TinySOL/Brass/Trombone",
    "TinySOL/Keyboards/Accordion",
]


def env_on_grid(snap) -> np.ndarray:
    """快照 → 固定格点 log 谐波包络。派生语义与 build_instrument 逐行一致:
    H_steady=H_env 中段 50% 帧中位数;分量绝对频率=partial_freqs(准谐波)或 k·f0_ref。"""
    H = np.asarray(snap["H_env"], np.float64)  # [K, T]
    K, T = H.shape
    lo_t, hi_t = int(T * 0.25), max(int(T * 0.75), int(T * 0.25) + 1)
    H_steady = np.median(H[:, lo_t:hi_t], axis=1)
    pf = snap.get("partial_freqs")
    freqs = (
        np.asarray(pf, np.float64)[:K]
        if pf is not None and len(pf) >= K
        else np.arange(1, K + 1) * float(snap["f0_ref"])
    )
    H_steady = np.maximum(H_steady, 1e-7)
    return np.log(np.interp(GRID, freqs, H_steady, left=H_steady[0], right=H_steady[-1])).astype(
        np.float32
    )


def noise_on_grid(snap) -> np.ndarray:
    """快照 → 固定格点 log 噪声包络(N_env 各带中段中位数,锚在带中心绝对频率)。
    分解实验(7.31)证明:谐波与噪声必须**联合**修正才能兑现 headroom(单侧 recovery≈0,
    双侧 0.82)——归一化谱距把两层的相对电平耦合在一起。"""
    N = np.asarray(snap.get("N_env"), np.float64)
    if N is None or N.size == 0:
        return np.full(len(GRID), np.log(1e-7), np.float32)
    B, T = N.shape
    lo_t, hi_t = int(T * 0.25), max(int(T * 0.75), int(T * 0.25) + 1)
    N_steady = np.maximum(np.median(N[:, lo_t:hi_t], axis=1), 1e-7)
    centers = snap.get("noise_centers_hz")
    c = np.asarray(centers, np.float64)[:B] if centers is not None else np.linspace(0, 22050, B)
    return np.log(np.interp(GRID, c, N_steady, left=N_steady[0], right=N_steady[-1])).astype(
        np.float32
    )


def snap_cache_path(i: int) -> Path:
    return SNAP_DIR / f"note_{i:04d}.npz"


def load_snap(i: int) -> dict:
    z = np.load(snap_cache_path(i), allow_pickle=True)
    return {k: z[k] if z[k].shape != () else z[k].item() for k in z.files}


def build_note_table(force=False):
    """TinySOL 全部音 → meta + envs[N,128](64 谐波 + 64 噪声);全快照落盘 SNAP_DIR。"""
    if CACHE.exists() and not force:
        z = np.load(CACHE, allow_pickle=True)
        return json.loads(str(z["meta"])), z["envs"]
    from eval.fidelity import SR, tinysol_groups

    groups = tinysol_groups(Path.home() / "Desktop" / "MT" / "datasets" / "TinySOL", max_pitch=9)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    meta, envs = [], []
    t0 = time.time()
    for name, samples in groups:
        for midi, audio in samples:
            y = np.asarray(audio, np.float32)
            f0v = f0_scalar(track_f0(y, sr=SR)[0])
            if not (np.isfinite(f0v) and f0v > 0):
                continue
            try:
                snap = analyze_timbre(y, sr=SR, name=f"{name}/{midi}")
            except Exception:
                continue
            i = len(envs)
            np.savez_compressed(
                snap_cache_path(i), **{k: v for k, v in snap.items() if k != "meta"}
            )
            meta.append({"instrument": name, "midi": int(midi), "f0": float(f0v), "i": i})
            envs.append(np.concatenate([env_on_grid(snap), noise_on_grid(snap)]))
        print(f"  {name}: 累计 {len(envs)} 音 ({time.time() - t0:.0f}s)", flush=True)
    envs = np.stack(envs).astype(np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, envs=envs, meta=json.dumps(meta))
    return meta, envs


def episodes(meta, envs, instruments=None):
    """全对 episode:x=[Δst, log2 f0_sup, env_sup(64)] → y=env_tgt−env_sup(64)。"""
    by_inst = {}
    for m in meta:
        by_inst.setdefault(m["instrument"], []).append(m)
    X, Y, tags = [], [], []
    for inst, notes in by_inst.items():
        if instruments is not None and inst not in instruments:
            continue
        for s in notes:
            for t in notes:
                if s["i"] == t["i"]:
                    continue
                dst = 12.0 * np.log2(t["f0"] / s["f0"])
                if abs(dst) > MAX_SEMIS:
                    continue
                es, et = envs[s["i"]], envs[t["i"]]  # 128 维(谐波64+噪声64)
                X.append(np.concatenate([[dst, np.log2(s["f0"])], es]))
                Y.append(et - es)
                tags.append(
                    {
                        "instrument": inst,
                        "sup_midi": s["midi"],
                        "tgt_midi": t["midi"],
                        "dst": round(float(dst), 2),
                        "sup_i": s["i"],
                        "tgt_i": t["i"],
                    }
                )
    return (np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32), tags)


def train_test_split(meta, envs):
    all_inst = sorted({m["instrument"] for m in meta})
    train_inst = [i for i in all_inst if i not in HELD_OUT]
    test_inst = [i for i in all_inst if i in HELD_OUT]
    return (
        episodes(meta, envs, train_inst),
        episodes(meta, envs, test_inst),
        train_inst,
        test_inst,
    )


if __name__ == "__main__":
    meta, envs = build_note_table(force="--force" in sys.argv)
    (Xtr, Ytr, _), (Xte, Yte, _), tr_i, te_i = train_test_split(meta, envs)
    print(
        f"notes={len(meta)}  train eps={len(Xtr)}({len(tr_i)} inst)  "
        f"test eps={len(Xte)}({len(te_i)} inst: {te_i})"
    )
