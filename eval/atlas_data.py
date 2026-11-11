"""Atlas data pipeline: Sprint 2 first piece (episodic support/query dataset).

Representation (v0, exactly compatible with L0):
  each note -> TimbreSnapshot -> spectral envelope sampled on a **fixed log-frequency grid** (80Hz–12kHz, 64 points, log amplitude).
  The static prediction of formant-preserving rendering (mode="formant") in envelope space = the support's own envelope,
  so **residual = env(target) − env(support)** (log domain).
  Render back: H_steady'_k = H_steady_k · exp(residual(freq_k)) (freq_k = absolute frequency of each support component),
  residual≡0 => snapshot bit-identical => α=0 falls back to L0 exactly (zero engine changes).

episode (v0 = single support; the 1-vs-3 ablation comes later):
  (instrument, support_note, target_note),|Δsemitone| ≤ 18
  x = [Δst, support_f0(log2), env_support(64)]
  y = env_target − env_support(64)

split rule: split by **instrument instance**, test instruments are never seen (sprint notebook §5.6).
cache: results/atlas/snapshots.npz (per-note envelopes + metadata; analyze_timbre ~0.7s per note, computed once).
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

GRID = np.geomspace(80.0, 12000.0, 64)  # fixed envelope grid (Hz)
MAX_SEMIS = 18.0
CACHE = ROOT / "results" / "atlas" / "notes_v2.npz"
SNAP_DIR = ROOT / "results" / "atlas" / "snaps"  # full snapshot cache (changing the representation needs no reanalysis)

# preregistered held-out (covering families: 1-2 each of strings/woodwinds/brass; Gate 2 requires improvement beyond training instruments)
HELD_OUT = [
    "TinySOL/Strings/Violoncello",
    "TinySOL/Winds/Oboe",
    "TinySOL/Brass/Trombone",
    "TinySOL/Keyboards/Accordion",
]


def env_on_grid(snap) -> np.ndarray:
    """Snapshot -> log harmonic envelope on the fixed grid. Derivation semantics match build_instrument line by line:
    H_steady = median of the middle 50% of H_env frames; component absolute frequencies = partial_freqs (quasi-harmonic) or k·f0_ref."""
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
    """Snapshot -> log noise envelope on the fixed grid (median of the middle section of each N_env band, anchored at the band-centre absolute frequency).
    The decomposition experiment (7.31) showed that harmonics and noise must be corrected **jointly** to realise the headroom (one side alone recovery≈0,
    both sides 0.82): the normalised spectral distance couples the relative levels of the two layers."""
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
    """All TinySOL notes -> meta + envs[N,128] (64 harmonic + 64 noise); full snapshots written to SNAP_DIR."""
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
        print(f"  {name}: {len(envs)} notes so far ({time.time() - t0:.0f}s)", flush=True)
    envs = np.stack(envs).astype(np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, envs=envs, meta=json.dumps(meta))
    return meta, envs


def episodes(meta, envs, instruments=None):
    """All-pairs episodes: x=[Δst, log2 f0_sup, env_sup(64)] -> y=env_tgt−env_sup(64)."""
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
                es, et = envs[s["i"]], envs[t["i"]]  # 128 dims (harmonic 64 + noise 64)
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
