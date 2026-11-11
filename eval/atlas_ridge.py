"""Gate 1 · Ridge linear baseline (v2 two-channel), Sprint 2.

v2 changes (driven by the 7.31 decomposition experiment):
  - residual = [harmonic envelope residual(64), noise envelope residual(64)], corrected **jointly**
    (one side alone recovery≈0, both sides 0.82: the normalised spectral distance couples the relative levels of the two layers);
  - the harmonic side is applied at the target absolute frequencies (render env_correction entry point);
    the noise side scales the N_env bands directly (noise does not shift with pitch, band centre = absolute frequency);
  - render evaluation reads the full snapshot cache (SNAP_DIR) instead of rerunning analyze_timbre;
  - the true-patch upper bound (the limit this mechanism can realise) is reported alongside for comparison.

Preregistered reading (unchanged): median ridge_recovery ≥0.5 KEEP-RIDGE-BAR-HIGH /
0.15–0.5 NEURAL-WORTH-TRYING / <0.15 CHECK-REPRESENTATION.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.atlas_data import GRID, build_note_table, load_snap, train_test_split  # noqa: E402
from eval.fidelity import SR, spectral_distance, tinysol_groups  # noqa: E402
from synth.render import render  # noqa: E402

BOUND = 3.0  # residual clipping (log nat ≈ ±26dB; with v1's ±1.5 the clipping rate was 33%)
RENDER_EVAL_N = 48


def apply_residual(snap: dict, res128: np.ndarray) -> dict:
    """residual -> corrected snapshot copy. The harmonic side is left to render's env_correction (target side);
    the noise side scales N_env here (band centre = absolute frequency, no shifting with pitch). Returns (snap', harm_res)."""
    res = np.clip(np.asarray(res128, np.float64), -BOUND, BOUND)
    harm_res, noise_res = res[:64], res[64:]
    out = dict(snap)
    N = np.asarray(snap.get("N_env"), np.float64)
    if N is not None and N.size:
        centers = snap.get("noise_centers_hz")
        B = N.shape[0]
        c = np.asarray(centers, np.float64)[:B] if centers is not None else np.linspace(0, 22050, B)
        fac = np.exp(np.interp(c, GRID, noise_res))
        out["N_env"] = N * fac[:, None]
    return out, harm_res


def corrected_render(sup_i: int, f0_sup_real, f0_tgt_real, res128, n):
    snap = load_snap(sup_i)
    snap2, harm_res = apply_residual(snap, res128)
    f0_tgt = float(snap["f0_ref"]) * (f0_tgt_real / f0_sup_real)
    return render(
        snap2, f0_tgt, length=n, mode="formant", add_attack=False, env_correction=(GRID, harm_res)
    )


def fit_ridge(Xtr, Ytr, alpha=10.0):
    from sklearn.linear_model import Ridge

    m = Ridge(alpha=alpha)
    m.fit(Xtr, Ytr)
    return m


def main():
    meta, envs = build_note_table()
    (Xtr, Ytr, tags_tr), (Xte, Yte, tags_te), tr_i, te_i = train_test_split(meta, envs)
    print(f"train {len(Xtr)} eps / {len(tr_i)} inst;  test {len(Xte)} eps / {te_i}")

    model = fit_ridge(Xtr, Ytr)
    P = model.predict(Xte).astype(np.float32)

    e_static = float(np.mean(np.abs(Yte)))
    e_ridge = float(np.mean(np.abs(P - Yte)))
    print(
        f"envelope space (128 dims) mean|res|: static={e_static:.3f}  ridge={e_ridge:.3f}  "
        f"(down {100 * (1 - e_ridge / e_static):.0f}%)"
    )

    audio_by = {}
    for name, samples in tinysol_groups(
        Path.home() / "Desktop" / "MT" / "datasets" / "TinySOL", max_pitch=9
    ):
        if name in te_i:
            audio_by[name] = {m: np.asarray(a, np.float32) for m, a in samples}
    order = np.argsort([-abs(t["dst"]) for t in tags_te])[:RENDER_EVAL_N]
    rows = []
    t0 = time.time()
    for j, idx in enumerate(order):
        t = tags_te[idx]
        tgt = audio_by.get(t["instrument"], {}).get(t["tgt_midi"])
        if tgt is None:
            continue
        f0_sup = 2.0 ** Xte[idx][1]
        f0_tgt = f0_sup * 2.0 ** (t["dst"] / 12.0)
        n = len(tgt)
        try:
            y_ridge = corrected_render(t["sup_i"], f0_sup, f0_tgt, P[idx], n)
            y_true = corrected_render(t["sup_i"], f0_sup, f0_tgt, Yte[idx], n)
            snap_sup = load_snap(t["sup_i"])
            y_static = render(
                snap_sup,
                float(snap_sup["f0_ref"]) * (f0_tgt / f0_sup),
                length=n,
                mode="formant",
                add_attack=False,
            )
            snap_o = load_snap(t["tgt_i"])
            y_oracle = render(
                snap_o, float(snap_o["f0_ref"]), length=n, mode="formant", add_attack=False
            )
        except Exception:
            continue
        rows.append(
            {
                "instrument": t["instrument"],
                "dst": t["dst"],
                "d_static": spectral_distance(y_static, tgt, SR),
                "d_ridge": spectral_distance(y_ridge, tgt, SR),
                "d_true": spectral_distance(y_true, tgt, SR),
                "d_oracle": spectral_distance(y_oracle, tgt, SR),
            }
        )
        if (j + 1) % 12 == 0:
            print(f"  render evaluation {j + 1}/{len(order)} ({time.time() - t0:.0f}s)", flush=True)

    ds = np.array([r["d_static"] for r in rows])
    dr = np.array([r["d_ridge"] for r in rows])
    dt = np.array([r["d_true"] for r in rows])
    do = np.array([r["d_oracle"] for r in rows])
    hs = ds - do
    rec_r = np.where(hs > 1e-9, (ds - dr) / hs, 0.0)
    rec_t = np.where(hs > 1e-9, (ds - dt) / hs, 0.0)
    med_rec = float(np.median(rec_r))
    out = {
        "n_render_eval": len(rows),
        "held_out": te_i,
        "bound": BOUND,
        "env_static": e_static,
        "env_ridge": e_ridge,
        "render_median": {
            "static": float(np.median(ds)),
            "ridge": float(np.median(dr)),
            "true_patch": float(np.median(dt)),
            "oracle": float(np.median(do)),
        },
        "ridge_recovery_median": med_rec,
        "truepatch_recovery_median": float(np.median(rec_t)),
        "ridge_beats_static_rate": float(np.mean(dr < ds)),
        "verdict": (
            "KEEP-RIDGE-BAR-HIGH"
            if med_rec >= 0.5
            else "NEURAL-WORTH-TRYING"
            if med_rec >= 0.15
            else "CHECK-REPRESENTATION"
        ),
        "rows": rows,
    }
    p = ROOT / "results" / "atlas" / "ridge_gate1_v2.json"
    p.write_text(json.dumps(out, indent=1))
    print(
        f"\nrender-space median: static={out['render_median']['static']:.3f}  "
        f"ridge={out['render_median']['ridge']:.3f}  "
        f"true={out['render_median']['true_patch']:.3f}  "
        f"oracle={out['render_median']['oracle']:.3f}"
    )
    print(
        f"recovery median: ridge={med_rec:.2f}  true-patch upper bound={out['truepatch_recovery_median']:.2f}  "
        f"ridge win rate={out['ridge_beats_static_rate']:.2f}"
    )
    print(f">>> GATE 1 (v2) reading: {out['verdict']}   artifact: {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
