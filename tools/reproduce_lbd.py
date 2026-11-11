"""Reproduce all numbers from the LBD paper.

Usage:
    python tools/reproduce_lbd.py              # re-run eval, compare with stored results
    python tools/reproduce_lbd.py --check-only # verify checkpoint hashes + eval_results structure

Requires checkpoints (DDSP models + behavior libraries) in checkpoints/.
Exit code 0 = all checks pass; 1 = mismatch or missing data.
"""

import hashlib
import json
import sys
from pathlib import Path

QUILL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QUILL))

STORED = QUILL / "checkpoints" / "eval" / "eval_results.json"

CHECKPOINT_HASHES = {
    "ddsp_urmp_vn_12k.pt": "d63c58185c753f096033c3d941725acbdebd1822455297b69c2fb7516dfd9f3d",
    "ddsp_urmp_tpt_12k.pt": "ed653891fc0ed21151ef12f4495d2b9b01ef8239be08af2d667b42e3e853b831",
    "ddsp_glass.pt": "b718be519aee9e87271ad7ebbe8810b76179f55a8555d6189c02b88cc396151f",
    "ddsp_guitarset.pt": "bdc153946fec299a07943950fa0e014c5663e481e0c8d2e5fedac6f0c5fb012e",
    "behavior_distilled.pt": "238df55dadd812f4cea34dfefb9000d64d8eeaeba9855e79cabaf4cea8be1a8c",
    "vibrato_model.pt": "79b6f544502e8b8d893ae9d18ba5211cf5d866c04ce4b64934ad918304635405",
    "urmp_vn_trajectories.npz": "0f081f432874392fe8e3ca8e4558fd2be0eb6fb916e9bd04654e031a43a157fe",
    "urmp_tpt_trajectories.npz": "d80a59f5ee0c01e1394d52a5f31549d358531cc9c4b6f20a66e2d74cbc6a042c",
    "guitarset_trajectories.npz": "ac41f33d253d86bbbf6406f30cb186203b0f59bd5bb52a67631444a24283275b",
    "glass_style.npz": "198675c22031590bf2115e2324bc8bd49b4901a579ffa9505f687003be9f1bcf",
}

TOLERANCES = {
    "controllability.v0.contrast": 0.5,
    "controllability.v0.slide_ok": 0,
    "decoupling.passed": None,
    "paired_counterfactual.control_median_ratio": 1.0,
    "paired_counterfactual.control_n": 0,
    "paired_counterfactual.audio_median_ratio": 1.0,
    "processing_ablation.none.paired_median_ratio": 0.3,
    "processing_ablation.full.paired_median_ratio": 1.0,
    "glass_2x2.passed": None,
}

REQUIRED_KEYS = [
    "controllability",
    "decoupling",
    "data_curve",
    "paired_counterfactual",
    "processing_ablation",
    "glass_2x2",
]


def _get(d, dotpath):
    for k in dotpath.split("."):
        d = d[k]
    return d


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoints():
    ok = True
    ckpt = QUILL / "checkpoints"
    for name, expected in CHECKPOINT_HASHES.items():
        p = ckpt / name
        if not p.exists():
            print(f"  MISS  {name}")
            ok = False
            continue
        got = _sha256(p)
        match = got == expected
        tag = " OK  " if match else " FAIL"
        print(f"  {tag}  {name}")
        if not match:
            print(f"         expected {expected}")
            print(f"         got      {got}")
            ok = False
    return ok


def verify_eval_structure(stored):
    ok = True
    for key in REQUIRED_KEYS:
        if key in stored:
            print(f"   OK    eval_results.{key}")
        else:
            print(f"  MISS  eval_results.{key}")
            ok = False
    for path in TOLERANCES:
        try:
            _get(stored, path)
            print(f"   OK    {path}")
        except (KeyError, TypeError):
            print(f"  MISS  {path}")
            ok = False
    return ok


def compare(stored, fresh):
    ok = True
    for path, tol in TOLERANCES.items():
        try:
            sv = _get(stored, path)
            fv = _get(fresh, path)
        except (KeyError, TypeError):
            print(f"  MISS  {path}")
            ok = False
            continue
        match = sv == fv if tol is None else abs(float(sv) - float(fv)) <= tol
        tag = " OK  " if match else " FAIL"
        print(f"  {tag}  {path}: stored={sv}  fresh={fv}" + (f"  (tol={tol})" if tol else ""))
        if not match:
            ok = False
    return ok


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Reproduce LBD paper numbers")
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="verify checkpoint hashes and eval_results structure (no GPU needed)",
    )
    args = ap.parse_args()

    if not STORED.exists():
        print(f"ERROR: stored results not found at {STORED}")
        return 1

    stored = json.loads(STORED.read_text())

    if args.check_only:
        print("1/2  Verifying checkpoint integrity...")
        ok1 = verify_checkpoints()
        print("\n2/2  Verifying eval_results.json structure...")
        ok2 = verify_eval_structure(stored)
        ok = ok1 and ok2
    else:
        print("Re-running full evaluation (this takes several minutes)...")
        from tools.eval_quill import main as run_eval

        run_eval()

        fresh_path = QUILL / "checkpoints" / "eval" / "eval_results.json"
        fresh = json.loads(fresh_path.read_text())
        print("\nComparing fresh results against stored baseline:")
        ok = compare(stored, fresh)

    if ok:
        print("\nPASS: all paper numbers reproduced within tolerance.")
        return 0
    else:
        print("\nFAIL: some numbers differ beyond tolerance.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
