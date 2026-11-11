"""升级行为库 npz 到 v1.0 格式:加 format_version / source / license / fps 元字段。

幂等:已有 format_version 的文件跳过。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

CKPT = Path(__file__).resolve().parent.parent / "checkpoints"

SPECS = {
    "urmp_vn_trajectories.npz": {
        "source": "URMP violin (University of Rochester Multi-Modal Music Performance dataset)",
        "license": "Derived trajectories (no audio); source dataset license unconfirmed — requires registration and citation",
    },
    "urmp_tpt_trajectories.npz": {
        "source": "URMP trumpet (University of Rochester Multi-Modal Music Performance dataset)",
        "license": "Derived trajectories (no audio); source dataset license unconfirmed — requires registration and citation",
    },
    "guitarset_trajectories.npz": {
        "source": "GuitarSet acoustic guitar (Xi et al., ISMIR 2018)",
        "license": "Derived trajectories (no audio); source CC BY-NC 4.0",
    },
    "glass_style.npz": {
        "source": "Self-recorded wine glass (twelve takes, three glasses + one fishbowl)",
        "license": "MIT",
    },
}

FORMAT_VERSION = "1.0"
FPS = 250


def upgrade(fname, spec):
    path = CKPT / fname
    if not path.exists():
        print(f"  SKIP  {fname} (not found)")
        return
    d = dict(np.load(str(path), allow_pickle=False))
    if "format_version" in d:
        print(f"  SKIP  {fname} (already v{d['format_version']})")
        return
    d["format_version"] = np.array(FORMAT_VERSION)
    d["source"] = np.array(spec["source"])
    d["license"] = np.array(spec["license"])
    d["fps"] = np.array(FPS, dtype=np.int32)
    d["traj_units"] = np.array(["cents_from_median", "z_score", "log2_residual", "log10_residual"])
    np.savez_compressed(str(path), **d)
    print(f"  OK    {fname} → v{FORMAT_VERSION}")


def main():
    print("Upgrading behavior libraries to v1.0 format...\n")
    for fname, spec in SPECS.items():
        upgrade(fname, spec)
    print("\nDone.")


if __name__ == "__main__":
    main()
