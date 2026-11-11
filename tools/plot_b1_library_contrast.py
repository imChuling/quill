"""B1: contrast of the library itself: bar chart binned by library and note duration.

Outputs: docs/report/fig_b1_library_contrast.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

QUILL = Path(__file__).resolve().parent.parent
DATA = QUILL / "checkpoints" / "eval" / "library_contrast.json"
OUT = QUILL / "docs" / "report" / "fig_b1_library_contrast.png"


def main():
    data = json.loads(DATA.read_text())

    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    fig.suptitle("B1: Vibrato / Plain Contrast Ratio by Library", fontsize=11, y=1.0)

    lib_order = ["violin", "trumpet", "glass", "guitar"]
    colors = ["#2c7bb6", "#d7191c", "#fdae61", "#abd9e9"]

    for ax, lib_name, color in zip(axes, lib_order, colors):
        d = data[lib_name]
        if not d.get("has_labels", False):
            ax.set_title(f"{lib_name}\n(no vib/plain labels)", fontsize=9)
            ax.text(0.5, 0.5, f"n={d['n']}\nmedian energy\n{d['median_vib_energy']:.2f}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            continue

        overall = d["overall"]
        by_dur = d["by_duration"]

        labels = [overall["label"]] + [b["label"] for b in by_dur]
        ratios = [overall["contrast_ratio"]] + [b["contrast_ratio"] for b in by_dur]
        counts = [f"v{overall['n_vibrato']}/p{overall['n_plain']}"] + \
                 [f"v{b['n_vibrato']}/p{b['n_plain']}" for b in by_dur]

        x = np.arange(len(labels))
        bars = ax.bar(x, ratios, color=color, edgecolor="white", width=0.7)

        for xi, bar, cnt in zip(x, bars, counts):
            ax.text(xi, bar.get_height() + 0.3, f"{bar.get_height():.1f}×",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
            ax.text(xi, -0.8, cnt, ha="center", va="top", fontsize=5.5,
                    color="gray", rotation=0)

        ax.set_title(lib_name, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.axhline(2.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylim(0, max(ratios) * 1.4)

    axes[0].set_ylabel("Contrast ratio (vibrato / plain)", fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
