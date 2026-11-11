"""E5: behaviour library size curve: contrast vs library size.

Outputs: docs/report/fig_e5_data_curve.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUILL = Path(__file__).resolve().parent.parent
DATA = QUILL / "checkpoints" / "eval" / "eval_results.json"
OUT = QUILL / "docs" / "report" / "fig_e5_data_curve.png"


def main():
    results = json.loads(DATA.read_text())
    dc = results["data_curve"]

    sizes = [e["n"] for e in dc]
    contrasts = [e["contrast"] for e in dc]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.plot(sizes, contrasts, "o-", color="#2c7bb6", linewidth=2, markersize=7, zorder=3)

    for s, c in zip(sizes, contrasts):
        ax.annotate(f"{c:.1f}×", (s, c), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")

    ax.axhline(2.0, color="gray", ls="--", lw=0.8, alpha=0.6, label="2× threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Library size (notes)", fontsize=10)
    ax.set_ylabel("Contrast ratio (8-seed mean)", fontsize=10)
    ax.set_title("E5: Contrast vs Library Size", fontsize=11)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_ylim(0, max(contrasts) * 1.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
