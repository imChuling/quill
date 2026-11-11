"""LBD 用图:读 eval_results.json 生成可编辑的矢量图。

  fig_datacurve.pdf   数据效率曲线(库规模 → 对比度)
  fig_decoupling.pdf/.svg  行为×音色来源归属比
  fig_pipeline.pdf    系统 pipeline 示意

跑: PATH="/Library/TeX/texbin:$PATH" conda run -n mt python tools/make_figures.py
"""

import json
import os
from pathlib import Path

os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ.get("PATH", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle

QUILL = Path(__file__).resolve().parent.parent
EVAL = QUILL / "checkpoints" / "eval"
OUT = QUILL / "docs" / "figures"
PAPER_OUT = QUILL / "docs" / "ismir_lbd_2026"

# ---- SciencePlots + IEEE + LaTeX 渲染 ---- #
plt.style.use(["science", "ieee"])

# Preserve labels as editable text in SVG and embed TrueType fonts in PDF.
plt.rcParams.update(
    {
        "text.usetex": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    }
)

# ---- Quill publication palette ---- #
C_LIGHT_GREEN = "#D4E6A9"
C_LIGHT_PURPLE = "#D4BFDC"
C_LIGHT_BLUE = "#9BCAE7"
C_PINK = "#E3A8AF"
C_PALE_PINK = "#F9D1D2"
C_BLUE = "#477DC0"
C_ORANGE = "#ED9B69"
C_CORAL = "#C9676D"
C_GRAY = "#7A7A7A"

INST_LABEL = {"vi": "violin", "gu": "guitar", "tr": "trumpet"}


def export_figure(fig, stem, paper=False):
    """Export preview plus editable vectors, including the paper-local copy."""
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)
        if paper:
            fig.savefig(PAPER_OUT / f"{stem}.{suffix}", **kwargs)


def fig_datacurve(res):
    curve = res["data_curve"]
    n = [r["n"] for r in curve]
    c = [r["contrast"] for r in curve]
    e = [r.get("contrast_std", 0.0) for r in curve]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.fill_between(
        n,
        [ci - ei for ci, ei in zip(c, e)],
        [ci + ei for ci, ei in zip(c, e)],
        alpha=0.18,
        color=C_BLUE,
        linewidth=0,
    )
    ax.plot(
        n,
        c,
        "o-",
        color=C_BLUE,
        markersize=3.5,
        linewidth=1.0,
        markeredgecolor="white",
        markeredgewidth=0.4,
        label="Vibrato / plain contrast",
        zorder=3,
    )

    ax.axhline(1.0, color=C_CORAL, lw=0.5, ls="--", zorder=1)
    ax.text(n[0] * 0.85, 1.35, r"no control ($1\times$)", fontsize=6, color=C_CORAL, style="italic")

    ax.axvline(500, color=C_ORANGE, lw=0.5, ls=":", zorder=0, alpha=0.7)
    ax.annotate(
        r"$\approx$2 min",
        xy=(500, c[2]),
        fontsize=6.5,
        color=C_ORANGE,
        ha="left",
        va="bottom",
        weight="bold",
        xytext=(750, c[2] + 1.0),
        arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=0.5, mutation_scale=7),
    )

    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(x) for x in n])
    ax.set_xlabel("Style library size (notes)")
    ax.set_ylabel(r"Articulation contrast ($\times$)")
    ax.set_ylim(0, max(c) * 1.25)
    ax.legend(loc="upper left")
    ax.tick_params(right=False, top=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    export_figure(fig, "fig_datacurve")
    plt.close(fig)
    print(f"  fig_datacurve: sizes={n} contrast={[round(x, 1) for x in c]}")


def fig_decoupling(res):
    dec = {k: v for k, v in res["decoupling"].items() if k != "passed"}
    order = ["vi", "gu", "tr"]
    names = ["V", "G", "T"]
    bmat = np.full((3, 3), np.nan)
    tmat = np.full((3, 3), np.nan)
    for pair, values in dec.items():
        b, t = pair.split("_")
        i, j = order.index(b[:2]), order.index(t[:2])
        bmat[i, j] = values["b_beh"] / max(values["b_tim"], 1e-9)
        tmat[i, j] = values["t_tim"] / max(values["t_beh"], 1e-9)

    cmap = LinearSegmentedColormap.from_list(
        "quill_attribution",
        [C_LIGHT_GREEN, C_LIGHT_BLUE, C_LIGHT_PURPLE, C_PINK, C_CORAL],
    )
    cmap.set_bad("#F4F4F4")
    norm = LogNorm(vmin=0.02, vmax=1.0)
    fig = plt.figure(figsize=(3.45, 2.35))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.58)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cax = fig.add_subplot(grid[0, 2])

    for panel, (ax, mat, title) in enumerate(
        zip(axes, (bmat, tmat), ("Behavior ratio", "Timbre ratio"))
    ):
        # pcolormesh keeps each cell as an editable vector rectangle in SVG/PDF.
        masked = np.ma.masked_invalid(mat)
        ax.pcolormesh(
            np.arange(4) - 0.5, np.arange(4) - 0.5, masked, cmap=cmap, norm=norm, shading="flat"
        )
        ax.set_title(title, fontsize=7.5, pad=3, weight="bold")
        ax.set_xticks(range(3), names, fontsize=7)
        ax.set_yticks(range(3), names, fontsize=7)
        ax.set_xlabel("Timbre target", fontsize=6.5, labelpad=2)
        if panel == 0:
            ax.set_ylabel("Behavior source", fontsize=6.5, labelpad=2)
        ax.tick_params(length=0, pad=2)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(3):
            for j in range(3):
                if i == j:
                    ax.add_patch(
                        Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            facecolor="#F4F4F4",
                            edgecolor="white",
                            lw=1.1,
                            zorder=2,
                        )
                    )
                    ax.text(
                        j,
                        i,
                        "same",
                        ha="center",
                        va="center",
                        fontsize=5.6,
                        color="#666666",
                        zorder=3,
                    )
                elif np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="black",
                        zorder=3,
                    )
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Draw the legend from vector rectangles so the SVG has no embedded bitmap.
    levels = np.geomspace(0.02, 1.0, 48)
    for lo, hi in zip(levels[:-1], levels[1:]):
        cax.add_patch(
            Rectangle((0, lo), 1, hi - lo, facecolor=cmap(norm(np.sqrt(lo * hi))), edgecolor="none")
        )
    cax.set_yscale("log")
    cax.set_ylim(0.02, 1.0)
    cax.set_xlim(0, 1)
    cax.set_xticks([])
    cax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    cax.set_yticklabels([".02", ".05", ".1", ".2", ".5", "1"], fontsize=5.8)
    cax.tick_params(length=2, pad=1)
    cax.set_ylabel("Distance ratio", fontsize=6.5, labelpad=2)
    for spine in cax.spines.values():
        spine.set_linewidth(0.4)
    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.22, top=0.88)

    passed = res["decoupling"]["passed"]
    export_figure(fig, "fig_decoupling", paper=True)
    plt.close(fig)
    print(f"  fig_decoupling: {passed} passed")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    with open(EVAL / "eval_results.json") as f:
        res = json.load(f)
    fig_datacurve(res)
    fig_decoupling(res)
    print(f"figures -> {OUT}/")


if __name__ == "__main__":
    main()
