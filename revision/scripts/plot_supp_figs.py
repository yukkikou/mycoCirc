#!/usr/bin/env python3
"""Supplementary figures for the mycoCirc revision (BMC Bioinformatics).

Fig S1  Calibration of genome-wide predictions on external species
        (source: revision/results/a4_calibration.csv)
Fig S2  Species-embedding (strain-slot) sensitivity sweep
        (source: revision/results/a7_strainid_sensitivity.csv)

Publication contract: width 183 mm (BMC max ~7.24 in), Arial, editable
text (svg.fonttype=none, pdf.fonttype=42), base font 7 pt, no top/right
spines, SVG+PDF+PNG(600 dpi) export.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",      # editable text in SVG
    "pdf.fonttype": 42,          # editable TrueType in PDF
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

PROJ = Path(__file__).resolve().parents[2]
RES = PROJ / "revision" / "results"
FIG = PROJ / "revision" / "figures"
FIG.mkdir(exist_ok=True)

PALETTE = {
    "blue": "#0F4D92",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#E8A33D",
    "red": "#B64342",
    "grey": "#767676",
    "grey_light": "#CFCECE",
    "red_shade": "#F6CFCB",
}

MM = 1 / 25.4


def save_pub(fig, stem):
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=600, bbox_inches="tight")
    print(f"saved {stem}.svg/.pdf/.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig S1 — reliability diagrams (4 external species, Filamentous checkpoint)
# ═══════════════════════════════════════════════════════════════════════════

def fig_calibration():
    df = pd.read_csv(RES / "a4_calibration.csv")
    ece_rows = df[df["bin"] == "ECE"]
    bins = df[df["bin"] != "ECE"].copy()

    order = ["Aspergillus_cristatus", "Trichophyton_mentagrophytes",
             "Trichophyton_rubrum", "Talaromyces_marneffei"]
    disp = {
        "Aspergillus_cristatus": ("A. cristatus", "≥2 tools"),
        "Trichophyton_mentagrophytes": ("T. mentagrophytes", "≥2 tools"),
        "Trichophyton_rubrum": ("T. rubrum", "≥2 tools"),
        "Talaromyces_marneffei": ("T. marneffei", "CIRIquant"),
    }
    colors = {
        "Aspergillus_cristatus": PALETTE["blue"],
        "Trichophyton_mentagrophytes": PALETTE["teal"],
        "Trichophyton_rubrum": PALETTE["violet"],
        "Talaromyces_marneffei": PALETTE["gold"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 128 * MM))
    panel_labels = ["A", "B", "C", "D"]

    for k, (sp, ax) in enumerate(zip(order, axes.flat)):
        name, gt = disp[sp]
        b = bins[(bins.species == sp)].dropna(subset=["mean_pred"])
        b = b[b.n > 0]
        x = b["mean_pred"].values
        y = b["obs_rate"].values
        n = b["n"].values.astype(float)
        w = 6 + 22 * (n / n.max())

        # deviation shading between curve and diagonal
        xs = np.r_[0, x, 1]
        ys = np.r_[0, y, 1]
        ax.fill_between(xs, xs, ys, color=PALETTE["red_shade"], alpha=0.55,
                        lw=0, zorder=1)
        ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=PALETTE["grey"],
                lw=0.8, zorder=2)
        ax.plot(x, y, color=colors[sp], lw=1.0, zorder=3, alpha=0.85)
        ax.scatter(x, y, s=w, color=colors[sp], edgecolors="white",
                   linewidths=0.5, zorder=4)

        e = ece_rows[ece_rows.species == sp].iloc[0]
        ax.text(0.03, 0.97,
                f"ECE {e['mean_pred']:.3f}\nBrier {e['obs_rate']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=6,
                color=PALETTE["grey"])
        tot_n = int(n.sum())
        n_pos = int(round(tot_n * (b["obs_rate"] * b.n).sum() / n.sum()))
        ax.text(0.97, 0.05, f"n = {tot_n:,} genes",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=6,
                color=PALETTE["grey"])
        ax.set_title(f"{name}  (GT: {gt})", loc="left", style="italic",
                     fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_aspect("equal", adjustable="box")
        ax.text(-0.16, 1.04, panel_labels[k], transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

    for ax in axes.flat:
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction detected")

    fig.tight_layout(w_pad=2.2, h_pad=2.0)
    save_pub(fig, "supp_fig_calibration")


# ═══════════════════════════════════════════════════════════════════════════
# Fig S2 — strain-slot sensitivity sweep
# ═══════════════════════════════════════════════════════════════════════════

def fig_strainid():
    df = pd.read_csv(RES / "a7_strainid_sensitivity.csv")

    specs = [
        ("Aspergillus_cristatus", "A. cristatus", "Candida checkpoint"),
        ("Trichophyton_rubrum", "T. rubrum", "Filamentous checkpoint"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 68 * MM))
    panel_labels = ["A", "B"]
    colors = [PALETTE["blue"], PALETTE["teal"]]

    for k, ((sp, name, ckpt), ax) in enumerate(zip(specs, axes)):
        g = df[df.species == sp].sort_values("strain_id")
        sid = g["strain_id"].values
        auc = g["auroc"].values
        n_eval = int(g["n"].iloc[0])

        ax.axhline(auc[0], ls=(0, (4, 3)), color=PALETTE["grey_light"],
                   lw=0.8, zorder=1)
        ax.plot(sid, auc, color=colors[k], lw=0.9, alpha=0.8, zorder=2)
        ax.scatter(sid, auc, s=12, color=colors[k], edgecolors="white",
                   linewidths=0.4, zorder=3)

        best = int(sid[np.argmax(auc)])
        # highlight default slot 0 (red diamond) and best slot (gold ring)
        ax.scatter([0], [auc[0]], s=46, marker="D", color=PALETTE["red"],
                   edgecolors="white", linewidths=0.6, zorder=4)
        ax.annotate("default (slot 0)", xy=(0, auc[0]),
                    xytext=(0.9, auc[0] - 0.075), fontsize=6,
                    color=PALETTE["red"],
                    arrowprops=dict(arrowstyle="-", lw=0.6,
                                    color=PALETTE["red"]))
        if best != 0:
            ax.scatter([best], [auc.max()], s=52, facecolors="none",
                       edgecolors=PALETTE["gold"], linewidths=1.4, zorder=4)
            ax.annotate(f"best (slot {best})", xy=(best, auc.max()),
                        xytext=(best - 1.2, auc.max() + 0.018), fontsize=6,
                        ha="right", color="#9a6b1f")

        rng = auc.max() - auc.min()
        ax.text(0.97, 0.05,
                f"range {rng:.3f}\ndefault − best = {auc.max()-auc[0]:.3f}"
                f"\nn = {n_eval:,} genes",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=6,
                color=PALETTE["grey"])
        ax.set_title(f"{name}  ({ckpt})", loc="left", style="italic",
                     fontsize=7)
        ax.set_xlabel("Assigned strain-embedding slot")
        ax.set_ylabel("AUROC (balanced eval set)")
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xlim(-0.8, 20.8)
        pad = 0.03
        ax.set_ylim(auc.min() - pad, auc.max() + pad)
        ax.text(-0.14, 1.05, panel_labels[k], transform=ax.transAxes,
                fontsize=8, fontweight="bold", va="top", ha="left")

    fig.tight_layout(w_pad=2.4)
    save_pub(fig, "supp_fig_strainid_sweep")


if __name__ == "__main__":
    fig_calibration()
    fig_strainid()
