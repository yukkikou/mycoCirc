"""
mycoCirc: Publication-quality integrated comparison figure
===============================================================
Combines:
  (A) Model comparison — PanCirc vs JEDI vs CircPCBL (AUROC, held-out test)
  (B) Expression ablation — CircExp/GeneExp contribution (AUROC, CV)
  (C) Mode A vs Mode B dot-line (AUROC, held-out test)
  (D) Ablation AUROC drop barh
  (E) F1 comparison across methods
  (F) Numerical summary table

Usage:
    python scripts/visualize_comparison.py --output figures/fig_comparison.pdf
"""

import argparse
import io
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

PROJ_ROOT = Path(__file__).resolve().parent.parent
GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
TEST_STRAIN_LABELS = {"Candida": "C. albicans (P4)", "Cryptococcus": "C. neoformans (C4)",
                      "Filamentous": "F. venenatum (F6)"}

# ─── Color palette (Tol — colorblind-friendly) ───────────────────────────────
C_GRP = {"Candida": "#D55E00", "Cryptococcus": "#0072B2", "Filamentous": "#009E73"}

# Methods — consistent across ALL panels
C_METHODS = ["PanCirc (Genome+GTF)", "PanCirc (+GeneExp)", "JEDI", "CircPCBL"]
C_COLORS = ["#0072B2", "#56B4E9", "#E69F00", "#CC79A7"]
C_HATCHES = [None, None, "////", "...."]

# Ablation modes
C_ABL = ["original", "shuffle_circexp", "shuffle_geneexp", "zero_both", "no_expression"]
C_ABL_COLORS = ["#009E73", "#89CFF0", "#F0C989", "#D55E00", "#56B4E9"]
C_ABL_LABELS = ["Full (CircExp+GeneExp)", "Shuffle CircExp", "Shuffle GeneExp",
                "Zero both", "No expression"]


def load_pancirc_data():
    import torch
    data = {}
    for grp in GROUPS:
        ckpt_path = PROJ_ROOT / "checkpoints" / "finetune" / grp / "final.pt"
        if not ckpt_path.exists():
            print(f"  WARN: {ckpt_path} not found")
            continue
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        base = ckpt.get("metrics_baseline", {})
        gen  = ckpt.get("metrics_genexp", {})
        data[grp] = {
            "mode_a": {k: base.get(k, float("nan")) for k in ["auroc","auprc","f1","accuracy","mcc"]},
            "mode_b": {k: gen.get(k, float("nan"))  for k in ["auroc","auprc","f1","accuracy","mcc"]},
        }
    return data


def load_jedi_data():
    data = {}
    for grp in GROUPS:
        p = PROJ_ROOT / "results" / "jedi_comparison" / grp / "summary.json"
        if not p.exists():
            continue
        with open(p) as f:
            m = json.load(f).get("jedi_metrics", {})
        data[grp] = {k: m.get(k, float("nan")) for k in ["auroc","auprc","f1","test_accuracy"]}
    return data


def load_circpcbl_data():
    data = {}
    for grp in GROUPS:
        p = PROJ_ROOT / "results" / "circpcbl_comparison" / grp / "summary.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        o = d.get("results", {}).get("overall", {})
        data[grp] = {"auroc": o.get("auroc_approx", float("nan")),
                     "auprc": o.get("auprc_approx", float("nan")),
                     "f1":    o.get("f1", float("nan"))}
    return data


def load_ablation_data():
    path = PROJ_ROOT / "checkpoints" / "results" / "expression_ablation.tsv"
    if not path.exists():
        print(f"  WARN: ablation TSV not found at {path}")
        return None
    raw = path.read_text()
    idx = raw.find("group\t")
    if idx < 0:
        return None
    raw = raw[idx:]
    clean = "\n".join(l for l in raw.split("\n") if not l.startswith("#") and l.strip())
    return pd.read_csv(io.StringIO(clean), sep="\t")


def get_val(source, grp, key, subkey=None):
    """Safe getter for nested metric dicts."""
    if source is None:
        return float("nan")
    d = source.get(grp, {})
    if subkey:
        d = d.get(subkey, {})
    v = d.get(key, float("nan"))
    return v if not math.isnan(v) else float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure(pancirc, jedi, circpcbl, ablation, output_path):
    n = len(GROUPS)
    x = np.arange(n)
    # Bar width: requested 0.35 for 4 methods = 0.35/4 = 0.0875 each
    # But that's too narrow. Let's use 0.15 per bar with 4 bars spanning ~0.6
    bw = 0.15
    off = [-1.5*bw, -0.5*bw, 0.5*bw, 1.5*bw]

    fig = plt.figure(figsize=(7.0, 8.8))
    # 5 rows: [A: single-wide] [B: single-wide] [C+D+E: three-col] [spacer] [F: single-wide]
    gs = fig.add_gridspec(5, 3, hspace=0.30, wspace=0.30,
                          height_ratios=[1.0, 1.0, 1.0, 0.03, 0.65],
                          left=0.09, right=0.97, top=0.94, bottom=0.06)

    # ══════════════════════════════════════════════════════════════════════
    # A: Model comparison — AUROC (full width)
    # ══════════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_title("A  Cross-species circRNA prediction on held-out test strains",
                  loc="left", fontweight="bold", fontsize=10)

    for idx in range(4):
        vals = []
        for g in GROUPS:
            if idx == 0:
                v = get_val(pancirc, g, "auroc", "mode_a")
            elif idx == 1:
                v = get_val(pancirc, g, "auroc", "mode_b")
            elif idx == 2:
                v = get_val(jedi, g, "auroc")
            else:
                v = get_val(circpcbl, g, "auroc")
            vals.append(v)
        bars = ax1.bar(x + off[idx], vals, bw, label=C_METHODS[idx],
                       color=C_COLORS[idx], edgecolor="white",
                       linewidth=0.3, alpha=0.90, zorder=3)
        if C_HATCHES[idx]:
            for b in bars:
                b.set_hatch(C_HATCHES[idx])
                b.set_edgecolor("white")

    ax1.axhline(y=0.5, color="grey", ls="--", lw=0.6, alpha=0.5, zorder=1)
    ax1.text(n + off[-1] + bw, 0.5, "Random", fontsize=6.5, color="grey",
             va="bottom", ha="left")
    ax1.set_xticks(x)
    ax1.set_xticklabels([TEST_STRAIN_LABELS[g] for g in GROUPS], fontstyle="italic")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0, 1.03)
    ax1.legend(loc="upper right", framealpha=0.88, edgecolor="#cccccc",
               fontsize=6.5, ncol=4, columnspacing=0.8)
    ax1.grid(axis="y", alpha=0.2, lw=0.4)
    ax1.set_axisbelow(True)

    # AUROC values on Mode A bars
    for gi, g in enumerate(GROUPS):
        v = get_val(pancirc, g, "auroc", "mode_a")
        if not math.isnan(v):
            ax1.text(x[gi] + off[0], v + 0.018, f"{v:.3f}",
                     fontsize=6.5, ha="center", va="bottom", color="#0072B2",
                     fontweight="bold")

    # ══════════════════════════════════════════════════════════════════════
    # B: Expression ablation (full width)
    # ══════════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title("B  Expression ablation — AUROC on within-group cross-validation",
                  loc="left", fontweight="bold", fontsize=10)

    if ablation is not None:
        n_abl = len(C_ABL)
        bw_abl = 0.85 / n_abl
        for ai, mode in enumerate(C_ABL):
            vals = []
            for grp in GROUPS:
                row = ablation[(ablation["group"] == grp) & (ablation["mode"] == mode)]
                vals.append(row["auroc"].values[0] if len(row) > 0 else float("nan"))
            off_abl = (ai - n_abl/2 + 0.5) * bw_abl
            bars = ax2.bar(x + off_abl, vals, bw_abl * 0.9,
                           label=C_ABL_LABELS[ai],
                           color=C_ABL_COLORS[ai], edgecolor="white",
                           linewidth=0.2, alpha=0.92, zorder=3)

        ax2.set_xticks(x)
        ax2.set_xticklabels(GROUPS, fontstyle="italic")
        ax2.set_ylabel("AUROC (CV)")
        ax2.set_ylim(0.5, 1.05)
        ax2.axhline(y=0.5, color="grey", ls="--", lw=0.6, alpha=0.5, zorder=1)

        # Legend BELOW x-axis labels (outside plot area)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                   framealpha=0.85, edgecolor="#cccccc", fontsize=6.5,
                   ncol=5, columnspacing=0.8)
        ax2.grid(axis="y", alpha=0.2, lw=0.4)
        ax2.set_axisbelow(True)
    else:
        ax2.text(0.5, 0.5, "Ablation data not available", ha="center", va="center",
                 transform=ax2.transAxes, fontstyle="italic", color="grey")

    # ══════════════════════════════════════════════════════════════════════
    # C: Mode A vs Mode B dot-line (AUROC)
    # ══════════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_title("C  Mode A vs B\n(Test AUROC)", loc="left", fontweight="bold", fontsize=9)

    for gi, grp in enumerate(GROUPS):
        ma = get_val(pancirc, grp, "auroc", "mode_a")
        mb = get_val(pancirc, grp, "auroc", "mode_b")
        c = C_GRP[grp]
        ax3.plot([0, 1], [ma, mb], "-o", color=c, lw=1.5, ms=7,
                 mfc=c, mec="white", mew=0.5, zorder=4, label=grp)
        # Labels offset: alternate sides to avoid overlap
        yoff_a = 0.022 if gi != 1 else -0.018
        yoff_b = -0.022 if gi != 1 else 0.018
        if not math.isnan(ma):
            ax3.text(0, ma + yoff_a, f"{ma:.3f}", fontsize=6,
                     ha="center", va="bottom" if yoff_a > 0 else "top",
                     color=c, fontweight="bold")
        if not math.isnan(mb):
            ax3.text(1, mb + yoff_b, f"{mb:.3f}", fontsize=6,
                     ha="center", va="bottom" if yoff_b > 0 else "top",
                     color=c, fontweight="bold")

    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["G+GTF", "+GE"], fontsize=7.5)
    ax3.set_ylabel("AUROC")
    ax3.set_ylim(0.30, 0.90)
    ax3.axhline(y=0.5, color="grey", ls="--", lw=0.6, alpha=0.5)
    ax3.legend(fontsize=6, framealpha=0.85, edgecolor="#cccccc",
               loc="lower left", handlelength=1)
    ax3.grid(axis="y", alpha=0.2, lw=0.4)
    ax3.set_axisbelow(True)

    # ══════════════════════════════════════════════════════════════════════
    # D: AUROC drop barh
    # ══════════════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_title("D  AUROC drop from\nfull model (CV)", loc="left", fontweight="bold", fontsize=9)

    if ablation is not None:
        drop_modes = ["zero_both", "no_expression"]
        drop_colors = [C_ABL_COLORS[3], C_ABL_COLORS[4]]
        drop_labels_short = ["Zero both", "No expression"]

        for gi, grp in enumerate(GROUPS):
            ref_row = ablation[(ablation["group"] == grp) & (ablation["mode"] == "original")]
            ref_v = ref_row["auroc"].values[0] if len(ref_row) > 0 else float("nan")
            for di, mode in enumerate(drop_modes):
                row = ablation[(ablation["group"] == grp) & (ablation["mode"] == mode)]
                v = row["auroc"].values[0] if len(row) > 0 else float("nan")
                drop = ref_v - v
                y_pos = gi * 2 + di
                ax4.barh(y_pos, drop, height=0.35, color=drop_colors[di],
                         edgecolor="white", lw=0.3, zorder=3,
                         label=drop_labels_short[di] if gi == 0 else "")
                if drop > 0.01:
                    ax4.text(drop + 0.005, y_pos, f"{drop:.3f}",
                             fontsize=6, va="center", ha="left", color=drop_colors[di],
                             fontweight="bold")

        ax4.set_yticks([gi * 2 + 0.5 for gi in range(n)])
        ax4.set_yticklabels(GROUPS, fontstyle="italic", fontsize=7.5)
        ax4.set_xlabel("AUROC drop")
        ax4.set_xlim(0, 0.45)
        ax4.legend(fontsize=6, framealpha=0.85, edgecolor="#cccccc",
                   loc="lower right")
        ax4.grid(axis="x", alpha=0.2, lw=0.4)
        ax4.set_axisbelow(True)
    else:
        ax4.text(0.5, 0.5, "N/A", ha="center", va="center",
                 transform=ax4.transAxes, fontstyle="italic", color="grey")

    # ══════════════════════════════════════════════════════════════════════
    # E: F1 comparison
    # ══════════════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.set_title("E  F1 score\n(held-out test)", loc="left", fontweight="bold", fontsize=9)

    f1_methods = [0, 2, 3]  # Mode A, JEDI, CircPCBL — same colors
    bw_f1 = 0.25
    off_f1 = [-bw_f1, 0, bw_f1]
    for mi, idx in enumerate(f1_methods):
        vals = []
        for g in GROUPS:
            if idx == 0:
                v = get_val(pancirc, g, "f1", "mode_a")
            elif idx == 2:
                v = get_val(jedi, g, "f1")
            else:
                v = get_val(circpcbl, g, "f1")
            vals.append(v)
        bars = ax5.bar(x + off_f1[mi], vals, bw_f1 * 0.85, label=C_METHODS[idx],
                       color=C_COLORS[idx], edgecolor="white", lw=0.3,
                       alpha=0.90, zorder=3)
        if C_HATCHES[idx]:
            for b in bars:
                b.set_hatch(C_HATCHES[idx])

    ax5.set_xticks(x)
    ax5.set_xticklabels(GROUPS, fontstyle="italic")
    ax5.set_ylabel("F1")
    ax5.set_ylim(0, 0.76)
    ax5.legend(fontsize=6, framealpha=0.85, edgecolor="#cccccc")
    ax5.grid(axis="y", alpha=0.2, lw=0.4)
    ax5.set_axisbelow(True)

    # ══════════════════════════════════════════════════════════════════════
    # F: Numerical summary table (full width, with spacer row above)
    # ══════════════════════════════════════════════════════════════════════
    ax6 = fig.add_subplot(gs[4, :])
    ax6.axis("off")
    ax6.set_title("F  Numerical summary of all metrics (held-out test)",
                  loc="left", fontweight="bold", fontsize=9, pad=4)

    col_labels = ["Group", "Method", "AUROC", "AUPRC", "F1", "Acc", "MCC"]
    table_data = []
    ncols = len(col_labels)

    for grp in GROUPS:
        c = C_GRP[grp]
        for idx in range(4):
            if idx == 0:
                src, key = pancirc, "mode_a"
            elif idx == 1:
                src, key = pancirc, "mode_b"
            elif idx == 2:
                src, key = jedi, None
            else:
                src, key = circpcbl, None

            if key:
                d = src.get(grp, {}).get(key, {})
            else:
                d = src.get(grp, {})
            vals = [f"{d.get(k, float('nan')):.3f}" for k in ["auroc","auprc","f1","accuracy","mcc"]]
            row = [grp if idx == 0 else "", C_METHODS[idx]] + vals
            table_data.append(row)

    table = ax6.table(cellText=table_data, colLabels=col_labels,
                      loc="center", cellLoc="center",
                      colWidths=[0.12, 0.18, 0.10, 0.10, 0.08, 0.08, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)

    # Style cells
    group_starts = [i * 4 for i in range(n)]
    for i, start in enumerate(group_starts):
        c = C_GRP[GROUPS[i]]
        for j in range(start, start + 4):
            for col in range(ncols):
                cell = table[j, col]
                cell.set_facecolor("#f8f8f8") if j % 2 == 0 else cell.set_facecolor("white")
                cell.set_edgecolor("#dddddd")
                cell.set_linewidth(0.3)
            # Style method column
            cell = table[j, 1]
            if j == start:
                cell.set_text_props(fontweight="bold", color=c, fontsize=6.5)

    # Header style
    for col in range(ncols):
        cell = table[0, col]
        cell.set_text_props(fontweight="bold", fontsize=7)
        cell.set_facecolor("#eeeeee")

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(str(output_path.parent), exist_ok=True)
    fig.savefig(str(output_path))
    fig.savefig(str(output_path.with_suffix(".png")), dpi=300)
    print(f"Figure saved to {output_path}")
    print(f"Preview: {output_path.with_suffix('.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default=str(PROJ_ROOT / "figures" / "fig_comparison.pdf"))
    args = parser.parse_args()
    out = Path(args.output)

    print("Loading data...")
    pancirc = load_pancirc_data()
    jedi = load_jedi_data()
    circpcbl = load_circpcbl_data()
    ablation = load_ablation_data()

    if not pancirc:
        print("ERROR: No mycoCirc data loaded.")
        sys.exit(1)

    print(f"  PanCirc: {list(pancirc.keys())}")
    print(f"  JEDI: {list(jedi.keys())}")
    print(f"  CircPCBL: {list(circpcbl.keys())}")
    print(f"  Ablation: {ablation.shape if ablation is not None else None}")

    make_figure(pancirc, jedi, circpcbl, ablation, out)


if __name__ == "__main__":
    main()
