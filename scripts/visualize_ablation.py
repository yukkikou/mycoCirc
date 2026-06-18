#!/usr/bin/env python3
"""
Visualize expression ablation results.

Reads ``checkpoints/results/expression_ablation.tsv`` and generates:
1. Grouped bar: AUROC per ablation mode, grouped by fungal group
2. Grouped bar: AUPRC per ablation mode, grouped by fungal group
3. Drop chart: AUROC drop from "original" for each ablation, per group
4. Heatmap: full metrics matrix for each group

Usage:
    python scripts/visualize_ablation.py \\
        [--input checkpoints/results/expression_ablation.tsv] \\
        [--output-dir checkpoints/results/plots]
"""

import argparse
import os
import sys

import numpy as np

# ── Lazy imports ───────────────────────────────────────────────────────────

HAS_MPL = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    HAS_MPL = False

HAS_PD = True
try:
    import pandas as pd
except ImportError:
    HAS_PD = False

HAS_SNS = True
try:
    import seaborn as sns
except ImportError:
    HAS_SNS = False


# ── Config ─────────────────────────────────────────────────────────────────

MODE_LABELS = {
    "original": "Original\n(all inputs)",
    "shuffle_circexp": "Shuffle\nCircExp",
    "shuffle_geneexp": "Shuffle\nGeneExp",
    "zero_both": "Zero\n(both expr)",
    "no_expression": "No expression\n(Genome+GTF)",
}

MODE_COLORS = {
    "original": "#2ecc71",
    "shuffle_circexp": "#f39c12",
    "shuffle_geneexp": "#e67e22",
    "zero_both": "#e74c3c",
    "no_expression": "#95a5a6",
}

GROUP_COLORS = {
    "Candida": "#3498db",
    "Cryptococcus": "#9b59b6",
    "Filamentous": "#1abc9c",
}

METRIC_LABELS = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "f1": "F1 Score",
    "accuracy": "Accuracy",
    "mcc": "MCC",
}


def load_ablation(path: str):
    """Read ablation TSV, skipping terminal warnings and comment lines."""
    lines = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            # Skip warnings, comment lines, blank lines, and indented stacks
            if (not stripped
                    or stripped.startswith("/media")
                    or stripped.startswith("#")
                    or stripped.startswith("File ")
                    or stripped.startswith("warnings.")
                    or stripped.startswith("  ")  # continuation lines
                    or "Error" in stripped
                    or "Traceback" in stripped):
                continue
            lines.append(stripped)

    header = lines[0].split("\t")
    data = [line.split("\t") for line in lines[1:]]
    # Filter out any malformed rows that don't match header width
    data = [row for row in data if len(row) == len(header)]
    df = pd.DataFrame(data, columns=header)
    for col in ["auroc", "auprc", "f1", "accuracy", "mcc"]:
        df[col] = df[col].astype(float)
    return df


def plot_grouped_bars(df, metric, title, ylabel, output_path):
    """Grouped bar: metric per mode, groups side-by-side."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    groups = df["group"].unique()
    modes = [m for m in MODE_LABELS if m in df["mode"].unique()]
    n_modes = len(modes)
    n_groups = len(groups)
    width = 0.8 / n_modes

    x = np.arange(n_groups)
    for i, mode in enumerate(modes):
        vals = [df.loc[(df["group"] == g) & (df["mode"] == mode), metric].values[0]
                for g in groups]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            vals,
            width,
            label=MODE_LABELS.get(mode, mode),
            color=MODE_COLORS.get(mode, "#7f8c8d"),
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left", ncol=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  → {output_path}")


def plot_drop_chart(df, metric, title, ylabel, output_path):
    """Drop from original mode → ablation mode per group."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    groups = df["group"].unique()
    modes = [m for m in MODE_LABELS if m != "original" and m in df["mode"].unique()]

    x = np.arange(len(modes))
    n_groups = len(groups)
    width = 0.7 / n_groups

    for i, group in enumerate(groups):
        orig = df.loc[(df["group"] == group) & (df["mode"] == "original"), metric].values[0]
        drops = []
        for mode in modes:
            val = df.loc[(df["group"] == group) & (df["mode"] == mode), metric].values[0]
            drops.append(orig - val)
        bars = ax.bar(
            x + i * width - 0.35 + width / 2,
            drops,
            width,
            label=group,
            color=GROUP_COLORS.get(group, "#7f8c8d"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, d in zip(bars, drops):
            label = f"{d:.3f}"
            y_pos = bar.get_height() + 0.005 if d >= 0 else bar.get_height() - 0.025
            va = "bottom" if d >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos, label,
                ha="center", va=va, fontsize=7, rotation=90,
            )

    ax.set_xticks(x)
    abbrev = {
        "shuffle_circexp": "Shuffle\nCircExp",
        "shuffle_geneexp": "Shuffle\nGeneExp",
        "zero_both": "Zero\nboth",
        "no_expression": "No\nexpr",
    }
    ax.set_xticklabels([abbrev.get(m, m) for m in modes], fontsize=10)
    ax.set_ylabel(f"{ylabel} drop ↓", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  → {output_path}")


def plot_metric_heatmap(df, output_dir, metric="auroc"):
    """Heatmap: group × mode for a single metric, with annotations."""
    modes_order = [m for m in MODE_LABELS if m in df["mode"].unique()]
    groups_order = ["Candida", "Cryptococcus", "Filamentous"]

    pivot = df.pivot_table(
        index="group", columns="mode", values=metric, aggfunc="first"
    )
    pivot = pivot.reindex(index=groups_order, columns=modes_order)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            color = "white" if val < 0.6 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(range(len(modes_order)))
    ax.set_xticklabels([m.replace("_", "\n") for m in modes_order], fontsize=9)
    ax.set_yticks(range(len(groups_order)))
    ax.set_yticklabels(groups_order, fontsize=11)
    ax.set_title(f"Ablation — {METRIC_LABELS.get(metric, metric)}", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = os.path.join(output_dir, f"ablation_heatmap_{metric}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  → {out}")


def plot_all_metrics_heatmap(df, output_path):
    """One big heatmap: group × mode × all metrics concatenated."""
    modes = [m for m in MODE_LABELS if m in df["mode"].unique()]
    groups = ["Candida", "Cryptococcus", "Filamentous"]
    metrics = ["auroc", "auprc", "f1", "accuracy", "mcc"]

    # Build matrix: rows = groups, columns = mode_metric combinations
    rows = []
    row_labels = []
    col_labels = []
    for mode in modes:
        for metric in metrics:
            col_labels.append(f"{mode[:4]}\n{metric[:4]}")
    for group in groups:
        row = []
        for mode in modes:
            for metric in metrics:
                val = df.loc[(df["group"] == group) & (df["mode"] == mode), metric].values
                row.append(val[0] if len(val) > 0 else 0.0)
        rows.append(row)
        row_labels.append(group)

    arr = np.array(rows)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_title("Expression Ablation — All Metrics", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ablation results")
    parser.add_argument(
        "--input", default="checkpoints/results/expression_ablation.tsv",
        help="Ablation results TSV",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/results/plots",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    if not HAS_PD:
        print("ERROR: pandas required. Install with: pip install pandas")
        sys.exit(1)
    if not HAS_MPL:
        print("ERROR: matplotlib required. Install with: pip install matplotlib")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ablation data from {args.input} ...")
    df = load_ablation(args.input)
    print(f"  {len(df)} rows, groups: {df['group'].unique().tolist()}")
    print(f"  modes: {df['mode'].unique().tolist()}")

    # ── 1. Grouped bars: AUROC ──────────────────────────────────────────
    print("\n[1/5] Grouped bar — AUROC")
    plot_grouped_bars(
        df, "auroc",
        "Expression Ablation — AUROC by Group",
        "AUROC",
        os.path.join(args.output_dir, "ablation_auroc_bars.png"),
    )

    # ── 2. Grouped bars: AUPRC ──────────────────────────────────────────
    print("[2/5] Grouped bar — AUPRC")
    plot_grouped_bars(
        df, "auprc",
        "Expression Ablation — AUPRC by Group",
        "AUPRC",
        os.path.join(args.output_dir, "ablation_auprc_bars.png"),
    )

    # ── 3. Drop chart: AUROC ────────────────────────────────────────────
    print("[3/5] Drop chart — AUROC")
    plot_drop_chart(
        df, "auroc",
        "AUROC Drop from Original Expression",
        "AUROC",
        os.path.join(args.output_dir, "ablation_auroc_drop.png"),
    )

    # ── 4. Heatmap: AUROC ──────────────────────────────────────────────
    print("[4/5] Heatmap — AUROC")
    plot_metric_heatmap(df, args.output_dir, metric="auroc")

    # ── 5. Full metrics heatmap ─────────────────────────────────────────
    print("[5/5] Full metrics heatmap")
    plot_all_metrics_heatmap(
        df,
        os.path.join(args.output_dir, "ablation_all_metrics.png"),
    )

    print(f"\n✓ All plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
