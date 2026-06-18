#!/usr/bin/env python3
"""
Visualize PanCirc-Fungi experiment results.

Reads ``checkpoints/results/experiments.csv`` and generates 5 diagnostic plots:

  1. param_vs_auroc.png — Each hyperparameter vs AUROC, grouped by group
  2. param_heatmap.png  — flank_size × k heatmap (AUROC) per group
  3. ablation_bar.png   — ablation experiment comparisons
  4. cv_boxplot.png     — cross-validation AUROC distribution per group
  5. flank_tuning.png   — flank_size vs AUROC line plot per group

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --csv path/to/experiments.csv --output plots/
    python scripts/visualize_results.py --group Candida  # filter to one group
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# All heavy imports (pandas, matplotlib, seaborn) are loaded inside the
# functions that need them. This keeps --help working without dependencies.


# ─── Configuration ────────────────────────────────────────────────────────────
DEFAULT_CSV = "checkpoints/results/experiments.csv"
DEFAULT_OUTPUT = "checkpoints/results/plots"
PALETTE = {  # group colors
    "Candida": "#E74C3C",
    "Cryptococcus": "#3498DB",
    "Filamentous": "#2ECC71",
}


def _check_deps():
    """Verify plotting dependencies."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("ERROR: matplotlib is required for visualization.")
        print("Install: pip install matplotlib seaborn")
        sys.exit(1)


def _setup_style():
    """Set consistent matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150,
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass


def _load_csv(csv_path: str):
    """Load experiments CSV."""
    import pandas as pd

    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        print("Run auto_train.py first to generate results.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"ERROR: CSV is empty: {csv_path}")
        sys.exit(1)
    print(f"Loaded {len(df)} experiment records from {csv_path}")
    return df


def _filter_ok(df):
    """Filter to completed experiments only."""
    ok = df[df["status"] == "ok"].copy()
    dropped = len(df) - len(ok)
    if dropped:
        print(f"  (dropped {dropped} non-ok entries)")
    return ok


def _savefig(path: str):
    """Save figure to file."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  Plot 1 — param_vs_auroc
# ═════════════════════════════════════════════════════════════════════════════

def plot_param_vs_auroc(df, output_dir: str):
    """For each hyperparameter, scatter plot of param value vs AUROC.

    Each point is one experiment, colored by group.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    params = [
        ("flank_size", "Flank size (bp)"),
        ("k", "k-mer size"),
        ("embed_dim", "Embedding dimension"),
        ("gru_hidden", "GRU hidden size"),
    ]

    n_params = len(params)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (col, xlabel) in zip(axes, params):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        for group in df["group"].unique():
            sub = df[df["group"] == group]
            if sub.empty:
                continue
            x = sub[col].values.astype(float)
            y = sub["auroc"].values
            # Add tiny horizontal jitter for discrete params
            jitter = np.random.RandomState(hash(group) % 2**32).uniform(
                -0.15, 0.15, size=len(x)
            ) if sub[col].nunique() <= 5 else np.zeros(len(x))
            color = PALETTE.get(group, "#888888")
            ax.scatter(x + jitter, y, alpha=0.6, s=40, c=color,
                       label=group, edgecolors="none")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("AUROC")
        ax.set_title(f"AUROC vs {xlabel}")
        ax.set_ylim(max(0.4, df["auroc"].min() - 0.05),
                     min(1.0, df["auroc"].max() + 0.05))
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hyperparameter vs AUROC", fontsize=16, y=1.02)
    fig.tight_layout()
    _savefig(os.path.join(output_dir, "param_vs_auroc.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  Plot 2 — param_heatmap
# ═════════════════════════════════════════════════════════════════════════════

def plot_param_heatmap(df, output_dir: str):
    """2D heatmap: flank_size x k, color = mean AUROC.

    Creates one panel per group.
    """
    import matplotlib.pyplot as plt

    groups = df["group"].unique()
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    try:
        import seaborn as sns
    except ImportError:
        print("  WARNING: seaborn not installed — using imshow for heatmap")
        sns = None

    for ax, group in zip(axes, groups):
        sub = df[df["group"] == group]
        if sub.empty:
            ax.set_visible(False)
            continue

        pivot = sub.pivot_table(
            index="k", columns="flank_size", values="auroc",
            aggfunc="mean"
        )
        if pivot.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        if sns is not None:
            sns.heatmap(
                pivot, annot=True, fmt=".3f", cmap="viridis",
                ax=ax, cbar_kws={"label": "Mean AUROC"},
                vmin=max(0.5, pivot.min().min()),
                vmax=max(pivot.max().max(), 0.7),
            )
        else:
            # Fallback with imshow
            import numpy as np
            data = pivot.values.astype(float)
            im = ax.imshow(data, cmap="viridis", aspect="auto",
                           vmin=max(0.5, np.nanmin(data)),
                           vmax=max(np.nanmax(data), 0.7))
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            plt.colorbar(im, ax=ax, label="Mean AUROC")

        ax.set_title(f"{group}")
        ax.set_xlabel("Flank size (bp)")
        ax.set_ylabel("k-mer")

    fig.suptitle("AUROC: flank_size x k (mean across embed_dim/gru_hidden)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(os.path.join(output_dir, "param_heatmap.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  Plot 3 — ablation_bar
# ═════════════════════════════════════════════════════════════════════════════

def plot_ablation_bar(df, output_dir: str):
    """Bar chart showing AUROC for each configuration."""
    import matplotlib.pyplot as plt
    import numpy as np

    ok = _filter_ok(df)
    if len(ok) < 2:
        print("  Skipping ablation_bar: need >=2 experiments")
        return

    # Use the best per (group, flank_size, k)
    best = ok.loc[
        ok.groupby(["group", "flank_size", "k"])["auroc"].idxmax()
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    groups = best["group"].unique()
    width = 0.25

    for i, group in enumerate(groups):
        sub = best[best["group"] == group].sort_values("auroc", ascending=False)
        if sub.empty:
            continue
        xs = np.arange(len(sub)) + i * width
        ax.bar(xs, sub["auroc"], width=width, color=PALETTE.get(group, "#888"),
               alpha=0.8, label=group)
        # Annotate with config
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.text(
                j + i * width, row["auroc"] + 0.005,
                f"fs={int(row['flank_size'])}\nk={int(row['k'])}",
                ha="center", va="bottom", fontsize=6, rotation=90,
            )

    ax.set_ylabel("AUROC")
    ax.set_title("Experiment AUROC comparison (best per config)")
    ax.set_xticks([])
    ax.legend(loc="lower right")
    ax.set_ylim(max(0.4, best["auroc"].min() - 0.05), 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _savefig(os.path.join(output_dir, "ablation_bar.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  Plot 4 — cv_boxplot
# ═════════════════════════════════════════════════════════════════════════════

def plot_cv_boxplot(df, output_dir: str):
    """Boxplot of AUROC across groups."""
    import matplotlib.pyplot as plt
    import numpy as np

    ok = _filter_ok(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: AUROC distribution per group
    ax = axes[0]
    groups = ok["group"].unique()
    data = [ok[ok["group"] == g]["auroc"].dropna().values for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True,
                    showfliers=True, showmeans=True)
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(PALETTE.get(group, "#888"))
        patch.set_alpha(0.6)
    # Overlay strip plot
    for i, (g, d) in enumerate(zip(groups, data)):
        if len(d) > 0:
            jitter = np.random.RandomState(42).uniform(-0.15, 0.15, size=len(d))
            ax.scatter(np.full_like(d, i + 1) + jitter, d,
                       alpha=0.5, s=20, color=PALETTE.get(g, "#888"),
                       edgecolors="none")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC distribution per group")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: CV stats (if available)
    ax = axes[1]
    if "cv_auroc_mean" in ok.columns and "cv_auroc_std" in ok.columns:
        cv_data = ok[ok["cv_auroc_mean"].notna()]
        if not cv_data.empty:
            groups_cv = cv_data["group"].unique()
            for g in groups_cv:
                sub = cv_data[cv_data["group"] == g]
                if sub.empty:
                    continue
                x = np.full(len(sub), float(groups_cv.tolist().index(g)) + 1)
                jitter = np.random.RandomState(42).uniform(-0.15, 0.15, size=len(sub))
                ax.errorbar(
                    x + jitter, sub["cv_auroc_mean"],
                    yerr=sub["cv_auroc_std"],
                    fmt="o", alpha=0.7, capsize=5,
                    color=PALETTE.get(g, "#888"),
                    label=g,
                )
            ax.set_xticks(np.arange(1, len(groups_cv) + 1))
            ax.set_xticklabels(groups_cv)
            ax.set_ylabel("CV AUROC (mean +/- std)")
            ax.set_title("Cross-validation results (per group)")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "No CV data available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=14, color="#888")
            ax.set_title("Cross-validation results")
    else:
        ax.text(0.5, 0.5, "CV columns not found in CSV",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="#888")
        ax.set_title("Cross-validation results (no CV data)")

    fig.tight_layout()
    _savefig(os.path.join(output_dir, "cv_boxplot.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  Plot 5 — flank_tuning
# ═════════════════════════════════════════════════════════════════════════════

def plot_flank_tuning(df, output_dir: str):
    """flank_size vs AUROC, one line per group, with error bars."""
    import matplotlib.pyplot as plt
    import numpy as np

    ok = _filter_ok(df)
    if "flank_size" not in ok.columns or "auroc" not in ok.columns:
        print("  Skipping flank_tuning: missing required columns")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    groups = ok["group"].unique()
    flank_order = sorted(ok["flank_size"].unique())

    for group in groups:
        sub = ok[ok["group"] == group]
        if sub.empty:
            continue

        agg = sub.groupby("flank_size")["auroc"].agg(["mean", "std", "count"])
        agg = agg.reindex(flank_order)

        x = agg.index.values
        y = agg["mean"].values
        yerr = agg["std"].values
        if yerr is not None:
            yerr = np.nan_to_num(yerr)
        valid = ~np.isnan(y) & (agg["count"] > 0)

        color = PALETTE.get(group, "#888")
        ax.errorbar(
            x[valid], y[valid], yerr=yerr[valid] if yerr is not None else None,
            fmt="-o", capsize=5, capthick=1.5, linewidth=2,
            color=color, label=group, markersize=8,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=1,
        )
        # Annotate best flank_size
        if valid.any():
            best_idx = np.argmax(y[valid])
            best_x = x[valid][best_idx]
            best_y = y[valid][best_idx]
            ax.annotate(
                f"best: {int(best_x)} bp",
                xy=(best_x, best_y),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center", fontsize=9,
                color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    ax.set_xlabel("Flank size (bp)")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Flank size tuning - AUROC vs flank length")
    ax.set_xticks(flank_order)
    ax.set_xticklabels([str(x) for x in flank_order])
    ax.set_ylim(max(0.4, ok["auroc"].min() - 0.05),
                 min(1.0, ok["auroc"].max() + 0.05))
    ax.legend(loc="lower right", title="Group")
    ax.grid(True, alpha=0.3)

    # Reference line for JEDI default
    ax.axvline(x=150, color="#888", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(150, ax.get_ylim()[1] - 0.02, "JEDI default (150bp)",
            ha="center", fontsize=8, color="#888", alpha=0.7)

    fig.tight_layout()
    _savefig(os.path.join(output_dir, "flank_tuning.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  Summary table
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(df):
    """Print a text summary of results."""
    ok = _filter_ok(df)
    if ok.empty:
        return

    print("\n" + "=" * 70)
    print("Experiment Summary")
    print("=" * 70)

    # Best per group
    print(f"\n{'Best AUROC per group':^70}")
    print("-" * 70)
    print(f"{'Group':<15s} {'AUROC':>7s} {'AUPRC':>7s} {'F1':>7s} "
          f"{'flank':>6s} {'k':>3s} {'embed':>6s} {'gru':>5s}")
    print("-" * 70)
    for group in ["Candida", "Cryptococcus", "Filamentous"]:
        sub = ok[ok["group"] == group]
        if sub.empty:
            continue
        best_idx = sub["auroc"].idxmax() if "auroc" in sub.columns else None
        if best_idx is not None:
            row = sub.loc[best_idx]
            print(f"{group:<15s} {row.get('auroc', 0):>7.4f} "
                  f"{row.get('auprc', 0):>7.4f} "
                  f"{row.get('f1', 0):>7.4f} "
                  f"{int(row.get('flank_size', 0)):>6d} "
                  f"{int(row.get('k', 0)):>3d} "
                  f"{int(row.get('embed_dim', 0)):>6d} "
                  f"{int(row.get('gru_hidden', 0)):>5d}")

    # Aggregate statistics
    print(f"\n{'Aggregate statistics':^70}")
    print("-" * 70)
    for metric in ["auroc", "auprc", "f1", "accuracy"]:
        if metric not in ok.columns:
            continue
        vals = ok[metric].dropna()
        if not vals.empty:
            print(f"  {metric:20s}: mean={vals.mean():.4f} "
                  f"+/-{vals.std():.4f}  "
                  f"[{vals.min():.4f} - {vals.max():.4f}]")

    # Best overall configuration
    if "auroc" in ok.columns and not ok["auroc"].dropna().empty:
        best_all = ok.loc[ok["auroc"].idxmax()]
        print(f"\n  {'* Best overall':20s}: AUROC={best_all['auroc']:.4f}")
        print(f"    Config: flank={int(best_all['flank_size'])}bp, "
              f"k={int(best_all['k'])}, "
              f"embed_dim={int(best_all['embed_dim'])}, "
              f"gru_hidden={int(best_all['gru_hidden'])}, "
              f"group={best_all['group']}")

    print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize PanCirc-Fungi experiment results"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="Path to experiments.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output directory for plots")
    parser.add_argument("--group", default=None,
                        help="Filter to one group "
                             "(Candida/Cryptococcus/Filamentous)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots (save only)")
    args = parser.parse_args()

    _check_deps()
    _setup_style()

    df = _load_csv(args.csv)

    if args.group:
        df = df[df["group"] == args.group]
        print(f"  Filtered to group: {args.group} ({len(df)} records)")

    # Print summary
    print_summary_table(df)

    # Generate plots
    print(f"\nGenerating plots -> {args.output}")
    os.makedirs(args.output, exist_ok=True)

    plot_param_vs_auroc(df, args.output)
    plot_param_heatmap(df, args.output)
    plot_ablation_bar(df, args.output)
    plot_cv_boxplot(df, args.output)
    plot_flank_tuning(df, args.output)

    print(f"\nAll plots saved to {os.path.abspath(args.output)}/")

    # Report missing data
    ok = _filter_ok(df)
    if len(ok) < 5:
        print(f"\nNOTE: Only {len(ok)} completed experiments - "
              f"plots will be sparse. Run more experiments with auto_train.py "
              f"for richer visualizations.")


if __name__ == "__main__":
    main()
