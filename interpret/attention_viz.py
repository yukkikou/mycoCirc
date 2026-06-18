"""
Fusion attention → ablation-based modality importance + Junction cross-attention.

Since the FusionModule is Concat-MLP (no built-in attention weights),
this module provides two alternative analyses:

  1. Modality ablation bar chart (from results/ablations.tsv)
  2. JunctionEncoder cross-attention heatmaps for top-scoring positive genes
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

MODALITY_NAMES = ["GenomeCtx", "GTF", "Species", "Junction", "Expression"]
MODALITY_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#937860"]


def plot_modality_ablation(
    ablation_tsv: str,
    save_path: str = "figures/fig_modality_ablation.png",
    figsize=(7, 4),
):
    """Modality importance from ablation results — horizontal bar chart.

    Reads results/ablations.tsv and plots the AUROC drop when each
    modality (or component) is removed, per group.
    """
    df = pd.read_csv(ablation_tsv, sep="\t")

    conditions_of_interest = [
        ("no_gtf", "Remove GTF", "#DD8452"),
        ("no_genome", "Remove GenomeCtx", "#4C72B0"),
        ("no_species", "Remove Species", "#55A868"),
        ("no_junction", "Remove Junction", "#C44E52"),
        ("no_expression", "Remove Expression", "#937860"),
    ]

    groups = df["group"].unique()
    n_groups = len(groups)
    n_conds = len(conditions_of_interest)

    fig, axes = plt.subplots(1, n_groups, figsize=figsize, sharey=True)
    if n_groups == 1:
        axes = [axes]

    for gi, grp in enumerate(groups):
        ax = axes[gi]
        grp_df = df[df["group"] == grp]
        full_row = grp_df[grp_df["condition"] == "full"]
        full_auroc = full_row["auroc"].values[0] if len(full_row) > 0 else 0.0

        drops = []
        labels = []
        colors = []
        for cond_name, label, color in conditions_of_interest:
            row = grp_df[grp_df["condition"] == cond_name]
            if len(row) > 0:
                drop = full_auroc - row["auroc"].values[0]
            else:
                drop = 0.0
            drops.append(drop)
            labels.append(label)
            colors.append(color)

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, drops, color=colors, edgecolor="white",
                       height=0.6, zorder=3)
        # Add value labels
        for bar, v in zip(bars, drops):
            if v > 0.005:
                ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                        f"{v:.3f}", fontsize=7, va="center")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels if gi == 0 else [], fontsize=7)
        ax.set_title(grp, fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("AUROC drop" if gi == n_groups - 1 else "", fontsize=8)
        ax.grid(axis="x", alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)

    fig.suptitle("Modality Importance — AUROC Drop on Ablation",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Modality ablation plot saved to {save_path}")
    plt.close()


@torch.no_grad()
def plot_cross_attention_examples(
    model,
    loader,
    device,
    save_path: str = "figures/fig_cross_attention.png",
    n_examples: int = 5,
):
    """Visualize JunctionEncoder cross-attention for top-scoring positive genes.

    For the top-n_examples positive genes (highest prediction score),
    plots the donor×acceptor cross-attention weight heatmap with
    the true backsplice junction marked.
    """
    import math

    model.eval()
    examples = []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        outputs = model(batch, task="pretrain")
        probs = torch.sigmoid(outputs["gene_logits"].squeeze(-1))
        cross_w = outputs["cross_weights"]  # (batch, N_d, N_a)
        labels = batch.get("is_positive", torch.zeros_like(probs))
        cross_labels = batch.get("cross_labels")  # (batch, N_d, N_a)
        donor_mask = batch.get("donor_mask")      # (batch, N_d)
        acceptor_mask = batch.get("acceptor_mask") # (batch, N_a)

        for i in range(len(probs)):
            if labels[i].item() > 0:
                examples.append({
                    "prob": probs[i].item(),
                    "cross_w": cross_w[i].cpu().numpy(),
                    "cross_labels": cross_labels[i].cpu().numpy() if cross_labels is not None else None,
                    "donor_mask": donor_mask[i].cpu().numpy() if donor_mask is not None else None,
                    "acceptor_mask": acceptor_mask[i].cpu().numpy() if acceptor_mask is not None else None,
                })
        if len(examples) >= n_examples * 2:
            break

    # Sort by probability descending, take top-n
    examples.sort(key=lambda x: -x["prob"])
    examples = examples[:n_examples]

    if not examples:
        logger.warning("No positive examples found for cross-attention plot")
        return

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 2.5 * n),
                             gridspec_kw={"width_ratios": [3, 1]})
    if n == 1:
        axes = axes.reshape(1, 2)

    for idx, ex in enumerate(examples):
        ax_heat = axes[idx, 0]
        ax_bar = axes[idx, 1]
        cross_w = ex["cross_w"]
        cl = ex["cross_labels"]
        d_mask = ex["donor_mask"]
        a_mask = ex["acceptor_mask"]

        if d_mask is not None and a_mask is not None:
            # Mask invalid positions
            valid = d_mask[:, None] & a_mask[None, :]
            masked_w = np.where(valid, cross_w, np.nan)
        else:
            masked_w = cross_w

        # Heatmap
        im = ax_heat.imshow(masked_w, aspect="auto", cmap="YlOrRd",
                            interpolation="nearest", origin="upper")
        plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        ax_heat.set_xlabel("Acceptor site index")
        ax_heat.set_ylabel("Donor site index")
        ax_heat.set_title(f"Cross-attention (P(circ)={ex['prob']:.3f})",
                          fontsize=8)

        # Mark true junction with star
        if cl is not None:
            true_d, true_a = np.where(cl > 0.5)
            for td, ta in zip(true_d, true_a):
                ax_heat.plot(ta, td, "*", color="cyan", markersize=8,
                             markeredgecolor="white", markeredgewidth=0.5)

        # Donor + Acceptor marginal attention
        d_attn = np.nanmean(masked_w, axis=1) if a_mask is not None else np.nanmean(masked_w, axis=1)
        a_attn = np.nanmean(masked_w, axis=0) if d_mask is not None else np.nanmean(masked_w, axis=0)

        ax_bar.barh(range(len(d_attn)), d_attn if d_mask is None else np.where(d_mask, d_attn, 0),
                    color="#C44E52", alpha=0.7, height=0.6)
        ax_bar.set_title("Donor margin", fontsize=7)
        ax_bar.set_xlim(0, None)
        ax_bar.tick_params(labelsize=6)

    plt.suptitle("Junction Cross-Attention (★ = true backsplice)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Cross-attention plot saved to {save_path}")
    plt.close()
