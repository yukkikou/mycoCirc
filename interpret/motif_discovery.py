"""
Sequence motif discovery from JunctionEncoder attention weights.

Clusters high-attention k-mers to identify sequence patterns
that drive backsplice junction recognition.

Output: sequence logo plots for top motifs.
"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def extract_high_attention_kmers(
    attention_weights: np.ndarray,
    kmer_tokens: np.ndarray,
    k: int = 3,
    top_pct: float = 10.0,
) -> List[str]:
    """Extract k-mer strings with highest attention weights.

    Parameters
    ----------
    attention_weights : (L,) array of attention scores
    kmer_tokens : (L,) array of k-mer integer IDs
    k : int
        k-mer size.
    top_pct : float
        Top percentage of positions to keep (default 10%).

    Returns
    -------
    List[str] of high-attention k-mer strings.
    """
    threshold = np.percentile(attention_weights, 100 - top_pct)
    high_attn_idx = np.where(attention_weights >= threshold)[0]

    # Decode k-mer IDs
    bases = "ACGT"
    kmers = []
    for idx in high_attn_idx:
        tid = kmer_tokens[idx]
        kmer = ""
        for _ in range(k):
            kmer = bases[tid % 4] + kmer
            tid //= 4
        kmers.append(kmer)

    return kmers


def find_enriched_motifs(
    positive_kmers: List[str],
    background_kmers: List[str],
    n_top: int = 5,
) -> Dict[str, float]:
    """Find k-mers enriched in positive vs background.

    Returns dict of motif -> log2 enrichment ratio.
    """
    pos_counts = Counter(positive_kmers)
    bg_counts = Counter(background_kmers)

    total_pos = len(positive_kmers)
    total_bg = len(background_kmers)

    enrichment = {}
    for kmer, count in pos_counts.items():
        pos_freq = count / max(total_pos, 1)
        bg_freq = bg_counts.get(kmer, 1) / max(total_bg, 1)
        ratio = pos_freq / max(bg_freq, 0.001)
        enrichment[kmer] = np.log2(ratio)

    # Sort by enrichment
    sorted_motifs = sorted(
        enrichment.items(), key=lambda x: -x[1]
    )[:n_top]

    return dict(sorted_motifs)


def plot_sequence_logo(
    motifs: Dict[str, float],
    save_path: str = "sequence_motifs.png",
):
    """Create a simple frequency-based sequence logo.

    If logomaker is available, uses it for proper logo plots.
    Otherwise creates a bar chart of top motifs.
    """
    try:
        import logomaker
        has_logomaker = True
    except ImportError:
        has_logomaker = False

    if has_logomaker:
        _plot_logo_logomaker(motifs, save_path)
    else:
        _plot_motif_bars(motifs, save_path)


def _plot_logo_logomaker(motifs: Dict[str, float], save_path: str):
    """Plot sequence logo using logomaker."""
    import logomaker as lm

    # Create position frequency matrix from motifs
    # Pad shorter motifs to max length
    max_len = max(len(m) for m in motifs)
    n_motifs = min(len(motifs), 10)

    fig, axes = plt.subplots(n_motifs, 1, figsize=(4, 2 * n_motifs))

    if n_motifs == 1:
        axes = [axes]

    for ax, (motif, enrichment) in zip(axes, list(motifs.items())[:n_motifs]):
        # Build PWM
        pwm = np.zeros((len(motif), 4))
        for i, base in enumerate(motif.upper()):
            idx = "ACGT".find(base)
            if idx >= 0:
                pwm[i, idx] = 1.0
        pwm_df = pd.DataFrame(pwm, columns=["A", "C", "G", "T"])

        lm.Logo(pwm_df, ax=ax)
        ax.set_ylabel(f"enr={enrichment:.1f}")
        ax.set_xticks([])

    plt.suptitle("Enriched Sequence Motifs near Backsplice Junctions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Sequence motif logo saved to {save_path}")
    plt.close()


def _plot_motif_bars(motifs: Dict[str, float], save_path: str):
    """Fallback: bar chart of top motifs."""
    names = list(motifs.keys())
    scores = list(motifs.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0" if s > 0 else "#C44E52" for s in scores]
    ax.bar(range(len(names)), scores, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylabel("log2 Enrichment (positive vs background)")
    ax.set_title("Enriched Sequence Motifs near Backsplice Junctions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Motif bar chart saved to {save_path}")
    plt.close()
