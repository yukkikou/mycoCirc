"""
Junction attention heatmap visualization.

Shows which k-mer positions in the donor/acceptor flanking
sequences contribute most to the backsplice prediction.

Extracts k-mer attention weights from JunctionEncoder's
JEDI pathway (kmer_embed → jedi_gru → jedi_attention).
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_junction_attention(
    model,
    donor_kmers: np.ndarray,
    acceptor_kmers: np.ndarray,
    device: str = "cpu",
):
    """Extract k-mer attention weights for donor and acceptor sequences.

    Parameters
    ----------
    model : PanCircModel
    donor_kmers : (batch, N_d, L) array of k-mer token IDs (one donor site)
    acceptor_kmers : (batch, N_a, L) array of k-mer token IDs
    device : str

    Returns
    -------
    donor_attn, acceptor_attn : list of (L,) attention weight arrays
    """
    model.eval()
    je = model.junction_encoder

    d_in = torch.tensor(donor_kmers, device=device, dtype=torch.long)
    a_in = torch.tensor(acceptor_kmers, device=device, dtype=torch.long)

    # Handle batched input: process each donor/acceptor site individually
    batch, N_d, L_d = d_in.shape
    _, N_a, L_a = a_in.shape

    # Flatten sites into batch dimension for JEDI pathway
    d_flat = d_in.view(batch * N_d, L_d)  # (batch*N_d, L)
    a_flat = a_in.view(batch * N_a, L_a)  # (batch*N_a, L)

    # Embed
    d_emb = je.kmer_embed(d_flat)   # (batch*N_d, L, embed_dim)
    a_emb = je.kmer_embed(a_flat)   # (batch*N_a, L, embed_dim)

    # GRU
    d_gru, _ = je.jedi_gru(d_emb)   # (batch*N_d, L, gru_out)
    a_gru, _ = je.jedi_gru(a_emb)   # (batch*N_a, L, gru_out)

    # Get attention weights per site
    d_attn_list = []
    a_attn_list = []

    for i in range(batch * N_d):
        # Single site: (1, L, gru_out)
        seq = d_gru[i:i + 1]
        keys = je.jedi_attention.key(seq)  # (1, L, attn_dim)
        attn = torch.matmul(je.jedi_attention.query, keys.transpose(-2, -1))
        attn = torch.softmax(attn / np.sqrt(je.jedi_attention.query.size(-1)), dim=-1)
        attn = torch.nan_to_num(attn, 0.0)
        d_attn_list.append(attn.squeeze().cpu().numpy())

    for i in range(batch * N_a):
        seq = a_gru[i:i + 1]
        keys = je.jedi_attention.key(seq)
        attn = torch.matmul(je.jedi_attention.query, keys.transpose(-2, -1))
        attn = torch.softmax(attn / np.sqrt(je.jedi_attention.query.size(-1)), dim=-1)
        attn = torch.nan_to_num(attn, 0.0)
        a_attn_list.append(attn.squeeze().cpu().numpy())

    return d_attn_list, a_attn_list


def plot_junction_heatmap(
    donor_attn: np.ndarray,
    acceptor_attn: np.ndarray,
    donor_seq: str = "",
    acceptor_seq: str = "",
    save_path: str = "junction_heatmap.png",
    title: str = "Junction k-mer Attention",
):
    """Plot attention heatmap over donor and acceptor flanking sequences."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))

    for ax, attn, seq, label in zip(
        axes, [donor_attn, acceptor_attn],
        [donor_seq, acceptor_seq],
        ["Donor (5'SS)", "Acceptor (3'SS)"]
    ):
        ax.bar(range(len(attn)), attn, width=1.0, color="#4C72B0",
               edgecolor="none", alpha=0.8)
        ax.set_ylabel("Attention")
        ax.set_title(label)
        ax.set_xlim(0, len(attn))
        if seq:
            # Only label every 10th position to avoid crowding
            step = max(1, len(seq) // 20)
            ticks = range(0, len(seq), step)
            ax.set_xticks(ticks)
            ax.set_xticklabels(
                [seq[i: i + 3] for i in ticks],
                fontsize=6, rotation=45,
            )

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Junction heatmap saved to {save_path}")
    plt.close()
