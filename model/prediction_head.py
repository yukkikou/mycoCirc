"""
Prediction heads for PanCirc-Fungi.

Two heads sharing the same fused representation:

1. GeneHead: binary classification — does this gene produce circRNA?
2. JunctionHead: ranking — which backsplice junction is most likely?

Both heads are used identically in pre-training and fine-tuning
(no head replacement needed for fine-tuning).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneHead(nn.Module):
    """Binary classifier: predicts if a gene produces circRNA.

    Input: fused gene representation (fusion_dim,)
    Output: P(circRNA | gene)
    """

    def __init__(self, fusion_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict circRNA probability.

        Args:
            x: (batch, fusion_dim) fused gene representation
        Returns:
            (batch, 1) logits (before sigmoid)
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probability in [0, 1]."""
        return torch.sigmoid(self.forward(x))


class JunctionHead(nn.Module):
    """Ranks candidate backsplice junctions for a gene.

    For each candidate junction, computes a score based on:
    - The junction embedding (from JunctionEncoder)
    - The gene representation (from FusionModule)

    Uses cross-attention between junction candidates and gene representation.
    """

    def __init__(self, junction_dim: int = 64,
                 fusion_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.junction_proj = nn.Linear(junction_dim, fusion_dim)

        self.score_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, gene_repr: torch.Tensor,
                junction_vecs: torch.Tensor,
                candidate_mask: torch.Tensor) -> torch.Tensor:
        """Score candidate junctions for a batch of genes.

        Args:
            gene_repr: (batch, fusion_dim) fused gene representation
            junction_vecs: (batch, max_candidates, junction_dim)
                junction embeddings for each candidate
            candidate_mask: (batch, max_candidates) True = valid candidate
        Returns:
            (batch, max_candidates) logit scores (before softmax)
        """
        # Project junction vectors to fusion_dim
        j_proj = self.junction_proj(junction_vecs)  # (batch, n_cand, fusion_dim)

        # Expand gene representation for each candidate
        gene_expanded = gene_repr.unsqueeze(1).expand(-1, j_proj.size(1), -1)

        # Concatenate gene repr + junction repr for each candidate
        combined = torch.cat([gene_expanded, j_proj], dim=-1)

        # Score each candidate
        scores = self.score_net(combined).squeeze(-1)  # (batch, n_cand)

        # Mask invalid candidates
        scores = scores.masked_fill(~candidate_mask, float("-inf"))

        return scores

    def rank_junctions(self, gene_repr: torch.Tensor,
                       junction_vecs: torch.Tensor,
                       candidate_mask: torch.Tensor) -> torch.Tensor:
        """Return normalized probability distribution over candidates."""
        scores = self.forward(gene_repr, junction_vecs, candidate_mask)
        probs = F.softmax(scores, dim=-1)
        return probs
