"""
GTF feature encoder for PanCirc-Fungi.

Encodes gene structure features (exon count, intron length, etc.)
into a fixed-dimensional embedding.
"""

import torch
import torch.nn as nn

from model.layers import MLPBlock


class GTFEncoder(nn.Module):
    """MLP encoder for gene annotation features.

    Parameters
    ----------
    n_features : int
        Number of input GTF features (default: 17).
    hidden_dim : int
        Hidden dimension of MLP.
    output_dim : int
        Output embedding dimension.
    n_biotypes : int
        Number of gene biotype classes (for embedding).
    dropout : float
    """

    def __init__(self, n_features: int = 17, hidden_dim: int = 64,
                 output_dim: int = 128, n_biotypes: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        # Biotype embedding (biotype is the last feature)
        self.biotype_embed = nn.Embedding(n_biotypes, 8)

        # The first 8 features are numeric, biotype is separate
        # total input: 8 numeric + 8 biotype_embed = 16 (since one feature is biotype idx)
        self.numeric_dim = n_features - 1  # exclude biotype (last feature)

        self.net = MLPBlock(
            in_dim=self.numeric_dim + 8,  # numeric + biotype_embed
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            n_layers=2,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode GTF features.

        Args:
            x: (batch, n_features) GTF feature vectors
        Returns:
            (batch, output_dim) gene embedding
        """
        # Split numeric and biotype
        numeric = x[:, :self.numeric_dim]  # (batch, 8)
        biotype_idx = x[:, self.numeric_dim].long()  # (batch,)
        biotype_emb = self.biotype_embed(biotype_idx)  # (batch, 8)

        combined = torch.cat([numeric, biotype_emb], dim=-1)
        return self.net(combined)
