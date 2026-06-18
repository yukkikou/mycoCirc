"""
Expression encoder for PanCirc-Fungi (fine-tuning only).

Encodes CircExp (BSJ counts) and GeneExp (gene counts) as
additional input features during fine-tuning.

Only instantiated during fine-tuning — not used in pre-training.
Note: CircExp is available during training but NOT during inference
(you don't know the circRNA expression before predicting it).
"""

import torch
import torch.nn as nn

from model.layers import MLPBlock, CrossAttentionFusion


class ExpressionEncoder(nn.Module):
    """Encodes CircExp + GeneExp for fine-tuning.

    Parameters
    ----------
    max_replicates : int
        Maximum number of expression replicates to support.
    hidden_dim : int
        Hidden dimension of MLP.
    output_dim : int
        Output embedding dimension.
    fusion_dim : int
        Dimension for cross-attention fusion (from backbone).
    dropout : float
    """

    def __init__(self, max_replicates: int = 3,
                 hidden_dim: int = 32,
                 output_dim: int = 64,
                 fusion_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()

        # Per-replicate encoding
        self.circ_mlp = MLPBlock(
            in_dim=max_replicates,
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            n_layers=2,
            dropout=dropout,
        )
        self.gene_mlp = MLPBlock(
            in_dim=max_replicates,
            hidden_dim=hidden_dim,
            out_dim=output_dim,
            n_layers=2,
            dropout=dropout,
        )

        # Cross-attention between circRNA expression and gene expression
        self.cross_attn = CrossAttentionFusion(
            d_model=output_dim, n_heads=2, dropout=dropout
        )

        # Project to fusion_dim for backbone compatibility
        self.output_proj = nn.Linear(output_dim, fusion_dim)

    def forward(self, circ_exp: torch.Tensor,
                gene_exp: torch.Tensor) -> torch.Tensor:
        """Encode expression data.

        During inference, pass circ_exp=zeros when CircExp is unavailable.

        Args:
            circ_exp: (batch, n_replicates) circRNA BSJ counts (log1p + z-score)
            gene_exp: (batch, n_replicates) gene expression counts (log1p + z-score)
        Returns:
            (batch, fusion_dim) expression embedding (to be fused with other modalities)
        """
        # Encode each modality
        circ_vec = self.circ_mlp(circ_exp)      # (batch, output_dim)
        gene_vec = self.gene_mlp(gene_exp)      # (batch, output_dim)

        # Cross-attention: circ query vs gene key/value
        fused = self.cross_attn(
            circ_vec.unsqueeze(1),
            gene_vec.unsqueeze(1),
            kv_mask=torch.ones(circ_exp.size(0), 1, device=circ_exp.device,
                               dtype=torch.bool),
        )  # (batch, output_dim)

        return self.output_proj(fused)  # (batch, fusion_dim)
