"""
Multi-modal fusion module for PanCirc-Fungi.

Fuses all modality embeddings via concatenation + MLP.

This avoids the junction-as-query cross-attention bottleneck,
which was problematic when the junction encoder outputs are
near-constant (especially for single-exon genes, which make
up >80% of fungal genes).

Modalities fused:
- Junction embedding (64-dim, projected to fusion_dim)
- Genomic context embedding (128-dim, projected to fusion_dim)
- GTF embedding (128-dim, projected to fusion_dim)
- Species embedding (32-dim, projected to fusion_dim)
- [optional] Expression embedding (128-dim, projected to fusion_dim)
"""

from typing import Optional

import torch
import torch.nn as nn

from model.layers import MLPBlock


class FusionModule(nn.Module):
    """Concat-MLP multi-modal fusion.

    All modality embeddings are projected to fusion_dim independently,
    concatenated, then refined through an MLP to produce the fused
    gene representation.

    Parameters
    ----------
    fusion_dim : int
        Dimension of the fused representation (default: 128).
    dropout : float
    """

    MODALITY_NAMES = ["junction", "genome_context", "gtf", "species", "expression"]

    def __init__(self, fusion_dim: int = 128,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fusion_dim = fusion_dim

        # Projections from each modality to fusion_dim
        self.projections = nn.ModuleDict({
            "junction": nn.Linear(64, fusion_dim),
            "genome_context": nn.Linear(128, fusion_dim),
            "gtf": nn.Linear(128, fusion_dim),
            "species": nn.Linear(32, fusion_dim),
            "expression": nn.Linear(128, fusion_dim),
        })

        # Post-fusion MLP (concatenated -> fused)
        # n_modalities: 4 base + 1 optional expression = 4 or 5
        self.refine = MLPBlock(
            in_dim=fusion_dim * 5,  # always 5 (expression padded if absent)
            hidden_dim=fusion_dim * 2,
            out_dim=fusion_dim,
            n_layers=2,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        junction_vec: torch.Tensor,
        genome_context_vec: torch.Tensor,
        gtf_vec: torch.Tensor,
        species_vec: torch.Tensor,
        expression_vec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse multi-modal information.

        All modalities are projected to fusion_dim, concatenated,
        and passed through an MLP to produce the fused representation.

        Args:
            junction_vec: (batch, 64) from JunctionEncoder
            genome_context_vec: (batch, 128) from GenomicContextEncoder
            gtf_vec: (batch, 128) from GTFEncoder
            species_vec: (batch, 32) from SpeciesEmbedding
            expression_vec: optional (batch, 128) from ExpressionEncoder
        Returns:
            (batch, fusion_dim) fused gene representation
        """
        # Project all to fusion_dim
        junction = self.projections["junction"](junction_vec)
        genome = self.projections["genome_context"](genome_context_vec)
        gtf = self.projections["gtf"](gtf_vec)
        species = self.projections["species"](species_vec)

        concat_list = [junction, genome, gtf, species]
        if expression_vec is not None:
            concat_list.append(self.projections["expression"](expression_vec))
        else:
            # Zero-pad for expression when not available
            concat_list.append(torch.zeros_like(junction))

        # Concatenate all modalities
        combined = torch.cat(concat_list, dim=-1)  # (batch, 5 * fusion_dim)

        # Refine through MLP
        fused = self.refine(combined)
        return self.dropout(fused)
