"""
Species embedding for PanCirc-Fungi.

Combines:
- Phylogenetic PCA embedding (from ultrametric tree distance matrix)
- Learned species embedding table
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SpeciesEmbedding(nn.Module):
    """Species-aware embedding combining phylogeny and learned vectors.

    Parameters
    ----------
    n_species : int
        Number of active species/strains.
    phylo_dim : int
        Dimension of PCA-reduced phylogenetic features.
    embed_dim : int
        Output dimension of combined embedding.
    """

    def __init__(self, n_species: int = 21,
                 phylo_dim: int = 8,
                 embed_dim: int = 32):
        super().__init__()
        self.n_species = n_species
        self.phylo_dim = phylo_dim
        self.embed_dim = embed_dim

        # Learned species embedding table
        self.species_embed = nn.Embedding(n_species, embed_dim)

        # Phylogeny projection (if phylo features provided)
        self.phylo_proj = nn.Linear(phylo_dim, embed_dim)

        # Combine
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Precomputed phylogeny features (set externally)
        # Shape: (n_species, phylo_dim)
        self.register_buffer("phylo_features", torch.zeros(n_species, phylo_dim))

    def load_phylo_from_npy(self, path: str):
        """Load precomputed phylogenetic PCA features.

        Expects a .npy file of shape (n_species, phylo_dim).
        """
        import numpy as np
        data = np.load(path)
        assert data.shape == (self.n_species, self.phylo_dim), \
            f"Expected ({self.n_species}, {self.phylo_dim}), got {data.shape}"
        self.phylo_features = torch.from_numpy(data).float()
        logger.info(f"Loaded phylogenetic features from {path}")

    def forward(self, species_ids: torch.Tensor) -> torch.Tensor:
        """Compute species embeddings.

        Args:
            species_ids: (batch,) long tensor of species indices
        Returns:
            (batch, embed_dim) species embedding vectors
        """
        # Learned embedding
        learned = self.species_embed(species_ids)  # (batch, embed_dim)

        # Phylogenetic projection
        phylo_vecs = self.phylo_features[species_ids]  # (batch, phylo_dim)
        phylo_proj = self.phylo_proj(phylo_vecs)  # (batch, embed_dim)

        # Combine
        combined = torch.cat([learned, phylo_proj], dim=-1)
        return self.combine(combined)
