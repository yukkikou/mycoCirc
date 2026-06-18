"""
Genomic context encoder for PanCirc-Fungi.

Encodes the broad genomic environment around a gene using
dilated convolutional layers followed by a BiGRU to model
sequential properties along the gene body.

Architecture:
  Profile (200 bins × 8 ch) → Conv1d (dilated, per-bin features)
  → BiGRU (bin-level sequence modeling)
  → attention pooling → gene context vector

Input:  200-bin × 8-channel profile (GC%, mono-nt%, CpG, entropy)
Output: (batch, output_dim) context vector
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenomicContextEncoder(nn.Module):
    """Dilated CNN + BiGRU over binned genome context profile.

    Parameters
    ----------
    in_channels : int
        Number of input profile features per bin (default: 8).
    filters : int
        Number of convolutional filters per layer.
    kernel_size : int
        Kernel size for convolutions.
    dilations : List[int]
        Dilation rates for each layer.
    gru_hidden : int
        BiGRU hidden dimension (default: 64).
    gru_layers : int
        Number of GRU layers (default: 1).
    output_dim : int
        Output embedding dimension (default: 128).
    dropout : float
    """

    def __init__(self, in_channels: int = 8, filters: int = 64,
                 kernel_size: int = 7, dilations=None,
                 gru_hidden: int = 64, gru_layers: int = 1,
                 output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]

        self.input_proj = nn.Conv1d(in_channels, filters, kernel_size=1)

        # Dilated CNN: extract local per-bin features
        conv_layers = []
        for d in dilations:
            conv_layers.append(
                nn.Conv1d(filters, filters, kernel_size,
                          dilation=d, padding="same")
            )
            conv_layers.append(nn.BatchNorm1d(filters))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
        self.conv_blocks = nn.Sequential(*conv_layers)

        # BiGRU: model sequential order of bins along the gene body
        self.gru = nn.GRU(
            filters, gru_hidden, num_layers=gru_layers,
            bidirectional=True, dropout=dropout if gru_layers > 1 else 0,
            batch_first=True,
        )
        gru_out = gru_hidden * 2  # bidirectional

        # Attention pooling over bin positions
        self.attn_query = nn.Parameter(torch.randn(1, 1, 16) * 0.02)
        self.attn_key = nn.Linear(gru_out, 16, bias=False)

        # Output projection
        self.output = nn.Linear(gru_out, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode genome context.

        Args:
            x: (batch, n_bins, n_channels) genome context profile
        Returns:
            (batch, output_dim) context embedding
        """
        # (batch, n_channels, n_bins) for Conv1d
        x = x.transpose(1, 2)
        x = self.input_proj(x)       # (batch, filters, n_bins)
        x = self.conv_blocks(x)      # (batch, filters, n_bins)

        # (batch, n_bins, filters) for GRU
        x = x.transpose(1, 2)
        x, _ = self.gru(x)           # (batch, n_bins, gru_out)

        # Attention pooling over bins
        keys = self.attn_key(x)      # (batch, n_bins, 16)
        attn = torch.matmul(self.attn_query, keys.transpose(-2, -1))
        attn = attn / math.sqrt(16)
        attn = F.softmax(attn, dim=-1)  # (batch, 1, n_bins)
        pooled = torch.matmul(attn, x).squeeze(1)  # (batch, gru_out)

        x = self.output(pooled)      # (batch, output_dim)
        x = self.norm(x)
        return x
