"""
Shared layer components for PanCirc-Fungi.

Provides:
- KmerEmbedding: Embedding lookup + learned positional encoding
- PositionalEncoding: Learned positional encodings
- KmerAttention: Single-head k-mer attention (JEDI-style)
- CrossAttentionFusion: Multi-head cross-attention block
- UncertaintyWeightedLoss: Learned loss weights for multi-task learning
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Learnable positional encoding — dynamically expands if needed."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input, expanding dynamically if needed.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        if seq_len <= self.max_len:
            return x + self.encoding[:, :seq_len, :]
        else:
            # Dynamically extend encoding via interpolation
            enc = self.encoding.transpose(1, 2)  # (1, d_model, max_len)
            enc = F.interpolate(enc, size=seq_len, mode="linear", align_corners=False)
            enc = enc.transpose(1, 2)  # (1, seq_len, d_model)
            return x + enc


class KmerEmbedding(nn.Module):
    """JEDI-style k-mer embedding with learned positional encoding.

    Parameters
    ----------
    vocab_size : int
        Number of unique k-mers (4^k).
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length (in k-mer positions).
    padding_idx : int, optional
    """

    def __init__(self, vocab_size: int, d_model: int,
                 max_len: int = 500, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model,
                                      padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed token IDs and add positional encoding.

        Args:
            x: (batch, seq_len) integer token IDs
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.norm(x)
        return self.dropout(x)


class KmerAttention(nn.Module):
    """JEDI-style k-mer attention: single-head attention over k-mer positions."""

    def __init__(self, d_model: int, attention_dim: int = 16):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, attention_dim) * 0.02)
        self.key = nn.Linear(d_model, attention_dim, bias=False)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        keys = self.key(x)
        attn = torch.matmul(self.query, keys.transpose(-2, -1))
        attn = attn / math.sqrt(self.query.size(-1))
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        pooled = torch.matmul(attn, x)
        return pooled.squeeze(1)


class CrossAttentionFusion(nn.Module):
    """Multi-head cross-attention fusion block."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor,
                kv: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        attn_out, _ = self.cross_attn(
            query, kv, kv,
            key_padding_mask=~kv_mask if kv_mask is not None else None
        )
        x = self.norm1(query + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.squeeze(1)


class UncertaintyWeightedLoss(nn.Module):
    """Learned loss weights for multi-task learning."""

    def __init__(self, n_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        precision = torch.exp(-self.log_vars)
        total = (precision * losses + 0.5 * self.log_vars).sum()
        return total


class MLPBlock(nn.Module):
    """Simple MLP with dropout and LayerNorm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, dropout: float = 0.1, use_norm: bool = True):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2 or use_norm:
                if use_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
