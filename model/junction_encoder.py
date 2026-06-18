"""
Junction encoder for PanCirc-Fungi — JEDI + CircPCBL hybrid.

For each gene, processes ALL potential donor and acceptor splice sites
collectively through three parallel encoding pathways:

  JEDI pathway:
    k-mer tokenization → Embedding → BiGRU → k-mer attention → site vectors

  CircPCBL Pathway 1 (one-hot CNN-BiGRU):
    raw sequence → one-hot encoding → multiscale CNN (kernels 3,5,7)
    → BiGRU → attention → site vectors

  CircPCBL Pathway 2 (k-mer frequency GLT):
    k=1..4 frequency features (340-dim) → MLP → site vectors

All three fused → Cross-attention (donor↔acceptor pairwise)
→ Final attention pooling (which sites matter most)
→ Junction vector

Reference:
  JEDI:    Jiang et al. Bioinformatics 2021 (cross-attention)
  CircPCBL: Wu et al. Plants 2023 (one-hot CNN-BiGRU + k-mer GLT)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import KmerEmbedding


# ═══════════════════════════════════════════════════════════════════════════
# Shared attention components (JEDI)
# ═══════════════════════════════════════════════════════════════════════════


class KmerAttention(nn.Module):
    """Single-head dot-product attention over k-mer positions within one site."""

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
        attn = torch.nan_to_num(attn, 0.0)
        pooled = torch.matmul(attn, x)
        return pooled.squeeze(1)


class CrossAttention(nn.Module):
    """Bidirectional cross-attention between two sets of vectors (JEDI core).

    Returns (attended_vector, softmax_weights, raw_logits).
    """

    def __init__(self, d_model: int, attn_dim: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.k_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.scale = math.sqrt(attn_dim)

    def forward(self, Q: torch.Tensor, K: torch.Tensor,
                Q_mask: Optional[torch.Tensor] = None,
                K_mask: Optional[torch.Tensor] = None):
        q = self.q_proj(Q)
        k = self.k_proj(K)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if K_mask is not None:
            attn = attn.masked_fill(~K_mask.unsqueeze(1), float("-inf"))
        if Q_mask is not None:
            attn = attn.masked_fill(~Q_mask.unsqueeze(-1), float("-inf"))
        attn_weights = F.softmax(attn, dim=-1)
        # Safety: replace NaN (all-masked inputs) with zeros
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        attn = torch.nan_to_num(attn, 0.0)
        Q_attended = torch.matmul(attn_weights, K)
        return Q_attended, attn_weights, attn  # raw attn = logits for loss


class FinalAttentionPooling(nn.Module):
    """JEDI-style final attention: learned query selects important sites."""

    def __init__(self, d_model: int, attn_dim: int = 16):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, attn_dim) * 0.02)
        self.key = nn.Linear(d_model, attn_dim, bias=False)
        self.scale = math.sqrt(attn_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = self.key(x)
        attn = torch.matmul(self.query, keys.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn_w = F.softmax(attn, dim=-1)
        # Safety: replace NaN (all-masked inputs) with zeros
        attn_w = torch.nan_to_num(attn_w, 0.0)
        pooled = torch.matmul(attn_w, x).squeeze(1)
        return pooled, attn_w.squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# CircPCBL-style detectors
# ═══════════════════════════════════════════════════════════════════════════


class OneHotCNNBiGRU(nn.Module):
    """CircPCBL Detector 1: one-hot → multiscale CNN → BiGRU → attention."""

    def __init__(self, in_channels: int = 4,
                 cnn_filters: int = 32,
                 kernel_sizes=None,
                 gru_hidden: int = 32,
                 gru_layers: int = 1,
                 attention_dim: int = 16,
                 output_dim: int = 64):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, cnn_filters, ks, padding=ks // 2),
                    nn.BatchNorm1d(cnn_filters),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
            )
        cnn_out = cnn_filters * len(kernel_sizes)

        self.gru = nn.GRU(
            cnn_out, gru_hidden, num_layers=gru_layers,
            bidirectional=True, dropout=0,
            batch_first=True,
        )
        gru_out = gru_hidden * 2

        self.attention = KmerAttention(gru_out, attention_dim)
        self.output = nn.Linear(gru_out, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, N, C, L = x.shape
        x_flat = x.view(batch * N, C, L)

        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x_flat))
        cnn_out = torch.cat(conv_outs, dim=1)

        cnn_out = cnn_out.transpose(1, 2)
        gru_out, _ = self.gru(cnn_out)

        pooled = self.attention(gru_out)
        projected = self.norm(self.output(pooled))
        return projected.view(batch, N, -1)


class KmerFreqEncoder(nn.Module):
    """CircPCBL Detector 2: k-mer frequency features → GLT."""

    def __init__(self, input_dim: int = 340,
                 hidden_dim: int = 64,
                 output_dim: int = 64,
                 n_groups: int = 4):
        super().__init__()
        self.n_groups = n_groups
        group_size = input_dim // n_groups
        self.group_projections = nn.ModuleList([
            nn.Linear(group_size, output_dim // n_groups)
            for _ in range(n_groups)
        ])
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, N, D = x.shape
        x_flat = x.view(batch * N, D)
        group_size = D // self.n_groups
        outs = []
        for i, proj in enumerate(self.group_projections):
            start = i * group_size
            end = start + group_size
            outs.append(proj(x_flat[:, start:end]))
        out = torch.cat(outs, dim=-1)
        out = self.norm(out)
        return out.view(batch, N, -1)


# ═══════════════════════════════════════════════════════════════════════════
# Main JunctionEncoder
# ═══════════════════════════════════════════════════════════════════════════


class JunctionEncoder(nn.Module):
    """Multi-site junction encoder with JEDI cross-attention + CircPCBL dual encoding."""

    def __init__(self, k: int = 3, embed_dim: int = 64,
                 gru_hidden: int = 64, gru_layers: int = 2,
                 gru_dropout: float = 0.1,
                 cnn_filters: int = 32,
                 kernel_sizes=None,
                 cnn_gru_hidden: int = 32,
                 kmer_freq_dim: int = 340,
                 freq_hidden: int = 64,
                 attention_dim: int = 16,
                 cross_attn_dim: int = 16,
                 output_dim: int = 64,
                 max_len: int = 500):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.k = k
        self.gru_out = gru_hidden * 2

        # ── Pathway 1: JEDI k-mer embedding + BiGRU
        vocab_size = 4 ** k
        self.kmer_embed = KmerEmbedding(vocab_size, embed_dim, max_len=max_len)
        self.jedi_gru = nn.GRU(
            embed_dim, gru_hidden, num_layers=gru_layers,
            bidirectional=True, dropout=gru_dropout,
            batch_first=True,
        )
        self.jedi_attention = KmerAttention(self.gru_out, attention_dim)

        # ── Pathway 2: CircPCBL one-hot CNN-BiGRU
        self.onehot_encoder = OneHotCNNBiGRU(
            in_channels=4, cnn_filters=cnn_filters,
            kernel_sizes=kernel_sizes, gru_hidden=cnn_gru_hidden,
            attention_dim=attention_dim,
            output_dim=output_dim,
        )

        # ── Pathway 3: CircPCBL k-mer frequency GLT
        self.freq_encoder = KmerFreqEncoder(
            input_dim=kmer_freq_dim, hidden_dim=freq_hidden,
            output_dim=output_dim,
        )

        # ── Fusion
        fusion_dim = self.gru_out + output_dim + output_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, self.gru_out),
            nn.LayerNorm(self.gru_out),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # ── Cross-attention (JEDI core)
        self.cross_attn = CrossAttention(self.gru_out, cross_attn_dim)

        # ── Final attention pooling
        self.final_donor_attn = FinalAttentionPooling(self.gru_out, cross_attn_dim)
        self.final_acceptor_attn = FinalAttentionPooling(self.gru_out, cross_attn_dim)

        # ── Output
        self.output = nn.Linear(self.gru_out * 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        donor_kmers: torch.Tensor,
        acceptor_kmers: torch.Tensor,
        donor_onehot: Optional[torch.Tensor] = None,
        acceptor_onehot: Optional[torch.Tensor] = None,
        donor_kmer_freq: Optional[torch.Tensor] = None,
        acceptor_kmer_freq: Optional[torch.Tensor] = None,
        donor_mask: Optional[torch.Tensor] = None,
        acceptor_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch, N_d, Ld = donor_kmers.shape
        _, N_a, La = acceptor_kmers.shape
        device = donor_kmers.device

        def _jedi_encode(tokens, N):
            flat = tokens.view(batch * N, -1)
            emb = self.kmer_embed(flat)
            gru, _ = self.jedi_gru(emb)
            vecs = self.jedi_attention(gru)
            return vecs.view(batch, N, -1)

        d_jedi = _jedi_encode(donor_kmers, N_d)
        a_jedi = _jedi_encode(acceptor_kmers, N_a)

        # ── CircPCBL Path 1: one-hot CNN-BiGRU
        d_cnn = self.onehot_encoder(donor_onehot, donor_mask) if donor_onehot is not None \
                else torch.zeros(batch, N_d, self.onehot_encoder.output.out_features, device=device)
        a_cnn = self.onehot_encoder(acceptor_onehot, acceptor_mask) if acceptor_onehot is not None \
                else torch.zeros(batch, N_a, self.onehot_encoder.output.out_features, device=device)

        # ── CircPCBL Path 2: k-mer frequency GLT
        d_freq = self.freq_encoder(donor_kmer_freq, donor_mask) if donor_kmer_freq is not None \
                 else torch.zeros(batch, N_d, self.freq_encoder.norm.normalized_shape[0], device=device)
        a_freq = self.freq_encoder(acceptor_kmer_freq, acceptor_mask) if acceptor_kmer_freq is not None \
                 else torch.zeros(batch, N_a, self.freq_encoder.norm.normalized_shape[0], device=device)

        # ── Fuse all three pathways
        d_fused = self.fusion_proj(torch.cat([d_jedi, d_cnn, d_freq], dim=-1))
        a_fused = self.fusion_proj(torch.cat([a_jedi, a_cnn, a_freq], dim=-1))

        # ── Bidirectional cross-attention (JEDI core)
        d_attended_a, cross_weights, d2a_logits = self.cross_attn(
            d_fused, a_fused,
            Q_mask=donor_mask, K_mask=acceptor_mask,
        )
        a_attended_d, _, a2d_logits = self.cross_attn(
            a_fused, d_fused,
            Q_mask=acceptor_mask, K_mask=donor_mask,
        )

        # ── Residual fusion
        d_final = d_fused + d_attended_a
        a_final = a_fused + a_attended_d

        # ── Final attention (which sites matter most)
        d_pooled, donor_attn = self.final_donor_attn(d_final, donor_mask)
        a_pooled, acceptor_attn = self.final_acceptor_attn(a_final, acceptor_mask)

        # ── Output
        combined = torch.cat([d_pooled, a_pooled], dim=-1)
        junction_vec = self.norm(self.output(combined))

        return {
            "junction_vec": junction_vec,
            "donor_attn": donor_attn,
            "acceptor_attn": acceptor_attn,
            "cross_weights": cross_weights,
            "junction_logits": d2a_logits,  # raw logits for loss
            "donor_vecs": d_final,
            "acceptor_vecs": a_final,
        }
