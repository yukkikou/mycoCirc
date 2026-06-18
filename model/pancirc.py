"""
mycoCirc: Multi-modal circRNA foundation model.

Main model class assembling all components:
  SpeciesEmbedding → GenomicContextEncoder → GTFEncoder
  → JunctionEncoder → ExpressionEncoder (FT only)
  → FusionModule → GeneHead + JunctionHead

Usage
-----
>>> model = PanCircModel(config)
>>> out = model(batch, task="pretrain")
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.species_embedding import SpeciesEmbedding
from model.genome_encoder import GenomicContextEncoder
from model.gtf_encoder import GTFEncoder
from model.junction_encoder import JunctionEncoder
from model.expression_encoder import ExpressionEncoder
from model.fusion import FusionModule
from model.prediction_head import GeneHead, JunctionHead

logger = logging.getLogger(__name__)


class PanCircModel(nn.Module):
    """PanCirc-Fungi multi-modal circRNA foundation model.

    Parameters
    ----------
    model_config : dict
        Model configuration (from config/default.yaml).
    n_species : int
        Number of active species/strains (default: 21).
    """

    def __init__(self, model_config: dict, n_species: int = 21):
        super().__init__()
        cfg = model_config
        d_fusion = cfg.get("fusion_dim", 128)

        # ── Encoders ────────────────────────────────────────────────────
        self.species_embedding = SpeciesEmbedding(
            n_species=n_species,
            phylo_dim=cfg["species"]["phylo_pca_dims"],
            embed_dim=cfg["species"]["embed_dim"],
        )

        self.genome_ctx_encoder = GenomicContextEncoder(
            in_channels=cfg["genome_ctx"]["in_channels"],
            filters=cfg["genome_ctx"]["conv_filters"],
            kernel_size=cfg["genome_ctx"]["conv_kernel"],
            dilations=cfg["genome_ctx"]["dilations"],
            gru_hidden=cfg["genome_ctx"]["gru_hidden"],
            gru_layers=cfg["genome_ctx"]["gru_layers"],
            output_dim=cfg["genome_ctx"]["output_dim"],
            dropout=cfg.get("dropout", 0.1),
        )

        self.gtf_encoder = GTFEncoder(
            n_features=cfg["gtf"]["n_features"],
            hidden_dim=cfg["gtf"]["hidden_dim"],
            output_dim=cfg["gtf"]["output_dim"],
            n_biotypes=cfg["gtf"]["n_biotypes"],
            dropout=cfg.get("dropout", 0.1),
        )

        self.junction_encoder = JunctionEncoder(
            k=cfg["junction"]["k"],
            embed_dim=cfg["junction"]["embed_dim"],
            gru_hidden=cfg["junction"]["gru_hidden"],
            gru_layers=cfg["junction"]["gru_layers"],
            gru_dropout=cfg["junction"].get("gru_dropout", 0.1),
            attention_dim=cfg["junction"]["attention_dim"],
            cross_attn_dim=cfg["junction"].get("cross_attn_dim", 16),
            # CircPCBL one-hot CNN-BiGRU
            cnn_filters=cfg["junction"].get("cnn_filters", 32),
            kernel_sizes=cfg["junction"].get("kernel_sizes", [3, 5, 7]),
            cnn_gru_hidden=cfg["junction"].get("cnn_gru_hidden", 32),
            # CircPCBL k-mer frequency GLT
            kmer_freq_dim=cfg["junction"].get("kmer_freq_dim", 340),
            freq_hidden=cfg["junction"].get("freq_hidden", 64),
            output_dim=cfg["junction"]["output_dim"],
            max_len=cfg.get("flank_size", 150) * 2,  # upstream + downstream
        )

        # Optional expression encoder (fine-tuning only)
        self.expression_encoder = ExpressionEncoder(
            max_replicates=cfg["expression"]["max_replicates"],
            hidden_dim=cfg["expression"]["hidden_dim"],
            output_dim=cfg["expression"]["output_dim"],
            fusion_dim=d_fusion,
            dropout=cfg.get("dropout", 0.1),
        )

        # ── Fusion ──────────────────────────────────────────────────────
        self.fusion = FusionModule(
            fusion_dim=d_fusion,
            n_heads=cfg.get("fusion_heads", 4),
            dropout=cfg.get("dropout", 0.1),
        )

        # ── Prediction Heads ─────────────────────────────────────────────
        self.gene_head = GeneHead(
            fusion_dim=d_fusion,
            dropout=cfg.get("dropout", 0.1),
        )
        self.junction_head = JunctionHead(
            junction_dim=cfg["junction"]["output_dim"],
            fusion_dim=d_fusion,
            dropout=cfg.get("dropout", 0.1),
        )

        # ── Masked k-mer head (Stage 1 pre-training) ────────────────────
        self.masked_kmer_head = nn.Linear(
            cfg["junction"]["embed_dim"], 4 ** cfg["junction"]["k"]
        )

        self._config = cfg

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        task: str = "pretrain",
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: dict with keys:
                strain_id, gtf_features, genome_context,
                is_positive,
                donor_kmers, acceptor_kmers,          ← exon boundary sets
                donor_mask, acceptor_mask,
                (fine-tuning also: circ_exp, gene_exp)
            task: "pretrain" | "finetune"

        Returns:
            dict with task-specific outputs
        """
        device = next(self.parameters()).device

        # ── Encode modalities ───────────────────────────────────────────
        species_vec = self.species_embedding(batch["strain_id"])
        genome_vec = self.genome_ctx_encoder(batch["genome_context"])
        gtf_vec = self.gtf_encoder(batch["gtf_features"])

        # ── Junction encoder (JEDI cross-attention + CircPCBL dual encoding) ─
        je_out = self.junction_encoder(
            donor_kmers=batch["donor_kmers"],
            acceptor_kmers=batch["acceptor_kmers"],
            donor_onehot=batch.get("donor_onehot"),
            acceptor_onehot=batch.get("acceptor_onehot"),
            donor_kmer_freq=batch.get("donor_kmer_freq"),
            acceptor_kmer_freq=batch.get("acceptor_kmer_freq"),
            donor_mask=batch.get("donor_mask"),
            acceptor_mask=batch.get("acceptor_mask"),
        )
        junction_vec = je_out["junction_vec"]  # (batch, output_dim)

        # ── Fuse (gene-level) ────────────────────────────────────────────
        fused = self.fusion(
            junction_vec=junction_vec,
            genome_context_vec=genome_vec,
            gtf_vec=gtf_vec,
            species_vec=species_vec,
        )

        # ── Gene-level prediction ────────────────────────────────────────
        gene_logits = self.gene_head(fused)  # (batch, 1)

        if task == "finetune":
            if "circ_exp" in batch and "gene_exp" in batch:
                expr_vec = self.expression_encoder(
                    batch["circ_exp"], batch["gene_exp"]
                )
                fused = self.fusion(
                    junction_vec=junction_vec,
                    genome_context_vec=genome_vec,
                    gtf_vec=gtf_vec,
                    species_vec=species_vec,
                    expression_vec=expr_vec,
                )
                gene_logits = self.gene_head(fused)
            return {
                "gene_logits": gene_logits,
                "fused_repr": fused,
                **{k: v for k, v in je_out.items() if k != "junction_vec"},
            }

        # ── Junction prediction from raw cross-attention logits ──────────
        jl = je_out["junction_logits"]  # (batch, N_d, N_a) raw logits
        batch_size, N_d, N_a = jl.shape
        junction_scores = jl.view(batch_size, -1)  # (batch, N_d*N_a)
        cross_w = je_out["cross_weights"]  # softmax weights (for interpretability)

        return {
            "gene_logits": gene_logits,
            "junction_scores": junction_scores,
            "junction_logits": jl,                # (batch, N_d, N_a) un-flattened
            "fused_repr": fused,
            "donor_attn": je_out["donor_attn"],
            "acceptor_attn": je_out["acceptor_attn"],
            "cross_weights": cross_w,
        }

    def encode_gene(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract fused gene representation (no heads)."""
        species_vec = self.species_embedding(batch["strain_id"])
        genome_vec = self.genome_ctx_encoder(batch["genome_context"])
        gtf_vec = self.gtf_encoder(batch["gtf_features"])

        je_out = self.junction_encoder(
            donor_kmers=batch["donor_kmers"],
            acceptor_kmers=batch["acceptor_kmers"],
            donor_mask=batch.get("donor_mask"),
            acceptor_mask=batch.get("acceptor_mask"),
        )
        fused = self.fusion(
            junction_vec=je_out["junction_vec"],
            genome_context_vec=genome_vec,
            gtf_vec=gtf_vec,
            species_vec=species_vec,
        )
        return fused

    def compute_pretrain_loss(
        self, outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute pre-training losses.

        Losses:
        1. Gene classification (BCE)
        2. Junction ranking (Cross-entropy on valid candidates)
        3. (Future) Masked k-mer prediction
        """
        losses = {}

        # 1. Gene-level BCE (skip dummy samples with is_positive == -1.0)
        labels = batch["is_positive"]  # (batch,)
        logits = outputs["gene_logits"].squeeze(-1)  # (batch,)
        valid = labels >= 0  # skip dummy samples
        if valid.any():
            losses["gene_bce"] = F.binary_cross_entropy_with_logits(
                logits[valid], labels[valid], reduction="mean"
            )
        else:
            # Zero loss with gradient graph intact
            losses["gene_bce"] = (logits * 0).mean()

        # 2. Junction ranking loss (cross_labels from dataset)
        if outputs.get("junction_scores") is not None and "cross_labels" in batch:
            scores = outputs["junction_scores"]  # (batch, N_d*N_a)
            targets = batch["cross_labels"]     # (batch, N_d, N_a)
            batch_size, N_d, N_a = targets.shape

            # Flatten targets to match scores
            t_flat = targets.view(batch_size, -1)  # (batch, N_d*N_a)

            # Create valid-pairs mask: donor_mask[i] & acceptor_mask[j]
            d_mask = batch.get("donor_mask")  # (batch, N_d)
            a_mask = batch.get("acceptor_mask")  # (batch, N_a)
            if d_mask is not None and a_mask is not None:
                pair_mask = d_mask.unsqueeze(-1) & a_mask.unsqueeze(1)  # (batch, N_d, N_a)
                pair_mask = pair_mask.view(batch_size, -1)  # (batch, N_d*N_a)
            else:
                pair_mask = torch.ones_like(t_flat, dtype=torch.bool)

            # BCE on valid pairs only
            valid_scores = scores[pair_mask]
            valid_targets = t_flat[pair_mask]
            if valid_scores.numel() > 0:
                losses["junction_bce"] = F.binary_cross_entropy_with_logits(
                    valid_scores, valid_targets, reduction="mean"
                )

        return losses

    def compute_finetune_loss(
        self, outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute fine-tuning loss (gene-level BCE, skip dummy samples)."""
        labels = batch["is_positive"]  # (batch,)
        logits = outputs["gene_logits"].squeeze(-1)  # (batch,)
        valid = labels >= 0  # skip dummy samples (is_positive == -1)
        if valid.any():
            return {
                "gene_bce": F.binary_cross_entropy_with_logits(
                    logits[valid], labels[valid], reduction="mean"
                ),
            }
        return {"gene_bce": (logits * 0).mean()}


def count_parameters(model: PanCircModel) -> int:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
