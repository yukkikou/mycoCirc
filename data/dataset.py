"""
PyTorch Dataset classes for PanCirc-Fungi — JEDI-aligned.

Each sample is a **gene** with:
- donor_kmers: (N_donors, L) k-mer tokenized flank sequences for each exon 5' boundary
- acceptor_kmers: (N_acceptors, L) for each exon 3' boundary
- donor_mask / acceptor_mask: variable-length masks
- cross_labels: (N_donors, N_acceptors) — 1 at (i,j) if exon_i→exon_j is a known circRNA junction
- gtf_features, genome_context, strain_id
"""

import logging
import os
import random
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset

from data.circ_info_encoding import (
    load_circ_info,
    filter_circ_info,
    get_positive_genes,
)
from data.expression_encoding import (
    load_expression_csv,
    align_circ_to_gene_expression,
    pad_to_max_replicates,
)
from data.tsv_parser import StrainEntry
from data.genome_encoding import extract_all_exon_flanks
from utils.seq_utils import kmer_tokenize

logger = logging.getLogger(__name__)


class CircRNAPretrainDataset(Dataset):
    """Gene-level dataset for pre-training.

    For each gene, extracts ALL exon boundary flank sequences and
    processes them through k-mer tokenization for the JunctionEncoder.

    Known circRNA junctions are encoded as labels on the cross-attention
    matrix: cross_labels[i,j] = 1 if exon_i→exon_j is backspliced.
    """

    def __init__(
        self,
        entries: List[StrainEntry],
        gene_models: Dict[str, "GeneModelIndexer"],
        genome_indexers: Dict[str, "GenomeIndexer"],
        positive_genes: Dict[str, Dict[str, pd.DataFrame]],
        negative_genes: Dict[str, List[str]],
        config: Optional[dict] = None,
        augment: bool = False,
        features_dir: Optional[str] = None,
    ):
        self.entries = entries
        self.gene_models = gene_models
        self.genome_indexers = genome_indexers
        self.positive_genes = positive_genes
        self.negative_genes = negative_genes
        self.config = config or {}
        self.augment = augment
        self.features_dir = features_dir

        self.flank_size = self.config.get("flank_size", 150)
        self.genome_window_size = self.config.get("genome_window_size", 10000)
        self.gtf_feature_dim = self.config.get("gtf_feature_dim", 17)
        self.k = self.config.get("k", 3)  # k-mer size
        self.max_exons = self.config.get("max_exons_per_gene", 50)

        self.samples: List[Tuple[StrainEntry, str, bool]] = []
        self._build_index()

    def _build_index(self):
        for entry in self.entries:
            strain = entry.strain
            pos_genes = self.positive_genes.get(strain, {})
            for gid in pos_genes:
                self.samples.append((entry, gid, True))
            neg_genes = self.negative_genes.get(strain, [])
            for gid in neg_genes:
                self.samples.append((entry, gid, False))

        logger.info(
            f"Dataset: {len(self.samples)} samples "
            f"({sum(1 for _, _, p in self.samples if p)} pos, "
            f"{sum(1 for _, _, p in self.samples if not p)} neg)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry, gene_id, is_positive = self.samples[idx]
        strain = entry.strain
        gm = self.gene_models[strain]
        gene = gm.get_gene(gene_id)

        if gene is None or gene.exon_count < 1:
            return self._get_dummy(entry.strain)

        # ── GTF features ────────────────────────────────────────────────
        gtf_feats = gm.extract_features(gene)

        # ── Genome context ──────────────────────────────────────────────
        genome_profile = np.zeros((200, 8), dtype=np.float32)
        genome_idx = self.genome_indexers.get(strain)
        if genome_idx is not None:
            mid = (gene.start + gene.end) // 2
            window = genome_idx.extract_window(
                gene.chrom, mid, self.genome_window_size // 2
            )
            if window:
                from data.genome_encoding import compute_genome_context_features
                profile = compute_genome_context_features(window)
                if profile.shape[0] >= 200:
                    genome_profile = profile[:200]
                else:
                    genome_profile[:profile.shape[0]] = profile

        # ── Species index ───────────────────────────────────────────────
        from data.tsv_parser import build_strain_index
        strain_idx_map = build_strain_index(self.entries)
        species_id = strain_idx_map.get(strain, 0)

        # ── Exon boundary flanks (donor + acceptor sets) ────────────────
        feat = self._load_junction_features(strain, gene_id, gene)

        # ── Cross-attention labels ──────────────────────────────────────
        N_d = feat["donor_kmers"].shape[0]
        N_a = feat["acceptor_kmers"].shape[0]
        cross_labels = np.zeros((N_d, N_a), dtype=np.float32)

        if is_positive:
            pos_df = self.positive_genes[strain].get(gene_id, pd.DataFrame())
            for _, row in pos_df.iterrows():
                parsed = _parse_circ_id_for_exon_pair(row.get("circ_id", ""), gene)
                if parsed:
                    d_idx, a_idx = parsed
                    if 0 <= d_idx < N_d and 0 <= a_idx < N_a:
                        cross_labels[d_idx, a_idx] = 1.0

        return {
            "strain_id": torch.tensor(species_id, dtype=torch.long),
            "gene_id": gene_id,
            "gtf_features": torch.from_numpy(gtf_feats).float(),
            "genome_context": torch.from_numpy(genome_profile).float(),
            "is_positive": torch.tensor(float(is_positive), dtype=torch.float),
            # JEDI path
            "donor_kmers": torch.from_numpy(feat["donor_kmers"]).long(),
            "acceptor_kmers": torch.from_numpy(feat["acceptor_kmers"]).long(),
            # CircPCBL one-hot path
            "donor_onehot": torch.from_numpy(feat["donor_onehot"]).float(),
            "acceptor_onehot": torch.from_numpy(feat["acceptor_onehot"]).float(),
            # CircPCBL k-mer frequency path
            "donor_kmer_freq": torch.from_numpy(feat["donor_kmer_freq"]).float(),
            "acceptor_kmer_freq": torch.from_numpy(feat["acceptor_kmer_freq"]).float(),
            # Masks
            "donor_mask": torch.from_numpy(feat["donor_mask"]).bool(),
            "acceptor_mask": torch.from_numpy(feat["acceptor_mask"]).bool(),
            # Labels
            "cross_labels": torch.from_numpy(cross_labels).float(),
        }

    def _load_junction_features(self, strain: str, gene_id: str, gene) \
            -> Dict[str, np.ndarray]:
        """Load or compute all junction features: kmers, onehot, kmer_freq."""
        default = self._empty_features()

        # Try pre-computed
        if self.features_dir:
            feat_path = os.path.join(
                self.features_dir, strain, f"{gene_id}_junction.npz"
            )
            if os.path.isfile(feat_path):
                return self._load_from_npz(feat_path)

        # On-the-fly extraction
        genome_idx = self.genome_indexers.get(strain)
        if genome_idx is None:
            return default

        flanks = extract_all_exon_flanks(genome_idx, gene, self.flank_size)
        d_seqs = flanks["donor_seqs"]
        a_seqs = flanks["acceptor_seqs"]
        N_d = min(len(d_seqs), self.max_exons)
        N_a = min(len(a_seqs), self.max_exons)

        # JEDI: k-mer tokens
        L = self.flank_size * 2
        tok_len = L - self.k + 1
        d_k = np.zeros((self.max_exons, max(tok_len, 1)), dtype=np.int64)
        a_k = np.zeros((self.max_exons, max(tok_len, 1)), dtype=np.int64)

        # CircPCBL: one-hot (C=4, L)
        d_oh = np.zeros((self.max_exons, 4, L), dtype=np.float32)
        a_oh = np.zeros((self.max_exons, 4, L), dtype=np.float32)

        # CircPCBL: k-mer frequencies (340-dim)
        d_fq = np.zeros((self.max_exons, 340), dtype=np.float32)
        a_fq = np.zeros((self.max_exons, 340), dtype=np.float32)

        d_mask = np.zeros(self.max_exons, dtype=bool)
        a_mask = np.zeros(self.max_exons, dtype=bool)

        from data.genome_encoding import encode_flanks_onehot, compute_kmer_frequencies
        from utils.seq_utils import kmer_tokenize, one_hot_encode

        for i in range(N_d):
            seq = d_seqs[i]
            # JEDI
            tokens = kmer_tokenize(seq, k=self.k)
            d_k[i, :min(len(tokens), tok_len)] = tokens[:tok_len]
            # CircPCBL one-hot
            oh = one_hot_encode(seq)
            d_oh[i, :, :min(oh.shape[1], L)] = oh[:, :L]
            d_mask[i] = True

        for i in range(N_a):
            seq = a_seqs[i]
            tokens = kmer_tokenize(seq, k=self.k)
            a_k[i, :min(len(tokens), tok_len)] = tokens[:tok_len]
            oh = one_hot_encode(seq)
            a_oh[i, :, :min(oh.shape[1], L)] = oh[:, :L]
            a_mask[i] = True

        # k-mer frequencies (batch computation)
        if N_d > 0:
            fq = compute_kmer_frequencies(d_seqs[:N_d])
            d_fq[:N_d] = fq
        if N_a > 0:
            fq = compute_kmer_frequencies(a_seqs[:N_a])
            a_fq[:N_a] = fq

        return {
            "donor_kmers": d_k, "acceptor_kmers": a_k,
            "donor_onehot": d_oh, "acceptor_onehot": a_oh,
            "donor_kmer_freq": d_fq, "acceptor_kmer_freq": a_fq,
            "donor_mask": d_mask, "acceptor_mask": a_mask,
        }

    def _load_from_npz(self, path: str) -> Dict[str, np.ndarray]:
        data = np.load(path)
        me = self.max_exons
        L = self.flank_size * 2
        tok_len = L - self.k + 1

        def pad(arr, max_n, pad_shape):
            out = np.zeros((max_n,) + pad_shape, dtype=arr.dtype)
            n = min(arr.shape[0], max_n)
            out[:n] = arr[:n]
            return out

        N_d = min(data["donor_kmers"].shape[0], me)
        N_a = min(data["acceptor_kmers"].shape[0], me)

        return {
            "donor_kmers": pad(data["donor_kmers"], me, (tok_len,)),
            "acceptor_kmers": pad(data["acceptor_kmers"], me, (tok_len,)),
            "donor_onehot": pad(data.get("donor_onehot") or np.zeros((0, 4, L)),
                                me, (4, L)),
            "acceptor_onehot": pad(data.get("acceptor_onehot") or np.zeros((0, 4, L)),
                                   me, (4, L)),
            "donor_kmer_freq": pad(data.get("donor_kmer_freq") or np.zeros((0, 340)),
                                   me, (340,)),
            "acceptor_kmer_freq": pad(data.get("acceptor_kmer_freq") or np.zeros((0, 340)),
                                      me, (340,)),
            "donor_mask": np.arange(me) < N_d,
            "acceptor_mask": np.arange(me) < N_a,
        }

    def _empty_features(self) -> Dict[str, np.ndarray]:
        L = self.flank_size * 2
        tok_len = L - self.k + 1
        return {
            "donor_kmers": np.zeros((self.max_exons, max(tok_len, 1)), dtype=np.int64),
            "acceptor_kmers": np.zeros((self.max_exons, max(tok_len, 1)), dtype=np.int64),
            "donor_onehot": np.zeros((self.max_exons, 4, L), dtype=np.float32),
            "acceptor_onehot": np.zeros((self.max_exons, 4, L), dtype=np.float32),
            "donor_kmer_freq": np.zeros((self.max_exons, 340), dtype=np.float32),
            "acceptor_kmer_freq": np.zeros((self.max_exons, 340), dtype=np.float32),
            "donor_mask": np.zeros(self.max_exons, dtype=bool),
            "acceptor_mask": np.zeros(self.max_exons, dtype=bool),
        }

    def _get_dummy(self, strain: str) -> Dict[str, torch.Tensor]:
        feats = self._empty_features()
        from data.tsv_parser import build_strain_index
        strain_idx_map = build_strain_index(self.entries)
        species_id = strain_idx_map.get(strain, 0)
        return {
            "strain_id": torch.tensor(species_id, dtype=torch.long),
            "gene_id": "dummy",
            "gtf_features": torch.zeros(self.gtf_feature_dim, dtype=torch.float),
            "genome_context": torch.zeros(200, 8, dtype=torch.float),
            "is_positive": torch.tensor(-1.0, dtype=torch.float),
            **{k: torch.from_numpy(v) for k, v in feats.items()},
            "cross_labels": torch.zeros(self.max_exons, self.max_exons, dtype=torch.float),
        }


class CircRNAFinetuneDataset(CircRNAPretrainDataset):
    """Fine-tuning dataset — adds CircExp/GeneExp as input features."""

    def __init__(
        self,
        entries: List[StrainEntry],
        gene_models: Dict[str, "GeneModelIndexer"],
        genome_indexers: Dict[str, "GenomeIndexer"],
        positive_genes: Dict[str, Dict[str, pd.DataFrame]],
        negative_genes: Dict[str, List[str]],
        expression_data: Optional[Dict[str, Dict]] = None,
        config: Optional[dict] = None,
        augment: bool = False,
        features_dir: Optional[str] = None,
        max_replicates: int = 3,
    ):
        self.expression_data = expression_data or {}
        self.max_replicates = max_replicates
        super().__init__(
            entries, gene_models, genome_indexers,
            positive_genes, negative_genes,
            config, augment, features_dir,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        entry, gene_id, is_positive = self.samples[idx]

        circ_exp = np.zeros(self.max_replicates, dtype=np.float32)
        gene_exp = np.zeros(self.max_replicates, dtype=np.float32)

        if self.expression_data:
            strain_data = self.expression_data.get(entry.strain, {})
            gene_exp_data = strain_data.get("gene_exp", {})
            if is_positive:
                pos_data = strain_data.get("aligned", {})
                if gene_id in pos_data:
                    ce = pos_data[gene_id].get("circ_exp", np.array([]))
                    if ce.size > 0:
                        ce = ce.mean(axis=0)
                        circ_exp = pad_to_max_replicates(ce, self.max_replicates)
            if gene_id in gene_exp_data:
                ge = gene_exp_data[gene_id]
                if isinstance(ge, np.ndarray) and ge.size > 0:
                    gene_exp = pad_to_max_replicates(ge, self.max_replicates)

        item["circ_exp"] = torch.from_numpy(circ_exp).float()
        item["gene_exp"] = torch.from_numpy(gene_exp).float()
        return item


def collate_pretrain(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader with variable exon counts."""
    out = {}
    for key in batch[0].keys():
        if key == "gene_id":
            out[key] = [b[key] for b in batch]
            continue

        if key in ("donor_kmers", "acceptor_kmers"):
            # Already padded to max_exons — just stack
            max_L = max(b[key].shape[1] for b in batch)
            tensors = []
            for b in batch:
                t = b[key]
                if t.shape[1] < max_L:
                    t = torch.nn.functional.pad(t, (0, max_L - t.shape[1]))
                tensors.append(t)
            out[key] = torch.stack(tensors)
            continue

        if isinstance(batch[0][key], torch.Tensor):
            tensors = [b[key] for b in batch]
            try:
                out[key] = torch.stack(tensors)
            except RuntimeError:
                max_shape = max(t.shape for t in tensors)
                padded = []
                for t in tensors:
                    pad = [0] * (2 * t.ndim)
                    for dim in range(t.ndim):
                        pad[2 * dim + 1] = max_shape[dim] - t.shape[dim]
                    t = torch.nn.functional.pad(t, pad[::-1])
                    padded.append(t)
                out[key] = torch.stack(padded)
        else:
            out[key] = [b[key] for b in batch]

    return out


ConcatStrainDataset = ConcatDataset


# ─── Helpers ────────────────────────────────────────────────────────────────

def _parse_circ_id_for_exon_pair(circ_id: str, gene) -> Optional[Tuple[int, int]]:
    """Find which exon index pair a circ_id corresponds to.

    Returns (donor_exon_idx, acceptor_exon_idx) or None.
    """
    if not circ_id or not isinstance(circ_id, str):
        return None
    try:
        coord_part = circ_id.split(":")[-1]
        if "|" in coord_part:
            c1, c2 = coord_part.split("|")
        elif "-" in coord_part:
            c1, c2 = coord_part.split("-")
        else:
            return None
        donor_pos = int(c1)
        acceptor_pos = int(c2)
    except (ValueError, IndexError):
        return None

    # Find exon indices matching these positions
    d_idx, a_idx = None, None
    for i, (s, e) in enumerate(zip(gene.exon_starts, gene.exon_ends)):
        if gene.strand == "+":
            if s == donor_pos:
                d_idx = i
            if e == acceptor_pos:
                a_idx = i
        else:
            if e == donor_pos:
                d_idx = i
            if s == acceptor_pos:
                a_idx = i

    if d_idx is not None and a_idx is not None:
        return (d_idx, a_idx)
    return None
