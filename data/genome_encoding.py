"""
Genome sequence encoding for PanCirc-Fungi.

Provides:
- GenomeIndexer: lazy-load genome via pyfaidx
- Junction sequence extraction (donor/acceptor flanks)
- Genomic context window extraction
- GC and k-mer profile computation
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pyfaidx
    HAS_PYFAIDX = True
except ImportError:
    HAS_PYFAIDX = False

from utils.seq_utils import one_hot_encode, kmer_tokenize, reverse_complement

logger = logging.getLogger(__name__)


class GenomeIndexer:
    """Wrapper around pyfaidx.Fasta for indexed genome access.

    Parameters
    ----------
    fasta_path : str
        Path to genome FASTA (with .fai index).
    """

    def __init__(self, fasta_path: str):
        if not HAS_PYFAIDX:
            raise ImportError("pyfaidx is required for GenomeIndexer. "
                              "Install: pip install pyfaidx")
        if not os.path.isfile(fasta_path):
            raise FileNotFoundError(f"Genome FASTA not found: {fasta_path}")
        if not os.path.isfile(fasta_path + ".fai"):
            logger.warning(f"No .fai index found for {fasta_path}, building...")
        self.fasta = pyfaidx.Fasta(fasta_path, rebuild=True)
        self.path = fasta_path

    def close(self):
        self.fasta.close()

    def get_seq(self, chrom: str, start: int, end: int,
                strand: str = "+") -> str:
        """Extract sequence [start, end) from a chromosome.

        Coordinates are 1-based, end-exclusive (GTF standard).
        If start < 1 or end > chrom_len, pads with 'N'.
        """
        try:
            seq = self.fasta[chrom][start - 1 : end - 1].seq
        except (KeyError, ValueError) as e:
            logger.debug(f"Failed to fetch {chrom}:{start}-{end}: {e}")
            return "N" * (end - start)

        seq = seq.upper()
        if strand == "-":
            seq = reverse_complement(seq)
        return seq

    def extract_window(self, chrom: str, center: int, half_width: int,
                       strand: str = "+") -> str:
        """Extract a symmetric window centered on a position.

        Pads with N if near chromosome ends.
        """
        half = half_width
        start = max(1, center - half)
        end = center + half
        return self.get_seq(chrom, start, end, strand)

    def get_chrom_lengths(self) -> Dict[str, int]:
        """Return dict of chromosome -> length."""
        return {name: len(self.fasta[name]) for name in self.fasta.keys()}


def extract_junction_flanks(
    genome: GenomeIndexer,
    chrom: str,
    donor_pos: int,
    acceptor_pos: int,
    strand: str,
    flank_size: int = 150,
) -> Dict[str, str]:
    """Extract four flanking windows around a backsplice junction.

    Returns dict with keys:
        donor_upstream, donor_downstream,
        acceptor_upstream, acceptor_downstream
    """
    if strand == "+":
        d5 = donor_pos
        a3 = acceptor_pos
        donor_up = genome.get_seq(chrom, d5 - flank_size, d5, "+")
        donor_down = genome.get_seq(chrom, d5, d5 + flank_size, "+")
        acceptor_up = genome.get_seq(chrom, a3 - flank_size, a3, "+")
        acceptor_down = genome.get_seq(chrom, a3, a3 + flank_size, "+")
    else:
        d5 = donor_pos
        a3 = acceptor_pos
        donor_up = genome.get_seq(chrom, d5, d5 + flank_size, "-")
        donor_down = genome.get_seq(chrom, d5 - flank_size, d5, "-")
        acceptor_up = genome.get_seq(chrom, a3, a3 + flank_size, "-")
        acceptor_down = genome.get_seq(chrom, a3 - flank_size, a3, "-")

    return {
        "donor_upstream": donor_up,
        "donor_downstream": donor_down,
        "acceptor_upstream": acceptor_up,
        "acceptor_downstream": acceptor_down,
    }


def extract_all_exon_flanks(
    genome: GenomeIndexer,
    gene,
    flank_size: int = 150,
) -> Dict[str, list]:
    """Extract flank sequences for ALL exon boundaries of a gene.

    Each exon provides:
      - A BACKSPLICE DONOR site (5' boundary on + strand, 3' boundary on - strand)
      - A BACKSPLICE ACCEPTOR site (3' boundary on + strand, 5' boundary on - strand)

    Returns
    -------
    donor_seqs : list of str — flank seq (upstream+downstream) per donor site
    acceptor_seqs : list of str — flank seq per acceptor site
    donor_positions : list of int — genomic coordinates for interpretability
    acceptor_positions : list of int
    """
    donor_seqs, acceptor_seqs = [], []
    donor_positions, acceptor_positions = [], []

    for s, e in zip(gene.exon_starts, gene.exon_ends):
        if gene.strand == "+":
            d_pos = s   # donor = exon start (5' boundary)
            a_pos = e   # acceptor = exon end (3' boundary)
        else:
            d_pos = e   # donor = exon end (on - strand, 5' in transcript)
            a_pos = s   # acceptor = exon start

        d_up = genome.get_seq(gene.chrom, d_pos - flank_size, d_pos, gene.strand)
        d_down = genome.get_seq(gene.chrom, d_pos, d_pos + flank_size, gene.strand)
        donor_seqs.append(d_up + d_down)
        donor_positions.append(d_pos)

        a_up = genome.get_seq(gene.chrom, a_pos - flank_size, a_pos, gene.strand)
        a_down = genome.get_seq(gene.chrom, a_pos, a_pos + flank_size, gene.strand)
        acceptor_seqs.append(a_up + a_down)
        acceptor_positions.append(a_pos)

    return {
        "donor_seqs": donor_seqs,
        "acceptor_seqs": acceptor_seqs,
        "donor_positions": donor_positions,
        "acceptor_positions": acceptor_positions,
    }


def encode_flanks_onehot(
    seqs: List[str],
    max_len: Optional[int] = None,
) -> np.ndarray:
    """One-hot encode a list of flank sequences for CircPCBL CNN-BiGRU path.

    Returns array of shape (n_seqs, 4, L) where L = max_len or
    the longest sequence length. Shorter sequences are zero-padded.
    """
    if not seqs:
        return np.zeros((0, 4, 0), dtype=np.float32)
    L = max_len or max(len(s) for s in seqs)
    n = len(seqs)
    arr = np.zeros((n, 4, L), dtype=np.float32)
    for i, seq in enumerate(seqs):
        oh = one_hot_encode(seq)
        length = min(oh.shape[1], L)
        arr[i, :, :length] = oh[:, :length]
    return arr


def compute_kmer_frequencies(
    seqs: List[str],
    max_k: int = 4,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute k-mer frequency features for CircPCBL GLT path.

    For each sequence, computes frequency of all k-mers for k=1..max_k.
    Returns array of shape (n_seqs, sum(4^k for k=1..max_k)) = (n_seqs, 340).
    Frequencies are normalized per sequence (sum to 1 per k value).
    """
    if not seqs:
        return np.zeros((0, 340), dtype=np.float32)

    from itertools import product
    bases = "ACGT"

    n = len(seqs)
    features = np.zeros((n, 340), dtype=np.float32)
    offset = 0

    for k in range(1, max_k + 1):
        kmers = ["".join(p) for p in product(bases, repeat=k)]
        idx_map = {kmer: i for i, kmer in enumerate(kmers)}
        n_k = len(kmers)

        for i, seq in enumerate(seqs):
            seq = seq.upper().replace("U", "T")
            total = 0
            for j in range(len(seq) - k + 1):
                kmer = seq[j:j + k]
                if kmer in idx_map:
                    features[i, offset + idx_map[kmer]] += 1.0
                    total += 1
            if total > 0:
                features[i, offset:offset + n_k] /= total

        offset += n_k

    return features


def extract_genome_context(
    genome: GenomeIndexer,
    chrom: str,
    center: int,
    strand: str,
    window_size: int = 10000,
) -> str:
    """Extract a symmetric genome context window centered on a position."""
    return genome.extract_window(chrom, center, window_size // 2, strand)


# ---------------------------------------------------------------------------
# Profile computation (for GenomicContextEncoder input)
# ---------------------------------------------------------------------------


def compute_gc_profile(seq: str, bin_size: int = 50,
                       stride: Optional[int] = None) -> np.ndarray:
    """Compute sliding-window GC content profile."""
    if stride is None:
        stride = bin_size
    L = len(seq)
    n_bins = max(1, (L - bin_size) // stride + 1)
    profile = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        start = i * stride
        end = start + bin_size
        window = seq[start:end]
        gc = window.upper().count("G") + window.upper().count("C")
        profile[i] = gc / max(len(window), 1)
    return profile


def compute_kmer_profile(seq: str, k: int = 2,
                         bin_size: int = 50) -> np.ndarray:
    """Compute k-mer frequency profile in sliding bins."""
    from itertools import product

    bases = "ACGT"
    kmers = ["".join(p) for p in product(bases, repeat=k)]
    kmer_to_idx = {k: i for i, k in enumerate(kmers)}
    n_kmers = len(kmers)

    L = len(seq)
    n_bins = max(1, (L - bin_size) // (bin_size // 2) + 1)
    stride = max(1, (L - bin_size) // n_bins) if n_bins > 1 else 1

    profile = np.zeros((n_bins, n_kmers), dtype=np.float32)
    seq_upper = seq.upper().replace("U", "T")

    for i in range(n_bins):
        start = i * stride
        end = min(start + bin_size, L)
        window = seq_upper[start:end]
        total = 0
        for j in range(len(window) - k + 1):
            kmer = window[j : j + k]
            if kmer in kmer_to_idx:
                profile[i, kmer_to_idx[kmer]] += 1.0
                total += 1
        if total > 0:
            profile[i] /= total

    return profile


def compute_genome_context_features(
    genome_window: str, bin_size: int = 50
) -> np.ndarray:
    """Compute combined profile features for a genome window.

    Returns array of shape (n_bins, 8).
    """
    n_bins = max(1, (len(genome_window) - bin_size) // (bin_size // 2) + 1)
    stride = max(1, (len(genome_window) - bin_size) // n_bins) if n_bins > 1 else 1

    features = np.zeros((n_bins, 8), dtype=np.float32)
    seq = genome_window.upper().replace("U", "T")

    for i in range(n_bins):
        start = i * stride
        end = min(start + bin_size, len(seq))
        window = seq[start:end]
        L = len(window)
        if L == 0:
            continue

        counts = {
            "A": window.count("A"),
            "C": window.count("C"),
            "G": window.count("G"),
            "T": window.count("T"),
        }
        total = sum(counts.values())
        if total == 0:
            continue

        features[i, 0] = (counts["G"] + counts["C"]) / total
        features[i, 1] = counts["A"] / total
        features[i, 2] = counts["C"] / total
        features[i, 3] = counts["G"] / total
        features[i, 4] = counts["T"] / total

        cpg_obs = window.count("CG")
        cpg_exp = (counts["C"] * counts["G"]) / (total**2) * (L - 1) if total > 0 else 1
        features[i, 5] = cpg_obs / max(cpg_exp, 0.01)
        features[i, 6] = (counts["A"] + counts["T"]) / max(counts["G"] + counts["C"], 1)

        ps = np.array([v / total for v in counts.values()])
        ps = ps[ps > 0]
        features[i, 7] = -np.sum(ps * np.log2(ps))

    return features
