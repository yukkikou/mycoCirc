#!/usr/bin/env python3
"""
Predict circRNA for a new fungal genome using a mycoCirc model.

Given a genome FASTA + GTF annotation (and optionally GeneExp counts),
outputs a TSV with each gene's circRNA probability.

Usage
-----
    python predict.py \\
        --genome genome.fa \\
        --gtf annotation.gtf \\
        --checkpoint checkpoints/finetune/Candida/final.pt \\
        --config config/default.yaml \\
        --output predictions.tsv \\
        [--genexp gene_exp.csv] \\
        [--device cuda]

Output TSV columns:
    gene_id     — gene identifier from GTF
    chrom       — chromosome/scaffold
    start, end  — gene coordinates
    strand      — +/-
    p_circ      — P(circRNA | gene)  [0–1]
    n_exons     — number of exons in the gene

Notes for new users
-------------------
- For a NEW species (not in the 21 training strains), the species embedding
  defaults to the first training strain's slot with a warning. The prediction
  should still be informative; for best results use a group-matched checkpoint
  (e.g., Candida checkpoint for a Candida genome).
- Checkpoint is loaded with strict=False. Any mismatched keys are reported
  as warnings for diagnostics.
- No RNA-seq data required for Mode A (genome+GTF only). Gene expression
  is optional.
"""

import argparse
import logging
import os
import sys
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("predict")


# ───── Lazy imports (fail with clear messages) ─────────────────────────────

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml required. pip install pyyaml")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

from data.genome_encoding import (
    GenomeIndexer, compute_genome_context_features, extract_all_exon_flanks,
)
from data.expression_encoding import pad_to_max_replicates
from model.pancirc import PanCircModel, count_parameters
from utils.gtf_utils import GeneModelIndexer
from utils.seq_utils import kmer_tokenize, one_hot_encode


# ───── Gene-level prediction dataset ──────────────────────────────────────


class PredictDataset(Dataset):
    """Iterates over all genes in a GTF and yields their features."""

    def __init__(self, gene_model, genome_index, config, k=3,
                 max_exons=50, genexp_data=None, max_replicates=3):
        self.gm = gene_model
        self.gi = genome_index
        self.config = config
        self.flank_size = config.get("flank_size", 150)
        self.genome_window = config.get("genome_window_size", 10000)
        self.k = k
        self.max_exons = max_exons
        self.max_replicates = max_replicates
        self.genexp_data = genexp_data or {}

        # Build gene list (skip genes with <1 exon)
        self.gene_ids = []
        for gid in list(gm.genes.keys()):
            g = gm.get_gene(gid)
            if g is not None and g.exon_count >= 1:
                self.gene_ids.append(gid)
        logger.info(f"  {len(self.gene_ids)} genes to predict")

    def __len__(self):
        return len(self.gene_ids)

    def __getitem__(self, idx):
        gene_id = self.gene_ids[idx]
        gene = self.gm.get_gene(gene_id)

        # ── GTF features ──────────────────────────────────────────────
        gtf_feats = self.gm.extract_features(gene)

        # ── Genome context ────────────────────────────────────────────
        genome_profile = np.zeros((200, 8), dtype=np.float32)
        mid = (gene.start + gene.end) // 2
        window = self.gi.extract_window(gene.chrom, mid, self.genome_window // 2)
        if window:
            profile = compute_genome_context_features(window)
            if profile.shape[0] >= 200:
                genome_profile = profile[:200]
            else:
                genome_profile[:profile.shape[0]] = profile

        # ── Exon boundary flanks ──────────────────────────────────────
        feat = self._load_junction_features(gene)

        # ── GeneExp (optional) ────────────────────────────────────────
        gene_exp = np.zeros(self.max_replicates, dtype=np.float32)
        if self.genexp_data and gene_id in self.genexp_data:
            ge = self.genexp_data[gene_id]
            if isinstance(ge, np.ndarray) and ge.size > 0:
                gene_exp = pad_to_max_replicates(ge, self.max_replicates)

        # ── dummy circ_exp (zeroed — not available at inference) ─────
        circ_exp = np.zeros(self.max_replicates, dtype=np.float32)

        return {
            "gene_id": gene_id,
            "chrom": gene.chrom,
            "start": gene.start,
            "end": gene.end,
            "strand": gene.strand,
            "n_exons": gene.exon_count,
            "gtf_features": torch.from_numpy(gtf_feats).float(),
            "genome_context": torch.from_numpy(genome_profile).float(),
            "donor_kmers": torch.from_numpy(feat["donor_kmers"]).long(),
            "acceptor_kmers": torch.from_numpy(feat["acceptor_kmers"]).long(),
            "donor_onehot": torch.from_numpy(feat["donor_onehot"]).float(),
            "acceptor_onehot": torch.from_numpy(feat["acceptor_onehot"]).float(),
            "donor_kmer_freq": torch.from_numpy(feat["donor_kmer_freq"]).float(),
            "acceptor_kmer_freq": torch.from_numpy(feat["acceptor_kmer_freq"]).float(),
            "donor_mask": torch.from_numpy(feat["donor_mask"]).bool(),
            "acceptor_mask": torch.from_numpy(feat["acceptor_mask"]).bool(),
            "circ_exp": torch.from_numpy(circ_exp).float(),
            "gene_exp": torch.from_numpy(gene_exp).float(),
            # dummy tensors for collate compatibility
            "strain_id": torch.tensor(0, dtype=torch.long),
            "is_positive": torch.tensor(0.0, dtype=torch.float),
            "cross_labels": torch.zeros(self.max_exons, self.max_exons, dtype=torch.float),
        }

    def _load_junction_features(self, gene):
        L = self.flank_size * 2
        tok_len = L - self.k + 1
        default = {
            "donor_kmers": np.zeros((self.max_exons, max(tok_len, 1)), dtype=np.int64),
            "acceptor_kmers": np.zeros((self.max_exons, max(tok_len, 1)), dtype=np.int64),
            "donor_onehot": np.zeros((self.max_exons, 4, L), dtype=np.float32),
            "acceptor_onehot": np.zeros((self.max_exons, 4, L), dtype=np.float32),
            "donor_kmer_freq": np.zeros((self.max_exons, 340), dtype=np.float32),
            "acceptor_kmer_freq": np.zeros((self.max_exons, 340), dtype=np.float32),
            "donor_mask": np.zeros(self.max_exons, dtype=bool),
            "acceptor_mask": np.zeros(self.max_exons, dtype=bool),
        }

        flanks = extract_all_exon_flanks(self.gi, gene, self.flank_size)
        d_seqs = flanks.get("donor_seqs", [])
        a_seqs = flanks.get("acceptor_seqs", [])
        N_d = min(len(d_seqs), self.max_exons)
        N_a = min(len(a_seqs), self.max_exons)

        for i in range(N_d):
            seq = d_seqs[i]
            tokens = kmer_tokenize(seq, k=self.k)
            default["donor_kmers"][i, :min(len(tokens), tok_len)] = tokens[:tok_len]
            oh = one_hot_encode(seq)
            default["donor_onehot"][i, :, :min(oh.shape[1], L)] = oh[:, :L]
            default["donor_mask"][i] = True
        for i in range(N_a):
            seq = a_seqs[i]
            tokens = kmer_tokenize(seq, k=self.k)
            default["acceptor_kmers"][i, :min(len(tokens), tok_len)] = tokens[:tok_len]
            oh = one_hot_encode(seq)
            default["acceptor_onehot"][i, :, :min(oh.shape[1], L)] = oh[:, :L]
            default["acceptor_mask"][i] = True

        return default


# ───── Collate ─────────────────────────────────────────────────────────────


def collate_predict(batch):
    """Collate function for PredictDataset."""
    out = {}
    keys = list(batch[0].keys())
    for key in keys:
        if key in ("gene_id", "chrom", "strand"):
            out[key] = [b[key] for b in batch]
            continue
        if key in ("start", "end", "n_exons"):
            out[key] = torch.tensor([b[key] for b in batch])
            continue
        if key in ("donor_kmers", "acceptor_kmers"):
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
                pass  # fallback for variable-size tensors
    return out


# ───── Main ────────────────────────────────────────────────────────────────


def load_gene_expression(path, max_replicates=3):
    """Load GeneExp CSV -> {gene_id: np.ndarray}.
    Returns empty dict if file not found."""
    if not path or not os.path.isfile(path):
        return {}
    from data.expression_encoding import load_expression_csv, encode_expression_values
    df = load_expression_csv(path)
    if df is None:
        return {}
    vals, ids, _ = encode_expression_values(df, log1p=True, zscore=True)
    return {gid: pad_to_max_replicates(vals[i], max_replicates)
            for i, gid in enumerate(ids)}


def main():
    parser = argparse.ArgumentParser(description="Predict circRNA with mycoCirc")
    parser.add_argument("--genome", required=True, help="Genome FASTA")
    parser.add_argument("--gtf", required=True, help="Gene annotation GTF")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint (.pt)")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Model config YAML")
    parser.add_argument("--output", default="predictions.tsv", help="Output TSV path")
    parser.add_argument("--genexp", default=None,
                        help="Optional: GeneExp CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (default: auto-detect)")
    parser.add_argument("--strain-id", type=int, default=0,
                        help="Species embedding index (0–20, default 0). "
                             "Only matters if checkpoint was trained on multiple species.")
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)
    k = config["model"]["junction"]["k"]
    max_rep = config["model"]["expression"]["max_replicates"]
    data_cfg = config["data"]

    # ── Index genome & GTF ──────────────────────────────────────────────
    logger.info(f"Indexing genome: {args.genome}")
    gi = GenomeIndexer(args.genome)
    logger.info(f"Indexing GTF: {args.gtf}")
    gm = GeneModelIndexer(args.gtf)
    logger.info(f"  {gm.n_genes()} genes in GTF")

    # ── Optional GeneExp ────────────────────────────────────────────────
    genexp_data = load_gene_expression(args.genexp, max_rep)
    if genexp_data:
        logger.info(f"  Loaded GeneExp: {len(genexp_data)} genes")
    else:
        logger.info("  No GeneExp provided — using genome+GTF only (Mode A)")

    # ── Dataset ─────────────────────────────────────────────────────────
    ds = PredictDataset(gm, gi, data_cfg, k=k,
                        max_replicates=max_rep,
                        genexp_data=genexp_data)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=collate_predict,
                        num_workers=args.num_workers)

    # ── Model ───────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = PanCircModel(config["model"])
    # Report missing/unexpected keys explicitly (don't silently swallow)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        miss_ok = {"masked_kmer_head.weight", "masked_kmer_head.bias",
                   "junction_head.score_net.0.weight", "junction_head.score_net.0.bias",
                   "junction_head.score_net.3.weight", "junction_head.score_net.3.bias",
                   "junction_head.junction_proj.weight", "junction_head.junction_proj.bias"}
        critical = [k for k in missing if k not in miss_ok]
        if critical:
            logger.warning(f"  Missing keys (may affect prediction): {critical}")
        optional = [k for k in missing if k in miss_ok]
        if optional:
            logger.info(f"  Missing optional keys (inference not affected): {optional}")
    if unexpected:
        logger.info(f"  Unexpected keys: {len(unexpected)}")
    model.to(device)
    model.eval()
    total, trainable = count_parameters(model)
    logger.info(f"  {total:,} total parameters, {trainable:,} trainable")

    # Warn about species embedding
    if args.strain_id == 0:
        logger.info("  Species embedding: using index 0 (default). "
                    "Use --strain-id if you know the group.")

    # ── Predict ─────────────────────────────────────────────────────────
    logger.info("Running prediction...")
    t0 = time.time()
    results = []
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

            # Ensure strain_id uses the user-specified index
            b["strain_id"] = torch.full(
                (len(next(iter(v for v in b.values() if isinstance(v, torch.Tensor)))),),
                args.strain_id, dtype=torch.long, device=device
            )

            # Try finetune mode (with GeneExp); if genexp not provided, fallback
            if genexp_data:
                outputs = model(b, task="finetune")
            else:
                outputs = model(b, task="pretrain")

            probs = torch.sigmoid(outputs["gene_logits"].squeeze(-1)).cpu().numpy()

            for i in range(len(probs)):
                results.append({
                    "gene_id": batch["gene_id"][i],
                    "chrom": batch["chrom"][i],
                    "start": batch["start"][i].item() if isinstance(batch["start"][i], torch.Tensor) else batch["start"][i],
                    "end": batch["end"][i].item() if isinstance(batch["end"][i], torch.Tensor) else batch["end"][i],
                    "strand": batch["strand"][i],
                    "p_circ": f"{probs[i]:.6f}",
                    "n_exons": batch["n_exons"][i].item() if isinstance(batch["n_exons"][i], torch.Tensor) else batch["n_exons"][i],
                })

    elapsed = time.time() - t0
    logger.info(f"  Predicted {len(results)} genes in {elapsed:.1f}s "
                f"({len(results)/max(elapsed, 0.01):.0f} genes/s)")

    # ── Write output ────────────────────────────────────────────────────
    header = ["gene_id", "chrom", "start", "end", "strand", "p_circ", "n_exons"]
    with open(args.output, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in results:
            row = [str(r[col]) for col in header]
            f.write("\t".join(row) + "\n")

    logger.info(f"Predictions saved to {args.output}")

    # Summary stats
    probs = np.array([float(r["p_circ"]) for r in results])
    logger.info(f"  Probability distribution: min={probs.min():.4f}, "
                f"max={probs.max():.4f}, mean={probs.mean():.4f}, "
                f"median={np.median(probs):.4f}")
    n_high = (probs >= 0.5).sum()
    logger.info(f"  Genes predicted to produce circRNA (p>=0.5): {n_high}/{len(results)}")

    # Cleanup
    gi.close()


if __name__ == "__main__":
    main()
