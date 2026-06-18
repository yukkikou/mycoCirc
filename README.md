# 🍄 mycoCirc — Pan-Fungi CircRNA Foundation Model

> *myco* (fungus) + *Circ* (circular RNA) — a multi-modal foundation model for end-to-end circRNA prediction from fungal genomes.

**Author:** Yukkikou ([xueyanhu@pku.edu.cn](mailto:xueyanhu@pku.edu.cn))

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()

**mycoCirc** (previously PanCirc-Fungi) is a multi-modal deep learning model that predicts **which genes produce circular RNAs (circRNAs)** and **which backsplice junction is most likely**, directly from a fungal genome sequence and gene annotation.

Unlike existing single-modal methods (JEDI, CircPCBL), mycoCirc integrates **five distinct modalities** during pre-training and fine-tuning, achieving superior cross-species generalization across 22 fungal strains from *Candida*, *Cryptococcus*, and *Filamentous* groups.

---

## Architecture Overview

```
Gene-level input:
  GTF  (gene structure)  ──→ GTFEncoder ──┐
  Species ID             ──→ SpeciesEmb ───┤
  Genome window (10 kb)  ──→ GenomicCtxEnc ─┼─→ FusionModule ─→ GeneHead → P(circRNA|gene)
  [FT] CircExp           ──→ ExpEncoder ───┘
  [FT] GeneExp           ──→ ExpEncoder ───┘

Junction-level (per gene):
  All exon 5' boundaries  ──→ k-mer embed → BiGRU → k-mer attn → donor vectors (N_d)
  All exon 3' boundaries  ──→ k-mer embed → BiGRU → k-mer attn → acceptor vectors (N_a)
                                    ↓
                    Bidirectional cross-attention (JEDI core)
                          (donor↔acceptor pairwise scores)
                                    ↓
                    Final attention pooling → junction vector → fusion
```

### Components

| Component | Dim. | Description |
|-----------|------|-------------|
| **GenomicContextEncoder** | 128 | Dilated CNN + BiGRU on GC/k-mer profile bins of the ±5 kb gene window |
| **GTFEncoder** | 128 | MLP on 17 gene-structure features (exon count, intron length, CDS, etc.) |
| **JunctionEncoder** | 64 | JEDI (BiGRU + cross-attention) + CircPCBL (CNN-BiGRU + GLT) hybrid |
| **SpeciesEmbedding** | 32 | Phylogenetic PCA + learnable embedding (21 species) |
| **ExpressionEncoder** | 64 | log1p + z-score of CircExp & GeneExp (fine-tuning only) |
| **FusionModule** | 128 | Concat-MLP: project each modality, concatenate, refine |
| **GeneHead** | → 1 | Linear → sigmoid → P(circRNA\|gene) |
| **JunctionHead** | → scores | MLP scoring of candidate exon pairs (backup ranking) |

**Total parameters:** 775,858

---

## Training Protocol

### Two-Stage Pre-training

| Stage | Epochs | Trainable Modules | Loss |
|-------|--------|-------------------|------|
| **Stage 1** | 50 | GTFEncoder + GenomicCtxEncoder + Fusion + GeneHead | BCE(gene) |
| **Stage 2** | 100 | All (unfreeze JunctionEncoder) | BCE(gene) + BCE(junction) |

### Fine-tuning (per taxonomic group)

- **5-fold cross-validation** by strain: train on N-1 strains, validate on 1
- Best fold model → evaluate on held-out test strain
- Two modes:
  - **Mode A** (Genome+GTF): no expression data needed
  - **Mode B** (+GeneExp): with host gene expression

### Test Strains

| Group | Training Strains | Held-out Test |
|-------|-----------------|---------------|
| **Candida** | *C. albicans*, *C. tropicalis*, *C. glabrata* (×2), *P. kudriavzevii*, *S. cerevisiae* | **_C. auris_** |
| **Cryptococcus** | *C. neoformans* var. *grubii* (×2), *C. floricola*, *C. gattii* (×2), *C. laurentii* | **_C. neoformans_ var. _neoformans_** |
| **Filamentous** | *F. proliferatum*, *F. dimerum*, *N. crassa*, *A. fumigatus*, *P. chrysogenum* | **_F. venenatum_** |

---

## Results Summary

### Cross-species circRNA Prediction (AUROC)

| Method | Candida | Cryptococcus | Filamentous |
|--------|:-------:|:------------:|:-----------:|
| **mycoCirc** (Genome+GTF) | **0.6985** | **0.6902** | **0.6976** |
| mycoCirc (+GeneExp) | 0.4526 | 0.5698 | 0.7227 |
| From-scratch (no pretrain) | 0.4992 | 0.5411 | 0.5053 |
| JEDI | 0.5057 | 0.5341 | 0.5656 |
| CircPCBL | 0.5329 | 0.4925 | 0.4927 |

### Key Findings

- **Pre-training is essential**: from-scratch collapses to ~random (AUROC ~0.50–0.54)
- **GTFEncoder is the dominant modality**: removing it drops AUROC by 0.06–0.14
- **JunctionEncoder contributes modestly**: consistent +0.01–0.04 AUROC across groups
- **Expression helps within-group but not cross-species**: Mode B inconsistent
- **Cross-species generalization** to unseen *Talaromyces marneffei* (PM1): AUROC 0.6955

### Junction Prediction Accuracy

*(Results from `results/interpretability/junction_topk.tsv` — see below)*

---

## Quick Start for Users

> **只想用 mycoCirc 做预测？** 看这一步就够了。

```bash
# 1. 下载预训练权重（以 Filamentous 为例）
wget https://github.com/yukkikou/mycoCirc/releases/latest/download/mycoCirc_filamentous.pt

# 2. 运行预测（需要 genome.fa + annotation.gtf）
python scripts/predict.py \
    --genome genome.fa \
    --gtf annotation.gtf \
    --checkpoint mycoCirc_filamentous.pt \
    --config config/default.yaml \
    --output predictions.tsv

# 3. 查看结果（p_circ ≥ 0.5 为预测阳性）
head predictions.tsv
```

📖 **详细教程见 [QUICKSTART.md](QUICKSTART.md)** — 包含安装步骤、权重选择、参数说明、常见问题。

---

## Architecture Overview

### Requirements

```
Python  ≥3.9
PyTorch ≥2.0 (CUDA recommended)
NumPy, Pandas, scikit-learn
Biopython, pyfaidx
matplotlib, seaborn
tqdm, pyyaml, scipy
```

### 1. Prepare Data

Ensure `all_lib_model_full.tsv` is in the project root with correct paths to:
- Reference genome FASTA files
- Gene annotation GTF files
- CircRNA metadata CSV (from CIRIquant or similar)
- Optional: CircExp BSJ count and GeneExp count matrices

```bash
# Validate the TSV
python scripts/parse_tsv.py
```

### 2. Pre-train

```bash
# Single GPU
python train/pretrain.py --config config/default.yaml

# Multi-GPU
accelerate launch train/pretrain.py --config config/default.yaml
```

### 3. Fine-tune

```bash
# Fine-tune with 5-fold CV for a specific group
python train/finetune.py --group Candida \
    --pretrained checkpoints/pretrain/best.pt \
    --config config/default.yaml

# Fine-tune without pre-training (from scratch)
python train/finetune.py --group Candida \
    --from-scratch --config config/default.yaml
```

### 4. Evaluate

```bash
# Evaluate on held-out test strain
python scripts/evaluate_junction.py \
    --config config/default.yaml \
    --checkpoint-dir checkpoints/finetune

# Test on completely unseen species
python scripts/test_pm1.py \
    --config config/default.yaml \
    --checkpoint-dir checkpoints/finetune
```

### 5. Interpretability

```bash
# Run full interpretability pipeline
python scripts/run_interpretability.py \
    --config config/default.yaml \
    --checkpoint-dir checkpoints/finetune \
    --fig-dir figures
```

### SLURM

```bash
sbatch scripts/run_pretrain.slurm
sbatch --array=1-3 scripts/run_finetune.slurm
sbatch scripts/run_interpretability.slurm
```

---

## Project Structure

```
4_model/
├── README.md                          ← this file
├── CLAUDE.md                          ← dev instructions
├── all_lib_model_full.tsv             ← strain registry (read-only)
├── config/
│   └── default.yaml                   ← model & training config
├── data/                              ← data loading & preprocessing
│   ├── dataset.py                     ← PyTorch Dataset classes
│   ├── genome_encoding.py             ← genome context features
│   ├── gtf_encoding.py                ← GTF/gene structure features
│   ├── circ_info_encoding.py          ← circRNA metadata
│   ├── expression_encoding.py         ← expression data
│   ├── negative_sampling.py           ← negative gene sampling
│   └── tsv_parser.py                  ← TSV registry parser
├── model/                             ← model architecture
│   ├── pancirc.py                     ← PanCircModel (main entry point)
│   ├── layers.py                      ← shared layers
│   ├── genome_encoder.py              ← GenomicContextEncoder
│   ├── gtf_encoder.py                 ← GTFEncoder
│   ├── junction_encoder.py            ← JunctionEncoder (JEDI+CircPCBL)
│   ├── species_embedding.py           ← SpeciesEmbedding
│   ├── expression_encoder.py          ← ExpressionEncoder
│   ├── fusion.py                      ← FusionModule (Concat-MLP)
│   └── prediction_head.py             ← GeneHead + JunctionHead
├── train/                             ← training code
│   ├── pretrain.py                    ← two-stage pre-training
│   ├── finetune.py                    ← fine-tuning with 5-fold CV
│   └── trainer.py                     ← training loop utilities
├── scripts/                           ← utility scripts
│   ├── evaluate.py                    ← evaluation scaffold
│   ├── evaluate_junction.py           ← junction prediction accuracy
│   ├── run_interpretability.py        ← interpretability pipeline
│   ├── test_pm1.py                    ← cross-species test (PM1)
│   ├── run_ablations.py               ← component ablation
│   ├── run_delong_test.py             ← statistical significance
│   ├── visualize_comparison.py        ← publication figure
│   └── *.slurm                        ← SLURM submission scripts
├── interpret/                         ← interpretability modules
│   ├── attention_viz.py               ← modality ablation + cross-attention plots
│   ├── feature_importance.py          ← Integrated Gradients for GTF
│   ├── junction_heatmap.py            ← k-mer attention heatmaps
│   ├── motif_discovery.py             ← sequence motif enrichment
│   └── genome_scan.py                 ← genome-wide prediction tracks
├── results/                           ← result tables (see below)
├── figures/                           ← publication figures
├── docs/
│   └── model_architecture.md          ← full architecture documentation
├── checkpoints/                       ← model weights (gitignored)
└── logs/                              ← training logs (gitignored)
```

---

## Results Files

| File | Description |
|------|-------------|
| `results/comparison_model_benchmark.tsv` | All methods × all groups |
| `results/ablations.tsv` | Component ablation (AUROC drop) |
| `results/comparison_ablation.tsv` | Expression ablation (CV AUROC) |
| `results/comparison_fromscratch_full.tsv` | From-scratch vs pretrained |
| `results/significance_tests.tsv` | DeLong statistical tests |
| `results/hyperparameter_sweep_summary.tsv` | Hyperparameter sensitivity |
| `results/pm1_cross_species_test.tsv` | PM1 cross-species generalization |
| `results/interpretability/junction_topk.tsv` | Junction prediction accuracy |
| `results/interpretability/feature_importance.tsv` | GTF feature importance |
| `results/interpretability/motif_enrichment.tsv` | Sequence motif enrichment |

---

## Comparison with Existing Methods

| Feature | mycoCirc | JEDI | CircPCBL |
|---------|----------|------|----------|
| Multi-modal | ✅ (5 modalities) | ❌ (sequence only) | ❌ (sequence only) |
| Cross-species | ✅ (22 fungi strains) | ❌ (species-specific) | ❌ (plant-trained) |
| Pre-training | ✅ (two-stage) | ❌ | ❌ |
| Junction ranking | ✅ (cross-attention) | ✅ | ❌ |
| Interpretability | ✅ (IG + attention) | ❌ | ❌ |
| Open-source | ✅ | ✅ | ✅ |

---

## Citation

If you use mycoCirc in your research, please cite the repository:

```
@software{mycoCirc2026,
  title = {mycoCirc: A Multi-modal Foundation Model for Pan-Fungal circRNA Prediction},
  year = {2026},
  url = {https://github.com/user/pancirc-fungi}
}
```

---

## License

This project is for academic research purposes. See `LICENSE` for details.
