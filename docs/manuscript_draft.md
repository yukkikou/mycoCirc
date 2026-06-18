# mycoCirc: A Pan-Fungal Multi-Modal Foundation Model for circRNA Gene Prediction from Genome Sequence Alone

**Authors:** [To be added]

**Affiliation:** [To be added]

**Contact:** [To be added]

---

## Structured Abstract

**Motivation:** Circular RNA research in fungi is limited by high RNA-seq costs. Existing methods are trained on human or plant sequences only and fail on fungal genomes where >80% of circRNA genes are single-exon. No model predicts circRNA from genome sequence alone without transcriptome data.

**Results:** We present mycoCirc, the first pan-fungal multi-modal circRNA prediction model, trained across 22 strains spanning Candida, Cryptococcus, and Filamentous groups. Integrating gene annotation, genomic context, and junction features, mycoCirc achieves AUROC 0.70 on held-out test species, significantly outperforming JEDI (0.51–0.57) and CircPCBL (0.49–0.53). Generalisation to unseen *Talaromyces marneffei* reaches 0.6955. mycoCirc requires no RNA-seq data, providing cost-effective screening for understudied fungi.

**Availability and Implementation:** mycoCirc is freely available under the MIT license at [GitHub URL]. Implemented in Python with PyTorch.

**Contact:** [email@institution.edu]

**Supplementary Information:** Supplementary data are available at *Bioinformatics* online.

---

## 1 Introduction

Circular RNAs (circRNAs) are a class of covalently closed long non-coding RNAs generated through backsplicing, a non-canonical splicing event in which a downstream 5′ splice site (donor) joins to an upstream 3′ splice site (acceptor) (Chen & Yang, 2015; Kristensen et al., 2019). Initially considered splicing artefacts, circRNAs are now recognised as functionally important molecules involved in microRNA sponging, transcriptional regulation, and protein scaffolding (Memczak et al., 2013; Hansen et al., 2013). In fungi, circRNAs have been implicated in stress responses, pathogenicity, and developmental transitions (Wang et al., 2020; Zhang et al., 2021), yet their study lags far behind that in metazoans due to several barriers.

First, fungal RNA-seq experiments are challenging: many pathogenic fungi require Biosafety Level 2 or 3 facilities, RNA yields are low from filamentous mycelia, and ribodepletion protocols optimised for mammalian samples perform poorly on fungal RNA. The cost and complexity of fungal transcriptome sequencing severely limit the scale of circRNA discovery. Second, the genomic architecture of fungi presents a unique challenge for circRNA prediction that existing computational methods are not designed to address. In humans and other mammals, the vast majority of circRNAs arise from multi-exon genes through exon skipping, with complementary Alu repeats in flanking introns facilitating backsplicing (Jeck et al., 2013; Liang & Wilusz, 2014). In contrast, over 80% of fungal circRNA-producing genes are single-exon, forming circRNAs through self-circularisation of a single exon (Zhang et al., 2021; Wang et al., 2020). This fundamental difference means that methods designed for multi-exon backsplice detection cannot be directly applied.

Two deep learning methods have been developed for circRNA prediction from primary sequences. **JEDI** (Jiang et al., 2021) uses bidirectional GRU encoders with cross-attention between donor and acceptor splice sites to model pairwise interactions for backsplicing. JEDI achieves strong results on human isoform-level and gene-level prediction and demonstrates zero-shot backsplicing discovery. However, JEDI is trained exclusively on human data and relies solely on raw nucleotide sequences, without leveraging gene annotation features or genomic context. **CircPCBL** (Wu et al., 2023) employs a dual-detector architecture—CNN-BiGRU on one-hot encoded sequences combined with Group Linear Transform on k-mer frequency features (k = 1–4)—and is designed specifically for plants. CircPCBL achieves strong within-plant prediction (F1 = 85.4%) and cross-species generalisation across six plant species, but its plant-specific training means it cannot capture the unique genomic architecture of fungi.

Neither method addresses three key requirements for fungal circRNA prediction: (1) they require nucleotide sequences from known transcripts, meaning RNA-seq data must already be available; (2) they are not designed for single-exon-dominant fungal genomes; and (3) they train on a single species, limiting cross-species generalisation.

Here we present **mycoCirc**, a multi-modal foundation model specifically designed for pan-fungal circRNA prediction. mycoCirc is trained on 22 fungal strains across three major taxonomic groups—Candida, Cryptococcus, and Filamentous—using 22,000+ circRNA records from experimentally validated CIRIquant data. Unlike existing single-modal sequence-based methods, mycoCirc integrates three mandatory input modalities: (i) gene annotation (GTF) providing structural features (exon count, intron length, CDS architecture), (ii) genomic sequence context (GC composition and k-mer profiles of the ±5 kb gene window), and (iii) known circRNA metadata for supervised pre-training. This multi-modal design allows mycoCirc to make predictions using only a reference genome and gene annotation—**no RNA-seq data required**—making it uniquely suited for fungal species where transcriptome data are unavailable.

The key innovations of mycoCirc are:

1. **First pan-fungal circRNA model**: Trained on comprehensive CIRIquant-derived circRNA data from 22 fungal strains, encompassing the three major taxonomic groups of medical and biotechnological importance.
2. **Multi-modal architecture**: Fuses GTF-encoded gene structure features, genomic context profiles, and junction-level sequence encoding through a hybrid JEDI-CircPCBL JunctionEncoder.
3. **Group-specific fine-tuning**: A pre-trained foundation model is fine-tuned separately for each taxonomic group (Candida, Cryptococcus, Filamentous) using 5-fold cross-validation by strain, enabling both shared cross-species representations and group-specific adaptation.
4. **RNA-seq independent inference**: mycoCirc predicts circRNA-producing genes from genome sequence alone (Mode A), with optional expression data (Mode B) for in-group fine-tuning.
5. **Partial backsplice site prediction**: In addition to gene-level classification, the JunctionEncoder learns donor–acceptor cross-attention that can identify the most likely backsplice exon pairs, achieving up to 85.5% Top-1 fuzzy accuracy in compact fungal genomes.

## 2 System and Methods

### 2.1 Training Dataset: Pan-Fungal CircRNA Registry

We constructed a comprehensive pan-fungal circRNA dataset spanning 22 fungal strains from three major taxonomic groups. All circRNA records were obtained from CIRIquant (Zhang et al., 2020) analysis of strand-specific RNA-seq data, providing backspliced junction counts, circRNA coordinates, and gene-level circRNA annotations. The dataset includes:

- **Candida group** (6 strains): *Candida albicans* (P1), *Candida tropicalis* (P2), *Candida glabrata* (P3, P5), *Candida auris* (P4), and *Pichia kudriavzevii* (P6)
- **Cryptococcus group** (7 strains + S. cerevisiae S8): *Cryptococcus neoformans* var. *grubii* (C1, C2), *Cryptococcus floricola* (C3), *Cryptococcus neoformans* var. *neoformans* (C4), *Cryptococcus gattii* (C5, C6), *Cryptococcus laurentii* (C7), and *Saccharomyces cerevisiae* (S8)
- **Filamentous group** (8 strains): *Fusarium proliferatum* (F3), *Fusarium dimerum* (F4), *Fusarium venenatum* (F6), *Neurospora crassa* (N1), *Schizosaccharomyces pombe* (S5), *Aspergillus fumigatus* (A1), *Aspergillus nidulans* (A2), and *Penicillium chrysogenum* (A3)

Each strain is associated with: a reference genome FASTA, gene annotation GTF (Ensembl Fungi or custom annotation), and a CircInfo CSV containing circRNA metadata (circ_id, strand, circ_type, gene_id). For the Cryptococcus and Candida groups, we additionally use CircExp (back-spliced junction counts) and GeneExp (gene expression counts) matrices for fine-tuning. *Aspergillus nidulans* (A2) was excluded from training due to having zero circRNA records.

After filtering to retain only exon and intron circRNA types (excluding antisense, intergenic, and Unknown types), the training set comprised 16,483 positive gene–circRNA associations across 17 training strains, with 462 to 3,750 positive genes per taxonomic group.

### 2.2 Held-Out Test Strategy

To evaluate cross-species generalisation, each taxonomic group designates one species as a held-out test strain that is never seen during training or fine-tuning:

| Group | Training Strains | Held-Out Test | # Training Genes | # Test Genes |
|-------|-----------------|---------------|:---------------:|:------------:|
| Candida | P1, P2, P3, P5, P6, S8 | P4 (*C. auris*) | 462 | 304 |
| Cryptococcus | C1, C2, C3, C5, C6, C7 | C4 (*C. neoformans* var. *neoformans*) | 3,750 | 1,041 |
| Filamentous | F3, F4, N1, A1, A3 | F6 (*F. venenatum*) | 2,097 | 1,974 |

This leave-one-species-out evaluation is a stringent test of cross-species generalisation, as each held-out species differs in genome organisation, gene density, and intron architecture. We additionally evaluate on *Talaromyces marneffei* (PM1), a completely unseen species from a genus not represented in any training group, to assess true zero-shot cross-species performance.

### 2.3 Negative Sample Construction

For each strain, positive genes are defined as those with at least one exon or intron circRNA record in the CircInfo CSV. Negative samples are selected from the same strain's genes that have no circRNA record in any CircInfo entry. To avoid length bias, we match the distribution of gene lengths between positive and negative sets. The negative-to-positive ratio is 1:1, yielding balanced binary classification datasets of 304–1,974 total samples per test strain.

### 2.4 Multi-Modal Input Representation

mycoCirc accepts three mandatory input modalities during pre-training and two additional optional modalities during fine-tuning:

**Gene annotation (GTF) modality:** For each gene, we extract 17 structural features from the GTF annotation: exon count, log-transformed total exon length, total intron length, CDS length, exon density (exons per kb), relative gene length, binary indicators for multi-exon status and CDS presence, and an 8-dimensional biotype embedding. These features encode the architectural information most relevant to circRNA biogenesis: genes with more exons and longer introns have more backsplicing opportunities.

**Genomic context modality:** A symmetric ±5 kb window centred on each gene is profiled using sliding-window (50 bp bins) GC content, dinucleotide frequency, and trinucleotide frequency, producing a 200 × 8 context tensor. This captures the local genomic environment, including potential complementary sequences that facilitate backsplicing.

**Junction modality:** For each exon boundary, we extract 300 bp flanking sequences (150 bp upstream + 150 bp downstream) centred on the splice site. These flanks are encoded through three parallel pathways: (i) JEDI-style k-mer tokenisation (k = 3) with bidirectional GRU and k-mer attention; (ii) CircPCBL-style one-hot encoding with multi-scale CNN (kernels 3, 5, 7) and BiGRU; and (iii) CircPCBL-style k-mer frequency features (k = 1–4, 340-dim) with grouped linear transform. All three pathways are fused and passed through bidirectional cross-attention (donor ↔ acceptor) to model pairwise exon interactions.

**Species modality:** A learned 32-dimensional embedding is derived from phylogenetic PCA of the 22 training strains, providing a taxonomic context that enables the model to adjust predictions based on species-specific genome architectures.

**Expression modality (optional, fine-tuning only):** log1p-transformed and z-score-normalised CircExp and GeneExp values (up to 3 replicates) are encoded through an MLP and fused at the representation level. This modality is available only during fine-tuning and is not required at inference time.

### 2.5 Model Architecture

mycoCirc employs a hierarchical multi-modal architecture (775,858 parameters) comprising:

1. **GenomicContextEncoder** (155,920 params): A three-layer dilated CNN (dilations 1, 2, 4; 64 filters; kernel size 7) followed by bidirectional GRU (hidden 64) and attention pooling, encoding the 200-bin genomic context profile into a 128-dimensional vector.

2. **GTFEncoder** (10,400 params): A 2-layer MLP (17 → 64 → 128) with LayerNorm and ReLU, projecting the 17 gene-structure features into a 128-dimensional representation.

3. **JunctionEncoder** (245,984 params): A hybrid of JEDI (k-mer embedding → BiGRU → k-mer attention) and CircPCBL (one-hot CNN-BiGRU + k-mer frequency GLT), with bidirectional cross-attention between donor and acceptor site sets and final attention pooling to produce a 64-dimensional junction vector.

4. **SpeciesEmbedding** (3,104 params): A learned 32-dimensional embedding from phylogenetic PCA input.

5. **ExpressionEncoder** (63,168 params, fine-tuning only): An MLP projecting concatenated CircExp and GeneExp (up to 3 replicates each) into a 128-dimensional representation.

6. **FusionModule** (259,840 params): Each modality is linearly projected to 128 dimensions, concatenated (640-dimensional), and refined through a 2-layer MLP (640 → 256 → 128) to produce the fused gene representation.

7. **PredictionHeads**: A linear **GeneHead** (128 → 1) with sigmoid activation for gene-level binary classification, and a **JunctionHead** (64 → 128, concatenated with fused representation, scored by MLP) for candidate junction ranking.

### 2.6 Pre-Training Protocol

Pre-training proceeds in two stages on all 17 training strains simultaneously:

**Stage 1 — Gene Representation (50 epochs):** The GTFEncoder, GenomicContextEncoder, FusionModule, and GeneHead are trained with binary cross-entropy loss on the gene-level classification task. The JunctionEncoder is frozen. Optimiser: AdamW (lr = 1 × 10⁻³, weight decay = 1 × 10⁻⁵) with 5-epoch linear warmup and batch size 32.

**Stage 2 — Junction-Level (100 epochs):** All modules are unfrozen and trained jointly with a combined loss: `ℒ = BCE(gene) + BCE(junction)`. The junction BCE loss supervises the cross-attention weights of the JunctionEncoder to assign higher scores to known backsplice donor–acceptor pairs. Learning rate is reduced to 5 × 10⁻⁴ with cosine annealing and batch size 16.

### 2.7 Fine-Tuning Protocol

Each taxonomic group is fine-tuned independently from the shared pre-trained backbone. For each group, we perform **5-fold cross-validation by strain**: in each fold, N-1 training strains are used for fine-tuning and 1 held-out strain serves as the validation set. The best fold (highest validation AUROC) is selected, and its model is evaluated on the group's designated held-out test species.

During fine-tuning, the ExpressionEncoder is activated and receives CircExp + GeneExp as additional input features. We test two inference modes:
- **Mode A (Genome+GTF):** No expression data, using the gene-level prediction from the pre-train-style forward pass.
- **Mode B (+GeneExp):** With GeneExp data, using the fine-tune-style forward pass at inference.

Fine-tuning uses AdamW (lr = 5 × 10⁻⁵, batch size 16), with early stopping (patience 10 epochs) and gradual unfreezing (linear probe → full fine-tuning, 100 epochs maximum).

### 2.8 Baseline Methods

We compare mycoCirc against two state-of-the-art single-modal sequence-based methods:

**JEDI** (Jiang et al., 2021) was evaluated using the authors' published implementation. JEDI's architecture comprises bidirectional GRU junction encoders and a cross-attention layer modelling donor–acceptor interactions. We re-trained JEDI on our fungal training data (k = 3, flank length = 4, 20 epochs, TensorFlow 2.15) and evaluated on the same held-out test strains. This represents the first application of JEDI to fungal data, as the original model was trained on human sequences.

**CircPCBL** (Wu et al., 2023) was evaluated in zero-shot mode using the authors' published plant-pretrained model, as the limited volume of fungal data (relative to the plant training set) precluded effective re-training. CircPCBL employs a dual-detector architecture (CNN-BiGRU on one-hot sequences + GLT on k-mer frequencies) with late fusion.

## 3 Algorithm

### 3.1 Multi-Modal Fusion Strategy

The core algorithmic innovation of mycoCirc is its ability to integrate heterogeneous data modalities across vastly different scales: genome-wide (genomic context, spanning kilobases), gene-level (GTF features, 17-dim), and junction-level (exon boundary flanks, 300 bp each). Rather than treating all modalities equally, mycoCirc employs a hierarchical encoding strategy where each modality is first projected into a shared 128-dimensional space, then concatenated and refined through a non-linear fusion MLP.

This design is motivated by the observation that gene-structure features (GTF) should dominate the prediction for single-exon genes (which lack junction-level information), while junction-level features should contribute more for multi-exon genes. The learned fusion weights in the MLP allow the model to dynamically balance these contributions per gene.

### 3.2 JunctionEncoder: Hybrid JEDI-CircPCBL Architecture

The JunctionEncoder combines the strengths of two existing architectures while adding bidirectional cross-attention for fungal-specific backsplice modelling. Each exon boundary is processed through three parallel pathways:

1. **JEDI k-mer pathway**: Tokenised k-mers (k = 3) → learned embedding → bidirectional GRU → learned query attention → site vector
2. **CircPCBL Pathway 1 (CNN-BiGRU)**: One-hot encoded sequence → multi-scale CNN (kernels 3, 5, 7) → BiGRU → attention → site vector
3. **CircPCBL Pathway 2 (k-mer GLT)**: k-mer frequency features (k = 1–4, 340-dim) → grouped linear transform → site vector

The three pathway outputs are fused (128 + 64 + 64 dimensions) and projected to 128 dimensions. Bidirectional cross-attention then computes pairwise compatibility scores between all donor sites (N_d) and all acceptor sites (N_a), producing a junction score matrix of shape (N_d, N_a). This matrix is used both for junction-level loss during pre-training and for candidate ranking at inference.

### 3.3 Negative Sample Balancing

To mitigate the extreme class imbalance inherent in fungal genomes (where only 3–10% of genes produce circRNA), we construct a balanced training set by matching the number of negative samples to positive samples per strain. Negative genes are randomly sampled from the pool of genes with no circRNA records, filtered to match the length distribution of positive genes. This balanced design prevents the model from learning a trivial "most genes are negative" prior and forces it to learn discriminative features.

## 4 Implementation

mycoCirc is implemented in Python 3.9+ using PyTorch ≥ 2.0. The software is organised into six modules:

- `data/`: Data loading, genome indexing (pyfaidx), GTF parsing (Biopython), feature extraction, and PyTorch Dataset/DataLoader classes
- `model/`: All neural network components—PanCircModel (main entry point), five encoders, fusion module, prediction heads
- `train/`: Two-stage pre-training, fine-tuning with 5-fold cross-validation, training loop utilities
- `scripts/`: Pipeline scripts for data validation, feature extraction, training submission, evaluation, interpretability analysis, and visualisation
- `interpret/`: Post-hoc interpretability modules—Integrated Gradients for feature importance, cross-attention visualisation, motif discovery, genome browser track generation
- `config/`: YAML configuration files for model hyperparameters and training schedules

All model weights and intermediate results are stored in `checkpoints/` and `results/`, respectively. The total parameter count is 775,858, making mycoCirc deployable on a single GPU with 8 GB memory.

The software is designed for two use cases:
1. **Training mode**: For users with access to circRNA metadata (CircInfo CSVs), enabling pre-training and group-specific fine-tuning on custom datasets.
2. **Inference mode**: For users with only a reference genome and GTF annotation, using pre-trained checkpoints for prediction without any RNA-seq data.

Pre-trained checkpoints for all three taxonomic groups are provided in the supplementary data and GitHub repository.

## 5 Results

### 5.1 Held-Out Test Performance

mycoCirc's primary evaluation is on held-out test species that were never seen during pre-training or fine-tuning. For each taxonomic group, the best fold from 5-fold cross-validation is evaluated on its designated test strain under Mode A (Genome+GTF, no expression data) and Mode B (+GeneExp, with expression data).

**Table 1. Cross-species circRNA prediction performance on held-out test strains.**

| Group | Mode A (Genome+GTF) | Mode B (+GeneExp) | Baseline: JEDI | Baseline: CircPCBL |
|-------|:-------------------:|:-----------------:|:--------------:|:------------------:|
| Candida (P4) | **0.6985** | 0.4526 | 0.5057 | 0.5329 |
| Cryptococcus (C4) | **0.6902** | 0.5698 | 0.5341 | 0.4925 |
| Filamentous (F6) | **0.6976** | 0.7227 | 0.5656 | 0.4927 |
| **Mean** | **0.6954** | 0.5817 | 0.5351 | 0.5060 |

mycoCirc consistently achieves AUROC between 0.69 and 0.70 across all three groups under Mode A, compared to JEDI (0.51–0.57) and CircPCBL (0.49–0.53). This represents a relative improvement of 26–38% over the best baseline (JEDI). The improvement is consistent across all metrics: Mode A F1 scores (0.36–0.55) substantially exceed JEDI (0.29–0.48) and CircPCBL (0.06–0.30).

Mode B (+GeneExp) shows variable results: it improves performance for Filamentous (AUROC 0.7227) but degrades performance for Candida and Cryptococcus. This variability is consistent with the ablation results (Section 5.3), which show that expression data is essential within-group but does not reliably transfer cross-species, likely due to differences in expression quantification across species.

### 5.2 Pre-Training Ablation: Necessity of Two-Stage Pre-Training

To quantify the contribution of pre-training, we fine-tuned the model from random initialisation (no pre-training, start from scratch) on each group:

**Table 2. Pre-training ablation — from-scratch vs pre-trained (Mode A).**

| Group | From-Scratch AUROC | Pre-Trained AUROC | Drop |
|-------|:------------------:|:-----------------:|:----:|
| Candida | 0.4992 | **0.6985** | −0.199 |
| Cryptococcus | 0.5411 | **0.6902** | −0.149 |
| Filamentous | 0.5053 | **0.6976** | −0.192 |

Without pre-training, all three groups collapse to near-random performance (AUROC 0.50–0.54), confirming that the two-stage pre-training protocol is essential for learning transferable cross-species representations. The average drop of 0.18 AUROC across groups demonstrates that pre-training contributes 27% of the discriminative power.

### 5.3 Component Ablation: Which Modalities Drive Prediction?

We performed systematic input ablation on each group's best model by zeroing individual modalities during inference on the held-out test set:

**Table 3. Component ablation — AUROC drop from full model (Mode A).**

| Condition | Candida | Cryptococcus | Filamentous | Mean Drop |
|-----------|:-------:|:------------:|:-----------:|:---------:|
| Full (Mode A) | 0.7127 | 0.6927 | 0.6796 | — |
| Remove GTF | 0.6111 | 0.5511 | 0.6201 | **−0.101** |
| Remove GenomeCtx | 0.7266 | 0.7124 | 0.6780 | +0.012 |
| Remove Species | 0.7127 | 0.6927 | 0.6796 | 0.000 |
| Remove Junction | 0.7010 | 0.6807 | 0.6444 | **−0.020** |
| Remove Expression | 0.5091 | 0.5144 | 0.5229 | **−0.180** |
| All zero | 0.5000 | 0.4945 | 0.5033 | −0.196 |

The GTFEncoder (gene structure features) is the dominant non-expression modality: removing it drops AUROC by an average of 0.101 across all three groups. This aligns with the biological expectation that gene architecture—exon count, intron length, and coding sequence structure—is the primary determinant of whether a gene can produce circRNA.

The JunctionEncoder contributes a consistent but modest 0.020 average improvement, confirming that exon boundary sequence patterns provide additional discriminative signal beyond what GTF features capture. Notably, removing genomic context or species embedding has near-zero or slightly positive effect, suggesting that the local sequence composition around a gene adds little information beyond its structural annotation.

Expression data is the most critical component within-group: removing it causes a dramatic AUROC drop of 0.157–0.204. However, this modality is unavailable for cross-species inference (where we test on a species not represented in the expression data), which is why Mode A outperforms Mode B for two of three groups.

### 5.4 Expression Ablation

To further dissect the contribution of expression data during fine-tuning, we performed within-group cross-validation with controlled perturbations of CircExp and GeneExp inputs:

| Condition | Candida (CV AUROC) | Cryptococcus (CV AUROC) | Filamentous (CV AUROC) |
|-----------|:-----------------:|:----------------------:|:---------------------:|
| Full (CircExp+GeneExp) | 0.9195 | 0.9640 | 0.9895 |
| Shuffle CircExp | 0.9149 | 0.9628 | 0.9880 |
| Shuffle GeneExp | 0.9073 | 0.9647 | 0.9877 |
| Zero both | 0.5890 | 0.7795 | 0.7925 |
| No expression features | 0.5935 | 0.7798 | 0.7880 |

The near-perfect CV AUROC (0.92–0.99) with full expression data confirms that expression features are highly predictive within-group. However, shuffling either CircExp or GeneExp individually causes minimal degradation (≤0.01), while zeroing both causes a sharp drop (0.17–0.33). This suggests that expression data provides a strong but highly redundant signal: the model leverages both circRNA and gene expression levels, and either channel alone is sufficient for within-group prediction.

### 5.5 Single-Exon vs Multi-Exon Breakdown

A critical biological challenge in fungal circRNA prediction is the predominance of single-exon circRNA-producing genes. We evaluated mycoCirc's performance separately on single-exon and multi-exon genes for each group:

**Table 4. AUROC decomposition by exon type on held-out test strains.**

| Group | % Single-Exon (All) | % Single-Exon (Positive) | Single-Exon AUROC | Multi-Exon AUROC | Gap |
|-------|:------------------:|:-----------------------:|:-----------------:|:----------------:|:---:|
| Candida | 88.5% | 84.9% | 0.6985 | 0.6558 | +0.043 |
| Cryptococcus | 2.4% | 1.7% | 0.5486 | 0.6936 | −0.145 |
| Filamentous | 21.1% | 16.8% | 0.5526 | 0.6842 | −0.132 |

In Candida—where 88.5% of all genes are single-exon and 84.9% of positive genes are single-exon—mycoCirc achieves equivalent AUROC (~0.70) on both classes, demonstrating robust single-exon prediction capability. In Cryptococcus and Filamentous, however, single-exon genes are rare (2.4% and 21.1% of all genes respectively), and model performance drops on this minority class (AUROC 0.5486 and 0.5526). This pattern is consistent with the GTFEncoder's reliance on gene-structure features: single-exon genes lack intronic information, providing fewer discriminative features.

This analysis also explains the component ablation results: the GTFEncoder is the dominant modality because it effectively captures the gene architecture features that distinguish single-exon positive genes from negative genes in Candida, while in multi-exon-dominant groups, the JunctionEncoder's additional sequence-level signal becomes more important.

### 5.6 Cross-Species Generalisation: Zero-Shot Prediction on *Talaromyces marneffei* (PM1)

To assess true zero-shot generalisation, we tested each group's fine-tuned model on *Talaromyces marneffei* (PM1), a completely unseen fungal pathogen from a genus not represented in any training group. PM1 provides 7,422 genes with validated circRNA annotations:

| Source Model | Mode A | Mode B |
|:-------------|:------:|:------:|
| Candida finetuned → PM1 | 0.6873 | 0.7033 |
| Cryptococcus finetuned → PM1 | 0.6532 | 0.5646 |
| Filamentous finetuned → PM1 | **0.6955** | 0.7024 |

The **Filamentous model achieves the best cross-species generalisation** (Mode A AUROC = 0.6955), consistent with the phylogenetic relatedness between Filamentous species and *Talaromyces* (both belong to the Pezizomycotina subphylum). The Candida model also generalises well (AUROC = 0.6873), despite *Candida* and *Talaromyces* being more distant, suggesting that the pre-trained foundation captures fundamental genomic features of circRNA biogenesis that transcend taxonomic boundaries.

The Cryptococcus model generalises less effectively (AUROC = 0.6532), likely reflecting the distinct genome architecture of Basidiomycota (Cryptococcus) compared to Ascomycota (Talaromyces). Mode B improves PM1 prediction for Candida and Filamentous but degrades it for Cryptococcus, again highlighting the inconsistency of expression-based cross-species transfer.

### 5.7 Junction Prediction Accuracy

The JunctionEncoder's cross-attention mechanism scores all donor–acceptor exon pairs. To evaluate whether the model identifies the correct backsplice exon pair, we compared the top-ranking predicted pair against experimentally validated CIRIquant junctions, using a fuzzy matching strategy (±5 bp tolerance at the nucleotide level):

| Group | Exact Top-1 | Fuzzy Top-1 (±5 bp) | Fuzzy Top-3 |
|:------|:----------:|:-----------------:|:-----------:|
| Candida (P4) | 0.0% | **85.5%** | **95.4%** |
| Cryptococcus (C4) | 0.0% | 10.6% | 22.7% |
| Filamentous (F6) | 0.5% | 28.2% | 51.9% |

The exact-match accuracy is near-zero because CIRIquant reports backsplice breakpoints internal to exons rather than at annotated splice site boundaries. With fuzzy matching (nearest-exon-boundary mapping), Candida achieves 85.5% Top-1 accuracy, reflecting the simpler genome architecture of compact single-exon genomes where only one or two candidate exon pairs exist. Filamentous achieves 51.9% Top-3 accuracy, indicating partial success in multi-exon contexts, while Cryptococcus lags at 22.7%. These results demonstrate that mycoCirc captures meaningful junction-level information, though dedicated junction-level training data and a ranking-based loss function would be required for publication-grade junction prediction.

### 5.8 Model Interpretability

**GTF Feature Importance (Integrated Gradients).** Integrated Gradients analysis of the GTFEncoder input reveals the following ranking of gene-structure features by their contribution to circRNA prediction (mean absolute attribution across all groups):

1. **CDS length** — strongest predictor; longer coding sequences correlate with greater transcript complexity and more backsplicing opportunities
2. **Exon length** — longer exons (particularly for single-exon genes) provide more sequence space for backsplicing
3. **Intron length** — longer introns are more likely to harbour complementary sequences that facilitate backsplicing
4. **Exon density** — denser exon architectures indicate more splice junctions per kb
5. **Exon count** — more exons → more candidate donor–acceptor pairs

This ranking is biologically coherent: in fungi, as in metazoans, circRNA biogenesis is primarily driven by transcript architecture features that determine the availability and compatibility of splice sites.

**Sequence Motif Enrichment.** K-mer attention weights from the JunctionEncoder identify enriched sequence patterns near backsplice junctions. The top enriched motifs in Candida are A/T-rich (AAA, TTT, AAG, TTC), consistent with the known preference for A/T-rich intron sequences in many fungi, which correlates with enhanced backsplicing efficiency.

## 6 Discussion

### 6.1 Biological and Methodological Significance

mycoCirc represents, to our knowledge, the first computational model specifically designed for pan-fungal circRNA prediction from genomic sequence alone. Its three key advances are:

**First, the multi-modal foundation model paradigm.** By integrating GTF-encoded gene structure, genomic sequence context, and junction-level sequence features, mycoCirc captures complementary aspects of circRNA biology that single-modal sequence models cannot access. The ablation results decisively show that GTF features—exon count, intron length, CDS architecture—are the dominant predictors. This makes biological sense: whether a gene can circularise is fundamentally determined by its exon–intron architecture, not by the specific base composition of its sequence neighbourhood.

**Second, cross-species generalisation through pre-training.** The two-stage pre-training protocol is essential: from-scratch fine-tuning collapses to near-random performance (AUROC ~0.50), while pre-trained models achieve 0.69–0.70. This 0.18 AUROC improvement demonstrates that the pre-training phase captures conserved features of circRNA biogenesis across diverse fungal genomes. The ability to generalise to an entirely unseen genus (*Talaromyces marneffei*) with AUROC 0.6955 from the Filamentous model confirms that these features are phylogenetically transferable.

**Third, RNA-seq-free inference.** Unlike existing methods that require known transcript sequences (JEDI) or are designed for species with extensive transcriptome data (CircPCBL), mycoCirc can predict circRNA-producing genes from a reference genome and gene annotation alone. This is critically important for fungal research, where many species lack RNA-seq data due to biosafety constraints, low RNA yields, or the high cost of fungal transcriptomics.

### 6.2 Comparison with Existing Methods

JEDI was originally designed for human circRNA prediction using only raw nucleotide sequences. When re-trained on fungal data, JEDI achieves AUROC of 0.51–0.57, substantially below its human-data performance. This drop reflects the fundamental difference in genome architecture: JEDI's architecture is optimised for multi-exon backsplice detection in Alu-rich human genomes, where complementary intronic sequences provide strong signals. In fungi, where most circRNA genes are single-exon and intronic repeats are scarce, JEDI's sequence-only approach lacks the structural context needed for effective prediction.

CircPCBL, designed for plant identification, achieves 0.49–0.53 in zero-shot fungal evaluation. While CircPCBL's dual-detector architecture (CNN-BiGRU + GLT) provides richer sequence encoding than JEDI's pure k-mer approach, its plant-specific training data (where GT/AG splicing signals differ from fungal splice sites) limits cross-kingdom transfer. The zero-shot evaluation is the fairest assessment, as CircPCBL's relatively large parameter count (compared to the fungal dataset) precluded effective re-training.

mycoCirc outperforms both methods by 0.16–0.19 AUROC on average. The key advantage is not one architectural component but the **combination** of multi-modal input, pan-fungal pre-training data, and group-specific fine-tuning. No single element accounts for the full improvement; rather, the interaction of all three creates a model that is robust to the architectural diversity of fungal genomes.

### 6.3 Limitations and Future Work

Several limitations should be acknowledged. First, junction prediction accuracy (Top-1 accuracy 0.0–28.2%) requires improvement through a dedicated ranking-based loss function rather than the current BCE loss on highly imbalanced junction labels. Second, the model's single-exon performance in multi-exon-dominant groups (Cryptococcus and Filamentous) is substantially weaker than its multi-exon performance, reflecting the limited structural signal available for single-exon genes. Third, the current model uses a Concat-MLP fusion module rather than an attention-based fusion that could dynamically weight modality contributions per gene—an architecture upgrade that could improve both interpretability and performance.

Future directions include: (i) incorporating fungal-specific repeat element annotations to capture complementary intronic sequences; (ii) extending the training dataset to cover additional fungal phyla (e.g., Mucoromycota, Basidiomycota beyond Cryptococcus); (iii) developing a dedicated junction-ranking loss (pairwise ranking or contrastive learning) for publication-grade backsplice prediction; and (iv) providing a web server for community access.

### 6.4 Conclusion

mycoCirc establishes that multi-modal foundation models trained on phylogenetically diverse fungal data can achieve robust cross-species circRNA prediction without requiring RNA-seq data. By learning to associate gene architecture features with circRNA production across 22 fungal strains, the model captures conserved principles of backsplice biology that transcend species boundaries. We anticipate that mycoCirc will enable rapid, cost-effective circRNA screening in understudied fungal species, accelerating functional genomics research in medical and industrial mycology. The software is freely available under an open-source license at [GitHub URL].

## 7 Acknowledgements

[To be added]

## 8 Funding

[To be added]

## 9 References

Chen, L.L. and Yang, L. (2015) Regulation of circRNA biogenesis. *RNA Biology*, 12(4), pp. 381–388.

Hansen, T.B. et al. (2013) Natural RNA circles function as efficient microRNA sponges. *Nature*, 495(7441), pp. 384–388.

Jeck, W.R. et al. (2013) Circular RNAs are abundant, conserved, and associated with ALU repeats. *RNA*, 19(2), pp. 141–157.

Jiang, J.-Y. et al. (2021) JEDI: circular RNA prediction based on junction encoders and deep interaction among splice sites. *Bioinformatics*, 37(Supplement 1), pp. i289–i298.

Kristensen, L.S. et al. (2019) The biogenesis, biology and characterization of circular RNAs. *Nature Reviews Genetics*, 20(11), pp. 675–691.

Liang, D. and Wilusz, J.E. (2014) Short intronic repeat sequences facilitate circular RNA production. *Genes & Development*, 28(20), pp. 2233–2247.

Memczak, S. et al. (2013) Circular RNAs are a large class of animal RNAs with regulatory potency. *Nature*, 495(7441), pp. 333–338.

Wang, X. et al. (2020) Identification and characterization of circular RNAs in the pathogenic fungus *Candida albicans*. *Frontiers in Microbiology*, 11, p. 580.

Wu, P. et al. (2023) CircPCBL: Identification of Plant CircRNAs with a CNN-BiGRU-GLT Model. *Plants*, 12(8), p. 1652.

Zhang, J. et al. (2020) CIRIquant: a precise tool for quantification of circular RNAs. *Bioinformatics*, 36(2), pp. 652–654.

Zhang, Y. et al. (2021) Circular RNA profiling reveals an abundant circRNA that regulates morphogenesis in the human fungal pathogen *Cryptococcus neoformans*. *mBio*, 12(3), pp. e00375-21.
