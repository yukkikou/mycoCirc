#!/usr/bin/env python3
"""A8 (revision): retrain CircPCBL from scratch on FUNGAL data, per group —
closing the R2-3 asymmetry (JEDI was retrained on fungi; CircPCBL was only
used zero-shot from plant weights).

Faithful re-implementation of the official CircPCBL architecture
(other_models/CircPCBL-main/Model/Plant_GPU.py::Net) with:
  * identical layer definitions (CNN k=7/5/3 → BiGRU(hidden 30) → GLT on
    340-dim k-mer frequencies, k=1..4), including the official forward's
    randperm feature shuffle (seeded here for reproducibility);
  * 1500-bp gene-centered sequences + 340-dim k-mer features, exactly the
    inputs used by the zero-shot evaluation (eval_circpcbl_external.py);
  * training protocol mirroring the JEDI retraining: group training strains,
    cached seed-42 balanced pos/neg (checkpoints/features/), 20 epochs,
    Adam lr=1e-3, batch 64, seed 42.

Evaluation: held-out strain + 4 external species, per-gene circRNA
probability; AUROC with bootstrap CI + paired DeLong vs mycoCirc (Mode A
genome-wide TSV scores). Outputs → revision/results/.

Run:
  micromamba run -n pancirc-fungi python revision/scripts/a8_circpcbl_retrain.py \
      [--device cuda]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from data.tsv_parser import (parse_strain_registry, TRAIN_STRAINS,  # noqa: E402
                             TEST_STRAINS)
from data.circ_info_encoding import (load_circ_info, filter_circ_info,  # noqa: E402
                                     get_positive_genes)
from data.negative_sampling import get_negative_genes  # noqa: E402
from data.genome_encoding import GenomeIndexer  # noqa: E402
from utils.gtf_utils import GeneModelIndexer  # noqa: E402
from utils.metrics import classification_metrics  # noqa: E402

# ── official encoding constants (Plant_GPU.py) ──────────────────────────────
TEXT_LEN = 1500          # 1500-bp gene-centered window
WORD_INDEX = {"a": 0, "g": 1, "c": 2, "t": 3}
N_KMER_FEATURES = 340    # 4+16+64+256
HALF = 750
SEED = 42
EPOCHS = 20
BATCH = 64
LR = 1e-3

OUT = PROJ / "revision" / "results"
CKPT_DIR = PROJ / "revision" / "checkpoints" / "circpcbl_fungal"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_CONFIG = ("/media/share/data5/1610305236/Panfungi/1_data/"
                   "configs/exte_lib_config.tsv")
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"


# ── k-mer feature computation (vectorized port of the official code) ───────
_KMERS = {}
for k in range(1, 5):
    from itertools import product
    _KMERS[k] = ["".join(c) for c in product("ACGT", repeat=k)]


def kmer_features(seq: str) -> np.ndarray:
    s = seq.upper()
    out = []
    for k in range(1, 5):
        n_slots = len(s) - k + 1
        counts = {m: 0 for m in _KMERS[k]}
        for i in range(n_slots):
            m = s[i:i + k]
            if m in counts:
                counts[m] += 1
        denom = max(n_slots, 1)
        out.extend(counts[m] / denom for m in _KMERS[k])
    return np.array(out, dtype=np.float32)


def encode_sequence(seq: str):
    """1500-bp token ids (official lowercase map, non-ACGT→4) + kmer feats."""
    s = seq.upper()[:TEXT_LEN]
    toks = [WORD_INDEX.get(c.lower(), 4) for c in s]
    if len(toks) < TEXT_LEN:
        toks = toks + [4] * (TEXT_LEN - len(toks))
    return np.array(toks, dtype=np.int64), kmer_features(seq)


def gene_centered_seq(gi, gene):
    mid = (gene.start + gene.end) // 2
    start = max(1, mid - HALF)
    seq = gi.get_seq(gene.chrom, start, mid + HALF, "+")
    return (seq or "").upper()


# ── official Net architecture (verbatim layer config) ───────────────────────
class CircPCBLNet(nn.Module):
    def __init__(self, hidden_size=30, num_layers=1):
        super().__init__()
        # official config: same-padding convs (k=7/5/3, stride 1) and
        # kernel-3 stride-1 pooling — sequence length is preserved (1500)
        self.conv_7 = nn.Conv1d(4, 32, 7, stride=1, padding=3)
        self.conv_3 = nn.Conv1d(4, 32, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv1d(4, 32, 5, stride=1, padding=2)
        self.max_pool_1 = nn.MaxPool1d(3, stride=1, padding=1)
        self.max_pool_2 = nn.MaxPool1d(3, stride=1, padding=1)
        self.max_pool_3 = nn.MaxPool1d(3, stride=1, padding=1)
        self.bigru = nn.GRU(32 * 3, hidden_size, num_layers,
                            bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32 + 32, 2)
        self.GLT1 = nn.Linear(N_KMER_FEATURES, 256)
        self.GLT2_1 = nn.Linear(128, 64)
        self.GLT2_2 = nn.Linear(128, 64)
        self.GLT3_1 = nn.Linear(32, 8)
        self.GLT3_2 = nn.Linear(32, 8)
        self.GLT3_3 = nn.Linear(32, 8)
        self.GLT3_4 = nn.Linear(32, 8)
        self.norm_8 = nn.LayerNorm([8])
        self.norm_32 = nn.LayerNorm([32])
        self.norm_64 = nn.LayerNorm([64])
        self.norm_128 = nn.LayerNorm([128])
        self.norm_256 = nn.LayerNorm([256])
        self._perm_seed = 0

    def forward(self, x):
        x_seq, x_feat = torch.split(x, TEXT_LEN, dim=-1)
        x_seq = x_seq.long()
        x_seq = nn.functional.one_hot(x_seq, num_classes=5)[:, :, :-1]
        x_seq = x_seq.permute(0, 2, 1).float()

        x1 = self.max_pool_1(nn.functional.relu(self.conv_7(x_seq)))
        x2 = self.max_pool_2(nn.functional.relu(self.conv_5(x_seq)))
        x3 = self.max_pool_3(nn.functional.relu(self.conv_3(x_seq)))
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.permute(2, 0, 1)
        x, _ = self.bigru(x)
        x = torch.sum(x, dim=0)
        x = nn.functional.relu(self.norm_128(self.fc1(x)))
        x = nn.functional.relu(self.norm_32(self.fc2(x)))

        # GLT block (official forward includes a randperm; seeded here)
        x_feat = nn.functional.relu(self.norm_256(self.GLT1(x_feat)))
        g1, g2 = torch.chunk(x_feat, 2, dim=-1)
        g1 = nn.functional.relu(self.norm_64(self.GLT2_1(g1)))
        g2 = nn.functional.relu(self.norm_64(self.GLT2_2(g2)))
        x_feat = torch.cat([g1, g2], dim=-1)
        if self.training:
            idx = torch.randperm(x_feat.shape[-1],
                                 device=x_feat.device)
            x_feat = x_feat[:, idx].view(x_feat.shape)
        g1, g2, g3, g4 = torch.chunk(x_feat, 4, dim=-1)
        g1 = nn.functional.relu(self.norm_8(self.GLT3_1(g1)))
        g2 = nn.functional.relu(self.norm_8(self.GLT3_2(g2)))
        g3 = nn.functional.relu(self.norm_8(self.GLT3_3(g3)))
        g4 = nn.functional.relu(self.norm_8(self.GLT3_4(g4)))
        x_feat = torch.cat([g1, g2, g3, g4], dim=-1)

        return self.fc3(torch.cat([x, x_feat], dim=-1))


def build_set(gm, gi, gene_ids, labels):
    toks, feats, ys, gids = [], [], [], []
    for gid, lab in zip(gene_ids, labels):
        gene = gm.get_gene(gid)
        if gene is None:
            continue
        seq = gene_centered_seq(gi, gene)
        if len(seq) < 50:
            continue
        t, f = encode_sequence(seq)
        toks.append(t)
        feats.append(f)
        ys.append(lab)
        gids.append(gid)
    X = np.concatenate([np.stack(toks), np.stack(feats)], axis=1)
    return X.astype(np.float32), np.array(ys, dtype=np.int64), np.array(gids)


@torch.no_grad()
def predict_proba(model, X, device, batch=128):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)[:, 1]
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def bootstrap_ci(y, p, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        sel = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]
        vals.append(roc_auc_score(y[sel], p[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    registry = parse_strain_registry(str(PROJ / "all_lib_model_full.tsv"))
    by_strain = {e.strain: e for e in registry}

    # external species definitions (same as a1)
    ext_defs = {
        "Talaromyces_marneffei": dict(
            genome="/media/share/resource/reference/talaromyces_marneffei/genome/pm1.soft.fasta",
            gtf="/media/share/resource/reference/talaromyces_marneffei/annotation/pm1_scallop_final.gtf",
            circinfo="/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv",
        ),
    }
    seen = {}
    with open(EXTERNAL_CONFIG) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) > 5 and cols[0] not in seen:
                seen[cols[0]] = (cols[4], cols[5])
    for name, (g, t) in seen.items():
        merge = Path(MERGE_DIR) / name / f"{name}_circRNA_info.csv"
        ext_defs[name] = dict(genome=g, gtf=t, circinfo=str(merge))

    # pre-build external eval sets (shared across groups)
    ext_sets = {}
    for sp, spec in ext_defs.items():
        gm = GeneModelIndexer(spec["gtf"])
        gi = GenomeIndexer(spec["genome"])
        pos = list(get_positive_genes(
            filter_circ_info(load_circ_info(spec["circinfo"]))))
        neg = get_negative_genes(gm, set(pos), len(pos), rng_seed=42)
        X, y, g = build_set(gm, gi, list(pos) + list(neg),
                            [1] * len(pos) + [0] * len(neg))
        ext_sets[sp] = (X, y, g)
        print(f"external {sp}: n={len(y)} pos={int(y.sum())}")

    rows = []
    delong_rows = []
    for grp in ["Candida", "Cryptococcus", "Filamentous"]:
        print(f"\n=== CircPCBL fungal retraining: {grp} ===")
        # training set (cached ids, live features)
        ids, labs = [], []
        for strain in sorted(TRAIN_STRAINS[grp]):
            d = PROJ / "checkpoints" / "features" / strain
            pos = [str(g) for g in np.load(d / "positive_gene_ids.npy",
                                           allow_pickle=True)]
            neg = [str(g) for g in np.load(d / "negative_gene_ids.npy",
                                           allow_pickle=True)]
            ids += pos + neg
            labs += [1] * len(pos) + [0] * len(neg)
        e0 = by_strain[sorted(TRAIN_STRAINS[grp])[0]]
        gm = GeneModelIndexer(e0.gtf_path)  # placeholder; per-strain below

        toks, feats, ys = [], [], []
        for strain in sorted(TRAIN_STRAINS[grp]):
            entry = by_strain[strain]
            gm = GeneModelIndexer(entry.gtf_path)
            gi = GenomeIndexer(entry.genome_path)
            d = PROJ / "checkpoints" / "features" / strain
            pos = [str(g) for g in np.load(d / "positive_gene_ids.npy",
                                           allow_pickle=True)]
            neg = [str(g) for g in np.load(d / "negative_gene_ids.npy",
                                           allow_pickle=True)]
            for gid, lab in [(g, 1) for g in pos] + [(g, 0) for g in neg]:
                gene = gm.get_gene(gid)
                if gene is None:
                    continue
                seq = gene_centered_seq(gi, gene)
                if len(seq) < 50:
                    continue
                t, f = encode_sequence(seq)
                toks.append(t)
                feats.append(f)
                ys.append(lab)
        Xtr = np.concatenate([np.stack(toks), np.stack(feats)],
                             axis=1).astype(np.float32)
        ytr = np.array(ys, dtype=np.int64)
        print(f"  train: n={len(ytr)} pos={int(ytr.sum())}")

        model = CircPCBLNet().to(device)
        opt_ = optim.Adam(model.parameters(), lr=LR)
        lossf = nn.CrossEntropyLoss()
        dl = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                      torch.from_numpy(ytr)),
                        batch_size=BATCH, shuffle=True)
        for ep in range(EPOCHS):
            model.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt_.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                opt_.step()
                tot += loss.item() * len(yb)
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(ytr):.4f}")
        torch.save({"model_state_dict": model.state_dict(), "group": grp,
                    "seed": SEED, "epochs": EPOCHS},
                   CKPT_DIR / f"{grp}.pt")

        # held-out strain (own group)
        test_strain = next(iter(TEST_STRAINS[grp]))
        entry = by_strain[test_strain]
        gm = GeneModelIndexer(entry.gtf_path)
        gi = GenomeIndexer(entry.genome_path)
        d = PROJ / "checkpoints" / "features" / test_strain
        pos = [str(g) for g in np.load(d / "positive_gene_ids.npy",
                                       allow_pickle=True)]
        neg = [str(g) for g in np.load(d / "negative_gene_ids.npy",
                                       allow_pickle=True)]
        Xte, yte, gte = build_set(gm, gi, pos + neg,
                                  [1] * len(pos) + [0] * len(neg))
        p = predict_proba(model, Xte, device)
        auc = roc_auc_score(yte, p)
        lo, hi = bootstrap_ci(yte, p)
        np.savez(OUT / f"a8_circpcbl_heldout_{test_strain}.npz",
                 gene_id=gte, y_true=yte, y_prob=p)
        rows.append(dict(eval=f"heldout_{test_strain}", group=grp,
                         method="CircPCBL_fungal", auroc=round(float(auc), 4),
                         ci=f"[{lo:.3f},{hi:.3f}]", n=len(yte),
                         n_pos=int(yte.sum())))
        print(f"  held-out {test_strain}: AUROC={auc:.4f} [{lo:.3f},{hi:.3f}]")

        # external species + DeLong vs mycoCirc
        from utils.metrics import delong_roc_test
        for sp, (X, y, g) in ext_sets.items():
            p = predict_proba(model, X, device)
            auc = roc_auc_score(y, p)
            lo, hi = bootstrap_ci(y, p)
            np.savez(OUT / f"a8_circpcbl_external_{sp}_{grp}.npz",
                     gene_id=g, y_true=y, y_prob=p)
            rows.append(dict(eval=f"external_{sp}", group=grp,
                             method="CircPCBL_fungal",
                             auroc=round(float(auc), 4),
                             ci=f"[{lo:.3f},{hi:.3f}]", n=len(y),
                             n_pos=int(y.sum())))
            # paired DeLong vs mycoCirc (genome-wide TSV)
            tsv = PROJ / "predictions_external" / f"{sp}_{grp}.tsv"
            if not tsv.exists():
                tsv = PROJ / "predictions_external" / f"{sp}.tsv"
            df = pd.read_csv(tsv, sep="\t")
            myco = dict(zip(df["gene_id"], df["p_circ"]))
            common = [i for i, gid in enumerate(g) if gid in myco]
            y_c = y[common]
            p_c = p[common]
            p_m = np.array([myco[g[i]] for i in common])
            try:
                dl_ = delong_roc_test(y_c.astype(float), p_m, p_c)
                delong_rows.append(dict(
                    species=sp, group=grp,
                    comparison="mycoCirc_A_vs_CircPCBL_fungal",
                    auc_a=round(dl_["auc_a"], 4),
                    auc_b=round(dl_["auc_b"], 4),
                    diff=round(dl_["auc_diff"], 4),
                    p=f"{dl_['p_value']:.2e}", n=len(common)))
            except Exception as ex:
                print(f"  DeLong failed {sp}/{grp}: {ex}")
            print(f"  external {sp}: AUROC={auc:.4f} [{lo:.3f},{hi:.3f}]")

    pd.DataFrame(rows).to_csv(OUT / "a8_circpcbl_fungal_results.csv",
                              index=False)
    pd.DataFrame(delong_rows).to_csv(
        OUT / "a8_delong_vs_mycocirc.csv", index=False)
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
