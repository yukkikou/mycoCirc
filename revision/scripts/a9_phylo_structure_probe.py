#!/usr/bin/env python3
"""A9 (revision): does the LEARNED species-lookup embedding encode
phylogenetic structure?

Motivation (R1-M3): the phylogenetic MDS branch is inactive in released
checkpoints; species information is carried by the learned lookup. If the
lookup embedding space itself is phylogenetically structured (distances
correlate with tree patristic distances; nearest neighbors are same-group),
then phylogeny is captured by the trained pathway — a positive answer that
complements the A7 sensitivity analysis.

Tests, per checkpoint (pretrain + 3 group finetunes):
  1. Mantel correlation between pairwise lookup-embedding distances and
     pairwise patristic distances from the ultrametric tree
     (permutation p, 5,000 perms).
  2. Nearest-neighbor analysis: for each strain, is its nearest lookup
     neighbor (cosine / euclidean) in the same taxonomic group?

Run:
  micromamba run -n pancirc-fungi python revision/scripts/a9_phylo_structure_probe.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from data.tsv_parser import parse_strain_registry, build_strain_index  # noqa: E402

OUT = PROJ / "revision" / "results"
TREE = Path("/media/share/data5/1610305236/Panfungi/1_data/AllFungi/"
            "phylogeny/1_nwk/Final_ultrametric_tree.nwk")

CKPTS = {
    "pretrain": PROJ / "checkpoints" / "pretrain" / "best.pt",
    "finetune_Candida": PROJ / "checkpoints" / "finetune" / "Candida" / "final.pt",
    "finetune_Cryptococcus": PROJ / "checkpoints" / "finetune" / "Cryptococcus" / "final_pretrained.pt",
    "finetune_Filamentous": PROJ / "checkpoints" / "finamentous_placeholder",
}
CKPTS["finetune_Filamentous"] = (
    PROJ / "checkpoints" / "finetune" / "Filamentous" / "final_pretrained.pt")


def lookup_weights(ckpt_path):
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = c.get("model_state_dict", c)
    key = "species_embedding.species_embed.weight"
    return sd[key].detach().numpy().astype(float)  # (21, 32)


def patristic_matrix(strains, entries_by_strain):
    from Bio import Phylo
    tree = Phylo.read(str(TREE), "newick")
    tips = {tip.name: tip for tip in tree.get_terminals()}
    n = len(strains)
    D = np.zeros((n, n))
    matched = 0
    for i, si in enumerate(strains):
        for j in range(i + 1, n):
            sj = strains[j]
            ei, ej = entries_by_strain[si], entries_by_strain[sj]
            ti = tips.get(ei.species.replace("_", " ")) or tips.get(ei.species)
            tj = tips.get(ej.species.replace("_", " ")) or tips.get(ej.species)
            if ti is not None and tj is not None:
                D[i, j] = D[j, i] = tree.distance(ti, tj)
                matched += 1
            else:
                D[i, j] = D[j, i] = np.nan
    rate = matched / (n * (n - 1) / 2)
    return D, rate, {s: (entries_by_strain[s].species.replace("_", " ")
                         in tips or entries_by_strain[s].species in tips)
                     for s in strains}


def mantel(D1, D2, n_perm=5000, seed=2026):
    """Mantel test on condensed distance matrices (rows with NaN dropped)."""
    from scipy.spatial.distance import squareform
    iu = ~np.isnan(D1) & ~np.isnan(D2)
    D1c = squareform(D1 * iu - np.where(iu, 0, np.nan) * 0, checks=False) \
        if False else None
    # safer: build condensed arrays manually over valid pairs
    n = D1.shape[0]
    a, b = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(D1[i, j]) and np.isfinite(D2[i, j]):
                a.append(D1[i, j])
                b.append(D2[i, j])
    a, b = np.array(a), np.array(b)
    r = np.corrcoef(a, b)[0, 1]
    rng = np.random.default_rng(seed)
    n_obs = int((np.sqrt(8 * len(a) - 1) + 1) / 2)  # infer matrix size
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(n_obs)
        Dp = D1[np.ix_(perm, perm)]
        ap = []
        for i in range(n_obs):
            for j in range(i + 1, n_obs):
                if np.isfinite(Dp[i, j]) and np.isfinite(D2[i, j]):
                    ap.append(Dp[i, j])
        if len(ap) == len(a):
            rp = np.corrcoef(np.array(ap), b)[0, 1]
            cnt += (rp >= r)
    return float(r), (cnt + 1) / (n_perm + 1)


def main():
    registry = parse_strain_registry(str(PROJ / "all_lib_model_full.tsv"))
    active = [e for e in registry if not e.is_excluded]
    idx = build_strain_index(active)
    strains = list(idx.keys())
    entries_by_strain = {e.strain: e for e in active}
    groups = np.array([entries_by_strain[s].group for s in strains])

    D_tree, match_rate, _ = patristic_matrix(strains, entries_by_strain)
    print(f"tree tip match rate: {match_rate:.2%} "
          f"({int(match_rate * len(strains) * (len(strains)-1) / 2)} pairs)")

    rows = []
    for name, path in CKPTS.items():
        if not path.exists():
            print(f"skip {name}: {path} missing")
            continue
        W = lookup_weights(path)  # rows follow build_strain_index order
        # euclidean distances in embedding space
        n = W.shape[0]
        D_emb = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(W[i] - W[j])
                D_emb[i, j] = D_emb[j, i] = d
        r, p = mantel(D_emb, D_tree)
        # nearest-neighbor same-group rate (euclidean)
        nn_same = 0
        for i in range(n):
            order = np.argsort([D_emb[i, j] if j != i else np.inf
                                for j in range(n)])
            nn_same += (groups[order[0]] == groups[i])
        nn_rate = nn_same / n
        # silhouette-ish: mean intra-group vs inter-group embedding distance
        intra, inter = [], []
        for i, j in combinations(range(n), 2):
            (intra if groups[i] == groups[j] else inter).append(D_emb[i, j])
        rows.append(dict(
            checkpoint=name,
            mantel_r=round(r, 4), mantel_p=f"{p:.4f}",
            nn_same_group=round(nn_rate, 3),
            mean_intra_group_dist=round(float(np.mean(intra)), 4),
            mean_inter_group_dist=round(float(np.mean(inter)), 4),
            intra_lt_inter=bool(np.mean(intra) < np.mean(inter))))
        print(f"{name:22s} Mantel r={r:.3f} (p={p:.4f}) | "
              f"NN same-group {nn_rate:.0%} | intra {np.mean(intra):.3f} "
              f"vs inter {np.mean(inter):.3f}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "a9_lookup_phylo_structure.csv",
                              index=False)
    np.savez(OUT / "a9_embedding_tree_distances.npz",
             strains=np.array(strains), groups=groups,
             D_tree=D_tree, match_rate=match_rate)
    print(f"\nDone → {OUT}")


if __name__ == "__main__":
    main()
