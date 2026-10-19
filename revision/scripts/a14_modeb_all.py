#!/usr/bin/env python3
"""A14: recompute ALL Mode B results under the corrected (CircExp-zeroed)
recipe, on a single consistent pipeline (scripts/predict.py --genexp).

Covered
  held-out strains P4 / C4 / F6   x own-group checkpoint   (manuscript fig 1A)
  external species (4)            x all 3 checkpoints      (manuscript table 1)
  -> for comparison, the original checkpoints are also scored on the held-out
     strains (same pipeline), so the before/after is apples-to-apples.

Outputs
  results/a14_modeb_scores/{species}_{ckpt}_{recipe}.tsv   (predict.py output)
  results/a14_modeb_results.csv     AUROC, bootstrap 95% CI, n, frac p>=0.5
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))
OUT = PROJ / "revision" / "results"
SCORES = OUT / "a14_modeb_scores"
SCORES.mkdir(parents=True, exist_ok=True)

from data.tsv_parser import parse_strain_registry  # noqa: E402
from data.circ_info_encoding import (load_circ_info, filter_circ_info,  # noqa: E402
                                     get_positive_genes)
from data.negative_sampling import get_negative_genes  # noqa: E402
from utils.gtf_utils import GeneModelIndexer  # noqa: E402

EXTERNAL_CONFIG = ("/media/share/data5/1610305236/Panfungi/1_data/"
                   "configs/exte_lib_config.tsv")
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
HELDOUT_OWN = {"P4": "Candida", "C4": "Cryptococcus", "F6": "Filamentous"}
BOOT, SEED = 2000, 2026


def bootstrap_ci(y, p):
    rng = np.random.default_rng(SEED)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(BOOT):
        sel = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]
        vals.append(roc_auc_score(y[sel], p[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def ckpt_path(recipe, group):
    if recipe == "original":
        base = PROJ / "checkpoints" / "finetune" / group
        for n in ("final_pretrained.pt", "final.pt"):
            if (base / n).exists():
                return base / n
    return PROJ / "revision" / "checkpoints" / "nocircexp" / group / "final.pt"


def run_predict(species, genome, gtf, genexp, recipe, group):
    tsv = SCORES / f"{species}_{group}_{recipe}.tsv"
    if tsv.exists() and tsv.stat().st_size > 0:
        print(f"  skip {tsv.name}")
        return tsv
    cmd = [sys.executable, str(PROJ / "scripts" / "predict.py"),
           "--genome", str(genome), "--gtf", str(gtf),
           "--checkpoint", str(ckpt_path(recipe, group)),
           "--genexp", str(genexp), "--output", str(tsv),
           "--device", "cpu", "--batch-size", "32"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED {tsv.name}: {r.stderr[-300:]}")
        return None
    return tsv


def prep_species(name, genome, gtf, circinfo, is_external):
    gm = GeneModelIndexer(gtf)
    circ = filter_circ_info(load_circ_info(circinfo))
    pos = list(get_positive_genes(circ))
    neg = get_negative_genes(gm, set(pos), len(pos), rng_seed=42)
    lab = {g: 1 for g in pos}
    lab.update({g: 0 for g in neg})
    return lab


def main():
    reg = parse_strain_registry(str(PROJ / "all_lib_model_full.tsv"))
    by = {e.strain: e for e in reg}
    rows = []

    # ---- held-out strains, own-group checkpoint, both recipes ----
    for strain, grp in HELDOUT_OWN.items():
        e = by[strain]
        lab = prep_species(strain, e.genome_path, e.gtf_path, e.circinfo_path, False)
        print(f"\n=== {strain} ({grp}) ===")
        for recipe in ("original", "nocircexp"):
            tsv = run_predict(strain, e.genome_path, e.gtf_path, e.geneexp_path,
                              recipe, grp)
            if tsv is None:
                continue
            df = pd.read_csv(tsv, sep="\t")
            common = [g for g in df.gene_id if g in lab]
            y = np.array([lab[g] for g in common])
            p = df.set_index("gene_id").loc[common, "p_circ"].values.astype(float)
            auc = roc_auc_score(y, p)
            lo, hi = bootstrap_ci(y, p)
            rows.append(dict(species=strain, checkpoint=grp, mode="B",
                             recipe=recipe, auroc=round(float(auc), 4),
                             ci=f"[{lo:.3f},{hi:.3f}]", n=len(y),
                             frac_p_ge_05=round(float((p >= 0.5).mean()), 3)))

    # ---- external species, all three checkpoints, corrected recipe ----
    ext = {"Talaromyces_marneffei": dict(
        genome="/media/share/resource/reference/talaromyces_marneffei/genome/pm1.soft.fasta",
        gtf="/media/share/resource/reference/talaromyces_marneffei/annotation/pm1_scallop_final.gtf",
        circinfo="/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv",
        genexp="/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_gene_count_matrix.csv",
    )}
    seen = {}
    with open(EXTERNAL_CONFIG) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > 7 and c[0] not in seen:
                seen[c[0]] = dict(genome=c[4], gtf=c[5], genexp=c[7])
    for name, d in seen.items():
        merge = Path(MERGE_DIR) / name / f"{name}_circRNA_info.csv"
        ext[name] = dict(**d, circinfo=str(merge))

    for sp, d in ext.items():
        lab = prep_species(sp, d["genome"], d["gtf"], d["circinfo"], True)
        print(f"\n=== external {sp} ===")
        for grp in GROUPS:
            tsv = run_predict(sp, d["genome"], d["gtf"], d["genexp"],
                              "nocircexp", grp)
            if tsv is None:
                continue
            df = pd.read_csv(tsv, sep="\t")
            common = [g for g in df.gene_id if g in lab]
            y = np.array([lab[g] for g in common])
            p = df.set_index("gene_id").loc[common, "p_circ"].values.astype(float)
            auc = roc_auc_score(y, p)
            lo, hi = bootstrap_ci(y, p)
            rows.append(dict(species=sp, checkpoint=grp, mode="B",
                             recipe="nocircexp", auroc=round(float(auc), 4),
                             ci=f"[{lo:.3f},{hi:.3f}]", n=len(y),
                             frac_p_ge_05=round(float((p >= 0.5).mean()), 3)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a14_modeb_results.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
