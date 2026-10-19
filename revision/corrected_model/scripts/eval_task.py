#!/usr/bin/env python3
"""One evaluation task: (species, checkpoint, mode) with the CORRECTED
(CircExp-zeroed) model. Invoked by a SLURM array; writes one JSON + npz.

Mode A -> predict.py without --genexp
Mode B -> predict.py with --genexp (circ_exp is always zeroed inside)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ))

BASE = PROJ / "revision" / "corrected_model"
SCORES = BASE / "eval" / "scores"
RES = BASE / "eval" / "results"
SCORES.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

from data.tsv_parser import parse_strain_registry  # noqa: E402
from data.circ_info_encoding import (load_circ_info, filter_circ_info,  # noqa: E402
                                     get_positive_genes)
from data.negative_sampling import get_negative_genes  # noqa: E402
from utils.gtf_utils import GeneModelIndexer  # noqa: E402

EXTERNAL_CONFIG = ("/media/share/data5/1610305236/Panfungi/1_data/"
                   "configs/exte_lib_config.tsv")
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
CKPT_ROOT = PROJ / "revision" / "checkpoints" / "nocircexp"
GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
BOOT, SEED = 2000, 2026


def species_table():
    """name -> dict(genome, gtf, circinfo, genexp, kind)."""
    reg = parse_strain_registry(str(PROJ / "all_lib_model_full.tsv"))
    out = {e.strain: dict(genome=e.genome_path, gtf=e.gtf_path,
                          circinfo=e.circinfo_path, genexp=e.geneexp_path,
                          kind="heldout") for e in reg}
    seen = {}
    with open(EXTERNAL_CONFIG) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > 5 and c[0] not in seen:
                seen[c[0]] = dict(genome=c[4], gtf=c[5])
    for name, d in seen.items():
        out[name] = dict(
            genome=d["genome"], gtf=d["gtf"],
            circinfo=f"{MERGE_DIR}/{name}/{name}_circRNA_info.csv",
            genexp=f"{MERGE_DIR}/{name}/{name}_gene_count.csv",
            kind="external")
    out["Talaromyces_marneffei"] = dict(
        genome="/media/share/resource/reference/talaromyces_marneffei/genome/pm1.soft.fasta",
        gtf="/media/share/resource/reference/talaromyces_marneffei/annotation/pm1_scallop_final.gtf",
        circinfo="/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv",
        genexp="/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_gene_count_matrix.csv",
        kind="external")
    return out


def labels_for(spec, species):
    """Balanced eval set. Held-out registry strains MUST use the cached
    ids (that is what the manuscript's held-out numbers were computed on);
    external species use the seed-42 sampling used by eval_external.py."""
    feat = PROJ / "checkpoints" / "features" / species
    if spec.get("kind") == "heldout" and (feat / "positive_gene_ids.npy").exists():
        pos = [str(g) for g in np.load(feat / "positive_gene_ids.npy", allow_pickle=True)]
        neg = [str(g) for g in np.load(feat / "negative_gene_ids.npy", allow_pickle=True)]
    else:
        gm = GeneModelIndexer(spec["gtf"])
        circ = filter_circ_info(load_circ_info(spec["circinfo"]))
        pos = list(get_positive_genes(circ))
        neg = get_negative_genes(gm, set(pos), len(pos), rng_seed=42)
    lab = {g: 1 for g in pos}
    lab.update({g: 0 for g in neg})
    return lab


def bootstrap_ci(y, p):
    rng = np.random.default_rng(SEED)
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(BOOT):
        sel = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]
        vals.append(roc_auc_score(y[sel], p[sel]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--group", required=True, choices=GROUPS)
    ap.add_argument("--mode", required=True, choices=["A", "B"])
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    spec = species_table()[args.species]
    tag = f"{args.species}_{args.group}_{args.mode}"
    tsv = SCORES / f"{tag}.tsv"
    outp = RES / f"{tag}.json"

    ckpt = CKPT_ROOT / args.group / "final.pt"
    if not tsv.exists() or tsv.stat().st_size == 0:
        cmd = [sys.executable, str(PROJ / "scripts" / "predict.py"),
               "--genome", spec["genome"], "--gtf", spec["gtf"],
               "--checkpoint", str(ckpt), "--output", str(tsv),
               "--device", "cpu", "--batch-size", "64",
               "--num-workers", str(args.num_workers)]
        if args.mode == "B":
            cmd += ["--genexp", spec["genexp"]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-800:])
            sys.exit(1)

    lab = labels_for(spec, args.species)
    df = pd.read_csv(tsv, sep="\t")
    common = [g for g in df.gene_id if g in lab]
    y = np.array([lab[g] for g in common])
    p = df.set_index("gene_id").loc[common, "p_circ"].values.astype(float)
    auc = float(roc_auc_score(y, p))
    lo, hi = bootstrap_ci(y, p)
    np.savez(RES / f"{tag}.npz", gene_id=np.array(common), y_true=y, y_prob=p)
    outp.write_text(json.dumps(dict(
        species=args.species, checkpoint=args.group, mode=args.mode,
        recipe="nocircexp", auroc=round(auc, 4),
        ci=[round(lo, 4), round(hi, 4)], n=int(len(y)),
        n_pos=int(y.sum()), frac_p_ge_05=round(float((p >= 0.5).mean()), 4),
        p_median=round(float(np.median(p)), 4))))
    print(f"{tag}: AUROC={auc:.4f} [{lo:.3f},{hi:.3f}] n={len(y)}")


if __name__ == "__main__":
    main()
