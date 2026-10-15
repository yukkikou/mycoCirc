#!/usr/bin/env python3
"""A2-ii prep: build a features directory with MATCHED negative gene ids for
every registry train/test strain, so train/finetune.py can be re-run
UNCHANGED via --features-dir (it only reads {strain}/{positive,negative}
_gene_ids.npy from there).

Matching (gene level, determinants MatchedExonSampler):
  log total-exon-length caliper +/-0.25 (relaxed +/-0.5); exon count exact
  when <=3 else +/-1; expression/GC channels NaN (structure-only).

Output: revision/features_matched/{strain}/positive_gene_ids.npy (copied
verbatim) + negative_gene_ids.npy (matched, n_per_pos=1, seed 42).

Run:
  micromamba run -n pancirc-fungi python revision/scripts/make_matched_features.py
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

_spec = importlib.util.spec_from_file_location(
    "matched_sampling",
    str(PROJ / "determinants" / "lib" / "matched_sampling.py"))
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)
MatchedExonSampler = ms.MatchedExonSampler

from data.tsv_parser import parse_strain_registry, TRAIN_STRAINS, TEST_STRAINS  # noqa: E402
from utils.gtf_utils import GeneModelIndexer  # noqa: E402

SRC = PROJ / "checkpoints" / "features"
DST = PROJ / "revision" / "features_matched"


def gene_rows(gm, gene_ids):
    rows = []
    for gid in gene_ids:
        g = gm.get_gene(gid)
        if g is None or g.exon_count < 1:
            continue
        exon_len = max(int(sum(g.exon_ends) - sum(g.exon_starts)
                           + g.exon_count), 1)
        rows.append(dict(
            gene_id=gid, chrom=g.chrom, exon_start=g.start, exon_end=g.end,
            strand=g.strand, exon_len=exon_len,
            host_exon_count=float(g.exon_count),
            host_expr=np.nan, flank_gc=np.nan,
            circ_len=exon_len, circ_id=gid,
        ))
    return rows


def main():
    registry = parse_strain_registry(str(PROJ / "all_lib_model_full.tsv"))
    by_strain = {e.strain: e for e in registry}
    strains = sorted(set().union(*TRAIN_STRAINS.values(), *TEST_STRAINS.values()))

    for strain in strains:
        src = SRC / strain
        if not (src / "positive_gene_ids.npy").exists():
            print(f"{strain}: no cached ids — skipped")
            continue
        pos = [str(g) for g in np.load(src / "positive_gene_ids.npy",
                                       allow_pickle=True)]
        if len(pos) < 10:
            print(f"{strain}: only {len(pos)} positives — copied as-is "
                  "(degenerate strain)")
            dst = DST / strain
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / "positive_gene_ids.npy",
                         dst / "positive_gene_ids.npy")
            shutil.copy2(src / "negative_gene_ids.npy",
                         dst / "negative_gene_ids.npy")
            continue

        gm = GeneModelIndexer(by_strain[strain].gtf_path)
        if len(gm.genes) < 500:
            raise RuntimeError(f"{strain}: GTF parse suspicious "
                               f"({len(gm.genes)} genes) — rerun")
        pos_df = pd.DataFrame(gene_rows(gm, pos))
        pos_set = set(pos)
        cands = [g for g in gm.genes if g not in pos_set]
        cand_df = pd.DataFrame(gene_rows(gm, cands))

        sampler = MatchedExonSampler(cand_df, rng_seed=42)
        neg_df = sampler.sample(pos_df, n_per_pos=1)
        neg_ids = sorted(set(neg_df["gene_id"]))
        diag = sampler.diagnostics()
        print(f"{strain}: pos={len(pos_df)} matched_neg={len(neg_ids)} {diag}")

        dst = DST / strain
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "positive_gene_ids.npy",
                     dst / "positive_gene_ids.npy")
        np.save(dst / "negative_gene_ids.npy",
                np.array(neg_ids, dtype=object), allow_pickle=True)

    print(f"\nDone → {DST}")


if __name__ == "__main__":
    main()
