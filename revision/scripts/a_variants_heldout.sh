#!/bin/bash
# Held-out per-gene scores for the two GPU variant model sets (A6-ii
# nocircexp, A2-ii matched) via the production predict.py path, then
# AUROC + bootstrap CI on the cached eval gene sets.
set -e
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /media/share/workdir/1610305236/Panfungi/4_model
OUT=revision/results/rescore
mkdir -p $OUT

declare -A GTF GENOME
while IFS=$'\t' read -r Group Species Strain Genome Gtf Circ CircExp GeneExp; do
  [ "$Group" = "Group" ] && continue
  GENOME[$Strain]="$Genome"; GTF[$Strain]="$Gtf"
done < all_lib_model_full.tsv

for VARIANT in nocircexp matched; do
  for STRAIN in P4 C4 F6; do
    GRP=$( [ "$STRAIN" = P4 ] && echo Candida || { [ "$STRAIN" = C4 ] && echo Cryptococcus || echo Filamentous; } )
    CK=revision/checkpoints/$VARIANT/$GRP/final.pt
    TSV=$OUT/${VARIANT}_${STRAIN}_${GRP}.tsv
    [ -s "$TSV" ] && { echo "skip $TSV"; continue; }
    python scripts/predict.py --genome "${GENOME[$STRAIN]}" --gtf "${GTF[$STRAIN]}" \
      --checkpoint "$CK" --output "$TSV" --batch-size 32 --device cpu
  done
done

python - <<'PYEOF'
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
PROJ=Path("/media/share/workdir/1610305236/Panfungi/4_model")
OUT=PROJ/"revision/results/rescore"
rng=np.random.default_rng(2026)
rows=[]
for variant in ["nocircexp","matched"]:
    for strain,grp in [("P4","Candida"),("C4","Cryptococcus"),("F6","Filamentous")]:
        f=OUT/f"{variant}_{strain}_{grp}.tsv"
        if not f.exists(): continue
        pos=[str(g) for g in np.load(PROJ/f"checkpoints/features/{strain}/positive_gene_ids.npy", allow_pickle=True)]
        neg=[str(g) for g in np.load(PROJ/f"checkpoints/features/{strain}/negative_gene_ids.npy", allow_pickle=True)]
        labels={g:1 for g in pos}; labels.update({g:0 for g in neg})
        df=pd.read_csv(f,sep="\t")
        # matched variants were TRAINED on matched negatives; eval on their own
        # matched training sets is handled separately — here use standard eval set
        common=[g for g in df.gene_id if g in labels]
        y=np.array([labels[g] for g in common]); p=df.set_index("gene_id").loc[common,"p_circ"].values
        auc=roc_auc_score(y,p)
        pi=np.where(y==1)[0]; ni=np.where(y==0)[0]
        vals=[roc_auc_score(y[np.r_[rng.choice(pi,len(pi)),rng.choice(ni,len(ni))]],
                            p[np.r_[rng.choice(pi,len(pi)),rng.choice(ni,len(ni))]]) for _ in range(2000)]
        lo,hi=np.percentile(vals,[2.5,97.5])
        rows.append(dict(variant=variant,strain=strain,checkpoint=grp,
                         auroc=round(float(auc),4),ci=f"[{lo:.3f},{hi:.3f}]",
                         n=len(y),n_pos=int(y.sum())))
        print(variant,strain,grp,f"AUROC={auc:.4f} CI=[{lo:.3f},{hi:.3f}] n={len(y)}")
pd.DataFrame(rows).to_csv(PROJ/"revision/results/a_variant_heldout_ci.csv", index=False)
PYEOF
