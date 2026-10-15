#!/bin/bash
# A1b + A7: per-gene held-out scores via the PRODUCTION predict.py path
# (same pipeline that produced predictions_external, whose scores matched
# the published npz exactly). Avoids the junction-feature instability seen
# with the ad-hoc rescore.py on this cluster.
#
# Held-out: P4/C4/F6 x 3 checkpoints (A1b: bootstrap CIs + DeLong vs baselines)
# A7 strain-id sweep: external species via --strain-id 0..20 (from
# predict.py directly; external TSVs for the default id already exist).
set -e
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /media/share/workdir/1610305236/Panfungi/4_model

OUT=revision/results/rescore
mkdir -p $OUT

# registry lookups
declare -A GTF GENOME
while IFS=$'\t' read -r Group Species Strain Genome Gtf Circ CircExp GeneExp; do
  [[ "$Group" == "Group" ]] && continue
  GENOME[$Strain]=$Genome; GTF[$Strain]=$Gtf
done < all_lib_model_full.tsv

declare -A CKPT=(
  [Candida]=checkpoints/finetune/Candida/final.pt
  [Cryptococcus]=checkpoints/finetune/Cryptococcus/final_pretrained.pt
  [Filamentous]=checkpoints/finetune/Filamentous/final_pretrained.pt
)

for STRAIN in P4 C4 F6; do
  for GRP in Candida Cryptococcus Filamentous; do
    OUTTSV=$OUT/predict_${STRAIN}_${GRP}.tsv
    if [ -s "$OUTTSV" ]; then echo "skip $OUTTSV"; continue; fi
    python scripts/predict.py \
      --genome "${GENOME[$STRAIN]}" \
      --gtf "${GTF[$STRAIN]}" \
      --checkpoint "${CKPT[$GRP]}" \
      --output "$OUTTSV" \
      --batch-size 32 --device cpu
    echo "done $OUTTSV"
  done
done

# convert to npz with cached labels
python - <<'PYEOF'
import numpy as np, pandas as pd, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
PROJ=Path("/media/share/workdir/1610305236/Panfungi/4_model")
OUT=PROJ/"revision/results/rescore"
for strain in ["P4","C4","F6"]:
    pos=[str(g) for g in np.load(PROJ/f"checkpoints/features/{strain}/positive_gene_ids.npy", allow_pickle=True)]
    neg=[str(g) for g in np.load(PROJ/f"checkpoints/features/{strain}/negative_gene_ids.npy", allow_pickle=True)]
    labels={g:1 for g in pos}; labels.update({g:0 for g in neg})
    for grp in ["Candida","Cryptococcus","Filamentous"]:
        f=OUT/f"predict_{strain}_{grp}.tsv"
        if not f.exists(): continue
        df=pd.read_csv(f,sep="\t")
        common=[g for g in df.gene_id if g in labels]
        y=np.array([labels[g] for g in common]); p=df.set_index("gene_id").loc[common,"p_circ"].values
        np.savez(OUT/f"heldout_{strain}_{grp}.npz", gene_id=np.array(common), y_true=y, y_prob=p)
        print(strain,grp,"n=",len(y),"AUROC=",round(roc_auc_score(y,p),4))
PYEOF
