#!/bin/bash
# PM1 (T. marneffei) Mode B with the corrected (nocircexp) checkpoints —
# checks whether the external Mode B conclusion is also artefact-depressed.
set -e
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi
cd /media/share/workdir/1610305236/Panfungi/4_model
GEN=/media/share/resource/reference/talaromyces_marneffei/genome/pm1.soft.fasta
GTF=/media/share/resource/reference/talaromyces_marneffei/annotation/pm1_scallop_final.gtf
GE=/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_gene_count_matrix.csv
for G in Candida Cryptococcus Filamentous; do
  OUT=revision/results/pm1_modeb_${G}_nocircexp.tsv
  [ -s "$OUT" ] && { echo "skip $OUT"; continue; }
  python scripts/predict.py --genome $GEN --gtf $GTF \
    --checkpoint revision/checkpoints/nocircexp/$G/final.pt \
    --genexp $GE --output $OUT --device cpu --batch-size 32
done
python - <<'PY'
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes
from data.negative_sampling import get_negative_genes
from utils.gtf_utils import GeneModelIndexer
gm=GeneModelIndexer("/media/share/resource/reference/talaromyces_marneffei/annotation/pm1_scallop_final.gtf")
pos=list(get_positive_genes(filter_circ_info(load_circ_info(
    "/media/share/data5/1610305236/circTS/3_CIRIquant/circTS_circRNA_info.csv"))))
neg=get_negative_genes(gm,set(pos),len(pos),rng_seed=42)
lab={g:1 for g in pos}; lab.update({g:0 for g in neg})
for G in ("Candida","Cryptococcus","Filamentous"):
    f=f"revision/results/pm1_modeb_{G}_nocircexp.tsv"
    try: df=pd.read_csv(f,sep="\t")
    except FileNotFoundError: continue
    c=[g for g in df.gene_id if g in lab]
    y=np.array([lab[g] for g in c]); p=df.set_index("gene_id").loc[c,"p_circ"].values
    print(f"PM1 Mode B, {G} (nocircexp): AUROC={roc_auc_score(y,p):.4f}  n={len(y)}  p>=0.5: {(p>=0.5).sum()}")
PY
