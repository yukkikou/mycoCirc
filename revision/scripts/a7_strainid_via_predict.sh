#!/bin/bash
# A7: species-embedding sensitivity via PRODUCTION predict.py --strain-id.
# For each of 2 external species (best checkpoint), predict genome-wide with
# every strain-id slot 0..20, then AUROC on the balanced seed-42 eval set.
set -e
source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /media/share/workdir/1610305236/Panfungi/4_model
OUT=revision/results/rescore/strainid
mkdir -p $OUT

declare -A SPECIES=(
  [Aspergillus_cristatus]="/media/share/workdir/1610305236/Panfungi/circ/External/genome/Aspergillus_cristatus_gca_001717485.ASM171748v1.dna.toplevel.fa /media/share/workdir/1610305236/Panfungi/circ/External/gtf/Aspergillus_cristatus_gca_001717485.ASM171748v1.60.gtf Candida"
  [Trichophyton_rubrum]="/media/share/workdir/1610305236/Panfungi/circ/External/genome/Trichophyton_rubrum_gca_001651445.CMCC_F_T1i.dna.toplevel.fa /media/share/workdir/1610305236/Panfungi/circ/External/gtf/Trichophyton_rubrum_gca_001651445.CMCC_F_T1i.60.gtf Filamentous"
)
declare -A CKPT=(
  [Candida]=checkpoints/finetune/Candida/final.pt
  [Filamentous]=checkpoints/finetune/Filamentous/final_pretrained.pt
)

for SP in "${!SPECIES[@]}"; do
  read -r GENOME GTF GRP <<< "${SPECIES[$SP]}"
  for SID in $(seq 0 20); do
    TSV=$OUT/${SP}_sid${SID}.tsv
    [ -s "$TSV" ] && { echo "skip $TSV"; continue; }
    python scripts/predict.py --genome "$GENOME" --gtf "$GTF" \
      --checkpoint "${CKPT[$GRP]}" --output "$TSV" \
      --strain-id "$SID" --batch-size 32 --device cpu
  done
  echo "swept $SP"
done

python - <<'PYEOF'
import numpy as np, pandas as pd, sys
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/media/share/workdir/1610305236/Panfungi/4_model")
PROJ=Path("/media/share/workdir/1610305236/Panfungi/4_model")
OUT=PROJ/"revision/results/rescore/strainid"
MERGE="/media/share/data5/1610305236/Panfungi/1_data/External/merge"
from data.circ_info_encoding import load_circ_info, filter_circ_info, get_positive_genes
from data.negative_sampling import get_negative_genes
from utils.gtf_utils import GeneModelIndexer
rows=[]
for sp in ["Aspergillus_cristatus","Trichophyton_rubrum"]:
    info=f"{MERGE}/{sp}/{sp}_circRNA_info.csv"
    gtf=[l.split('\t')[5] for l in open("/media/share/data5/1610305236/Panfungi/1_data/configs/exte_lib_config.tsv") if l.startswith(sp)][0]
    gm=GeneModelIndexer(gtf)
    pos=list(get_positive_genes(filter_circ_info(load_circ_info(info))))
    neg=get_negative_genes(gm,set(pos),len(pos),rng_seed=42)
    labels={g:1 for g in pos}; labels.update({g:0 for g in neg})
    for sid in range(21):
        f=OUT/f"{sp}_sid{sid}.tsv"
        if not f.exists(): continue
        df=pd.read_csv(f,sep="\t")
        common=[g for g in df.gene_id if g in labels]
        y=np.array([labels[g] for g in common]); p=df.set_index("gene_id").loc[common,"p_circ"].values
        rows.append(dict(species=sp,strain_id=sid,auroc=round(float(roc_auc_score(y,p)),5),n=len(y)))
df=pd.DataFrame(rows)
df.to_csv(PROJ/"revision/results/a7_strainid_sensitivity.csv",index=False)
if len(df):
    for sp,g in df.groupby("species"):
        print(sp, "min",g.auroc.min(),"max",g.auroc.max(),"range",round(g.auroc.max()-g.auroc.min(),5))
PYEOF
