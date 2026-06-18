#!/bin/bash
# Ablation: shuffle expression data to verify AUROC ~1.0 is real signal
# Usage: bash scripts/run_ablation_shuffle.sh

source /media/share/home/1610305236/.local/share/mamba/etc/profile.d/mamba.sh
micromamba activate pancirc-fungi

for GROUP in Candida Cryptococcus Filamentous; do
    echo "=== Shuffle ablation: $GROUP ==="
    python3 -c "
import sys, os, yaml, torch, numpy as np
sys.path.insert(0, '.')
import torch.multiprocessing as mp
try: mp.set_start_method('fork', force=True)
except: pass

from data.tsv_parser import parse_strain_registry, TRAIN_STRAINS, TEST_STRAINS
from utils.gtf_utils import GeneModelIndexer
from data.genome_encoding import GenomeIndexer
from data.circ_info_encoding import load_circ_info, filter_circ_info
from data.dataset import CircRNAFinetuneDataset, collate_pretrain
from data.expression_encoding import load_expression_csv, encode_expression_values, pad_to_max_replicates
from model.pancirc import PanCircModel
from utils.metrics import classification_metrics

config = yaml.safe_load(open('config/default.yaml'))
entries = [e for e in parse_strain_registry(config['data']['tsv_path']) if not e.is_excluded]
group = '$GROUP'
train_strains = TRAIN_STRAINS[group]
test_strains = TEST_STRAINS[group]
all_strains = train_strains | test_strains
group_entries = [e for e in entries if e.strain in all_strains]

# Load features
pos_genes, neg_genes = {}, {}
for e in group_entries:
    sd = os.path.join('checkpoints/features', e.strain)
    if os.path.isdir(sd):
        circ_df = filter_circ_info(load_circ_info(e.circinfo_path))
        circ_map = {str(gid): g for gid, g in circ_df.groupby('gene_id')}
        pos_ids = np.load(os.path.join(sd, 'positive_gene_ids.npy'), allow_pickle=True).tolist() if os.path.isfile(os.path.join(sd, 'positive_gene_ids.npy')) else []
        neg_ids = np.load(os.path.join(sd, 'negative_gene_ids.npy'), allow_pickle=True).tolist() if os.path.isfile(os.path.join(sd, 'negative_gene_ids.npy')) else []
        pos_genes[e.strain] = {gid: circ_map.get(gid) for gid in pos_ids}
        neg_genes[e.strain] = neg_ids

gm = {e.strain: GeneModelIndexer(e.gtf_path) for e in group_entries}
gi = {e.strain: GenomeIndexer(e.genome_path) for e in group_entries}

# Load expression data
expr_data = {}
for e in group_entries:
    gene_exp_map = {}
    ge_raw = load_expression_csv(e.geneexp_path)
    if ge_raw is not None:
        ge_vals, ge_ids, _ = encode_expression_values(ge_raw, True, True)
        for i, gid in enumerate(ge_ids):
            gene_exp_map[gid] = pad_to_max_replicates(ge_vals[i], 3)
    expr_data[e.strain] = {'gene_exp': gene_exp_map, 'aligned': {}}

# Load fine-tuned model
ckpt_path = f'checkpoints/finetune/{group}/final.pt'
model = PanCircModel(config['model'])
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Evaluate with normal expression
from torch.utils.data import DataLoader
test_entries = [e for e in group_entries if e.strain in test_strains]
ds_orig = CircRNAFinetuneDataset(test_entries, gm, gi, pos_genes, neg_genes,
    expression_data=expr_data, config=config['data'], max_replicates=3)
loader = DataLoader(ds_orig, batch_size=64, collate_fn=collate_pretrain, shuffle=False)

all_y, all_p = [], []
with torch.no_grad():
    for batch in loader:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(b, task='finetune')
        probs = torch.sigmoid(out['gene_logits'].squeeze(-1))
        all_y.append(b['is_positive'].cpu())
        all_p.append(probs.cpu())

y_true = torch.cat(all_y); y_prob = torch.cat(all_p)
valid = y_true >= 0
m_orig = classification_metrics(y_true[valid].numpy(), y_prob[valid].numpy())
print(f'  Original:         AUROC={m_orig[\"auroc\"]:.4f}')

# Evaluate with shuffled expression
ds_shuf = CircRNAFinetuneDataset(test_entries, gm, gi, pos_genes, neg_genes,
    expression_data=expr_data, config=config['data'], max_replicates=3)
# Shuffle circ_exp in-place
for item in ds_shuf.samples:
    expr = ds_shuf.expression_data.get(item[0].strain, {})
    if 'aligned' in expr:
        for gid in expr['aligned']:
            ce = expr['aligned'][gid].get('circ_exp', np.array([]))
            if ce.size > 0:
                ce_flat = ce.flatten()
                np.random.shuffle(ce_flat)
                expr['aligned'][gid]['circ_exp'] = ce_flat.reshape(ce.shape)

loader_shuf = DataLoader(ds_shuf, batch_size=64, collate_fn=collate_pretrain, shuffle=False)
all_y, all_p = [], []
with torch.no_grad():
    for batch in loader_shuf:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(b, task='finetune')
        probs = torch.sigmoid(out['gene_logits'].squeeze(-1))
        all_y.append(b['is_positive'].cpu())
        all_p.append(probs.cpu())

y_true = torch.cat(all_y); y_prob = torch.cat(all_p)
valid = y_true >= 0
m_shuf = classification_metrics(y_true[valid].numpy(), y_prob[valid].numpy())
print(f'  Shuffled (circ):  AUROC={m_shuf[\"auroc\"]:.4f}')

# Evaluate with expression zeroed out
ds_zero = CircRNAFinetuneDataset(test_entries, gm, gi, pos_genes, neg_genes,
    expression_data=None, config=config['data'], max_replicates=3)
loader_zero = DataLoader(ds_zero, batch_size=64, collate_fn=collate_pretrain, shuffle=False)
all_y, all_p = [], []
with torch.no_grad():
    for batch in loader_zero:
        b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(b, task='finetune')
        probs = torch.sigmoid(out['gene_logits'].squeeze(-1))
        all_y.append(b['is_positive'].cpu())
        all_p.append(probs.cpu())

y_true = torch.cat(all_y); y_prob = torch.cat(all_p)
valid = y_true >= 0
m_zero = classification_metrics(y_true[valid].numpy(), y_prob[valid].numpy())
print(f'  Zeroed (no expr): AUROC={m_zero[\"auroc\"]:.4f}')
"
done
