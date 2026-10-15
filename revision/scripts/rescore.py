#!/usr/bin/env python3
"""Re-score mycoCirc checkpoints per gene (Mode A) without touching existing
results. Used by revision analyses:

  * A1b — held-out strains (P4/C4/F6) per-gene scores → bootstrap CIs +
    DeLong vs baselines for the Fig 1A / benchmark table.
  * A7  — species-embedding sensitivity: re-score with every strain_id slot
    (0..n_species-1) to show predictions are insensitive to the unseen-species
    embedding fallback.

Writes per-run npz (gene_id, y_true, y_prob) under revision/results/rescore/.

Usage:
  micromamba run -n pancirc-fungi python revision/scripts/rescore.py \
      --target heldout                       # P4/C4/F6, all 3 group ckpts
  micromamba run -n pancirc-fungi python revision/scripts/rescore.py \
      --target external                      # 4 external species
  micromamba run -n pancirc-fungi python revision/scripts/rescore.py \
      --target strainid --species Aspergillus_cristatus   # A7 sweep
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from data.tsv_parser import parse_strain_registry, StrainEntry, build_strain_index  # noqa: E402
from data.dataset import CircRNAFinetuneDataset, collate_pretrain  # noqa: E402
from data.circ_info_encoding import (  # noqa: E402
    load_circ_info, filter_circ_info, get_positive_genes,
)
from data.negative_sampling import get_negative_genes  # noqa: E402
from data.genome_encoding import GenomeIndexer  # noqa: E402
from utils.gtf_utils import GeneModelIndexer  # noqa: E402
from model.pancirc import PanCircModel  # noqa: E402

OUTDIR = PROJ / "revision" / "results" / "rescore"
OUTDIR.mkdir(parents=True, exist_ok=True)

GROUPS = ["Candida", "Cryptococcus", "Filamentous"]
EXTERNAL_CONFIG = ("/media/share/data5/1610305236/Panfungi/1_data/"
                   "configs/exte_lib_config.tsv")
MERGE_DIR = "/media/share/data5/1610305236/Panfungi/1_data/External/merge"
FEAT_DIR = PROJ / "checkpoints" / "features"

HELDOUT = {"P4": "Candida", "C4": "Cryptococcus", "F6": "Filamentous"}


def external_species_defs():
    defs = {
        "Talaromyces_marneffei": dict(
            strain="PM1",
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
        defs[name] = dict(strain=name, genome=g, gtf=t, circinfo=str(merge))
    return defs


def build_target_dataset(spec, strain_id_override=None, config=None,
                         batch_size=16, use_cached_negatives=True):
    """Build a Mode-A dataset for one species/strain.

    Held-out registry strains MUST use the cached negative ids from
    checkpoints/features/{strain}/ (built unseeded at extraction time —
    that is what the published evaluations used). External species have no
    cache and are sampled with rng_seed=42 (same as eval_external.py).
    """
    gm = GeneModelIndexer(spec["gtf"])
    gi = GenomeIndexer(spec["genome"])
    circ_df = filter_circ_info(load_circ_info(spec["circinfo"]))
    pos = get_positive_genes(circ_df)
    neg_cache = FEAT_DIR / spec["strain"] / "negative_gene_ids.npy"
    if use_cached_negatives and neg_cache.exists():
        neg = [str(g) for g in np.load(neg_cache, allow_pickle=True)]
        print(f"  {spec['strain']}: using {len(neg)} cached negatives "
              f"(matches published evaluation)")
    else:
        neg = get_negative_genes(gm, set(pos.keys()), len(pos),
                                 rng_seed=42)

    entry = StrainEntry(
        group="External", species=spec.get("species", spec["strain"]),
        strain=spec["strain"], genome_path=spec["genome"],
        gtf_path=spec["gtf"], circinfo_path=spec["circinfo"],
        circexp_path=None, geneexp_path=None, is_excluded=False,
    )
    # the registry strain whose species-embedding slot is used; the dataset
    # resolves strain_id from the registry, so we pass a matching entry name
    # when overriding (A7) we post-patch strain_id in the batch instead.
    ds = CircRNAFinetuneDataset(
        entries=[entry],
        gene_models={entry.strain: gm},
        genome_indexers={entry.strain: gi},
        positive_genes={entry.strain: pos},
        negative_genes={entry.strain: neg},
        expression_data=None,
        config=config["data"] if "data" in config else config,
    )
    return ds, entry


def load_model(group, config, n_species, device, ckpt_root=None):
    base = Path(ckpt_root) if ckpt_root else PROJ / "checkpoints" / "finetune"
    ckpt_dir = base / group
    path = None
    for cand in [ckpt_dir / "final_pretrained.pt", ckpt_dir / "final.pt"]:
        if cand.exists():
            path = cand
            break
    state = torch.load(path, map_location=device, weights_only=False)
    model = PanCircModel(config["model"], n_species=n_species)
    missing, unexpected = model.load_state_dict(
        state.get("model_state_dict", state), strict=False)
    # A shape mismatch (e.g. species table size) would be silently skipped by
    # strict=False, leaving layers randomly initialized — fail loudly instead.
    ckpt_key = "species_embedding.lookup.weight"
    for name, p in model.named_parameters():
        if name == ckpt_key:
            src_sd = state.get("model_state_dict", state)
            for cand in (name, name.replace("lookup.", ""),
                         "species_embedding.embedding.weight"):
                if cand in src_sd and src_sd[cand].shape != p.shape:
                    raise RuntimeError(
                        f"checkpoint species table {src_sd[cand].shape} != "
                        f"model {p.shape} — refusing to run with random "
                        f"species embeddings")
    if missing:
        print(f"  [load_model] missing keys: {len(missing)} (e.g. "
              f"{missing[:3]})")
    model.to(device).eval()
    tag = "" if not ckpt_root else f"{Path(ckpt_root).name}_"
    return model, f"{tag}{path.name}"


@torch.no_grad()
def score(model, ds, device, strain_id_override=None, batch_size=16,
          _cache={}):
    """Score a dataset. RAW items are computed once and cached (keyed by
    dataset id) so the A7 strain-id sweep and checkpoint sweeps do not
    recompute junction features."""
    key = id(ds)
    if key not in _cache:
        # raw __getitem__ items (not collated); num_workers avoided because
        # GenomeIndexer holds a pyfaidx handle that cannot be pickled
        _cache[key] = [ds[i] for i in range(len(ds))]
        _i0 = _cache[key][0]
        print(f"  [score-items] item0 gtf[:4]="
              f"{_i0['gtf_features'][:4].tolist()} "
              f"km0[:5]={_i0['donor_kmers'][0][:5].tolist()} "
              f"sid={_i0['strain_id'].item()} "
              f"same_as_ds0={torch.equal(_i0['gtf_features'], ds[0]['gtf_features'])}",
              flush=True)
    items = _cache[key]

    genes, ys, ps = [], [], []
    for i in range(0, len(items), batch_size):
        batch = collate_pretrain(items[i:i + batch_size])
        if i == 0:
            # forward-twice idempotence probe on the SAME batch object
            with torch.no_grad():
                _p1 = torch.sigmoid(model(batch, task="pretrain")
                                    ["gene_logits"])[0].item()
                _p2 = torch.sigmoid(model(batch, task="pretrain")
                                    ["gene_logits"])[0].item()
            _b2 = collate_pretrain(items[i:i + batch_size])
            with torch.no_grad():
                _p3 = torch.sigmoid(model(_b2, task="pretrain")
                                    ["gene_logits"])[0].item()
            print(f"  [idem] fwd1={_p1:.4f} fwd2(same batch)={_p2:.4f} "
                  f"fwd3(recollated)={_p3:.4f}", flush=True)
            # replicate loop-body wrapping exactly, then forward
            _bw = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
            with torch.no_grad():
                _p4 = torch.sigmoid(model(_bw, task="pretrain")
                                    ["gene_logits"])[0].item()
            print(f"  [idem2] after .to(device) rewrap: {_p4:.4f} "
                  f"| batch sid0={batch['strain_id'][0].item()} "
                  f"rewrap sid0={_bw['strain_id'][0].item()}", flush=True)
            # stepwise: exact loop-body forward, one line at a time
            _b3 = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in batch.items()}
            _o3 = model(_b3, task="pretrain")
            _p5 = torch.sigmoid(_o3["gene_logits"].squeeze(-1)).cpu().numpy()
            print(f"  [idem3] exact loop-body replica: {_p5[0]:.4f}",
                  flush=True)
        if i == 0:
            print(f"  [score-debug] model.training={model.training} "
                  f"n_items={len(items)} "
                  f"batch_gtf0={batch['gtf_features'][0][:4].tolist()} "
                  f"batch_km0={batch['donor_kmers'][0][0][:5].tolist()}",
                  flush=True)
        if strain_id_override is not None:
            batch["strain_id"] = torch.full_like(batch["strain_id"],
                                                 strain_id_override)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        out = model(batch, task="pretrain")
        p = torch.sigmoid(out["gene_logits"].squeeze(-1)).cpu().numpy()
        if i == 0:
            print(f"  [loop-fwd] p[0]={p[0]:.4f} p[:3]={p[:3].round(4).tolist()} "
                  f"logits0={out['gene_logits'][0].item():.4f}",
                  flush=True)
        y = batch["is_positive"].cpu().numpy()
        gids = batch["gene_id"]
        # drop dummy items (is_positive == -1), matching evaluate() semantics
        keep = y >= 0
        if keep.all():
            genes.extend(gids)
            ys.append(y)
            ps.append(p)
        else:
            kp = p[keep]
            genes.extend([g for g, k in zip(gids, keep) if k])
            ys.append(y[keep])
            ps.append(kp if kp.ndim else kp.reshape(1))
    return (np.array(genes), np.concatenate(ys), np.concatenate(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True,
                    choices=["heldout", "external", "strainid"])
    ap.add_argument("--species", default=None,
                    help="strainid target: restrict sweep to this species")
    ap.add_argument("--config", default=str(PROJ / "config" / "default.yaml"))
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ckpt-root", default=None,
                    help="alternative checkpoint root, e.g. "
                         "revision/checkpoints/nocircexp (scores variant "
                         "models with the same held-out protocol)")
    args = ap.parse_args()
    device = torch.device(args.device)

    config = yaml.safe_load(open(args.config))
    registry = parse_strain_registry(str(PROJ / "all_lib_model_full.tsv"))
    active = [e for e in registry if not e.is_excluded]
    n_species = len(build_strain_index(active))

    if args.target == "heldout":
        by_strain = {e.strain: e for e in registry}
        for strain, group in HELDOUT.items():
            e = by_strain[strain]
            spec = dict(strain=strain, species=e.species,
                        genome=e.genome_path, gtf=e.gtf_path,
                        circinfo=e.circinfo_path)
            ds, _ = build_target_dataset(spec, config=config)
            # score with own-group checkpoint (benchmark setting) + all others
            for grp in GROUPS:
                model, src = load_model(grp, config, n_species, device,
                                        ckpt_root=args.ckpt_root)
                if strain == "P4" and grp == "Candida":
                    # bisect: score item0 immediately after load, before
                    # score()'s cache/collate path
                    import data.dataset as _dd
                    _b = _dd.collate_pretrain([ds[0]])
                    with torch.no_grad():
                        _p = torch.sigmoid(model(_b, task="pretrain")
                                           ["gene_logits"]).item()
                    print(f"  [bisect] direct item0 after load: {_p:.4f}",
                          flush=True)
                g, y, p = score(model, ds, device, args.batch_size)
                if strain == "P4" and grp == "Candida":
                    # debug fingerprint: first 3 genes + model param hash
                    import hashlib
                    h = hashlib.md5(
                        next(model.parameters()).detach().numpy()
                        .tobytes()).hexdigest()[:8]
                    print(f"  [debug] P4/Candida src={src} param_md5={h} "
                          f"first_scores={p[:3].round(4).tolist()} "
                          f"first_genes={list(g[:3])}", flush=True)
                    it = ds[0]
                    print(f"  [debug] item0 gtf[:6]="
                          f"{it['gtf_features'][:6].tolist()} "
                          f"genome_ctx[0][:4]="
                          f"{it['genome_context'][0][:4].tolist()} "
                          f"donor_kmers[0][:5]="
                          f"{it['donor_kmers'][0][:5].tolist()} "
                          f"strain_id={it['strain_id'].item()}",
                          flush=True)
                out = OUTDIR / f"heldout_{strain}_{grp}_{src}.npz"
                np.savez(out, gene_id=g, y_true=y, y_prob=p)
                print(f"{strain} x {grp} ({src}): n={len(y)} "
                      f"pos={int(y.sum())} → {out.name}")

    elif args.target == "external":
        defs = external_species_defs()
        for name, spec in defs.items():
            spec = dict(spec, species=name)
            ds, _ = build_target_dataset(spec, config=config)
            for grp in GROUPS:
                model, src = load_model(grp, config, n_species, device)
                g, y, p = score(model, ds, device, args.batch_size)
                out = OUTDIR / f"external_{name}_{grp}_{src}.npz"
                np.savez(out, gene_id=g, y_true=y, y_prob=p)
                print(f"{name} x {grp} ({src}): n={len(y)} "
                      f"pos={int(y.sum())} → {out.name}")

    else:  # strainid sweep (A7)
        defs = external_species_defs()
        targets = [args.species] if args.species else list(defs)[:2]
        for name in targets:
            spec = dict(defs[name], species=name)
            ds, _ = build_target_dataset(spec, config=config)
            # fix one checkpoint (group-matched best per external_results)
            best = {"Aspergillus_cristatus": "Candida",
                    "Trichophyton_mentagrophytes": "Candida",
                    "Trichophyton_rubrum": "Filamentous",
                    "Talaromyces_marneffei": "Filamentous"}.get(
                        name, "Filamentous")
            model, src = load_model(best, config, n_species, device)
            for sid in range(n_species):
                g, y, p = score(model, ds, device,
                                strain_id_override=sid,
                                batch_size=args.batch_size)
                out = OUTDIR / f"strainid_{name}_{best}_{sid:02d}.npz"
                np.savez(out, gene_id=g, y_true=y, y_prob=p)
            print(f"{name}: swept {n_species} strain ids ({best} ckpt)")


if __name__ == "__main__":
    main()
