#!/usr/bin/env python3
"""A6-ii: fine-tune with CircExp ZEROED DURING TRAINING (train/eval symmetric
with Mode B inference), to directly quantify the train-inference mismatch
reviewer 1 asks about (R1-M2).

Runs the ORIGINAL train/finetune.py unmodified, monkey-patching only
_load_expression_data so the aligned CircExp channel is emptied (the dataset
then feeds zero vectors for circ_exp — identical to Mode B inference). All
checkpoints/logs go to revision/, never touching checkpoints/ or logs/.

Usage (same CLI as train/finetune.py):
  python revision/scripts/finetune_nocircexp.py --group Candida \
      --pretrained checkpoints/pretrain/best.pt --device cuda
"""

import importlib.util
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]

spec = importlib.util.spec_from_file_location(
    "orig_finetune", str(PROJ / "train" / "finetune.py"))
ft = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(PROJ))
spec.loader.exec_module(ft)

_orig_load = ft._load_expression_data


def _load_expression_zeroed_circexp(entries, max_replicates=3):
    """Original loader + CircExp channel emptied (=> zero vectors)."""
    ed = _orig_load(entries, max_replicates)
    n_zeroed = 0
    for strain, d in ed.items():
        n_zeroed += len(d.get("aligned", {}))
        d["aligned"] = {}
    ft.logger.info(
        f"  [A6-ii] CircExp channel zeroed for training/validation "
        f"({n_zeroed} aligned positive-gene entries dropped; GeneExp kept)")
    return ed


ft._load_expression_data = _load_expression_zeroed_circexp

if __name__ == "__main__":
    # same argparse as the original main()
    ft.main()
