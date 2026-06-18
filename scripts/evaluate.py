#!/usr/bin/env python3
"""
Evaluate PanCirc-Fungi on held-out test strains.

Usage:
    python scripts/evaluate.py --group Candida \\
        --checkpoint checkpoints/finetune/Candida/best.pt \\
        --config config/default.yaml
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")

from data.tsv_parser import (
    parse_strain_registry, TEST_STRAINS,
)
from model.pancirc import PanCircModel
from utils.metrics import classification_metrics, report_metrics
from utils.logging import setup_logging

logger = logging.getLogger("pancirc.evaluate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True,
                        choices=["Candida", "Cryptococcus", "Filamentous"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ── Load model ──────────────────────────────────────────────────────
    all_entries = parse_strain_registry(config["data"]["tsv_path"])
    active = [e for e in all_entries if not e.is_excluded]
    from data.tsv_parser import build_strain_index
    n_species = len(build_strain_index(active))

    model = PanCircModel(config["model"], n_species=n_species)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # ── Get test strains ────────────────────────────────────────────────
    test_strains = TEST_STRAINS[args.group]
    logger.info(f"Evaluating {args.group} group, test strains: {test_strains}")

    # ── Note: In production, load test data from features directory ─────
    # This is a scaffold — real evaluation uses actual dataset.
    #
    # For a proper evaluation:
    #   1. Load test strain features from checkpoints/features/{strain}/
    #   2. Create CircRNAPretrainDataset for test strains
    #   3. Run model inference in batches
    #   4. Compute metrics

    logger.info("=" * 50)
    logger.info(f"Evaluation scaffold for {args.group}")
    logger.info("=" * 50)
    logger.info("")
    logger.info("To run a full evaluation:")
    logger.info("  1. Ensure features are pre-computed:")
    logger.info("     python scripts/extract_features.py --strains " +
                " ".join(test_strains))
    logger.info("  2. Create test dataset and run inference")
    logger.info("  3. Report metrics with utils.metrics.classification_metrics()")
    logger.info("")


if __name__ == "__main__":
    main()
