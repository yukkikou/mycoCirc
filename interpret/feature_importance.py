"""
Feature importance analysis using Integrated Gradients.

Computes per-feature attribution scores for GTFEncoder inputs
to show which gene-level features drive circRNA prediction.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

GTF_FEATURE_NAMES = [
    "exon_count", "log_exon_length", "log_intron_length",
    "log_cds_length", "exon_density", "relative_gene_length",
    "is_multi_exon", "has_cds", "biotype",  # +8 reserved
] + [f"reserved_{i}" for i in range(8)]


def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    target_class: int = 1,
) -> np.ndarray:
    """Compute Integrated Gradients for GTF input features.

    Parameters
    ----------
    model : nn.Module
        PanCircModel.
    input_tensor : torch.Tensor
        (batch, n_features) GTF features.
    baseline : torch.Tensor or None
        (batch, n_features) zero baseline if None.
    n_steps : int
        Number of integration steps (default: 50).
    target_class : int
        Target output class (1 = circRNA positive, 0 = negative).

    Returns
    -------
    np.ndarray (batch, n_features) attribution scores.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    model.eval()
    input_tensor = input_tensor.requires_grad_(True)
    scaled_inputs = [
        baseline + (float(i) / n_steps) * (input_tensor - baseline)
        for i in range(n_steps + 1)
    ]

    grads = []
    for scaled_input in scaled_inputs:
        with torch.enable_grad():
            scaled_input = scaled_input.clone().detach().requires_grad_(True)

            # Forward through GTF encoder
            gtf_output = model.gtf_encoder(scaled_input)
            # Sum as proxy for total effect on prediction
            output = gtf_output.sum()

            model.zero_grad()
            output.backward(retain_graph=True)

            grad = scaled_input.grad.detach().cpu().numpy()
            grads.append(grad)

    # Average gradients (trapezoidal rule)
    grads = np.stack(grads)
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = avg_grads.mean(axis=0)

    # Integrated gradients = (input - baseline) * avg_grads
    ig = (input_tensor.detach().cpu().numpy() - baseline.cpu().numpy()) * avg_grads

    return ig


def aggregate_importances(
    ig_scores: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Aggregate per-feature IG scores across samples.

    Returns dict mapping feature name -> mean |score|.
    """
    if feature_names is None:
        feature_names = GTF_FEATURE_NAMES

    n_features = ig_scores.shape[1]
    agg = {}
    for i in range(min(n_features, len(feature_names))):
        agg[feature_names[i]] = float(np.abs(ig_scores[:, i]).mean())

    return dict(sorted(agg.items(), key=lambda x: -x[1]))


def print_feature_ranking(importances: Dict[str, float], n_top: int = 10):
    """Pretty-print top feature importances."""
    print(f"\n{'='*50}")
    print(f"Top {min(n_top, len(importances))} GTF Features for circRNA Prediction")
    print(f"{'='*50}")
    for i, (name, score) in enumerate(
        list(importances.items())[:n_top], 1
    ):
        direction = "↑ promotes circRNA" if score > 0 else "↓ inhibits circRNA"
        print(f"  {i:2d}. {name:25s} {score:+.4f}  {direction}")
    print(f"{'='*50}")
